"""Tests for hit ledgers, eviction rounds, invariants and audit reports.

CPU-only: every case uses ``RadixCache.create_simulated`` with either no
allocator or a small fake allocator. No model is downloaded and no network is
used.
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd")

import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import unittest.mock
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_audit import (
    CacheEvictError,
    HitKind,
    HitLedger,
    HitStats,
    StopReason,
    hit_kind,
    namespace_of,
)
from sglang.srt.mem_cache.events import BlockRemoved
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.test.test_utils import CustomTestCase


def _key(tokens, extra_key=None, cache_salt=None):
    return RadixKey(
        array("q", list(tokens)),
        extra_key=extra_key,
        cache_salt=cache_salt,
    )


def _insert(cache, tokens, extra_key=None, cache_salt=None):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams

    return cache.insert(
        InsertParams(key=_key(tokens, extra_key, cache_salt))
    )


def _match(cache, tokens, extra_key=None, cache_salt=None):
    return cache.match_prefix(
        MatchPrefixParams(key=_key(tokens, extra_key, cache_salt))
    )


class FakeAllocator:
    """Minimal allocator surface RadixCache reads; records free order."""

    def __init__(self, page_size=1, fail_on_call=None):
        self.page_size = page_size
        self.device = torch.device("cpu")
        self.fail_on_call = fail_on_call
        self.free_calls = []
        self.call_count = 0

    def free_segment(self, value, start_pos=0):
        self.call_count += 1
        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise RuntimeError(f"fake failure on free_segment #{self.call_count}")
        self.free_calls.append((value, start_pos))


class TestNamespaceAndKind(CustomTestCase):
    def test_namespace_normalizes_only_salt(self):
        self.assertEqual(namespace_of("lora-a", None), ("lora-a", None))
        self.assertEqual(namespace_of("lora-a", ""), ("lora-a", None))
        self.assertEqual(namespace_of(None, "salt"), (None, "salt"))
        # extra_key is preserved verbatim, even when empty.
        self.assertEqual(namespace_of("", ""), ("", None))

    def test_hit_kind_tiers(self):
        self.assertIs(hit_kind(8, 0), HitKind.MISS)
        self.assertIs(hit_kind(5, 4), HitKind.PARTIAL)
        self.assertIs(hit_kind(8, 8), HitKind.FULL)

    def test_stats_derived_properties(self):
        stats = HitStats(3, 20, 2, 16, 1)
        self.assertEqual(stats.misses, 1)
        self.assertAlmostEqual(stats.hit_rate, 0.8)
        self.assertEqual(HitStats().hit_rate, 0.0)
        with self.assertRaises(AttributeError):
            stats.lookups = 9


class TestHitLedger(CustomTestCase):
    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            HitLedger(capacity=-1)

    def test_overflow_drops_oldest_and_keeps_aggregate(self):
        ledger = HitLedger(capacity=2)
        for i in range(3):
            ledger.record((None, None), 1, 1)
        taken = ledger.take()
        self.assertEqual(len(taken), 2)
        # Oldest (seq 1) dropped; retained seqs are 2 and 3.
        self.assertEqual([r.seq for r in taken], [2, 3])
        self.assertEqual(ledger.dropped, 1)
        self.assertEqual(ledger.overall.lookups, 3)
        # take drains details only; a second take is empty.
        self.assertEqual(ledger.take(), ())

    def test_capacity_zero_keeps_aggregate_only(self):
        ledger = HitLedger(capacity=0)
        record = ledger.record(("z", None), 3, 3)
        self.assertIsNotNone(record)
        self.assertEqual(ledger.take(), ())
        self.assertEqual(ledger.overall.lookups, 1)
        self.assertEqual(ledger.dropped, 1)

    def test_zero_lookup_returns_none_and_changes_nothing(self):
        ledger = HitLedger()
        self.assertIsNone(ledger.record((None, None), 0, 0))
        self.assertEqual(ledger.overall, HitStats())
        self.assertEqual(ledger.take(), ())
        with self.assertRaises(ValueError):
            ledger.record((None, None), -1, 0)

    def test_seq_continues_after_clear(self):
        ledger = HitLedger()
        first = ledger.record((None, None), 2, 2)
        self.assertEqual(first.seq, 1)
        ledger.clear()
        self.assertEqual(ledger.overall, HitStats())
        self.assertEqual(ledger.take(), ())
        second = ledger.record((None, None), 2, 2)
        self.assertEqual(second.seq, 2)

    def test_take_on_empty_is_empty_tuple(self):
        self.assertEqual(HitLedger().take(), ())

    def test_stats_are_keyed_by_namespace_of_tuples(self):
        ledger = HitLedger()
        ledger.record(namespace_of("a", ""), 4, 4)
        stats = ledger.stats
        self.assertIn(("a", None), stats)
        self.assertNotIn(("a", ""), stats)

    def test_concurrent_records(self):
        ledger = HitLedger(capacity=4096)

        def worker():
            for _ in range(500):
                ledger.record((None, None), 1, 1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(ledger.overall.lookups, 4000)
        retained = len(ledger.take())
        self.assertEqual(retained + int(ledger.dropped), 4000)


class TestHitRecordsFromMatch(CustomTestCase):
    def test_full_hit(self):
        cache = RadixCache.create_simulated()
        _insert(cache, range(1, 9))
        result = _match(cache, range(1, 9))

        records = cache.take_hit_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.lookup_tokens, 8)
        self.assertEqual(record.matched_tokens, 8)
        self.assertIs(record.kind, HitKind.FULL)
        self.assertEqual(record.splits, 0)
        self.assertEqual(record.page_size, 1)
        self.assertEqual(record.namespace, (None, None))
        self.assertEqual(record.last_node_id, result.last_device_node.id)

    def test_partial_hit_reports_one_split(self):
        cache = RadixCache.create_simulated()
        _insert(cache, range(1, 9))
        result = _match(cache, [1, 2, 3, 4, 99])

        record, = cache.take_hit_records()
        self.assertEqual(record.lookup_tokens, 5)
        self.assertEqual(record.matched_tokens, 4)
        self.assertIs(record.kind, HitKind.PARTIAL)
        self.assertEqual(record.splits, 1)
        self.assertEqual(record.last_node_id, result.last_device_node.id)

    def test_miss(self):
        cache = RadixCache.create_simulated()
        _insert(cache, range(1, 9))
        result = _match(cache, [9])

        record, = cache.take_hit_records()
        self.assertEqual(record.lookup_tokens, 1)
        self.assertEqual(record.matched_tokens, 0)
        self.assertIs(record.kind, HitKind.MISS)
        self.assertEqual(record.splits, 0)
        self.assertEqual(record.last_node_id, cache.root_node.id)
        self.assertEqual(record.last_node_id, result.last_device_node.id)

    def test_page_alignment_changes_lookup_len(self):
        cache = RadixCache.create_simulated(page_size=4)
        # 3 tokens align down to zero pages: no record at all.
        _match(cache, [1, 2, 3])
        self.assertEqual(cache.take_hit_records(), ())

        _match(cache, [1, 2, 3, 4, 5, 6])
        record, = cache.take_hit_records()
        self.assertEqual(record.lookup_tokens, 4)
        self.assertEqual(record.page_size, 4)

    def test_touch_writes_nothing_and_keeps_details(self):
        cache = RadixCache.create_simulated()
        _insert(cache, range(1, 5))

        cache.touch_prefix(_key([1, 2, 3, 4]))
        self.assertEqual(cache.take_hit_records(), ())

        _match(cache, [1, 2, 3, 4])
        cache.touch_prefix(_key([1, 2]))
        records = cache.take_hit_records()
        self.assertEqual(len(records), 1)

    def test_raising_match_leaves_no_record(self):
        cache = RadixCache.create_simulated()
        before = cache.hit_ledger.overall
        with unittest.mock.patch.object(
            cache, "_match_prefix_helper", side_effect=RuntimeError("boom")
        ), self.assertRaises(RuntimeError):
            _match(cache, [1, 2, 3])
        self.assertEqual(cache.take_hit_records(), ())
        self.assertEqual(cache.hit_ledger.overall, before)

    def test_namespace_stats_separate_and_coexist(self):
        cache = RadixCache.create_simulated()
        for extra_key in ("lora-a", "lora-b"):
            _insert(cache, [1, 2, 3, 4], extra_key=extra_key)
            _match(cache, [1, 2, 3, 4], extra_key=extra_key)

        stats = cache.hit_stats()
        self.assertEqual(
            set(stats.keys()), {("lora-a", None), ("lora-b", None)}
        )
        overall = cache.hit_ledger.overall
        self.assertEqual(
            overall.hit_tokens,
            sum(s.hit_tokens for s in stats.values()),
        )
        self.assertEqual(overall.lookups, 2)
        self.assertAlmostEqual(
            cache.hit_rate(),
            overall.hit_tokens / overall.lookup_tokens,
        )

    def test_reset_keeps_seq_monotonic(self):
        cache = RadixCache.create_simulated()
        _match(cache, [1])
        first_seq = cache.take_hit_records()[0].seq
        cache.reset()
        _match(cache, [2])
        second_seq = cache.take_hit_records()[0].seq
        self.assertEqual(second_seq, first_seq + 1)
        self.assertEqual(cache.hit_ledger.overall.lookups, 1)

    def test_ledger_capacity_param_wiring(self):
        self.assertEqual(
            RadixCache.create_simulated().hit_ledger.capacity, 1024
        )
        cache = RadixCache.create_simulated(hit_ledger_capacity=2)
        self.assertEqual(cache.hit_ledger.capacity, 2)


def _two_tail_tree(fail_on_call=None, use_allocator=True):
    allocator = FakeAllocator(fail_on_call=fail_on_call) if use_allocator else None
    cache = RadixCache.create_simulated(mock_allocator=allocator)
    _insert(cache, range(1, 9))
    _insert(cache, list(range(1, 9)) + [100])
    _insert(cache, list(range(1, 9)) + [200])
    leaves = sorted(cache.evictable_leaves, key=lambda n: n.id)
    assert len(leaves) == 2
    frozen = 1_000_000.0
    for leaf in leaves:
        leaf.last_access_time = frozen
    return cache, allocator, leaves


class TestEvictRound(CustomTestCase):
    def test_deterministic_id_order_and_namespace_accounting(self):
        old_counter = TreeNode.counter
        try:

            def build():
                TreeNode.counter = 0
                return _two_tail_tree()

            cache1, alloc1, leaves1 = build()
            round1 = cache1.evict_round(2)

            cache2, _alloc2, _leaves2 = build()
            round2 = cache2.evict_round(2)
        finally:
            TreeNode.counter = old_counter

        expected_ids = tuple(n.id for n in leaves1)
        self.assertEqual(round1.victims, expected_ids)
        self.assertEqual(round1.victims, tuple(sorted(round1.victims)))
        # Rebuilding the same tree replays the exact same victim ids.
        self.assertEqual(round1.victims, round2.victims)

        self.assertEqual(round1.requested, 2)
        self.assertEqual(round1.evicted, 2)
        self.assertEqual(round1.shortfall, 0)
        self.assertIs(round1.reason, StopReason.TARGET_MET)
        self.assertEqual(round1.allocator_calls, 2)
        self.assertEqual(len(round1.victims), round1.allocator_calls)
        self.assertEqual(round1.by_namespace, {(None, None): 2})
        # Exactly one free per victim, start_pos=0, in id order.
        self.assertEqual(len(alloc1.free_calls), 2)
        self.assertTrue(all(pos == 0 for _, pos in alloc1.free_calls))
        cache1.check_invariants()

    def test_exhausted_reports_shortfall(self):
        allocator = FakeAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        _insert(cache, [1, 2])

        report = cache.evict_round(5)
        self.assertIs(report.reason, StopReason.EXHAUSTED)
        self.assertEqual(report.evicted, 2)
        self.assertEqual(report.shortfall, 3)
        self.assertEqual(report.shortfall + report.evicted, report.requested)
        self.assertEqual(report.allocator_calls, 1)
        self.assertEqual(len(report.victims), 1)

        # Once empty, evict returns zero so reclaim_jointly terminates.
        result = cache.evict(EvictParams(num_tokens=10))
        self.assertEqual(result.num_tokens_evicted, 0)

    def test_requested_zero_is_empty_round_without_allocator_calls(self):
        allocator = FakeAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        _insert(cache, [1])
        report = cache.evict_round(0)
        self.assertEqual(report.victims, ())
        self.assertEqual(report.evicted, 0)
        self.assertEqual(report.shortfall, 0)
        self.assertEqual(report.allocator_calls, 0)
        self.assertIs(report.reason, StopReason.TARGET_MET)
        self.assertEqual(allocator.call_count, 0)

    def test_negative_requested_rejected(self):
        cache = RadixCache.create_simulated(mock_allocator=FakeAllocator())
        with self.assertRaises(ValueError):
            cache.evict_round(-1)

    def test_none_allocator_detaches_with_zero_calls(self):
        cache, _allocator, leaves = _two_tail_tree(use_allocator=False)
        report = cache.evict_round(2)
        self.assertEqual(report.allocator_calls, 0)
        self.assertEqual(report.evicted, 2)
        self.assertEqual(set(report.victims), {n.id for n in leaves})
        self.assertEqual(cache.total_size(), 8)  # shared prefix remains
        cache.check_invariants()

        # A later round promotes and reclaims the now-leaf shared prefix.
        second = cache.evict_round(10)
        self.assertEqual(second.allocator_calls, 0)
        self.assertEqual(second.evicted, 8)
        self.assertEqual(cache.total_size(), 0)
        cache.check_invariants()

    def test_disabled_returns_disabled_round_and_zero_lock_deltas(self):
        cache = RadixCache.create_simulated(disable=True)
        _match(cache, [1, 2, 3])
        self.assertEqual(cache.take_hit_records(), ())

        report = cache.evict_round(4)
        self.assertIs(report.reason, StopReason.DISABLED)
        self.assertEqual(report.evicted, 0)
        self.assertEqual(report.allocator_calls, 0)
        self.assertEqual(report.victims, ())

        self.assertEqual(
            cache.inc_lock_ref(cache.root_node).delta, 0
        )
        self.assertEqual(
            cache.dec_lock_ref(cache.root_node).delta, 0
        )

    def test_allocator_failure_is_transactional(self):
        # Hash computation needs a native extension unrelated to eviction;
        # stub it so this test stays hermetic on CPU boxes without ninja.
        import sglang.srt.mem_cache.events as events_mod

        def _fake_hash_values(node, page_size):
            count = max(1, (len(node.key) + page_size - 1) // page_size)
            return ["0" * 64 for _ in range(count)]

        with unittest.mock.patch.object(
            events_mod, "compute_node_hash_values", _fake_hash_values
        ):
            allocator = FakeAllocator(fail_on_call=2)
            cache = RadixCache.create_simulated(
                mock_allocator=allocator,
                enable_kv_cache_events=True,
            )
            _insert(cache, range(1, 9))
            _insert(cache, list(range(1, 9)) + [100])
            _insert(cache, list(range(1, 9)) + [200])
            leaves = sorted(cache.evictable_leaves, key=lambda n: n.id)
            for leaf in leaves:
                leaf.last_access_time = 1.0

            # Drain AllBlocksCleared / BlockStored before eviction.
            cache.take_events()

            with self.assertRaises(CacheEvictError) as ctx:
                cache.evict_round(2)

        error = ctx.exception
        first, second = leaves
        self.assertEqual(error.report.evicted, len(first.key))
        self.assertEqual(error.node_id, second.id)
        self.assertIsInstance(error.__cause__, RuntimeError)
        # Completed first victim is accounted; failed call is not.
        self.assertEqual(error.report.victims, (first.id,))
        self.assertEqual(error.report.allocator_calls, 1)
        # Failed node was never detached and stays evictable.
        self.assertIn(second, cache.evictable_leaves)
        self.assertNotIn(first, cache.evictable_leaves)
        cache.check_invariants()

        events = cache.take_events()
        removed = [e for e in events if isinstance(e, BlockRemoved)]
        self.assertEqual(len(removed), 1)

    def test_randomized_invariants(self):
        rng = random.Random(20261126)
        allocator = FakeAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        held = []

        def random_tokens():
            return [rng.randint(1, 6) for _ in range(rng.randint(1, 12))]

        def any_node():
            nodes = []
            stack = [cache.root_node]
            while stack:
                node = stack.pop()
                nodes.append(node)
                stack.extend(node.children.values())
            live = [n for n in nodes if n is not cache.root_node]
            return rng.choice(live) if live else None

        for _ in range(500):
            op = rng.random()
            if op < 0.55:
                _insert(cache, random_tokens())
            elif op < 0.70:
                node = any_node()
                if node is not None:
                    cache.inc_lock_ref(node)
                    held.append(node)
            elif op < 0.85:
                if held:
                    cache.dec_lock_ref(held.pop())
            else:
                report = cache.evict_round(rng.randint(0, 20))
                if allocator is not None:
                    self.assertEqual(
                        len(report.victims), report.allocator_calls
                    )
            cache.check_invariants()
            self.assertEqual(
                cache.evictable_size() + cache.protected_size(),
                cache.total_size(),
            )

        while held:
            cache.dec_lock_ref(held.pop())
        cache.check_invariants()

    def test_deep_chain_is_iterative_and_fast(self):
        allocator = FakeAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        prefix = []
        for token in range(1, 3001):
            prefix.append(token)
            _insert(cache, prefix)

        cache.check_invariants()
        start = time.perf_counter()
        report = cache.evict_round(3000)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(report.victims), 3000)
        self.assertEqual(report.allocator_calls, 3000)
        self.assertLess(elapsed, 2.0, f"eviction took {elapsed:.3f}s")
        cache.check_invariants()


class TestBasePrefixCacheDefaults(CustomTestCase):
    def test_base_has_no_ledger_and_returns_empty(self):
        self.assertIsNone(BasePrefixCache.hit_ledger)
        no_ledger = SimpleNamespace(hit_ledger=None)
        self.assertEqual(BasePrefixCache.take_hit_records(no_ledger), ())

        ledger = HitLedger()
        ledger.record((None, None), 1, 1)
        with_ledger = SimpleNamespace(hit_ledger=ledger)
        self.assertEqual(len(BasePrefixCache.take_hit_records(with_ledger)), 1)

    def test_cache_audit_module_loads_without_torch(self):
        radix_file = RadixCache.__module__
        module = sys.modules[radix_file]
        audit_path = os.path.join(
            os.path.dirname(os.path.abspath(module.__file__)),
            "cache_audit.py",
        )
        code = (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('ca_standalone', {audit_path!r}); "
            "m = importlib.util.module_from_spec(spec); "
            "sys.modules['ca_standalone'] = m; "
            "spec.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )


def test_evict_log_line_matches_report(caplog):
    old_counter = TreeNode.counter
    try:
        TreeNode.counter = 0
        cache, _allocator, _leaves = _two_tail_tree()
    finally:
        TreeNode.counter = old_counter

    caplog.set_level(logging.INFO)
    report = cache.evict_round(2)

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("radix-evict: requested=")
    ]
    assert len(lines) == 1, lines
    match = re.search(
        r"requested=(-?\d+) evicted=(\d+) shortfall=(\d+) reason=(\w+)",
        lines[0],
    )
    assert match is not None, lines[0]
    assert int(match.group(1)) == report.requested
    assert int(match.group(2)) == report.evicted
    assert int(match.group(3)) == report.shortfall
    assert match.group(4) == report.reason.name


def test_disabled_round_emits_no_evict_log(caplog):
    cache = RadixCache.create_simulated(disable=True)
    caplog.set_level(logging.INFO)
    cache.evict_round(5)
    assert not [
        record.getMessage()
        for record in caplog.records
        if "radix-evict: requested=" in record.getMessage()
    ]


if __name__ == "__main__":
    import unittest

    unittest.main()
