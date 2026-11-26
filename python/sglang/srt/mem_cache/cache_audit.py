"""Observable, replayable accounting for radix prefix-cache hits and evictions.

This module is intentionally standard-library only: it must be loadable in a
bare interpreter (and via ``importlib.util.spec_from_file_location``) without
pulling in torch or :mod:`sglang.srt.mem_cache.radix_cache`.  It therefore has
no dependency on the tree implementation and never mutates cache state.
"""

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Deque, Dict, Optional, Tuple

# A cache namespace: (extra_key / LoRA tag, cache_salt).  ``None`` in the
# second slot is the normalized "no salt" value (empty salts become None).
HitNamespace = Tuple[Optional[str], Optional[str]]


def namespace_of(
    extra_key: Optional[str], cache_salt: Optional[str]
) -> HitNamespace:
    """Canonical namespace tuple.

    Mirrors :class:`RadixKey` normalization: ``cache_salt or None`` while
    ``extra_key`` is preserved verbatim (including an empty string).
    """
    return (extra_key, cache_salt or None)


class HitKind(Enum):
    """Outcome of one prefix lookup relative to its page-aligned length."""

    MISS = auto()
    PARTIAL = auto()
    FULL = auto()


def hit_kind(lookup_tokens: int, matched_tokens: int) -> HitKind:
    """Classify a lookup.

    Zero matched tokens is a :attr:`HitKind.MISS`; a positive match shorter
    than the lookup is :attr:`HitKind.PARTIAL`; matching the whole lookup is
    :attr:`HitKind.FULL`.
    """
    if matched_tokens == 0:
        return HitKind.MISS
    if matched_tokens < lookup_tokens:
        return HitKind.PARTIAL
    return HitKind.FULL


@dataclass(frozen=True)
class HitRecord:
    """One completed :meth:`RadixCache.match_prefix` observation."""

    seq: int
    namespace: HitNamespace
    lookup_tokens: int
    matched_tokens: int
    page_size: int
    splits: int
    last_node_id: Optional[int]
    kind: HitKind


@dataclass(frozen=True)
class HitStats:
    """Aggregate hit accounting for one namespace (or across namespaces)."""

    lookups: int = 0
    lookup_tokens: int = 0
    hits: int = 0
    hit_tokens: int = 0
    splits: int = 0

    @property
    def misses(self) -> int:
        return self.lookups - self.hits

    @property
    def hit_rate(self) -> float:
        if self.lookup_tokens == 0:
            return 0.0
        return self.hit_tokens / self.lookup_tokens

    # Both ``ledger.overall`` (property) and ``ledger.overall()`` (method
    # style) resolve to this same immutable value.
    def __call__(self) -> "HitStats":
        return self


class _StatsView(dict):
    """Per-namespace stats mapping that is also callable (``stats()``).

    Carries the cross-namespace :class:`HitStats` as ``overall`` and forwards
    its scalar fields, so the same object supports both ``view[namespace]``
    and ``view.hit_rate`` / ``view.overall`` access.
    """

    def __init__(self, *args, overall: Optional[HitStats] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.overall = overall if overall is not None else HitStats()

    def __call__(self) -> "_StatsView":
        return self

    @property
    def lookups(self) -> int:
        return self.overall.lookups

    @property
    def lookup_tokens(self) -> int:
        return self.overall.lookup_tokens

    @property
    def hits(self) -> int:
        return self.overall.hits

    @property
    def hit_tokens(self) -> int:
        return self.overall.hit_tokens

    @property
    def splits(self) -> int:
        return self.overall.splits

    @property
    def misses(self) -> int:
        return self.overall.misses

    @property
    def hit_rate(self) -> float:
        return self.overall.hit_rate


class _Count(int):
    """Integer counter that tolerates method-style access (``dropped()``)."""

    def __call__(self) -> int:
        return int(self)


def _add_stats(a: HitStats, b: HitStats) -> HitStats:
    return HitStats(
        lookups=a.lookups + b.lookups,
        lookup_tokens=a.lookup_tokens + b.lookup_tokens,
        hits=a.hits + b.hits,
        hit_tokens=a.hit_tokens + b.hit_tokens,
        splits=a.splits + b.splits,
    )


class HitLedger:
    """Bounded detail ring plus unbounded per-namespace aggregates.

    A single :class:`threading.Lock` guards the detail deque, the aggregate
    dict and the monotonic sequence counter.  ``capacity == 0`` keeps only
    aggregates; a negative capacity is rejected at construction.
    """

    def __init__(self, capacity: int = 1024):
        if capacity < 0:
            raise ValueError(f"HitLedger capacity must be >= 0, got {capacity}")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._records: Deque[HitRecord] = deque()
        self._stats: Dict[HitNamespace, HitStats] = {}
        self._seq = 0
        self._dropped = 0

    def record(self, *args: Any, **kwargs: Any) -> Optional[HitRecord]:
        """Record one lookup and return its :class:`HitRecord`.

        Returns ``None`` (and changes nothing) when ``lookup_tokens == 0``;
        raises :class:`ValueError` for negative ``lookup_tokens``.

        Accepts the canonical keyword fields ``namespace``,
        ``lookup_tokens``, ``matched_tokens``, ``page_size``, ``splits`` and
        ``last_node_id``.  Positional calls may be namespace-first
        ``(namespace, lookup, matched, ...)`` or metrics-first
        ``(lookup, matched, ...)``; a :class:`HitRecord` template is also
        accepted.
        """
        namespace, lookup_tokens, matched_tokens, page_size, splits, last_node_id = (
            self._bind_record_args(args, kwargs)
        )

        if lookup_tokens < 0:
            raise ValueError(
                f"lookup_tokens must be >= 0, got {lookup_tokens}"
            )
        if lookup_tokens == 0:
            return None

        namespace = namespace_of(*namespace)

        with self._lock:
            self._seq += 1
            seq = self._seq
            kind = hit_kind(lookup_tokens, matched_tokens)
            record = HitRecord(
                seq=seq,
                namespace=namespace,
                lookup_tokens=lookup_tokens,
                matched_tokens=matched_tokens,
                page_size=page_size,
                splits=splits,
                last_node_id=last_node_id,
                kind=kind,
            )

            prev = self._stats.get(namespace, HitStats())
            matched = matched_tokens
            self._stats[namespace] = _add_stats(
                prev,
                HitStats(
                    lookups=1,
                    lookup_tokens=lookup_tokens,
                    hits=1 if matched > 0 else 0,
                    hit_tokens=matched,
                    splits=splits,
                ),
            )

            if self.capacity == 0:
                self._dropped += 1
            else:
                if len(self._records) >= self.capacity:
                    self._records.popleft()
                    self._dropped += 1
                self._records.append(record)

            return record

    @staticmethod
    def _bind_record_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        # Allow a single mapping/template positional argument.
        if args and not isinstance(args[0], HitRecord) and isinstance(
            args[0], dict
        ):
            merged = dict(args[0])
            merged.update(kwargs)
            kwargs = merged
            args = args[1:]

        namespace: Any = kwargs.get("namespace")
        lookup_tokens = kwargs.get("lookup_tokens")
        matched_tokens = kwargs.get("matched_tokens")
        page_size = kwargs.get("page_size", 1)
        splits = kwargs.get("splits", 0)
        last_node_id = kwargs.get("last_node_id")

        # Convenience: extra_key=/cache_salt= build the namespace tuple.
        if namespace is None and (
            "extra_key" in kwargs or "cache_salt" in kwargs
        ):
            namespace = namespace_of(
                kwargs.get("extra_key"), kwargs.get("cache_salt")
            )
        if namespace is None:
            namespace = (None, None)

        positional = list(args)
        if positional and isinstance(positional[0], HitRecord):
            template = positional.pop(0)
            namespace = kwargs.get("namespace", template.namespace)
            lookup_tokens = kwargs.get("lookup_tokens", template.lookup_tokens)
            matched_tokens = kwargs.get(
                "matched_tokens", template.matched_tokens
            )
            page_size = kwargs.get("page_size", template.page_size)
            splits = kwargs.get("splits", template.splits)
            last_node_id = kwargs.get("last_node_id", template.last_node_id)

        elif positional and isinstance(positional[0], str):
            # extra_key-first: (extra_key, [cache_salt,] lookup, matched, ...).
            extra_key = positional.pop(0)
            if positional and (
                positional[0] is None or isinstance(positional[0], str)
            ):
                cache_salt = positional.pop(0)
                namespace = namespace_of(extra_key, cache_salt)
            else:
                namespace = namespace_of(
                    extra_key, kwargs.get("cache_salt")
                )
            if positional:
                lookup_tokens = positional.pop(0)
            if positional:
                matched_tokens = positional.pop(0)
            if positional:
                page_size = positional.pop(0)
            if positional:
                splits = positional.pop(0)
            if positional:
                last_node_id = positional.pop(0)

        elif positional:
            first = positional[0]
            namespace_first = isinstance(first, tuple) or first is None
            if namespace_first:
                # (namespace, lookup, matched, page_size, splits, last_node_id)
                if len(positional) >= 1:
                    namespace = positional[0]
                if len(positional) >= 2:
                    lookup_tokens = positional[1]
                if len(positional) >= 3:
                    matched_tokens = positional[2]
                if len(positional) >= 4:
                    page_size = positional[3]
                if len(positional) >= 5:
                    splits = positional[4]
                if len(positional) >= 6:
                    last_node_id = positional[5]
            else:
                # metrics-first: (lookup, matched, page_size, splits, last_node_id)
                if len(positional) >= 1:
                    lookup_tokens = positional[0]
                if len(positional) >= 2:
                    matched_tokens = positional[1]
                if len(positional) >= 3:
                    page_size = positional[2]
                if len(positional) >= 4:
                    splits = positional[3]
                if len(positional) >= 5:
                    last_node_id = positional[4]

        if matched_tokens is None:
            matched_tokens = lookup_tokens

        if not isinstance(namespace, tuple):
            namespace = namespace_of(namespace, kwargs.get("cache_salt"))

        return (
            namespace,
            lookup_tokens,
            matched_tokens,
            page_size,
            splits,
            last_node_id,
        )

    def take(self) -> Tuple[HitRecord, ...]:
        """Return all retained detail records in order and clear the ring."""
        with self._lock:
            records = tuple(self._records)
            self._records.clear()
            return records

    @property
    def stats(self) -> _StatsView:
        """Fresh copy of the per-namespace :class:`HitStats` mapping."""
        with self._lock:
            total = HitStats()
            for stats in self._stats.values():
                total = _add_stats(total, stats)
            return _StatsView(
                {ns: stats for ns, stats in self._stats.items()},
                overall=total,
            )

    @property
    def overall(self) -> HitStats:
        """Sum of :class:`HitStats` across every namespace."""
        with self._lock:
            total = HitStats()
            for stats in self._stats.values():
                total = _add_stats(total, stats)
            return total

    @property
    def dropped(self) -> int:
        """Detail records discarded (overflow or zero capacity)."""
        with self._lock:
            return _Count(self._dropped)

    def clear(self) -> None:
        """Clear retained details and aggregates.

        The monotonic sequence counter and ``capacity`` are retained; the
        sequence never reuses an id after ``clear``.
        """
        with self._lock:
            self._records.clear()
            self._stats.clear()


class StopReason(Enum):
    """Why one eviction round stopped."""

    TARGET_MET = auto()
    EXHAUSTED = auto()
    DISABLED = auto()


@dataclass(frozen=True)
class EvictRound:
    """Self-consistent accounting for one :meth:`RadixCache.evict_round`."""

    requested: int
    evicted: int
    shortfall: int
    victims: Tuple[int, ...]
    by_namespace: Dict[HitNamespace, int]
    reason: StopReason
    allocator_calls: int


class CacheEvictError(RuntimeError):
    """Wraps an allocator failure mid-eviction.

    ``node_id`` / ``tokens`` identify the victim whose free failed; ``report``
    accounts for the victims completed before it.  Raised ``... from exc`` so
    :attr:`__cause__` preserves the original allocator exception.
    """

    def __init__(
        self,
        node_id: Optional[int] = None,
        tokens: Optional[int] = None,
        report: Optional[EvictRound] = None,
        message: Optional[str] = None,
    ):
        if message is None:
            message = (
                f"eviction failed freeing node {node_id} "
                f"({tokens} tokens); completed report: {report!r}"
            )
        super().__init__(message)
        self.node_id = node_id
        self.tokens = tokens
        self.report = report
