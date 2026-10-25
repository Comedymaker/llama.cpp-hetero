"""Conservative parser for the mixed Q2/Q4 phone traces.

The trace has a real, shared monotonic clock for ``mixg32 read``, cluster
``execute_sync``, and ``q4 async_evict``.  The later ``pipeline interval``
records use a different, generation-relative origin and are deliberately not
joined here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re


_READ = re.compile(
    r"^mixg32 read tensor=blk\.(?P<layer>\d+)\.[^ ]+ "
    r"payload=(?P<payload>\d+).*? q4=(?P<q4>\d+).*?"
    r"before_begin_us=(?P<begin>\d+).*? end_us=(?P<end>\d+)"
)
_EXEC = re.compile(
    r"^cluster trace phase=(?P<phase>\S+) call=(?P<call>\d+) "
    r"index=(?P<index>\d+) first=(?P<first>\d+) last=(?P<last>\d+) "
    r"cached=(?P<cached>\d+) event=execute_sync start_us=(?P<begin>\d+) end_us=(?P<end>\d+)"
)
_EVICT = re.compile(
    r"^q4 async_evict tensor=blk\.(?P<layer>\d+)\.[^ ]+ bytes=(?P<bytes>\d+) "
    r"begin_us=(?P<begin>\d+) end_us=(?P<end>\d+) rc=(?P<rc>-?\d+)"
)
_NODES = re.compile(r"^tree nodes = (?P<nodes>\d+)$")
_PIPE_EXEC = re.compile(
    r"^pipeline interval execute_host_sync layer=(?P<layer>\d+) "
    r"tokens=(?P<tokens>\d+) cached=(?P<cached>\d+) "
    r"begin_us=(?P<begin>\d+) end_us=(?P<end>\d+)"
)


def _interval_union(intervals: list[tuple[int, int]]) -> int:
    """Return the union width in microseconds (overlap is counted once)."""
    if not intervals:
        return 0
    total = 0
    start, end = sorted(intervals)[0]
    for left, right in sorted(intervals)[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def _selection_counts(selection_tsv: str | Path | None) -> Counter:
    counts: Counter = Counter()
    if not selection_tsv:
        return counts
    for raw in Path(selection_tsv).read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            counts[int(fields[0])] += 1
    return counts


def parse_case(path: str | Path, selection_tsv: str | Path | None = None) -> list[dict]:
    """Parse one log into compute-centric per-layer observations.

    ``first,last`` is the cluster event's half-open executed range: the
    observation therefore joins reads and evictions for ``layer=first``.
    ``compute_evict_union_ms`` is a true wall-clock union from the shared
    absolute timestamps; it includes execution plus its following evictions,
    not the preceding read.  ``nodes`` is the sequentially matched ``tokens``
    field from the generation-relative pipeline trace; this is deliberately
    not the case-wide ``tree nodes`` summary.
    """
    trace = Path(path)
    selected = _selection_counts(selection_tsv)
    reads: list[dict] = []
    execs: list[dict] = []
    evicts: list[dict] = []
    pipeline_execs: list[dict] = []
    nodes = None
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := _READ.match(line):
            reads.append({key: int(value) for key, value in match.groupdict().items()})
        elif match := _EXEC.match(line):
            item = {key: (int(value) if key != "phase" else value)
                    for key, value in match.groupdict().items()}
            execs.append(item)
        elif match := _EVICT.match(line):
            evicts.append({key: int(value) for key, value in match.groupdict().items()})
        elif match := _PIPE_EXEC.match(line):
            pipeline_execs.append({key: int(value) for key, value in match.groupdict().items()})
        elif match := _NODES.match(line):
            nodes = int(match["nodes"])

    # Group seven tensor-read lines into an occurrence of a logical layer.
    # Their order is not stable, so only their clock envelope is meaningful.
    by_layer: dict[int, list[list[dict]]] = defaultdict(list)
    open_groups: dict[int, list[dict]] = defaultdict(list)
    for item in reads:
        group = open_groups[item["layer"]]
        group.append(item)
        if len(group) == 7:
            by_layer[item["layer"]].append(group[:])
            open_groups[item["layer"]] = []

    # The post-compute eviction has seven tensor jobs for the computed layer.
    evict_by_layer: dict[int, list[list[dict]]] = defaultdict(list)
    open_evicts: dict[int, list[dict]] = defaultdict(list)
    for item in evicts:
        group = open_evicts[item["layer"]]
        group.append(item)
        if len(group) == 7:
            evict_by_layer[item["layer"]].append(group[:])
            open_evicts[item["layer"]] = []

    read_cursor: Counter = Counter()
    evict_cursor: Counter = Counter()
    generation_pipeline_cursor = 0
    round_no = -1
    records: list[dict] = []
    for event in execs:
        layer = event["first"]
        if layer == 8:
            round_no += 1
        flags = []
        groups = by_layer.get(layer, [])
        group = groups[read_cursor[layer]] if read_cursor[layer] < len(groups) else None
        read_cursor[layer] += 1
        if group is None:
            flags.append("missing_read_group")
            load_ms = None
            payload = None
            observed_q4_tiles = None
            load_begin_us = None
            load_end_us = None
        else:
            load_ms = _interval_union([(x["begin"], x["end"]) for x in group]) / 1000
            payload = sum(x["payload"] for x in group)
            observed_q4_tiles = sum(x["q4"] for x in group)
            load_begin_us = min(x["begin"] for x in group)
            load_end_us = max(x["end"] for x in group)
            if len(group) != 7:
                flags.append("incomplete_read_group")
        groups = evict_by_layer.get(layer, [])
        group = groups[evict_cursor[layer]] if evict_cursor[layer] < len(groups) else None
        evict_cursor[layer] += 1
        if group is None:
            flags.append("missing_evict_group")
            evict_ms = None
            union_ms = None
            evict_intervals_us = []
        else:
            evict_intervals = [(x["begin"], x["end"]) for x in group]
            evict_intervals_us = evict_intervals
            evict_ms = _interval_union(evict_intervals) / 1000
            union_ms = _interval_union([(event["begin"], event["end"]), *evict_intervals]) / 1000
            if any(x["rc"] != 0 for x in group):
                flags.append("evict_failure")
        if event["cached"]:
            flags.append("cached_execute")
        if event["phase"] == "generation":
            if generation_pipeline_cursor >= len(pipeline_execs):
                node_count = None
                flags.append("missing_generation_pipeline_interval")
            else:
                pipe = pipeline_execs[generation_pipeline_cursor]
                generation_pipeline_cursor += 1
                node_count = pipe["tokens"]
                if pipe["layer"] != layer:
                    flags.append("pipeline_layer_mismatch")
        else:
            node_count = None
        records.append({
            "case": trace.stem,
            "round": round_no,
            "layer": layer,
            "nodes": node_count,
            "q4_count": selected[layer],
            "load_ms": load_ms,
            "execute_ms": (event["end"] - event["begin"]) / 1000,
            "evict_ms": evict_ms,
            "compute_evict_union_ms": union_ms,
            "payload_bytes": payload,
            "load_begin_us": load_begin_us,
            "load_end_us": load_end_us,
            "execute_begin_us": event["begin"],
            "execute_end_us": event["end"],
            "evict_intervals_us": evict_intervals_us,
            "quality_flags": flags,
            "observed_q4_tiles": observed_q4_tiles,
            "phase": event["phase"],
            "call": event["call"],
            "index": event["index"],
        })
    return records
