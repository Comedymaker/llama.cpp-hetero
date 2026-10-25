"""Bounded, offline export of the runtime tree-initial cost profile.

The exported ``V`` curve is a whole Tail verification curve.  It is not a
load/compute union and deliberately makes no independently measured claim
about leaves: the available traces only support a fixed host residual.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from planner import LAYERS, collect, cost, simulate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_ROUND = re.compile(
    r"^batch gain round=(?P<round>\d+) actions=\d+ nodes=(?P<nodes>\d+) "
    r"leaves=\d+ depth=\d+ .*? verify-ms=(?P<verify>[0-9.]+)\b", re.M)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(value, label: str):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")
        return json.loads(path.read_text(encoding="utf-8")), path
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object or path")
    return value, None


def _normalise_table(value):
    saved, path = _read_json(value, "cost table")
    # Accept the frozen artifact only, or its explicitly extracted layers.
    if "layers" in saved:
        if saved.get("schema") != 1:
            raise ValueError("unknown cost-table schema/model")
        table = saved["layers"]
    else:
        table = saved
    if set(table) != {str(layer) for layer in LAYERS}:
        raise ValueError("unknown or incomplete planner layer model")
    for layer in LAYERS:
        row = table[str(layer)]
        if not isinstance(row, dict) or not {"load", "execute", "evict", "prep_ms"} <= set(row):
            raise ValueError(f"invalid cost model for layer {layer}")
    return table, path


def _normalise_counts(counts):
    if isinstance(counts, dict):
        # plan.json itself is convenient, but accepting a bare per-layer map
        # avoids coupling this exporter to planner's command-line entrypoint.
        counts = counts.get("per_layer_q4", counts)
        try:
            answer = [int(counts[str(layer)]) for layer in LAYERS]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("counts must name every planner layer") from exc
    else:
        if not isinstance(counts, (list, tuple)) or len(counts) != len(LAYERS):
            raise ValueError("counts must contain exactly 24 layer values")
        answer = list(counts)
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 32 for x in answer):
        raise ValueError("counts must be integer Q4 shard counts in 0..32")
    return answer


def _cpu_priors(profile):
    if not isinstance(profile, dict) or "cpu_batch_priors_ms" not in profile:
        raise ValueError("unknown CPU profile model")
    values = profile["cpu_batch_priors_ms"]
    if not isinstance(values, list) or len(values) != 16:
        raise ValueError("CPU profile must provide priors for batches 1..16")
    try:
        values = [float(x) for x in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("CPU batch priors must be numeric") from exc
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("CPU batch priors must be positive finite milliseconds")
    return values


def _training_residual(table, counts, io_scale):
    """Fit one host offset from precisely planner.py's six training cases."""
    rows, sources = collect()  # manifests() in planner.py; no planner146 input.
    by_case = {}
    for record in rows:
        if record["phase"] == "generation":
            by_case.setdefault(record["case"], {}).setdefault(record["round"], []).append(record)
    residuals = []
    calibration = []
    for _top, log, _selection in __import__("planner").manifests():
        rounds = list(_ROUND.finditer(log.read_text(encoding="utf-8", errors="replace")))
        observed = by_case.get(log.stem, {})
        for match in rounds:
            # The trace parser has already aligned the actual pipeline rows.
            # Reject a changed log instead of using the summary's node count.
            round_no = int(match["round"])
            group = observed.get(round_no)
            nodes = group[0]['nodes'] if group else None
            # ``nodes=`` in the batch-gain summary is the planner tree count;
            # timing must use the pipeline rows actually executed in the log.
            if nodes is None:
                raise ValueError(f"unaligned batch-gain/pipeline round in {log}: {round_no}")
            # Each training run used ITS OWN mixture, not the requested new
            # mixture. Otherwise the offset would erase true format costs.
            by_layer={r['layer']:r['q4_count'] for r in group}
            training_counts=[by_layer[l] for l in LAYERS]
            prefetch = cost(table, 8, training_counts[0], nodes, 1.0)["load"]
            predicted = simulate(table, training_counts, nodes, 1.0, prefetch)["predicted_tail_to_sync_ms"]
            actual = float(match["verify"])
            residuals.append(actual - predicted)
            calibration.append({"log_sha256": _sha256(log), "round": round_no,
                                "nodes": nodes, "actual_verify_ms": actual,
                                "pipeline_rows": nodes,
                                "simulated_prefetched_verify_ms": predicted})
    if not residuals:
        raise ValueError("no usable training batch-gain rounds")
    return float(statistics.median(residuals)), calibration, sources


def _upper_envelope(values):
    """Monotone upper envelope: extra rows never get a negative marginal cost."""
    out = []
    high = 0.0
    for value in values:
        high = max(high, value)
        out.append(high)
    return out


def export_profile(table, counts, cpu_profile_path, outdir, io_scale=1.0):
    """Write initial-profile.tsv and manifest.json, returning the manifest.

    ``table`` is the frozen cost-table object/path; ``counts`` is a 24-item
    vector, a per-layer map, or a plan.json object.  This function is offline:
    it reads only local artifacts and never launches a device or server call.
    """
    if isinstance(io_scale, bool) or not isinstance(io_scale, (int, float)) or not math.isfinite(io_scale) or io_scale <= 0:
        raise ValueError("io_scale must be positive and finite")
    table, table_path = _normalise_table(table)
    counts = _normalise_counts(counts)
    cpu_data, cpu_path = _read_json(cpu_profile_path, "CPU profile")
    priors = _cpu_priors(cpu_data)
    destination = Path(outdir)
    if destination.exists() and not destination.is_dir():
        raise ValueError("outdir exists but is not a directory")
    destination.mkdir(parents=True, exist_ok=True)

    residual, calibration, sources = _training_residual(table, counts, float(io_scale))
    raw = []
    points = []
    for rows in range(1, 65):
        prefetch = cost(table, 8, counts[0], rows, float(io_scale))["load"]
        simulated = simulate(table, counts, rows, float(io_scale), prefetch)["predicted_tail_to_sync_ms"]
        raw.append(simulated + residual)
        in_range = all(item["nodes_range"][0] <= rows <= item["nodes_range"][1] for item in table.values())
        points.append({"rows": rows, "raw_ms": simulated + residual,
                       "simulated_prefetched_verify_ms": simulated,
                       "inrange": in_range,
                       "range": "inrange" if in_range else "extrapolation"})
    curve = _upper_envelope(raw)
    if any(not math.isfinite(v) or v<=0 for v in curve):
        raise ValueError('nonpositive/nonfinite full verification prediction')
    for point, value in zip(points, curve):
        point["ms"] = value
        point["envelope_raised"] = value > point["raw_ms"]

    profile_path = destination / "initial-profile.tsv"
    lines = ["# bounded offline tree initial profile; units: milliseconds"]
    lines += [f"V {point['rows']} {point['ms']:.6f}" for point in points]
    lines += [f"C {batch} {value:.6f}" for batch, value in enumerate(priors, 1)]
    profile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashes = {"cpu_profile_sha256": _sha256(cpu_path) if cpu_path else hashlib.sha256(json.dumps(cpu_data,sort_keys=True).encode()).hexdigest(),
              "counts_sha256": hashlib.sha256(json.dumps(counts).encode()).hexdigest()}
    if table_path:
        hashes["cost_table_sha256"] = _sha256(table_path)
    # The frozen table carries the original per-selection digests.  Retain
    # them as provenance even when callers supplied only a count vector.
    for source in sources:
        if "selection_sha256" in source:
            hashes[f"training_selection_top{source['top']}_sha256"] = source["selection_sha256"]
    manifest = {"schema": 1, "status": "offline export only", "profile": profile_path.name,
                "format": {"V": "V rows ms; rows 1..64", "C": "C batch ms; batch 1..16"},
                "io_scale": float(io_scale), "counts": dict(zip(map(str, LAYERS), counts)),
                "fixed_host_residual_ms": residual,
                "residual_method": "median(actual verify-ms - simulated prefetched verify-ms) over planner.py six manifests only",
                "calibration_rounds": calibration, "sourceSHA": hashes,
                "training_sources": sources, "rows": points,
                "cpu": {"batches": 16, "extrapolated_batches": list(range(9, 17)),
                        "source": "CPU full-action priors; 9..16 are supplied extrapolations"},
                "limitations": ["Leaf effect is not independently modeled; the calibration is one fixed host residual.",
                                "Rows outside each layer's observed node range are extrapolations.",
                                "The V curve is a monotone upper envelope, so it never advertises negative marginal row cost."]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
