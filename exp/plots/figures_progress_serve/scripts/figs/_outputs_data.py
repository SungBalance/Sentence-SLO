from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = FIGURES_DIR / "data" / "output_sweep"
EXTRA_TOKEN_THROUGHPUT_FIELDS = [
    "decode_tokens_per_second",
    "prefill_tokens_per_second",
    "scheduled_tokens_per_second",
]
CONSUME_PROFILE_ORDER = (
    "Reading",
    "TTS (Supertonic-3)",
    "TTS (Kokoro-82M)",
)
MODEL_LABEL_ORDER = ("Qwen3.5-9B", "Qwen3.5-35B-A3B")
CONSUME_PROFILE_LABELS = {
    ("read", "none"): "Reading",
    ("tts", "Supertone__supertonic-3"): "TTS (Supertonic-3)",
    ("tts", "hexgrad__Kokoro-82M"): "TTS (Kokoro-82M)",
}
TTS_MODEL_PROFILE_LABELS = {
    "Supertone/supertonic-3": "TTS (Supertonic-3)",
    "hexgrad/Kokoro-82M": "TTS (Kokoro-82M)",
}


@dataclass(frozen=True)
class RunPath:
    path: Path
    model_dir: str
    consume_mode: str
    consume_profile: str
    consume_profile_label: str
    max_num_seqs: int
    policy: str
    seed: int
    lambda_req_s: float


def resolve_outputs_root(path: Path) -> Path:
    path = Path(path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(FIGURES_DIR / path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ValueError(f"Expected an outputs directory, but none exists at {path}.")


def load_summary_from_outputs(path: Path) -> pd.DataFrame:
    root = resolve_outputs_root(path)
    json_frame = _read_summary_json_runs(root)
    if not json_frame.empty:
        return json_frame

    csv_paths = sorted(root.glob("*/summary.csv"))
    if not csv_paths:
        raise ValueError(
            f"No run summary JSON files or model summary CSV files found under {root}. "
            "Expected outputs/<model>/<consume_mode>/<profile>/cap*/<policy>/run_*/"
            "rate_*/summary.json or outputs/<model>/summary.csv files."
        )

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = _read_summary_csv(csv_path)
        frame["source_summary_csv"] = csv_path.relative_to(root).as_posix()
        frame["model_dir"] = frame["model"].astype(str).str.rsplit("/", n=1).str[-1]
        frame["consume_mode"] = "legacy"
        frame["consume_profile"] = "legacy"
        frame["consume_profile_label"] = "Legacy"
        frame["consume_profile_order"] = len(CONSUME_PROFILE_ORDER)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def consume_profile_sort_value(label: object) -> int:
    try:
        return CONSUME_PROFILE_ORDER.index(str(label))
    except ValueError:
        return len(CONSUME_PROFILE_ORDER)


def ordered_consume_profiles(df: pd.DataFrame) -> list[str]:
    labels = [str(label) for label in df["consume_profile_label"].dropna().unique()]
    return sorted(labels, key=lambda label: (consume_profile_sort_value(label), label))


def display_consume_profile_label(label: str) -> str:
    if str(label) == "Reading":
        return "Human Reading"
    return str(label)


def short_model_label(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def model_sort_value(name: object) -> int:
    label = short_model_label(str(name))
    try:
        return MODEL_LABEL_ORDER.index(label)
    except ValueError:
        return len(MODEL_LABEL_ORDER)


def ordered_models(df: pd.DataFrame) -> list[str]:
    models = [str(model) for model in df["model"].dropna().unique()]
    return sorted(models, key=lambda model: (model_sort_value(model), short_model_label(model)))


def _consume_profile_label(
    consume_mode: str,
    consume_profile: str,
    *,
    tts_model: str | None = None,
) -> str:
    if tts_model in TTS_MODEL_PROFILE_LABELS:
        return TTS_MODEL_PROFILE_LABELS[tts_model]
    if (consume_mode, consume_profile) in CONSUME_PROFILE_LABELS:
        return CONSUME_PROFILE_LABELS[(consume_mode, consume_profile)]
    if consume_mode == "read":
        return "Reading"
    return consume_profile.replace("__", "/").replace("_", " ").strip() or consume_mode


def run_metadata_from_path(run_dir: Path, outputs_root: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    parts = (
        run_dir.relative_to(resolve_outputs_root(outputs_root)).parts
        if outputs_root is not None
        else run_dir.parts
    )
    if len(parts) >= 7:
        model_dir, consume_mode, consume_profile, cap_part, policy, run_part, rate_part = parts[-7:]
    elif len(parts) >= 5:
        model_dir, cap_part, policy, run_part, rate_part = parts[-5:]
        consume_mode = "legacy"
        consume_profile = "legacy"
    else:
        raise ValueError(f"Cannot parse run directory metadata from {run_dir}.")

    label = _consume_profile_label(consume_mode, consume_profile)
    return {
        "model_dir": model_dir,
        "consume_mode": consume_mode,
        "consume_profile": consume_profile,
        "consume_profile_label": label,
        "consume_profile_order": consume_profile_sort_value(label),
        "max_num_seqs": int(cap_part.removeprefix("cap")),
        "policy": policy,
        "seed": int(run_part.removeprefix("run_")),
        "lambda_req_s": float(rate_part.removeprefix("rate_")),
    }


def _read_summary_json_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/*/cap*/*/run_*/rate_*/summary.json")):
        run_dir = summary_path.parent
        metadata = run_metadata_from_path(run_dir, root)
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {summary_path}: {exc}") from exc
        rows.append(_summary_json_row(summary, metadata, root, run_dir))

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return _convert_numeric_columns(frame)


def _summary_json_row(
    summary: dict[str, Any],
    metadata: dict[str, Any],
    root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    config = summary.get("config") or {}
    policy = str(config.get("run_kind") or metadata["policy"])
    policy_config = (config.get("sslo_config") or {}).get(policy) or {}
    tts_model = policy_config.get("tts_model")
    label = _consume_profile_label(
        str(metadata["consume_mode"]),
        str(metadata["consume_profile"]),
        tts_model=tts_model,
    )

    row: dict[str, Any] = {
        "model": config.get("model") or metadata["model_dir"],
        "model_dir": metadata["model_dir"],
        "consume_mode": metadata["consume_mode"],
        "consume_profile": metadata["consume_profile"],
        "consume_profile_label": label,
        "consume_profile_order": consume_profile_sort_value(label),
        "tts_model": tts_model,
        "max_num_seqs": config.get("max_num_seqs", metadata["max_num_seqs"]),
        "lambda_req_s": config.get("request_rate", metadata["lambda_req_s"]),
        "policy": policy,
        "seed": metadata["seed"],
        "source_run_dir": run_dir.relative_to(root).as_posix(),
        "num_requests_in_measurement_window": summary.get("in_window_count"),
        "num_arrivals_total": summary.get("injected_count"),
        "measurement_window_seconds": summary.get("measurement_window_seconds"),
    }

    metrics = summary.get("metrics") or {}
    policy_metrics = (metrics.get(policy) or {}).copy()
    row.update(policy_metrics)

    workload = (metrics.get("workload") or {}).get(policy) or {}
    throughput = (metrics.get("throughput") or {}).get(policy) or {}
    scheduler = (metrics.get("scheduler") or {}).get(policy) or {}
    stall_time = (metrics.get("stall_time") or {}).get(policy) or {}
    req_viol = (metrics.get("request_cu_slo_violation") or {}).get(policy) or {}
    chunk_viol = (metrics.get("chunk_cu_slo_violation") or {}).get(policy) or req_viol

    row["tokens_per_second"] = _first_present(
        policy_metrics.get("corrected_processed_tokens_per_s"),
        workload.get("tokens_per_second"),
    )
    for key in (
        "output_tokens_per_second",
        "input_tokens_per_second",
        "total_tokens_per_second",
    ):
        row[key] = throughput.get(key)
    for key in (
        "decode_tokens_per_second",
        "prefill_tokens_per_second",
        "scheduled_tokens_per_second",
    ):
        row[key] = scheduler.get(key)
    urgent_fraction = scheduler.get("urgent_mode_fraction")
    row["urgent_mode_fraction_pct"] = (
        float(urgent_fraction) * 100.0 if urgent_fraction is not None else None
    )

    for src_prefix, dst_prefix in (
        ("request_max_stall_s", "max_stall_interval_s"),
        ("request_total_stall_s", "total_stall_time_s"),
    ):
        for stat in ("mean", "p50", "p90", "p99", "max"):
            value = policy_metrics.get(f"{src_prefix}_{stat}")
            if value is not None:
                row[f"{dst_prefix}_{stat}"] = value

    row["stall_fraction_mean"] = policy_metrics.get("request_stall_fraction_mean")
    row["unit_deadline_miss_rate"] = _unit_deadline_miss_rate(
        policy_metrics,
        workload,
        stall_time,
    )
    for tau in ("0_5", "1", "2"):
        row[f"chunk_slo_violation_rate_tau_{tau}s"] = _violation_pct(chunk_viol, tau)
        row[f"request_slo_violation_rate_tau_{tau}s"] = _violation_pct(req_viol, tau)

    _copy_distribution(row, workload.get("num_prompt_tokens") or {}, "input_tokens")
    _copy_distribution(row, workload.get("num_output_tokens") or {}, "output_tokens")
    _copy_distribution(row, workload.get("num_consumable_units") or {}, "units_per_request")
    _copy_distribution(row, workload.get("consume_duration") or {}, "consume_time_per_unit")
    return row


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _copy_distribution(row: dict[str, Any], dist: dict[str, Any], prefix: str) -> None:
    for stat in ("mean", "p50", "p90", "p99"):
        row[f"{prefix}_{stat}"] = dist.get(stat)


def _violation_pct(node: dict[str, Any], tau: str) -> float | None:
    key = f"tau_{tau.replace('_', '.')}"
    value = (node.get(key) or node.get(f"tau_{tau}") or {}).get("rate")
    return float(value) * 100.0 if value is not None else None


def _unit_deadline_miss_rate(
    policy_metrics: dict[str, Any],
    workload: dict[str, Any],
    stall_time: dict[str, Any],
) -> float | None:
    if stall_time.get("count"):
        return 100.0 * float(stall_time.get("violated_count") or 0.0) / float(
            stall_time["count"]
        )
    value = _first_present(
        policy_metrics.get("unit_deadline_miss_rate"),
        workload.get("unit_deadline_miss_rate"),
    )
    return float(value) * 100.0 if value is not None else None


def _convert_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() == frame[column].notna().sum():
            frame[column] = converted
    return frame


def _expanded_summary_header(header: list[str], row_width: int) -> list[str]:
    if (
        row_width == len(header) + len(EXTRA_TOKEN_THROUGHPUT_FIELDS)
        and "decode_tokens_per_second" not in header
        and "urgent_mode_fraction_pct" in header
    ):
        insert_at = header.index("urgent_mode_fraction_pct")
        return (
            header[:insert_at]
            + EXTRA_TOKEN_THROUGHPUT_FIELDS
            + header[insert_at:]
        )
    return header


def _read_summary_csv(csv_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for line_no, row in enumerate(reader, start=2):
            row_header = _expanded_summary_header(header, len(row))
            if len(row) != len(row_header):
                raise ValueError(
                    f"{csv_path} line {line_no} has {len(row)} fields, "
                    f"expected {len(row_header)}."
            )
            rows.append(dict(zip(row_header, row)))
    frame = pd.DataFrame(rows)
    return _convert_numeric_columns(frame)


def _format_rate(rate: float) -> str:
    return f"{float(rate):g}"


def run_dir_from_summary_row(outputs_root: Path, row: pd.Series) -> Path:
    root = resolve_outputs_root(outputs_root)
    source_run_dir = row.get("source_run_dir")
    if source_run_dir is not None and not pd.isna(source_run_dir):
        source = Path(str(source_run_dir))
        return source if source.is_absolute() else root / source

    model_dir = str(row["model"]).rsplit("/", 1)[-1]
    cap = int(row["max_num_seqs"])
    seed = int(row["seed"])
    rate = _format_rate(float(row["lambda_req_s"]))
    policy = str(row["policy"])
    return root / model_dir / f"cap{cap}" / policy / f"run_{seed}" / f"rate_{rate}"


def iter_run_dirs(path: Path) -> list[RunPath]:
    root = resolve_outputs_root(path)
    runs: list[RunPath] = []
    candidates = list(root.glob("*/*/*/cap*/*/run_*/rate_*"))
    if not candidates:
        candidates = list(root.glob("*/cap*/*/run_*/rate_*"))
    for run_dir in sorted(candidates):
        if not run_dir.is_dir():
            continue
        try:
            metadata = run_metadata_from_path(run_dir, root)
            runs.append(
                RunPath(
                    path=run_dir,
                    model_dir=metadata["model_dir"],
                    consume_mode=metadata["consume_mode"],
                    consume_profile=metadata["consume_profile"],
                    consume_profile_label=metadata["consume_profile_label"],
                    max_num_seqs=metadata["max_num_seqs"],
                    policy=metadata["policy"],
                    seed=metadata["seed"],
                    lambda_req_s=metadata["lambda_req_s"],
                )
            )
        except (ValueError, IndexError):
            continue
    return runs


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_request_id(request_id: object) -> str:
    return str(request_id).split("-", 1)[0]


def finite_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def request_map(run_dir: Path) -> dict[str, dict]:
    path = Path(run_dir) / "requests.jsonl"
    if not path.exists():
        return {}
    return {normalize_request_id(row["request_id"]): row for row in read_jsonl(path)}


def chunks_by_request(run_dir: Path) -> dict[str, list[dict]]:
    path = Path(run_dir) / "chunks.jsonl"
    if not path.exists():
        return {}
    grouped: dict[str, list[dict]] = {}
    for row in read_jsonl(path):
        grouped.setdefault(normalize_request_id(row["request_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=_chunk_index)
    return grouped


def _chunk_index(chunk: dict) -> int:
    return int(chunk.get("chunk_idx", chunk.get("unit_index", 0)) or 0)


def _chunk_token_start(chunk: dict) -> float:
    return float(chunk.get("token_start_idx", chunk.get("token_start", 0.0)) or 0.0)


def _chunk_token_end(chunk: dict) -> float:
    return float(
        chunk.get(
            "token_end_idx",
            chunk.get("token_end", chunk.get("cumulative_tokens_at_end", 0.0)),
        )
        or 0.0
    )


def _chunk_cumulative_tokens(chunk: dict) -> float:
    return float(chunk.get("cumulative_tokens_at_end", _chunk_token_end(chunk)) or 0.0)


def _chunk_gen_end(chunk: dict) -> float:
    return float(
        chunk.get("chunk_generation_end_ts", chunk.get("text_generation_end_time"))
    )


def _chunk_consume_duration(chunk: dict) -> float | None:
    return finite_float(chunk.get("chunk_consume_time_s", chunk.get("consume_duration")))


def chunk_stall_duration(chunk: dict) -> float:
    return max(
        0.0,
        finite_float(chunk.get("stall_duration_s", chunk.get("unit_deadline_miss_s")))
        or 0.0,
    )


def build_request_trace(chunks: list[dict], samples: int = 900) -> pd.DataFrame:
    if not chunks:
        raise ValueError("Cannot build a request trace without chunks.")

    gen_points: list[tuple[float, float]] = []
    start_candidates: list[float] = []
    end_candidates: list[float] = []
    ordered_chunks = sorted(chunks, key=_chunk_index)
    previous_gen_end: float | None = None
    for chunk in ordered_chunks:
        gen_end = _chunk_gen_end(chunk)
        demand_start = float(
            chunk.get("demand_window_start_ts", chunk.get("consumer_ready_time", gen_end))
        )
        consume_duration = _chunk_consume_duration(chunk)
        demand_end = float(
            chunk.get(
                "demand_window_end_ts",
                demand_start + (consume_duration if consume_duration is not None else 0.0),
            )
        )
        gen_start = float(
            chunk.get(
                "chunk_generation_start_ts",
                previous_gen_end if previous_gen_end is not None else min(gen_end, demand_start),
            )
        )
        previous_gen_end = gen_end
        token_start = _chunk_token_start(chunk)
        token_end = _chunk_token_end(chunk)
        gen_points.extend([(gen_start, token_start), (gen_end, token_end)])
        start_candidates.extend([gen_start, demand_start])
        end_candidates.extend([gen_end, demand_end])

    t0 = min(start_candidates)
    t1 = max(end_candidates)
    if t1 <= t0:
        t1 = t0 + 1.0
    t_abs = np.linspace(t0, t1, samples)
    time_s = t_abs - t0

    gen_points = sorted(set(gen_points))
    gen_t = np.array([item[0] for item in gen_points])
    gen_y = np.array([item[1] for item in gen_points])
    generated = np.interp(t_abs, gen_t, gen_y)

    available = np.zeros_like(t_abs)
    demand = np.zeros_like(t_abs)
    consumed = np.zeros_like(t_abs)
    for idx, chunk in enumerate(ordered_chunks):
        gen_end = _chunk_gen_end(chunk)
        token_start = _chunk_token_start(chunk)
        token_end = _chunk_token_end(chunk)
        demand_start = float(
            chunk.get("demand_window_start_ts", chunk.get("consumer_ready_time", gen_end))
        )
        consume_duration = _chunk_consume_duration(chunk)
        demand_end = float(
            chunk.get(
                "demand_window_end_ts",
                demand_start + (consume_duration if consume_duration is not None else 0.0),
            )
        )
        cumulative = _chunk_cumulative_tokens(chunk)
        available[t_abs >= gen_end] = np.maximum(available[t_abs >= gen_end], cumulative)

        if demand_end <= demand_start:
            demand_end = demand_start + 1e-9
        progress = (t_abs - demand_start) / (demand_end - demand_start)
        active_demand = token_start + np.clip(progress, 0.0, 1.0) * (
            token_end - token_start
        )
        chunk_demand = np.where(
            t_abs < demand_start,
            0.0,
            np.where(t_abs > demand_end, token_end, active_demand),
        )
        demand = np.maximum(demand, chunk_demand)

        stall_end = finite_float(
            chunk.get(
                "stall_end_ts",
                (
                    (finite_float(chunk.get("deadline")) or demand_start)
                    + chunk_stall_duration(chunk)
                )
                if chunk.get("deadline") is not None
                else None,
            )
        )
        stall_duration = chunk_stall_duration(chunk)
        consume_start = stall_end if stall_duration > 0 and stall_end is not None else demand_start
        if idx + 1 < len(ordered_chunks):
            next_chunk = ordered_chunks[idx + 1]
            consume_end = float(
                next_chunk.get(
                    "demand_window_start_ts",
                    next_chunk.get("consumer_ready_time", demand_end),
                )
            )
        else:
            if consume_duration is None:
                consume_duration = max(0.0, demand_end - demand_start)
            consume_end = consume_start + consume_duration
        if consume_end <= consume_start:
            consume_end = consume_start + 1e-9
        consume_progress = (t_abs - consume_start) / (consume_end - consume_start)
        active_consumed = token_start + np.clip(consume_progress, 0.0, 1.0) * (
            token_end - token_start
        )
        chunk_consumed = np.where(
            t_abs < consume_start,
            0.0,
            np.where(t_abs > consume_end, token_end, active_consumed),
        )
        consumed = np.maximum(consumed, chunk_consumed)
    consumed = np.minimum(consumed, available)

    return pd.DataFrame(
        {
            "time_s": time_s,
            "generated_tokens": generated,
            "available_tokens": available,
            "demand_tokens": demand,
            "consumed_tokens": consumed,
            "buffer_tokens": available - consumed,
        }
    )


def stall_intervals(chunks: Iterable[dict]) -> pd.DataFrame:
    chunk_list = list(chunks)
    rows = []
    start_origin = min(
        float(
            chunk.get(
                "chunk_generation_start_ts",
                chunk.get("text_generation_end_time", chunk.get("consumer_ready_time")),
            )
        )
        for chunk in chunk_list
        if (
            chunk.get("chunk_generation_start_ts") is not None
            or chunk.get("text_generation_end_time") is not None
            or chunk.get("consumer_ready_time") is not None
        )
    )
    for chunk in chunk_list:
        duration = chunk_stall_duration(chunk)
        if duration <= 0:
            continue
        start = finite_float(chunk.get("stall_start_ts", chunk.get("deadline")))
        end = finite_float(chunk.get("stall_end_ts"))
        if end is None and start is not None:
            end = start + duration
        if start is None or end is None:
            continue
        rows.append(
            {
                "request_id": normalize_request_id(chunk["request_id"]),
                "chunk_idx": _chunk_index(chunk),
                "stall_start_s": start - start_origin,
                "stall_end_s": end - start_origin,
                "stall_duration_s": duration,
            }
        )
    return pd.DataFrame(rows)
