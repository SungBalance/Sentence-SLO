"""Figure 5.1 - Token metric mismatch from measured outputs."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from figure_paths import (
    OUTPUT_SWEEP_DIR,
    PROCESSED_DIR,
    figure_output_path,
    processed_csv_path,
)
import paper_plot_style as pps
from _outputs_data import (
    build_request_trace,
    chunks_by_request,
    chunk_stall_duration,
    load_summary_from_outputs,
    request_map,
    run_dir_from_summary_row,
    stall_intervals,
)

_BASENAME = "fig5_1_token_metric_mismatch"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_STALLS_DATA = PROCESSED_DIR / f"{_BASENAME}_stalls.csv"
DEFAULT_REQUEST_METRICS_DATA = PROCESSED_DIR / f"{_BASENAME}_request_metrics.csv"
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)
SECONDS_PER_WORD = 0.28
X_MAX_S = 4.0
TARGET_MODEL_LABEL = "Qwen3.5-9B"
MIN_SELECTED_GENERATED_TOKENS = 300
GEN_CU_COLOR = "#0072B2"
CONSUMED_COLOR = "#D55E00"
STALL_COLOR = "red"
STALL_ALPHA = 0.12
READING_PROFILE = "Reading"
CONSUME_PROFILE_COLORS = {
    READING_PROFILE: CONSUMED_COLOR,
}
MIN_OUTPUT_TOKENS = 500
MAX_OUTPUT_TOKENS = 700
MAX_TOKEN_RELATIVE_GAP = 0.15
MIN_CHUNKS = 4
MAX_CHUNKS = 24
PREFERRED_SLOW_REQUEST_IDS = frozenset({"719", "593"})
ENGLISH_STOPWORDS = frozenset(
    """
    a about above across after against all also among an and any are as at back be
    because been before being between but by can come could day did do does during
    even first for from get give go good had has have he her him his how i if in
    into is it its just know like look make may me might most must my new no not
    now of on one only or other our out over people say see shall she should so
    some such take than that the their them then there these they think this through
    time to two under up us use want way we well were what when where which while
    who why will with within without work would year you your
    """.split()
)
CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u30ff\uac00-\ud7af]"
)
PRODUCT_KEY_RE = re.compile(r"\b[A-Z0-9]{5}(?:-[A-Z0-9]{5}){3,}\b")
LONG_DECIMAL_RE = re.compile(r"\b\d+\.\d{20,}\b")
JSON_KEY_RE = re.compile(r'^\s*"[^"]+"\s*:', re.MULTILINE)
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)]|[A-Za-z][.)])\s+", re.MULTILINE)
STRUCTURED_LIST_RE = re.compile(
    r"^\s*(?:>|#{1,6}\s*)?(?:\*\*\s*)?(?:option|version|alternative)\s+\d+\b"
    r"|^\s*(?:#{1,6}\s*)?(?:\*\*\s*)?(?:key correction|correction made)s?\b",
    re.IGNORECASE | re.MULTILINE,
)
NUMBERED_HEADING_RE = re.compile(r"^\s*#{1,6}\s*\d+[.)]\s+", re.MULTILINE)
OPTION_COUNT_RE = re.compile(r"\b(?:option|version|alternative)\s+\d+\s*:", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
COLON_FIELD_RE = re.compile(r"^\s*[A-Z][A-Za-z ]{1,24}:\s+", re.MULTILINE)
LIST_PHRASE_RE = re.compile(
    r"\b(?:here is|here are|below is|below are|the following is|the following are)"
    r"\b.{0,80}\b(?:list|ways|options|examples|steps)\b",
    re.IGNORECASE | re.DOTALL,
)
PROMPT_LIST_RE = re.compile(r"/imagine\s+prompt\b|\bprompt\s+\d+\s*:", re.IGNORECASE)
REFERENCE_RE = re.compile(
    r"^\s*(?:references|bibliography|works cited)\s*$"
    r"|\bdoi\b|\bet al\.\b|\bAPA\b|\bHarvard\b",
    re.IGNORECASE | re.MULTILINE,
)
MATH_RE = re.compile(
    r"\b(?:derivative|integral|equation|formula|solve|proof|theorem|matrix|"
    r"vector|logarithm|polynomial|fraction|decimal places?)\b|[$=^{}]",
    re.IGNORECASE,
)
CODE_PATH_RE = re.compile(
    r"(?:/[\w.\-]+/|[A-Za-z]:\\|```|</?\w+>|"
    r"\b(?:def|class|import|const|let|var|function)\b)"
)


def _short_model_label(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _candidate_rows(outputs_root: Path) -> pd.DataFrame:
    df = load_summary_from_outputs(outputs_root)
    df = df[df["policy"] == "baseline"].copy()
    df["stall_rank"] = pd.to_numeric(
        df.get("max_stall_interval_s_p99", 0.0),
        errors="coerce",
    ).fillna(0.0)
    df = df.sort_values(
        ["stall_rank", "lambda_req_s", "max_num_seqs"],
        ascending=False,
    )
    return df


def _request_text(request_chunks: list[dict]) -> str:
    return "".join(str(chunk.get("text") or "") for chunk in request_chunks)


def _is_english_text(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) < 80:
        return False

    stopword_ratio = sum(1 for word in words if word in ENGLISH_STOPWORDS) / len(words)
    letter_count = sum(char.isalpha() for char in text)
    ascii_letter_count = sum("a" <= char.lower() <= "z" for char in text)
    ascii_letter_ratio = ascii_letter_count / max(letter_count, 1)
    return stopword_ratio >= 0.105 and ascii_letter_ratio >= 0.985


def _looks_like_simple_list(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    bullet_lines = sum(1 for line in lines if BULLET_RE.match(line))
    structured_lines = sum(1 for line in lines if STRUCTURED_LIST_RE.match(line))
    short_lines = sum(1 for line in lines if len(line) < 90)
    return (
        bullet_lines >= 4
        or structured_lines >= 3
        or len(NUMBERED_HEADING_RE.findall(text)) >= 2
        or len(OPTION_COUNT_RE.findall(text)) >= 2
        or len(TABLE_ROW_RE.findall(text)) >= 2
        or len(COLON_FIELD_RE.findall(text)) >= 3
        or LIST_PHRASE_RE.search(text) is not None
        or (len(lines) >= 8 and short_lines / len(lines) >= 0.75)
    )


def _is_filtered_text(text: str) -> bool:
    stripped = text.strip()
    return (
        CJK_RE.search(text) is not None
        or not _is_english_text(text)
        or stripped.startswith("{")
        or JSON_KEY_RE.search(text) is not None
        or PRODUCT_KEY_RE.search(text) is not None
        or REFERENCE_RE.search(text) is not None
        or LONG_DECIMAL_RE.search(text) is not None
        or (
            re.search(r"\bpi\b|π", text, re.IGNORECASE) is not None
            and sum(char.isdigit() for char in text) >= 30
        )
        or MATH_RE.search(text) is not None
        or CODE_PATH_RE.search(text) is not None
        or PROMPT_LIST_RE.search(text) is not None
        or _looks_like_simple_list(text)
    )


def _token_relative_gap(left: dict, right: dict) -> float:
    return abs(left["num_output_tokens"] - right["num_output_tokens"]) / max(
        left["num_output_tokens"],
        right["num_output_tokens"],
    )


def _request_stats(chunks: dict[str, list[dict]], requests: dict[str, dict]) -> list[dict]:
    rows = []
    for request_id, request_chunks in chunks.items():
        if len(request_chunks) < 2:
            continue
        request_chunks = sorted(
            request_chunks,
            key=lambda item: int(item.get("chunk_idx", item.get("unit_index", 0))),
        )
        text = _request_text(request_chunks)
        if _is_filtered_text(text):
            continue
        req = requests.get(request_id, {})
        max_stall = max(chunk_stall_duration(item) for item in request_chunks)
        total_tokens = float(
            request_chunks[-1].get("cumulative_tokens_at_end")
            or request_chunks[-1].get("token_end")
            or req.get("num_output_tokens")
            or 0.0
        )
        consume_time_s = sum(
            float(chunk.get("chunk_consume_time_s", chunk.get("consume_duration")) or 0.0)
            for chunk in request_chunks
        )
        tpot_s = float(req.get("tpot") or 0.0)
        rows.append(
            {
                "request_id": request_id,
                "max_stall_s": max_stall,
                "num_chunks": len(request_chunks),
                "ttft_s": float(req.get("ttft") or 0.0),
                "tpot_s": tpot_s,
                "num_output_tokens": total_tokens,
                "consume_time_s": consume_time_s,
                "token_per_consume_s": (
                    total_tokens / consume_time_s if consume_time_s > 0.0 else 0.0
                ),
            }
        )
    return rows


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


def _trace_from_first_token(chunks: list[dict], request_info: dict) -> pd.DataFrame:
    trace = build_request_trace(chunks).copy()
    ordered_chunks = sorted(
        chunks,
        key=lambda item: int(item.get("chunk_idx", item.get("unit_index", 0)) or 0),
    )
    first_chunk = ordered_chunks[0]
    first_chunk_tokens = max(1.0, _chunk_token_end(first_chunk) - _chunk_token_start(first_chunk))
    tpot_s = float(request_info.get("tpot") or 0.0)
    first_token_lead_s = max(0.0, (first_chunk_tokens - 1.0) * tpot_s)
    if first_token_lead_s <= 0.0:
        return trace

    trace["time_s"] = trace["time_s"] + first_token_lead_s
    first_row = trace.iloc[0].copy()
    first_row["time_s"] = 0.0
    first_row["generated_tokens"] = max(1.0, _chunk_token_start(first_chunk) + 1.0)
    first_row["available_tokens"] = 0.0
    first_row["demand_tokens"] = 0.0
    first_row["consumed_tokens"] = 0.0
    first_row["buffer_tokens"] = 0.0
    return pd.concat([pd.DataFrame([first_row]), trace], ignore_index=True)


def _generation_finish_row(trace: pd.DataFrame) -> pd.Series:
    max_generated = float(trace["generated_tokens"].max())
    return trace[trace["generated_tokens"] >= max_generated].iloc[0]


def _select_pair_from_rows(
    outputs_root: Path,
    rows: pd.DataFrame,
) -> tuple[pd.Series, Path, list[dict], dict[str, dict]] | None:
    best: tuple[tuple[float, ...], Path, list[dict], dict[str, list[dict]]] | None = None
    best_row: pd.Series | None = None
    for _, summary_row in rows.iterrows():
        run_dir = run_dir_from_summary_row(outputs_root, summary_row)
        if not (run_dir / "chunks.jsonl").exists() or not (run_dir / "requests.jsonl").exists():
            continue
        chunks = chunks_by_request(run_dir)
        requests = request_map(run_dir)
        stats = _request_stats(chunks, requests)
        eligible = [
            row for row in stats
            if MIN_CHUNKS <= row["num_chunks"] <= MAX_CHUNKS
            and row["num_output_tokens"] >= MIN_OUTPUT_TOKENS
            and row["num_output_tokens"] <= MAX_OUTPUT_TOKENS
            and row["tpot_s"] > 0
            and row["consume_time_s"] > 0
        ]
        for slow_req in eligible:
            for fast_req in eligible:
                if fast_req["token_per_consume_s"] <= slow_req["token_per_consume_s"]:
                    continue
                token_gap = _token_relative_gap(slow_req, fast_req)
                if token_gap > MAX_TOKEN_RELATIVE_GAP:
                    continue
                score = (
                    1.0 if slow_req["request_id"] in PREFERRED_SLOW_REQUEST_IDS else 0.0,
                    fast_req["token_per_consume_s"] / slow_req["token_per_consume_s"],
                    -token_gap,
                    -(slow_req["max_stall_s"] + fast_req["max_stall_s"]),
                    -abs(slow_req["tpot_s"] - fast_req["tpot_s"]),
                )
                if best is None or score > best[0]:
                    best = (score, run_dir, [slow_req, fast_req], chunks)
                    best_row = summary_row

    if best is not None:
        _, run_dir, selected, chunks = best
        assert best_row is not None
        return best_row, run_dir, selected, chunks
    return None


def _select_reading_pair(
    outputs_root: Path,
) -> tuple[pd.Series, Path, list[dict], dict[str, dict], pd.DataFrame]:
    rows = _candidate_rows(outputs_root)
    reading_rows = rows[rows["consume_profile_label"] == READING_PROFILE].copy()
    target_rows = reading_rows[
        reading_rows["model"].map(lambda value: _short_model_label(str(value)) == TARGET_MODEL_LABEL)
    ]
    if target_rows.empty:
        target_rows = reading_rows

    best: tuple[tuple[float, float, float], pd.Series, Path, dict, dict[str, list[dict]]] | None = None
    for _, summary_row in target_rows.iterrows():
        run_dir = run_dir_from_summary_row(outputs_root, summary_row)
        if (
            not (run_dir / "chunks.jsonl").exists()
            or not (run_dir / "requests.jsonl").exists()
        ):
            continue
        chunks = chunks_by_request(run_dir)
        requests = request_map(run_dir)
        for request_id, request_chunks in chunks.items():
            if len(request_chunks) < 2:
                continue
            trace = _trace_from_first_token(request_chunks, requests.get(request_id, {}))
            max_generated = float(trace["generated_tokens"].max())
            if max_generated <= 0.0:
                continue
            if max_generated < MIN_SELECTED_GENERATED_TOKENS:
                continue
            finish_row = _generation_finish_row(trace)
            consumed_at_finish = float(finish_row["consumed_tokens"])
            if consumed_at_finish <= 0.0:
                continue
            ratio = max_generated / consumed_at_finish
            selection = {
                "request_id": request_id,
                "generation_finish_ratio": ratio,
                "generated_at_finish": max_generated,
                "consumed_at_finish": consumed_at_finish,
                "generation_finish_s": float(finish_row["time_s"]),
                "num_chunks": len(request_chunks),
            }
            score = (ratio, max_generated, -float(finish_row["time_s"]))
            if best is None or score > best[0]:
                best = (score, summary_row, run_dir, selection, chunks)

    if best is None:
        raise ValueError(
            "Figure 5.1 needs at least one 9B Reading request with positive "
            "consumed tokens at generation finish and at least "
            f"{MIN_SELECTED_GENERATED_TOKENS} generated tokens."
        )
    _, summary_row, run_dir, selected, chunks = best
    return summary_row, run_dir, [selected], chunks, rows


def _overlay_summary_rows(rows: pd.DataFrame, reference_row: pd.Series) -> list[pd.Series]:
    return [reference_row]


def _request_summary(chunks: list[dict], request_info: dict) -> dict:
    stalls = stall_intervals(chunks)
    total_words = sum(int(chunk.get("num_words", chunk.get("word_count")) or 0) for chunk in chunks)
    consume_time_s = sum(
        float(chunk.get("chunk_consume_time_s", chunk.get("consume_duration")) or 0.0)
        for chunk in chunks
    )
    total_stall_s = 0.0 if stalls.empty else float(stalls["stall_duration_s"].sum())
    generation_start_ts = min(
        float(
            chunk.get(
                "chunk_generation_start_ts",
                chunk.get("text_generation_end_time", chunk.get("consumer_ready_time")),
            )
        )
        for chunk in chunks
    )
    generation_finish_ts = max(
        float(chunk.get("chunk_generation_end_ts", chunk.get("text_generation_end_time")))
        for chunk in chunks
    )
    return {
        "tokens": int(
            chunks[-1].get("cumulative_tokens_at_end")
            or chunks[-1].get("token_end")
            or request_info.get("num_output_tokens")
            or 0
        ),
        "words": total_words,
        "consume_time_s": consume_time_s,
        "generation_finish_s": generation_finish_ts - generation_start_ts,
        "tpot_ms": 1000.0 * float(request_info.get("tpot") or 0.0),
        "total_stall_s": total_stall_s,
    }


def _draw_request(ax, trace: pd.DataFrame, stalls: pd.DataFrame) -> None:
    trace = trace.sort_values("time_s")
    reading_trace = trace[trace["consume_profile_label"] == READING_PROFILE]
    if reading_trace.empty:
        reading_trace = trace
    ax.plot(
        reading_trace["time_s"],
        reading_trace["generated_tokens"],
        color=GEN_CU_COLOR,
        linewidth=1.5,
        linestyle="-",
    )
    for profile in (READING_PROFILE,):
        profile_trace = trace[trace["consume_profile_label"] == profile]
        if profile_trace.empty:
            continue
        ax.plot(
            profile_trace["time_s"],
            profile_trace["consumed_tokens"],
            color=CONSUME_PROFILE_COLORS[profile],
            linewidth=1.5,
            linestyle="-",
        )
    for _, row in stalls.iterrows():
        ax.axvspan(
            row["stall_start_s"],
            row["stall_end_s"],
            color=STALL_COLOR,
            alpha=STALL_ALPHA,
            zorder=0,
        )


def preprocess_fig5_1(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    reference_row, reference_run_dir, selected, reference_chunks, all_rows = (
        _select_reading_pair(input_path)
    )
    overlay_rows = _overlay_summary_rows(all_rows, reference_row)
    selected_request_ids = [selection["request_id"] for selection in selected]

    csv_rows = []
    metric_rows = []
    stall_rows = []
    for summary_row in overlay_rows:
        run_dir = run_dir_from_summary_row(input_path, summary_row)
        chunks = reference_chunks if run_dir == reference_run_dir else chunks_by_request(run_dir)
        requests = request_map(run_dir)
        model = str(summary_row["model"])
        profile = str(summary_row["consume_profile_label"])
        for selection in selected:
            request_id = selection["request_id"]
            if request_id not in chunks:
                continue
            request_info = requests.get(request_id, {})
            request_chunks = chunks[request_id]
            trace = _trace_from_first_token(request_chunks, request_info)
            summary = _request_summary(request_chunks, request_info)
            trace = trace.assign(
                model=model,
                model_label=_short_model_label(model),
                consume_profile_label=profile,
                request_id=request_id,
                total_words=summary["words"],
                consume_time_s=summary["consume_time_s"],
                total_stall_s=summary["total_stall_s"],
            )
            csv_rows.append(trace)
            tpot_s = float(request_info.get("tpot") or 0.0)
            consume_time_s = summary["consume_time_s"]
            generation_finish_s = summary["generation_finish_s"]
            metric_rows.append(
                {
                    "model": model,
                    "model_label": _short_model_label(model),
                    "consume_profile_label": profile,
                    "request_id": request_id,
                    "tokens": summary["tokens"],
                    "tpot_ms": 1000.0 * tpot_s,
                    "words": summary["words"],
                    "consume_time_s": consume_time_s,
                    "generation_finish_s": generation_finish_s,
                    "generation_throughput_tps": (
                        summary["tokens"] / generation_finish_s
                        if generation_finish_s > 0.0
                        else 0.0
                    ),
                    "token_per_consume_s": (
                        summary["tokens"] / consume_time_s if consume_time_s > 0.0 else 0.0
                    ),
                    "generation_finish_ratio": selection.get("generation_finish_ratio"),
                    "generated_at_finish": selection.get("generated_at_finish"),
                    "consumed_at_finish": selection.get("consumed_at_finish"),
                    "generation_finish_trace_s": selection.get("generation_finish_s"),
                }
            )
            stalls = stall_intervals(request_chunks)
            if profile == READING_PROFILE and not stalls.empty:
                stall_rows.append(
                    stalls.assign(
                        model=model,
                        model_label=_short_model_label(model),
                        consume_profile_label=profile,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(csv_rows, ignore_index=True).to_csv(
        output_dir / f"{_BASENAME}.csv",
        index=False,
    )
    if stall_rows:
        stall_df = pd.concat(stall_rows, ignore_index=True)
    else:
        stall_df = pd.DataFrame(
            columns=[
                "stall_start_s",
                "stall_end_s",
                "stall_duration_s",
                "request_id",
                "model",
                "model_label",
                "consume_profile_label",
            ]
        )
    stall_df.to_csv(output_dir / f"{_BASENAME}_stalls.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / f"{_BASENAME}_request_metrics.csv",
        index=False,
    )
    if set(selected_request_ids) - set(pd.DataFrame(metric_rows)["request_id"].astype(str)):
        raise ValueError("Figure 5.1 failed to retain the selected Reading request metrics.")


def _draw_request_title(ax, request_id: str) -> None:
    ax.set_title(f"Request {request_id}", fontsize=11, fontweight="bold", pad=8)


def _model_footer_label(df: pd.DataFrame) -> str:
    values = df["model_label"].dropna().astype(str).unique()
    if len(values) > 0:
        return values[0]
    values = df["model"].dropna().astype(str).unique()
    return _short_model_label(values[0]) if len(values) > 0 else ""


def _visible_token_y_max(df: pd.DataFrame, x_max_s: float) -> float:
    visible = df[(df["time_s"] >= 0.0) & (df["time_s"] <= x_max_s)]
    if visible.empty:
        return 100.0
    max_tokens = float(visible[["generated_tokens", "consumed_tokens"]].max().max())
    return max(100.0, math.ceil(max_tokens * 1.08 / 50.0) * 50.0)


def _x_ticks(x_max_s: float) -> list[float]:
    return [float(tick) for tick in range(0, int(math.ceil(x_max_s)) + 1)]


def _format_time_tick(tick: float) -> str:
    return f"{tick:.0f}"


def _token_y_tick_step(y_max_tokens: float) -> int:
    if y_max_tokens <= 120:
        return 20
    if y_max_tokens <= 300:
        return 50
    return 100


def plot_fig5_1(
    input_path: Path,
    output_path: Path,
    stalls_path: Path = DEFAULT_STALLS_DATA,
    request_metrics_path: Path = DEFAULT_REQUEST_METRICS_DATA,
) -> None:
    pps.paper_theme()
    df = pd.read_csv(input_path)
    df["request_id"] = df["request_id"].astype(str)
    if stalls_path.exists():
        stalls_df = pd.read_csv(stalls_path)
        if not stalls_df.empty:
            stalls_df["request_id"] = stalls_df["request_id"].astype(str)
    else:
        stalls_df = pd.DataFrame()
    reading_df = df[df["consume_profile_label"] == READING_PROFILE]
    request_ids = list(pd.unique(reading_df["request_id"]))
    selected_reading_df = reading_df[reading_df["request_id"].isin(request_ids)]
    x_max_s = X_MAX_S
    x_ticks = _x_ticks(x_max_s)
    model_label = _model_footer_label(selected_reading_df)

    w, h = pps.fig_size("wide", ratio=0.44)
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    fig.subplots_adjust(bottom=0.24, top=0.82)
    grid = fig.add_gridspec(
        1,
        len(request_ids),
        width_ratios=[1.0] * len(request_ids),
        wspace=0.24,
    )

    axes = [
        fig.add_subplot(grid[0, idx])
        for idx in range(len(request_ids))
    ]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    y_max_tokens = _visible_token_y_max(selected_reading_df, x_max_s)
    y_tick_step = _token_y_tick_step(y_max_tokens)

    for ax, request_id in zip(axes, request_ids, strict=False):
        trace = df[df["request_id"] == request_id]
        request_stalls = (
            stalls_df[stalls_df["request_id"] == request_id]
            if not stalls_df.empty
            else stalls_df
        )
        _draw_request(ax, trace, request_stalls)
        _draw_request_title(ax, request_id)
        ax.set_xlim(0.0, x_max_s)
        pps.clean_axes(ax, legend=False)

    axes[0].set_ylabel("# of Tokens")
    axes[0].set_ylim(0.0, y_max_tokens)
    for ax in axes:
        ax.set_yticks(range(0, int(y_max_tokens) + 1, y_tick_step))
        ax.tick_params(labelleft=True)
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([_format_time_tick(tick) for tick in x_ticks])
        pps.clean_axes(ax, legend=False)
        for tick_value, tick_label in zip(ax.get_xticks(), ax.get_xticklabels()):
            if abs(tick_value) < 1e-6:
                tick_label.set_ha("left")
                tick_label.set_clip_on(False)
            elif abs(tick_value - x_max_s) < 1e-6:
                tick_label.set_ha("right")
                tick_label.set_clip_on(False)

    style_handles = [
        Line2D(
            [0],
            [0],
            color=GEN_CU_COLOR,
            lw=1.5,
            linestyle="-",
            label="Generated Tokens",
        ),
        Line2D(
            [0],
            [0],
            color=CONSUMED_COLOR,
            lw=1.5,
            linestyle="-",
            label="Consumed Tokens (Reading)",
        ),
    ]
    legend = fig.legend(
        handles=style_handles,
        fontsize=11,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.000),
        ncol=2,
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.4,
        labelspacing=0.3,
    )
    pps.frame_legend(legend)
    if model_label:
        text = fig.text(
            0.5,
            0.035,
            model_label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        pps.bold_text(text)

    output_path = Path(output_path)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Processed trace CSV.")
    parser.add_argument("--stalls-data", type=Path, default=DEFAULT_STALLS_DATA,
                        help="Processed stall interval CSV.")
    parser.add_argument("--request-metrics-data", type=Path,
                        default=DEFAULT_REQUEST_METRICS_DATA,
                        help="Processed request metric CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_1(args.data, args.out, args.stalls_data, args.request_metrics_data)


if __name__ == "__main__":
    main()
