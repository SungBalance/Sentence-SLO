"""Canonical CP-SLO export name aliases.

Internal engine code keeps legacy SSLO names (slack_s, gen_finish_ts,
deadline_ts, decoding_start_ts). This module is the single source of
truth for the CP-SLO export names used in paper figures and analysis
scripts. Export writers (run_test.py) emit BOTH the legacy key and
the canonical key during the migration; readers MAY use either.
"""

# Chunk-level
CHUNK_RENAMES = {
    "deadline_ts": "chunk_deadline_ts",
    "gen_finish_ts": "chunk_generation_end_ts",
    "start_time_ts": "chunk_generation_start_ts",
    "slack_s": "chunk_deadline_margin_s",
}

# Request-level
REQUEST_RENAMES = {
    "decoding_start_ts": "first_token_ts",
    "ttfc": "TTFC",
    "num_pending_iters_per_request": "num_pending_intervals",
    "total_pending_time_s": "pending_time_total_s",
}

# Token-bucket boundaries for request_class.
REQUEST_CLASS_BUCKETS = [
    (1, 100, "xs"),
    (100, 500, "s"),
    (500, 2000, "m"),
    (2000, None, "l"),  # None = unbounded upper
]


def classify_request(reference_output_tokens: int | None) -> str:
    """Bucket reference_output_tokens into xs/s/m/l. Returns "unknown"
    if reference is None."""
    if reference_output_tokens is None:
        return "unknown"
    n = int(reference_output_tokens)
    for lo, hi, label in REQUEST_CLASS_BUCKETS:
        if n >= lo and (hi is None or n < hi):
            return label
    return "unknown"
