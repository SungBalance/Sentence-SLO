"""R2/R3 SSLO export-name helpers."""

# Unit-level legacy-to-R2 names.
UNIT_RENAMES = {
    "chunk_idx": "unit_index",
    "deadline_ts": "deadline",
    "gen_finish_ts": "text_generation_end_time",
    "conversion_time_s": "conversion_time",
    "chunk_consume_time_s": "consume_duration",
    "stall_duration_s": "unit_deadline_miss_s",
    "cumulative_tokens_at_end": "token_boundary",
    "token_start_idx": "token_start",
    "token_end_idx": "token_end",
}

# Request-level legacy-to-R2 names.
REQUEST_RENAMES = {
    "num_chunks": "num_consumable_units",
    "max_stall_time": "request_max_stall_s",
    "total_stall_time": "request_total_stall_s",
    "stall_fraction": "request_stall_fraction",
    "demand_duration": "request_demand_duration_s",
}

# Token-bucket boundaries for request_class.
REQUEST_CLASS_BUCKETS = [
    (1, 100, "xs"),
    (100, 500, "s"),
    (500, 2000, "m"),
    (2000, None, "l"),  # None = unbounded upper
]


def request_class_for_output_tokens(reference_output_tokens: int | None) -> str:
    """Bucket reference_output_tokens into xs/s/m/l. Returns "unknown"
    if reference is None."""
    if reference_output_tokens is None:
        return "unknown"
    n = int(reference_output_tokens)
    for lo, hi, label in REQUEST_CLASS_BUCKETS:
        if n >= lo and (hi is None or n < hi):
            return label
    return "unknown"


# run_test.py still imports this legacy helper name; keep the alias local here
# until that caller is in scope for a rename.
classify_request = request_class_for_output_tokens
