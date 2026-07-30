# _policy_style.py
from __future__ import annotations

from paper_plot_style import POLICY_COLORS

POLICY_PLOT_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "sslo": "Adaptive",
}

POLICY_ORDER: list[str] = ["baseline", "sslo"]

BATCH_LINESTYLE: dict[str, str] = {
    "small": "--",
    "large": "-",
}

# Map raw CSV policy strings from measured run_sslo summaries to the internal
# policy keys used by figure scripts.
POLICY_CSV_NORMALIZE: dict[str, str] = {
    "baseline": "baseline",
    "sslo_mlp": "sslo",
    "sslo": "sslo",
    "progress_serve_adaptive": "sslo",
}


def normalize_policy(raw: str) -> str:
    """Map a raw policy string from measured outputs to the internal key."""
    return POLICY_CSV_NORMALIZE.get(raw, raw)
