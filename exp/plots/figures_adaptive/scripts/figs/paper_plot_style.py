# paper_plot_style.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


# ---- Paper figure widths ----
FIG_WIDTH = {
    "single": 3.35,
    "double": 6.90,
    "wide": 7.20,
}

GOLDEN_RATIO = 0.618
PRETENDARD_FONTS_DIR = Path(__file__).resolve().parents[3] / "fonts"
PRETENDARD_REGULAR_FONT = PRETENDARD_FONTS_DIR / "Pretendard-Regular.ttf"
PRETENDARD_BOLD_FONT = PRETENDARD_FONTS_DIR / "Pretendard-Bold.ttf"
PRETENDARD_FALLBACK_FONT = PRETENDARD_FONTS_DIR / "PretendardVariable.ttf"
AXIS_LABEL_WEIGHT = "bold"
COMMON_PALETTE = (
    "#31688e",
    "#b04a3a",
    "#5f8c52",
    "#7a5aa6",
    "#c4892e",
    "#4f8f88",
    "#8a6f4d",
    "#6f7580",
)
POLICY_COLORS = {
    "baseline": COMMON_PALETTE[0],
    "sslo": COMMON_PALETTE[1],
}
MODEL_PALETTE = COMMON_PALETTE
READING_COLOR = COMMON_PALETTE[4]
TTS_MODEL_COLORS = {
    "hexgrad/Kokoro-82M": MODEL_PALETTE[2],
    "Supertone/supertonic-3": MODEL_PALETTE[3],
    "rhasspy/piper-voices": MODEL_PALETTE[3],
}
TTS_MODEL_LABELS = {
    "hexgrad/Kokoro-82M": "TTS: Kokoro-82M (GPU)",
    "Supertone/supertonic-3": "TTS: Supertonic-3 (CPU)",
    "rhasspy/piper-voices": "piper-voices",
}


def _register_pretendard() -> str:
    font_files = sorted(PRETENDARD_FONTS_DIR.glob("Pretendard-*.ttf"))
    if font_files:
        for font_path in font_files:
            font_manager.fontManager.addfont(str(font_path))
        preferred_font = (
            PRETENDARD_REGULAR_FONT
            if PRETENDARD_REGULAR_FONT.exists()
            else font_files[0]
        )
        return font_manager.FontProperties(fname=str(preferred_font)).get_name()

    if PRETENDARD_FALLBACK_FONT.exists():
        font_manager.fontManager.addfont(str(PRETENDARD_FALLBACK_FONT))
        return font_manager.FontProperties(fname=str(PRETENDARD_FALLBACK_FONT)).get_name()

    return "Pretendard"


def paper_theme(
    *,
    font: str | None = None,
    font_size: int = 11,
    palette: str | tuple[str, ...] = COMMON_PALETTE,
) -> None:
    """
    Apply a compact paper-oriented seaborn/matplotlib style.

    Example:
        import paper_plot_style as pps
        pps.paper_theme()
    """

    font_name = font or _register_pretendard()
    rc = {
        # ---- Figure ----
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,

        # ---- Font ----
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "Pretendard", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.labelweight": AXIS_LABEL_WEIGHT,
        "axes.titlesize": font_size + 1,
        "axes.titleweight": AXIS_LABEL_WEIGHT,
        "figure.titleweight": AXIS_LABEL_WEIGHT,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size,

        # ---- Lines / markers ----
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "axes.linewidth": 1.2,

        # ---- Ticks ----
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

        # ---- Grid ----
        "axes.grid": True,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.35,

        # ---- Legend ----
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,

        # ---- Vector export friendliness ----
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    sns.set_theme(
        context="paper",
        style="ticks",
        palette=palette,
        font=font_name,
        rc=rc,
    )


def fig_size(
    width: Literal["single", "double", "wide"] = "single",
    *,
    ratio: float = GOLDEN_RATIO,
) -> tuple[float, float]:
    """Return a figure size matching common paper column widths."""
    w = FIG_WIDTH[width]
    return w, w * ratio


def clean_axes(ax, *, legend: bool = True):
    """Apply common paper-axis cleanup."""
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="major", pad=2)
    bold_axis_labels(ax)

    if legend and ax.get_legend() is not None:
        ax.legend(frameon=False)

    return ax


def bold_text(text):
    """Make a Matplotlib text object use the static Pretendard bold face."""
    if PRETENDARD_BOLD_FONT.exists():
        text.set_fontproperties(
            font_manager.FontProperties(
                fname=str(PRETENDARD_BOLD_FONT),
                size=text.get_fontsize(),
            )
        )
    text.set_fontweight(AXIS_LABEL_WEIGHT)
    text.set_path_effects([])
    return text


def bold_axis_labels(ax):
    """Make axis labels and axis title visibly bold."""
    bold_text(ax.xaxis.label)
    bold_text(ax.yaxis.label)
    bold_text(ax.title)
    return ax


def frame_legend(legend, *, linewidth: float = 1.5) -> None:
    """Apply the standard visible frame for external figure legends."""
    frame = legend.get_frame()
    frame.set_visible(True)
    frame.set_facecolor("white")
    frame.set_edgecolor("#555555")
    frame.set_linewidth(linewidth)
    frame.set_alpha(0.95)


def savefig(
    fig,
    path: str | Path,
    *,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 600,
) -> None:
    """
    Save a figure in one or more formats.

    Example:
        pps.savefig(fig, "result/figure1")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        fig.savefig(
            path.with_suffix(f".{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
