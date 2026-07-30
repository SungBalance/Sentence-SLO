"""Conceptual sketch: # of tokens vs batch size, two panels (Left/Right).

Two concave (diminishing-returns) curves per panel; the gap between the blue
and red curve is large on the left and small on the right. Schematic only —
arrow axes, no ticks.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#4f7cb0"
RED = "#b15953"

x = np.linspace(0.0, 1.0, 400)

# (amplitude, exponent) per curve. Lower x<1 with higher exponent => lower curve.
CURVES = {
    "Left": {
        "blue": (0.97, 0.46),
        "red": (0.50, 0.66),
    },
    "Right": {
        "blue": (0.64, 0.48),
        "red": (0.54, 0.56),
    },
}


def axis_arrows(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    arrow = dict(arrowstyle="-|>", lw=2.6, color="black", mutation_scale=22)
    ax.annotate("", xy=(1.12, 0.0), xytext=(0.0, 0.0), arrowprops=arrow,
                annotation_clip=False)
    ax.annotate("", xy=(0.0, 1.12), xytext=(0.0, 0.0), arrowprops=arrow,
                annotation_clip=False)


fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
for ax, title in zip(axes, ("Left", "Right")):
    for color, key in ((BLUE, "blue"), (RED, "red")):
        amp, exp = CURVES[title][key]
        ax.plot(x, amp * x ** exp, color=color, lw=4.0,
                solid_capstyle="round", solid_joinstyle="round")
    axis_arrows(ax)
    ax.set_xlim(-0.03, 1.22)
    ax.set_ylim(-0.03, 1.22)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=18)
    ax.set_xlabel("batch size", fontsize=22, labelpad=14)
    ax.set_ylabel("# of tokens", fontsize=22, labelpad=14)
    ax.set_box_aspect(10.0 / 16.0)  # each panel's plot box is 16:10 (w:h)

fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.88, wspace=0.25)
fig.savefig("/workspace/mlsys/exp/plots/sketch_batch_tokens.png", dpi=150)
print("saved sketch_batch_tokens.png")
