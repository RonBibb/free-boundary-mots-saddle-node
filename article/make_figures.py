#!/usr/bin/env python3
"""Generate manuscript figures from sealed JSON records."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
RESULTS = REPOSITORY / "data" / "historical"
OUTPUT = HERE / "figures" / "formation-and-separation.pdf"
P229_OUTPUT = (
    REPOSITORY / "protocols/"
    "protocol229-free-boundary-mots-saddle-node-v4-2026-08-24/candidate-output"
)
SADDLE_OUTPUT = HERE / "figures" / "saddle-node-closure.pdf"


def read_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


formation = read_json("corrected_A790_formation_time_refinement.json")
geometry = read_json("corrected_A790_surface_geometry_history.json")

times = np.asarray(formation["fine_times"], dtype=float)
counts = formation["count_histories"]

g8_times = []
g8_gap_percent = []
for item in geometry["records"]["G8"]:
    inner = float(item["branches"][0]["geometry"][1]["one_sided_cap_area"])
    outer = float(item["branches"][1]["geometry"][1]["one_sided_cap_area"])
    g8_times.append(float(item["time"]))
    g8_gap_percent.append(100.0 * (outer - inner) / (0.5 * (outer + inner)))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)

for label, marker, color in (("G7", "o", "#1f5a91"), ("G8", "s", "#a8482a")):
    axes[0].step(times, counts[label], where="post", color=color, linewidth=1.4)
    axes[0].plot(times, counts[label], marker=marker, linestyle="none", color=color,
                 markersize=3.8, label=label)
axes[0].axvspan(0.0005, 0.000625, color="#777777", alpha=0.14, linewidth=0)
axes[0].set_xlabel(r"$t/\ell$")
axes[0].set_ylabel("admitted roots")
axes[0].set_yticks((0, 1, 2))
axes[0].set_xlim(0.0001, 0.001025)
axes[0].set_ylim(-0.12, 2.22)
axes[0].legend(frameon=False, loc="upper left")
axes[0].text(0.0005625, 1.08, "formation\nbracket", ha="center", va="center",
             fontsize=7.5, color="#555555")
axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top",
             fontweight="bold")

axes[1].plot(g8_times, g8_gap_percent, color="#1f5a91", marker="o",
             linewidth=1.4, markersize=4)
axes[1].axhline(0.0, color="#777777", linewidth=0.8)
axes[1].set_xlabel(r"$t/\ell$")
axes[1].set_ylabel(r"$(\mathcal{A}_{\rm out}-\mathcal{A}_{\rm in})/\bar{\mathcal{A}}$ (\%)")
axes[1].set_xlim(0.00045, 0.00415)
axes[1].set_ylim(bottom=0.0)
axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top",
             fontweight="bold")

for axis in axes:
    axis.grid(True, color="#d8d8d8", linewidth=0.55)
    axis.tick_params(direction="in")
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
plt.close(fig)
print(OUTPUT)


grid_style = {
    "G9": ("o", "#1f5a91"),
    "G10": ("s", "#a8482a"),
    "G11": ("^", "#3d7f4e"),
}
records = {}
for label in grid_style:
    value = json.loads((P229_OUTPUT / f"protocol229_{label}.json").read_text(encoding="utf-8"))
    records[label] = value["result"]

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
for label, (marker, color) in grid_style.items():
    record = records[label]
    pairs = record["square_root_pairs"]
    critical_time = float(record["critical_time_estimate"])
    critical_area = float(record["critical_geometry"]["one_sided_cap_area"])
    delta_time = 1.0e6 * np.asarray(
        [float(item["time_over_ell"]) - critical_time for item in pairs]
    )
    outer_area = 100.0 * (
        np.asarray([float(item["positive_area"]) for item in pairs]) - critical_area
    ) / critical_area
    inner_area = 100.0 * (
        np.asarray([float(item["negative_area"]) for item in pairs]) - critical_area
    ) / critical_area
    outer_lambda = np.asarray([float(item["positive_eigenvalue"]) for item in pairs])
    inner_lambda = np.asarray([float(item["negative_eigenvalue"]) for item in pairs])

    axes[0].plot(delta_time, outer_area, color=color, marker=marker, markersize=3.8,
                 linewidth=1.0, label=label)
    axes[0].plot(delta_time, inner_area, color=color, marker=marker, markerfacecolor="white",
                 markersize=3.8, linewidth=1.0)
    axes[1].plot(delta_time, outer_lambda, color=color, marker=marker, markersize=3.8,
                 linewidth=1.0)
    axes[1].plot(delta_time, inner_lambda, color=color, marker=marker, markerfacecolor="white",
                 markersize=3.8, linewidth=1.0)

    fit = record["square_root_fit"]
    fit_time = float(fit["critical_time"])
    raw_time = np.asarray([float(item["time_over_ell"]) for item in pairs])
    separation = np.asarray([float(item["area_separation"]) for item in pairs])
    x = raw_time - fit_time
    axes[2].loglog(x, separation, linestyle="none", marker=marker, color=color,
                   markersize=4.0, label=label)
    xline = np.geomspace(float(np.min(x)), float(np.max(x)), 80)
    yline = np.sqrt(float(fit["slope"]) * xline)
    axes[2].loglog(xline, yline, color=color, linewidth=1.0)

axes[0].axhline(0.0, color="#777777", linewidth=0.75)
axes[0].set_xlabel(r"$10^6(t-t_*)/\ell$")
axes[0].set_ylabel(r"$(\mathcal{A}-\mathcal{A}_*)/\mathcal{A}_*$ (\%)")
axes[0].text(0.98, 0.94, "outer", transform=axes[0].transAxes, ha="right", va="top",
             fontsize=7.5)
axes[0].text(0.98, 0.08, "inner", transform=axes[0].transAxes, ha="right", va="bottom",
             fontsize=7.5)

axes[1].axhline(0.0, color="#777777", linewidth=0.75)
axes[1].set_xlabel(r"$10^6(t-t_*)/\ell$")
axes[1].set_ylabel(r"principal $\lambda$")

axes[2].set_xlabel(r"$(t-t_{*,\mathrm{fit}})/\ell$")
axes[2].set_ylabel(r"$\mathcal{A}_{\rm out}-\mathcal{A}_{\rm in}$")
axes[2].legend(frameon=False, loc="lower right")

for index, axis in enumerate(axes):
    axis.grid(True, which="both", color="#d8d8d8", linewidth=0.5)
    axis.tick_params(direction="in")
    axis.text(0.03, 0.96, f"({chr(ord('a') + index)})", transform=axis.transAxes,
              ha="left", va="top", fontweight="bold")
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

SADDLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SADDLE_OUTPUT, bbox_inches="tight")
plt.close(fig)
print(SADDLE_OUTPUT)
