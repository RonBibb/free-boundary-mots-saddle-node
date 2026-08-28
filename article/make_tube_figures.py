#!/usr/bin/env python3
"""Generate the post-fold tube figures from sealed Protocol 240/244/247/249/250 results.

The script reads result records without modifying them and verifies their exact
SHA-256 identities before plotting.  Pass the root of the separately preserved
``free-boundary-mots-horizon-tube`` archive with ``--tube-root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "figures"

RECORDS = {
    "p240": (
        "protocol240-dense-g10-outer-tube-v3-2026-08-25/"
        "candidate-output/protocol240_result.json",
        "b07a9fbe51af819594643427f37d945cf50c4acb43efaa2feeeec29086aff690",
    ),
    "p244": (
        "protocol244-full-dt-g10-dense-tube-replay-2026-08-25/"
        "candidate-output/protocol244_result.json",
        "60ca87bcb4bd45937d2babace30e50b1ea059839fa554cd21c9ff3803879958f",
    ),
    "p247": (
        "protocol247-g9-g11-bounded-spatial-transfer-2026-08-25/"
        "candidate-output/protocol247_result.json",
        "515e365bd7d8f1b1caeb7356724c0d3ca129364556de64898cf563cd8b122642",
    ),
    "p249": (
        "protocol249-three-grid-integrated-balance-2026-08-25/"
        "candidate-output/protocol249_result.json",
        "6d54755023ce09beaa992e57a1eef1c1d2fcd3f42db0e662d08c25bd557e9317",
    ),
    "p250": (
        "protocol250-g10-full-half-causal-signature-2026-08-26/"
        "candidate-output/protocol250_result.json",
        "6866d08889beb5107436c3975b7d69ff7f48a80d244ff185ea954292ede400b3",
    ),
}


def read_verified(root: Path, relative: str, expected_sha256: str) -> dict:
    path = root / relative
    payload = path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f"sealed identity differs for {path}: {observed}")
    return json.loads(payload)


def leaf_series(record: dict, steps: list[int]) -> dict[str, np.ndarray]:
    leaves = record["scientific"]["evaluation"]["leaves"]
    chosen = [leaves[str(step)] for step in steps]
    return {
        "time": np.asarray([item["time_over_ell"] for item in chosen], dtype=float),
        "area": np.asarray(
            [item["geometry"]["one_sided_cap_area"] for item in chosen], dtype=float
        ),
        "theta_minus": np.asarray(
            [
                item["negative_inward_resolution"][
                    "maximum_theta_minus_any_stencil"
                ]
                for item in chosen
            ],
            dtype=float,
        ),
        "lambda0": np.asarray(
            [item["stability"]["fine_principal_eigenvalue"] for item in chosen],
            dtype=float,
        ),
    }


def spatial_leaf_series(record: dict, grid: str, steps: list[int]) -> dict[str, np.ndarray]:
    leaves = record["scientific"]["grids"][grid]["leaves"]
    chosen = [leaves[str(step)] for step in steps]
    return {
        "time": np.asarray([item["time_over_ell"] for item in chosen], dtype=float),
        "area": np.asarray(
            [item["geometry"]["one_sided_cap_area"] for item in chosen], dtype=float
        ),
        "theta_minus": np.asarray(
            [
                item["negative_inward_resolution"][
                    "maximum_theta_minus_any_stencil"
                ]
                for item in chosen
            ],
            dtype=float,
        ),
        "lambda0": np.asarray(
            [item["stability"]["fine_principal_eigenvalue"] for item in chosen],
            dtype=float,
        ),
    }


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def finish_axes(axes) -> None:
    for index, axis in enumerate(np.ravel(axes)):
        axis.grid(True, color="#d8d8d8", linewidth=0.5)
        axis.tick_params(direction="in")
        axis.text(
            0.03,
            0.96,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)


def make_tube_evolution(records: dict[str, dict]) -> Path:
    full_steps = list(range(39, 48))
    half_steps = list(range(78, 96, 2))
    spatial_steps = list(range(43, 48))
    full = leaf_series(records["p244"], full_steps)
    half = leaf_series(records["p240"], half_steps)
    spatial = {
        grid: spatial_leaf_series(records["p247"], grid, spatial_steps)
        for grid in ("G9", "G11")
    }
    lanes = records["p250"]["scientific"]["lanes"]
    full_margin = np.asarray(
        [lanes["full"]["records"][str(step)]["minimum_resolution_margin"] for step in full_steps]
    )
    half_margin = np.asarray(
        [lanes["half"]["records"][str(step)]["minimum_resolution_margin"] for step in half_steps]
    )
    if not np.isclose(min(np.min(full_margin), np.min(half_margin)), 54.00554986042094):
        raise RuntimeError("causal-resolution minimum differs from the sealed record")

    colors = {"G9": "#6c6c6c", "G10": "#1f5a91", "G11": "#3d7f4e"}
    markers = {"G9": "o", "G10": "s", "G11": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)

    quantities = (
        ("area", r"one-sided $\mathcal{A}$"),
        ("theta_minus", r"largest $\theta_{(n)}$"),
        ("lambda0", r"principal $\lambda_0$"),
    )
    for axis, (key, ylabel) in zip(axes.ravel()[:3], quantities):
        axis.plot(
            full["time"], full[key], color=colors["G10"], marker=markers["G10"],
            linewidth=1.15, markersize=3.5, label=r"G10 $\Delta t$",
        )
        axis.plot(
            half["time"], half[key], color="#a8482a", marker="D", fillstyle="none",
            linestyle="--", linewidth=1.0, markersize=3.3, label=r"G10 $\Delta t/2$",
        )
        for grid in ("G9", "G11"):
            axis.plot(
                spatial[grid]["time"], spatial[grid][key], linestyle="none",
                marker=markers[grid], markerfacecolor="white", markeredgecolor=colors[grid],
                markersize=4.2, label=grid,
            )
        axis.set_xlabel(r"$t/\ell$")
        axis.set_ylabel(ylabel)

    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    axes[1, 1].plot(
        full["time"], full_margin, color=colors["G10"], marker=markers["G10"],
        linewidth=1.15, markersize=3.5, label=r"G10 $\Delta t$",
    )
    axes[1, 1].plot(
        half["time"], half_margin, color="#a8482a", marker="D", fillstyle="none",
        linestyle="--", linewidth=1.0, markersize=3.3, label=r"G10 $\Delta t/2$",
    )
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xlabel(r"$t/\ell$")
    axes[1, 1].set_ylabel("minimum causal-resolution margin")
    axes[1, 1].annotate(
        "minimum 54.01",
        xy=(full["time"][-1], full_margin[-1]),
        xytext=(-55, 18), textcoords="offset points", fontsize=7.5,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
    )
    finish_axes(axes)
    path = OUTPUT / "tube-evolution.pdf"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_tube_balance(records: dict[str, dict]) -> Path:
    science = records["p249"]["scientific"]
    grids = ("G9", "G10", "G11")
    widths = (5, 7, 9)
    colors = {"G9": "#6c6c6c", "G10": "#1f5a91", "G11": "#3d7f4e"}
    markers = {5: "o", 7: "s", 9: "^"}

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
    all_charge = []
    all_flux = []
    for grid in grids:
        charge = []
        flux = []
        discrepancy = []
        boundary_fraction = []
        for width in widths:
            item = science["grids"][grid][str(width)]
            charge.append(float(item["charge_change"]))
            flux.append(float(item["integrated_total_flux"]))
            discrepancy.append(100.0 * float(item["residuals"]["charge_flux_relative"]))
            boundary_fraction.append(
                100.0 * float(item["integrated_brane_endpoint_flux"])
                / float(item["integrated_total_flux"])
            )
            axes[0].plot(
                charge[-1], flux[-1], linestyle="none", marker=markers[width],
                color=colors[grid], markersize=5.0,
                label=f"{grid}, w={width}" if grid == "G10" else None,
            )
        axes[1].plot(widths, discrepancy, color=colors[grid], marker="o", linewidth=1.1,
                     markersize=3.8, label=grid)
        axes[2].plot(widths, boundary_fraction, color=colors[grid], marker="o", linewidth=1.1,
                     markersize=3.8, label=grid)
        all_charge.extend(charge)
        all_flux.extend(flux)

    lower = min(all_charge + all_flux)
    upper = max(all_charge + all_flux)
    pad = 0.08 * (upper - lower)
    axes[0].plot([lower - pad, upper + pad], [lower - pad, upper + pad],
                 color="#555555", linewidth=0.8, linestyle="--")
    axes[0].set_xlim(lower - pad, upper + pad)
    axes[0].set_ylim(lower - pad, upper + pad)
    axes[0].set_xlabel(r"charge change $\Delta Q_{\rm AdS}$")
    axes[0].set_ylabel(r"integrated flux $\int F\,dt$")
    axes[0].text(0.97, 0.08, "diagonal: exact closure", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=7.2, color="#555555")

    axes[1].axhline(1.0, color="#a8482a", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(widths)
    axes[1].set_xlabel("spatial stencil width")
    axes[1].set_ylabel(r"$|\Delta Q-\int Fdt|/\max|\cdot|$ (\%)")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(frameon=False, loc="upper right")

    axes[2].set_xticks(widths)
    axes[2].set_xlabel("spatial stencil width")
    axes[2].set_ylabel(r"brane-endpoint subtotal / total flux (\%)")
    axes[2].annotate(
        "59.44%",
        xy=(7, 100.0 * science["grids"]["G10"]["7"]["integrated_brane_endpoint_flux"]
            / science["grids"]["G10"]["7"]["integrated_total_flux"]),
        xytext=(8, 12), textcoords="offset points", fontsize=7.5,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
    )
    finish_axes(axes)
    path = OUTPUT / "tube-balance.pdf"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-root", required=True, type=Path)
    args = parser.parse_args()
    records = {
        name: read_verified(args.tube_root, relative, digest)
        for name, (relative, digest) in RECORDS.items()
    }
    set_plot_style()
    print(make_tube_evolution(records))
    print(make_tube_balance(records))


if __name__ == "__main__":
    main()
