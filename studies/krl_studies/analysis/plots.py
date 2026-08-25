"""Plotting functions for Task 5 publication figures."""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {
    "rl": "#1f77b4",
    "krl": "#ff7f0e",
    "hkrl": "#2ca02c",
    "dtv": "#d62728",
    "iy": "#9467bd",
    "post_smoothing": "#8c564b",
    "gtm": "#e377c2",
}


def _ensure_output(output: Path) -> None:
    """Ensure output directory exists and create a figure with proper cleanup."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)


def _empty_figure(output: Path, title: str) -> None:
    """Create an empty labelled figure for empty inputs."""
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("NRMSE")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    try:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_nrmse_convergence(summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write mean +/- std NRMSE versus iteration, grouped by method."""
    output = Path(output)
    _ensure_output(output)

    if summary.empty:
        _empty_figure(output, title)
        return

    fig, ax = plt.subplots()
    try:
        # Filter for NRMSE metric
        nrmse = summary[summary["metric"] == "nrmse"].copy()
        if nrmse.empty:
            _empty_figure(output, title)
            return

        # Group by method, condition, beta, guidance_condition, assumed_fwhm_mm
        for method in sorted(nrmse["method"].unique()):
            color = METHOD_COLORS.get(method, "#000000")
            method_data = nrmse[nrmse["method"] == method]

            # Sort by iteration
            method_data = method_data.sort_values("iteration")

            # Build label
            cond = method_data["condition"].iloc[0]
            beta = method_data["beta"].iloc[0]
            guidance = method_data["guidance_condition"].iloc[0]
            label_parts = [f"{method}", f"{cond}"]
            if beta is not None:
                label_parts.append(f"β={beta}")
            if guidance != "exact":
                label_parts.append(guidance)
            label = " ".join(label_parts)

            ax.errorbar(
                method_data["iteration"],
                method_data["value_mean"],
                yerr=method_data["value_std"],
                label=label,
                color=color,
                marker="o",
                capsize=3,
            )

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NRMSE")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_recovery_vs_cov(summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write CRC/NRMSE versus background variability."""
    output = Path(output)
    _ensure_output(output)

    if summary.empty:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Background Variability (%)")
        ax.set_ylabel("Recovery")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots()
    try:
        for method in sorted(summary["method"].unique()):
            color = METHOD_COLORS.get(method, "#000000")
            method_data = summary[summary["method"] == method].sort_values("iteration")

            if "bv_percent" not in method_data.columns:
                continue

            # Prefer crc_percent, fall back to nrmse
            if "crc_percent" in method_data.columns:
                x = method_data["bv_percent"]
                y = method_data["crc_percent"]
                ylabel = "CRC (%)"
            else:
                x = method_data["bv_percent"]
                y = method_data["nrmse"]
                ylabel = "NRMSE"

            label_parts = [method]
            cond = method_data["condition"].iloc[0]
            label_parts.append(cond)
            label = " ".join(label_parts)

            ax.plot(x, y, "o-", label=label, color=METHOD_COLORS.get(method, "#000000"))

        ax.set_title(title)
        ax.set_xlabel("Background Variability (%)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close()


def plot_crc_by_size(lesion_summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write mean +/- std CRC versus lesion diameter."""
    output = Path(output)
    _ensure_output(output)

    if lesion_summary.empty:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Lesion Diameter (mm)")
        ax.set_ylabel("CRC (%)")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots()
    try:
        crc = lesion_summary[lesion_summary["metric"] == "crc_percent"].copy()
        if crc.empty:
            fig.savefig(output, dpi=200, bbox_inches="tight")
            plt.close()
            return

        for method in sorted(crc["method"].unique()):
            color = METHOD_COLORS.get(method, "#000000")
            method_data = crc[crc["method"] == method].sort_values("lesion_diameter_mm")

            cond = method_data["condition"].iloc[0]
            guidance = method_data["guidance_condition"].iloc[0]
            label = f"{method} {cond}"
            if guidance != "exact":
                label += f" {guidance}"

            ax.errorbar(
                method_data["lesion_diameter_mm"],
                method_data["value_mean"],
                yerr=method_data["value_std"],
                label=label,
                color=color,
                marker="o",
                capsize=3,
            )

        ax.set_title(title)
        ax.set_xlabel("Lesion Diameter (mm)")
        ax.set_ylabel("CRC (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close()


def plot_mismatch_sensitivity(summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write the selected metric versus assumed deconvolution FWHM."""
    output = Path(output)
    _ensure_output(output)

    if summary.empty:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Assumed FWHM (mm)")
        ax.set_ylabel("Metric")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots()
    try:
        for method in sorted(summary["method"].unique()):
            color = METHOD_COLORS.get(method, "#000000")
            method_data = summary[summary["method"] == method].sort_values("assumed_fwhm_mm")

            cond = method_data["condition"].iloc[0]
            guidance = method_data["guidance_condition"].iloc[0]
            recon = method_data["recon_model_fwhm_json"].iloc[0]
            label_parts = [method, cond]
            if recon != "null":
                label_parts.append(f"recon={recon}")
            if guidance != "exact":
                label_parts.append(guidance)
            label = " ".join(label_parts)

            ax.plot(
                method_data["assumed_fwhm_mm"],
                method_data["value_mean"],
                "o-",
                label=label,
                color=METHOD_COLORS.get(method, "#000000"),
            )

        ax.set_title(title)
        ax.set_xlabel("Assumed Deconvolution FWHM (mm)")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close()


def plot_profile(
    images: dict[str, np.ndarray],
    output: Path,
    *,
    axis: int,
    index: tuple[int, int],
) -> None:
    """Write fixed-index one-dimensional profiles from supplied arrays."""
    output = Path(output)
    _ensure_output(output)

    if axis not in (0, 1, 2):
        raise ValueError("axis must be in {0, 1, 2}")

    fig, ax = plt.subplots()
    try:
        if not images:
            ax.set_title("Profile")
            ax.text(0.5, 0.5, "No images", ha="center", va="center", transform=ax.transAxes)
            fig.savefig(output, dpi=200, bbox_inches="tight")
            plt.close()
            return

        i1, i2 = index
        for name, img in images.items():
            img_arr = np.asarray(img)
            if axis == 0:
                if i1 >= img_arr.shape[1] or i2 >= img_arr.shape[2]:
                    raise ValueError(f"Index ({i1}, {i2}) out of bounds for image shape {img_arr.shape}")
                profile = img_arr[:, i1, i2]
            elif axis == 1:
                if i1 >= img_arr.shape[0] or i2 >= img_arr.shape[2]:
                    raise ValueError(f"Index ({i1}, {i2}) out of bounds for image shape {img_arr.shape}")
                profile = img_arr[i1, :, i2]
            else:  # axis == 2
                if i1 >= img_arr.shape[0] or i2 >= img_arr.shape[1]:
                    raise ValueError(f"Index ({i1}, {i2}) out of bounds for image shape {img_arr.shape}")
                profile = img_arr[i1, i2, :]

            ax.plot(profile, label=name)

        ax.set_title("Profile")
        ax.set_xlabel("Voxel index")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close()
