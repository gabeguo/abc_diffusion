import os
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image


def save_vil_grid(
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_dir: str,
    max_items: int = 4,
    normalize_mode: str = "zero_center",
):
    os.makedirs(output_dir, exist_ok=True)
    batch = min(int(context.shape[0]), max_items)
    all_frames = torch.cat([context[:batch], target[:batch], prediction[:batch]], dim=1)
    num_frames = all_frames.shape[1]
    if normalize_mode == "zero_center":
        value_range = (-1, 1)
    elif normalize_mode == "unit":
        value_range = (0, 1)
    else:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
    for t in range(num_frames):
        save_image(
            all_frames[:, t],
            os.path.join(output_dir, f"frame_{t:02d}.png"),
            nrow=batch,
            normalize=True,
            value_range=value_range,
        )


def save_plot_data(
    output_dir: str,
    method: str,
    batch_idx: int,
    pred_raw_mean: torch.Tensor,
    pred_raw_members: torch.Tensor,
    target_raw: torch.Tensor,
    conditioning_raw: torch.Tensor,
    eval_frame_mask: torch.Tensor,
    frame_times: torch.Tensor,
    metadata: Optional[Dict[str, object]] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    conditioning_mask = ~eval_frame_mask.bool()

    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 255.0).round().to(torch.uint8)

    artifact = {
        "method": method,
        "batch_idx": int(batch_idx),
        "pred_raw_mean": to_uint8(pred_raw_mean.cpu()),
        "pred_raw_members": to_uint8(pred_raw_members.cpu()),
        "target_raw": to_uint8(target_raw.cpu()),
        "conditioning_raw": to_uint8(conditioning_raw.cpu()),
        "eval_frame_mask": eval_frame_mask.cpu().bool(),
        "conditioning_mask": conditioning_mask.cpu().bool(),
        "frame_times": frame_times.cpu().float(),
        "metadata": metadata or {},
    }
    torch.save(artifact, os.path.join(output_dir, "plot_data.pt"))


def load_plot_data(path: str) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_vil_cmap():
    colors = [
        "#06141f",
        "#10324a",
        "#0c5f6b",
        "#1aa38d",
        "#6fd36a",
        "#f0d54e",
        "#f39b34",
        "#d94a2b",
        "#7d1638",
        "#f2f2f2",
    ]
    return LinearSegmentedColormap.from_list("sevir_vil", colors, N=512)


def _frame_status(conditioning_mask: torch.Tensor, frame_idx: int, causal: bool) -> str:
    if not bool(conditioning_mask[frame_idx]):
        return "forecast"
    if causal:
        return "context"
    return "pinned"


def _status_color(status: str) -> str:
    return {
        "context": "#0f8a8a",
        "pinned": "#6a4c93",
        "forecast": "#d1495b",
    }[status]


def render_rollout_figure(
    artifacts_by_label: Dict[str, Dict[str, object]],
    output_path: str,
    example_index: int = 0,
    frame_indices: Optional[Sequence[int]] = None,
    show_uncertainty: bool = True,
    show_title: bool = True,
    show_colorbar: bool = True,
    dpi: int = 300,
):
    if not artifacts_by_label:
        raise ValueError("Expected at least one artifact to render.")

    labels = list(artifacts_by_label.keys())
    reference = artifacts_by_label[labels[0]]
    target_raw = reference["target_raw"][example_index, :, 0].numpy().astype(np.float32)
    conditioning_raw = reference["conditioning_raw"][example_index, :, 0].numpy().astype(np.float32)
    conditioning_mask = reference["conditioning_mask"].numpy().astype(bool)
    frame_times = reference["frame_times"].numpy()
    if frame_indices is None:
        frame_indices = list(range(target_raw.shape[0]))
    if not frame_indices:
        raise ValueError("No frame indices selected for rendering.")

    pred_blocks = []
    uncertainty_blocks = []
    for label in labels:
        artifact = artifacts_by_label[label]
        pred_raw_mean = artifact["pred_raw_mean"][example_index, :, 0].numpy().astype(np.float32)
        pred_raw_members = artifact["pred_raw_members"][:, example_index, :, 0].numpy().astype(np.float32)
        pred_blocks.append((label, pred_raw_mean))
        uncertainty_blocks.append((label, pred_raw_members.std(axis=0)))

    rows = [("Ground Truth", target_raw)]
    rows.extend(pred_blocks)
    height_ratios = [1.0] * len(rows)
    if show_uncertainty:
        for label, std_map in uncertainty_blocks:
            rows.append((f"{label} Std. Dev.", std_map))
            height_ratios.append(0.8)

    n_rows = len(rows)
    n_cols = len(frame_indices)
    fig_w = max(8.0, 2.4 * n_cols)
    fig_h = 1.8 * sum(height_ratios)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(n_rows, n_cols),
        axes_pad=(0.20, 0.48),
        share_all=True,
        direction="row",
        cbar_mode=None,
    )
    axes = np.array(grid).reshape(n_rows, n_cols)
    fig.patch.set_facecolor("#ffffff")

    vil_cmap = _build_vil_cmap()
    vil_norm = Normalize(vmin=0.0, vmax=255.0)
    std_norm = Normalize(vmin=0.0, vmax=64.0)
    std_cmap = plt.get_cmap("magma")
    metadata = reference.get("metadata", {})
    causal = bool(metadata.get("causal", False))
    title_bits = []
    event_ids = metadata.get("event_id")
    if isinstance(event_ids, list) and example_index < len(event_ids):
        title_bits.append(f"Event {event_ids[example_index]}")
    timestamps = metadata.get("timestamp")
    if isinstance(timestamps, list) and example_index < len(timestamps):
        title_bits.append(str(timestamps[example_index]))
    if show_title and title_bits:
        fig.suptitle("  |  ".join(title_bits), fontsize=15, fontweight="bold", y=0.995)

    for row_idx, (row_label, row_frames) in enumerate(rows):
        is_uncertainty = row_label.endswith("Std. Dev.")
        for col_idx, frame_idx in enumerate(frame_indices):
            ax = axes[row_idx, col_idx]
            frame = row_frames[frame_idx]
            ax.imshow(frame, cmap=std_cmap if is_uncertainty else vil_cmap, norm=std_norm if is_uncertainty else vil_norm)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                status = _frame_status(torch.from_numpy(conditioning_mask), frame_idx, causal=causal)
                accent = _status_color(status)
                ax.set_title(
                    f"t={frame_idx}\n{status}",
                    fontsize=10,
                    color=accent,
                    pad=7,
                    fontweight="bold",
                )
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.5)
                    spine.set_edgecolor(accent)
            elif bool(conditioning_mask[frame_idx]):
                accent = _status_color(_frame_status(torch.from_numpy(conditioning_mask), frame_idx, causal=causal))
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.8)
                    spine.set_edgecolor(accent)

            if col_idx == 0:
                label_y = -0.16
                if row_label == "Noise-to-Data Diffusion":
                    label_y = -0.22
                ax.text(
                    0.0,
                    label_y,
                    row_label,
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="black",
                    ha="left",
                    va="top",
                    color="#1f2933",
                    clip_on=False,
                    bbox={
                        "facecolor": "#ffffff",
                        "edgecolor": "none",
                        "pad": 0.6,
                    },
                )

            if row_idx == n_rows - 1:
                ax.set_xlabel(f"{frame_times[frame_idx]:.2f}", fontsize=9, color="#374151")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#ffffff")

    right_margin = 0.9
    if show_colorbar:
        cbar_ax_main = fig.add_axes([0.92, 0.10, 0.018, 0.78])
        plt.colorbar(plt.cm.ScalarMappable(norm=vil_norm, cmap=vil_cmap), cax=cbar_ax_main)
        cbar_ax_main.set_title("VIL", fontsize=9, pad=6)
        if show_uncertainty:
            cbar_ax_std = fig.add_axes([0.945, 0.10, 0.012, 0.25])
            plt.colorbar(plt.cm.ScalarMappable(norm=std_norm, cmap=std_cmap), cax=cbar_ax_std)
            cbar_ax_std.set_title("Std", fontsize=8, pad=6)
        right_margin = 0.9
    else:
        right_margin = 0.985

    plt.subplots_adjust(left=0.12, right=right_margin, top=0.92, bottom=0.06)
    grid_left = axes[0, 0].get_position().x0
    grid_right = axes[0, -1].get_position().x1
    fig.text(
        0.5 * (grid_left + grid_right) + 0.025,
        0.015,
        "Normalized physical time",
        ha="center",
        fontsize=11,
        color="#374151",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
