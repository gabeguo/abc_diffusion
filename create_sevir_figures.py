#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

from sevir_benchmark.visualization import load_plot_data, render_rollout_figure


DEFAULT_METHOD_LABELS = {
    "abc": "ABC",
    "abc_noncausal": "ABC Non-Causal",
    "abc_causal": "ABC Causal",
    "conditional_diffusion_bridge": "Conditional Diffusion Bridge",
    "noise_to_data_diffusion": "Noise-to-Data Diffusion",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render polished SEVIR comparison figures from saved plot artifacts.")
    parser.add_argument("--eval-root", type=str, required=True, help="Root eval directory containing method subfolders.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["abc_noncausal", "abc_causal", "conditional_diffusion_bridge", "noise_to_data_diffusion"],
        help="Method subdirectories under --eval-root to include.",
    )
    parser.add_argument("--batch-name", type=str, default="batch_0000", help="Saved batch folder to render.")
    parser.add_argument("--example-index", type=int, default=0, help="Example index within the saved batch artifact.")
    parser.add_argument("--frame-stride", type=int, default=2, help="Stride when selecting frames to display.")
    parser.add_argument("--max-frames", type=int, default=8, help="Maximum number of frames to show after stride.")
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Optional explicit frame indices.")
    parser.add_argument("--out-dir", type=str, default="results/sevir/figures", help="Output directory for rendered figures.")
    parser.add_argument("--output-name", type=str, default="sevir_qualitative", help="Base name for rendered outputs.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-uncertainty", action="store_true", help="Hide ensemble standard deviation rows.")
    parser.add_argument("--no-title", action="store_true", help="Hide the figure title/metadata header.")
    parser.add_argument("--no-colorbar", action="store_true", help="Hide the right-hand colorbar.")
    return parser.parse_args()


def _artifact_path(eval_root: str, method: str, batch_name: str) -> str:
    return os.path.join(eval_root, method, batch_name, "plot_data.pt")


def load_artifacts(eval_root: str, methods: List[str], batch_name: str) -> Dict[str, Dict[str, object]]:
    artifacts = {}
    for method in methods:
        artifact_path = _artifact_path(eval_root=eval_root, method=method, batch_name=batch_name)
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(
                f"Missing plot artifact for method '{method}': {artifact_path}. "
                "Run eval_sevir_benchmark.py first, or re-run the benchmark without --replot."
            )
        artifacts[DEFAULT_METHOD_LABELS.get(method, method)] = load_plot_data(artifact_path)
    return artifacts


def choose_frame_indices(num_frames_total: int, frames: List[int], frame_stride: int, max_frames: int) -> List[int]:
    if frames is not None and len(frames) > 0:
        frame_indices = list(frames)
    else:
        frame_indices = list(range(0, num_frames_total, frame_stride))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError("No frame indices selected.")
    return frame_indices


def main():
    args = parse_args()
    artifacts = load_artifacts(eval_root=args.eval_root, methods=args.methods, batch_name=args.batch_name)
    first_artifact = next(iter(artifacts.values()))
    num_frames_total = int(first_artifact["target_raw"].shape[1])
    frame_indices = choose_frame_indices(
        num_frames_total=num_frames_total,
        frames=args.frames,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, f"{args.output_name}.png")
    render_rollout_figure(
        artifacts_by_label=artifacts,
        output_path=output_path,
        example_index=args.example_index,
        frame_indices=frame_indices,
        show_uncertainty=not args.no_uncertainty,
        show_title=not args.no_title,
        show_colorbar=not args.no_colorbar,
        dpi=args.dpi,
    )

    args_record = {
        **vars(args),
        "frame_indices": frame_indices,
        "artifacts": {
            label: _artifact_path(args.eval_root, method, args.batch_name)
            for method, label in ((m, DEFAULT_METHOD_LABELS.get(m, m)) for m in args.methods)
        },
    }
    with open(os.path.join(args.out_dir, f"{args.output_name}.json"), "w") as f:
        json.dump(args_record, f, indent=4)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
