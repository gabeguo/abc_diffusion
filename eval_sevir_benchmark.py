import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DiT_models
from sevir_benchmark.abc import get_method_flags, load_sde, rollout_bridge_model, validate_benchmark_config
from sevir_benchmark.data import DEFAULT_CLIP_STARTS, SevirVILDataset, inverse_normalize_vil
from sevir_benchmark.metrics import finalize_metric_state, init_metric_state, sync_metric_state, update_metric_state
from sevir_benchmark.utils import (
    barrier,
    cleanup_distributed,
    is_main_process,
    init_distributed,
    namespace_to_dict,
    save_json,
    set_seed,
    shard_dataset_for_rank,
    warmup_distributed_collectives,
)
from sevir_benchmark.visualization import save_plot_data, save_vil_grid


def build_dataset(eval_args, data_args):
    clip_starts = eval_args.clip_starts
    if clip_starts is None:
        clip_starts = getattr(data_args, "clip_starts", None)
    if clip_starts is None:
        clip_starts = list(DEFAULT_CLIP_STARTS)
    return SevirVILDataset(
        root_dir=eval_args.data_root,
        split=eval_args.split,
        image_size=eval_args.image_size if eval_args.image_size is not None else data_args.image_size,
        prefix_length=eval_args.prefix_length if eval_args.prefix_length is not None else data_args.prefix_length,
        target_length=eval_args.target_length if eval_args.target_length is not None else data_args.target_length,
        clip_starts=clip_starts,
        val_fraction=getattr(data_args, "val_fraction", 0.1),
        seed=getattr(data_args, "global_seed", 0),
        max_events=eval_args.max_events,
        normalize_mode=getattr(data_args, "normalize_mode", "zero_center"),
        return_native=True,
        return_metadata=True,
    )


def load_abc_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=1,
        num_classes=1,
        class_dropout_prob=0.0,
    ).to(device)
    state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state_dict)
    model.eval()
    sde = load_sde(args=args, model=model)
    if getattr(args, "brownian_bridge_residual", False):
        model.sde = sde
    return model, sde, args


def validate_eval_overrides(eval_args, data_args):
    if eval_args.image_size is not None and eval_args.image_size != data_args.image_size:
        raise ValueError(
            f"--image-size={eval_args.image_size} does not match checkpoint image_size={data_args.image_size}. "
            "DiT positional embeddings are tied to the training image size, so SEVIR evaluation "
            "must use the checkpoint's image size."
        )
    benchmark_fields = ("prefix_length", "target_length")
    for field in benchmark_fields:
        override = getattr(eval_args, field)
        trained = getattr(data_args, field)
        if override is not None and override != trained:
            raise ValueError(
                f"--{field.replace('_', '-')}={override} does not match checkpoint {field}={trained}. "
                "SEVIR evaluation must use the checkpoint's benchmark definition."
            )
    if eval_args.clip_starts is not None:
        override = list(eval_args.clip_starts)
        trained = list(getattr(data_args, "clip_starts", DEFAULT_CLIP_STARTS))
        if override != trained:
            raise ValueError(
                f"--clip-starts={override} does not match checkpoint clip_starts={trained}. "
                "SEVIR evaluation must use the checkpoint's benchmark definition."
            )


def average_member_metrics(member_metrics):
    if not member_metrics:
        raise ValueError("Expected at least one ensemble member metric dictionary.")
    numeric_keys = [
        key for key, value in member_metrics[0].items()
        if isinstance(value, (int, float)) and key != "num_examples"
    ]
    averaged = {
        key: sum(float(metrics[key]) for metrics in member_metrics) / len(member_metrics)
        for key in numeric_keys
    }
    averaged["num_examples"] = int(member_metrics[0]["num_examples"])
    return averaged


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEVIR VIL bridge models.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["abc", "conditional_diffusion_bridge", "noise_to_data_diffusion"],
        default=None,
        help="Optional override. Must match the method stored in the checkpoint.",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path for the bridge model.")
    parser.add_argument("--data-root", type=str, default="datasets/sevir")
    parser.add_argument("--out-dir", type=str, default="results/sevir/eval")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--prefix-length", type=int, default=None)
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--clip-starts", type=int, nargs="*", default=None)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[16.0, 74.0, 133.0, 160.0])
    parser.add_argument("--save-grid-every", type=int, default=100)
    parser.add_argument("--save-plot-data-every", type=int, default=None)
    parser.add_argument("--plot-max-items", type=int, default=4)
    parser.add_argument("--num-sampling-steps", type=int, default=48)
    parser.add_argument("--num-ensemble-members", type=int, default=1)
    parser.add_argument("--clip-dt-eps", type=float, default=1e-6)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--conditioning-prefix-length", type=int, default=1)
    parser.add_argument("--pin-every", type=int, default=8)
    parser.add_argument("--teacher-force-pinned", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = init_distributed()
    warmup_distributed_collectives(device)
    if is_main_process():
        save_json(namespace_to_dict(args), os.path.join(args.out_dir, "eval_args.json"))
    set_seed(args.seed)

    model, sde, data_args = load_abc_model(args.ckpt, device=device)
    checkpoint_method = getattr(data_args, "method", "abc")
    if args.method is not None and args.method != checkpoint_method:
        raise ValueError(
            f"--method={args.method} does not match checkpoint method={checkpoint_method}. "
            "SEVIR evaluation requires the checkpoint's training method."
        )
    method = checkpoint_method
    get_method_flags(method)
    validate_benchmark_config(args=data_args, method=method)
    validate_eval_overrides(eval_args=args, data_args=data_args)

    base_dataset = build_dataset(eval_args=args, data_args=data_args)
    dataset = shard_dataset_for_rank(base_dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    member_states = [init_metric_state(args.thresholds) for _ in range(args.num_ensemble_members)]
    saved_plot_batches = []
    save_plot_data_every = args.save_plot_data_every if args.save_plot_data_every is not None else args.save_grid_every
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Eval {method}", disable=(not is_main_process()))):
        videos, _, native_videos, metadata = batch
        videos = videos.to(device)
        native_videos = native_videos.to(device)

        ensemble = []
        ensemble_native = []
        eval_frame_mask = None
        conditioning_frames = None
        for member_idx in range(args.num_ensemble_members):
            rollout = rollout_bridge_model(
                model=model,
                sde=sde,
                videos=videos,
                num_sampling_steps=args.num_sampling_steps,
                method=method,
                causal=args.causal,
                prefix_length=args.conditioning_prefix_length,
                pin_every=args.pin_every,
                teacher_force_pinned=args.teacher_force_pinned,
                clip_dt_eps=args.clip_dt_eps,
            )
            pred = rollout["pred_frames"].clamp(-1.0, 1.0)
            ensemble.append(pred)
            eval_frame_mask = rollout["eval_frame_mask"]
            conditioning_frames = rollout["conditioning_frames"]
            pred_native = pred
            if pred_native.shape[-2:] != native_videos.shape[-2:]:
                pred_native = F.interpolate(
                    pred_native.flatten(0, 1),
                    size=native_videos.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).unflatten(0, pred_native.shape[:2])
            ensemble_native.append(pred_native)
            pred_raw = inverse_normalize_vil(pred_native, normalize_mode=base_dataset.normalize_mode).cpu()
            target_raw = inverse_normalize_vil(native_videos, normalize_mode=base_dataset.normalize_mode).cpu()
            update_metric_state(
                member_states[member_idx],
                pred_raw=pred_raw,
                target_raw=target_raw,
                eval_mask=eval_frame_mask.cpu() if eval_frame_mask is not None else None,
            )
        pred_members = torch.stack(ensemble, dim=0)
        pred_for_save = pred_members.mean(dim=0).clamp(-1.0, 1.0)
        pred_native_members = torch.stack(ensemble_native, dim=0)
        pred_native_mean = pred_native_members.mean(dim=0).clamp(-1.0, 1.0)

        if is_main_process() and batch_idx % args.save_grid_every == 0:
            save_vil_grid(
                context=conditioning_frames.cpu(),
                target=videos.cpu(),
                prediction=pred_for_save.cpu(),
                output_dir=os.path.join(args.out_dir, f"batch_{batch_idx:04d}"),
                normalize_mode=base_dataset.normalize_mode,
            )
        if is_main_process() and batch_idx % save_plot_data_every == 0:
            batch_dir = os.path.join(args.out_dir, f"batch_{batch_idx:04d}")
            conditioning_native = conditioning_frames
            if conditioning_native.shape[-2:] != native_videos.shape[-2:]:
                conditioning_native = F.interpolate(
                    conditioning_native.flatten(0, 1),
                    size=native_videos.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).unflatten(0, conditioning_native.shape[:2])
            plot_items = min(args.plot_max_items, pred_native_mean.shape[0])
            save_plot_data(
                output_dir=batch_dir,
                method=method,
                batch_idx=batch_idx,
                pred_raw_mean=inverse_normalize_vil(
                    pred_native_mean[:plot_items],
                    normalize_mode=base_dataset.normalize_mode,
                ),
                pred_raw_members=inverse_normalize_vil(
                    pred_native_members[:, :plot_items],
                    normalize_mode=base_dataset.normalize_mode,
                ),
                target_raw=inverse_normalize_vil(
                    native_videos[:plot_items],
                    normalize_mode=base_dataset.normalize_mode,
                ),
                conditioning_raw=inverse_normalize_vil(
                    conditioning_native[:plot_items],
                    normalize_mode=base_dataset.normalize_mode,
                ),
                eval_frame_mask=eval_frame_mask.cpu(),
                frame_times=torch.linspace(0, 1, videos.shape[1]),
                metadata={
                    "event_id": metadata.get("event_id", [])[:plot_items],
                    "timestamp": metadata.get("timestamp", [])[:plot_items],
                    "clip_start": metadata.get("clip_start", torch.tensor([]))[:plot_items].tolist()
                    if isinstance(metadata.get("clip_start"), torch.Tensor)
                    else metadata.get("clip_start", [])[:plot_items],
                    "dataset_index": metadata.get("dataset_index", torch.tensor([]))[:plot_items].tolist()
                    if isinstance(metadata.get("dataset_index"), torch.Tensor)
                    else metadata.get("dataset_index", [])[:plot_items],
                    "causal": args.causal,
                    "conditioning_prefix_length": args.conditioning_prefix_length,
                    "pin_every": args.pin_every,
                },
            )
            saved_plot_batches.append(f"batch_{batch_idx:04d}")

    barrier()
    member_states = [sync_metric_state(state) for state in member_states]
    per_member_metrics = [finalize_metric_state(state) for state in member_states]
    metrics = average_member_metrics(per_member_metrics)
    metrics["num_predictions"] = metrics["num_examples"] * args.num_ensemble_members
    metrics["ckpt"] = args.ckpt
    metrics["method"] = method
    metrics["num_ensemble_members"] = args.num_ensemble_members
    metrics["metric_aggregation"] = "mean_of_member_metrics"
    metrics["metric_resolution"] = "native"
    metrics["causal"] = args.causal
    metrics["conditioning_prefix_length"] = args.conditioning_prefix_length
    metrics["pin_every"] = args.pin_every
    metrics["saved_plot_batches"] = saved_plot_batches
    barrier()
    if is_main_process():
        save_json(metrics, os.path.join(args.out_dir, "metrics.json"))
        print(metrics)
    cleanup_distributed()


if __name__ == "__main__":
    main()
