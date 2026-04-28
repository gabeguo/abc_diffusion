from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist


EPS = 1e-8


def init_metric_state(thresholds: Iterable[float]) -> Dict[str, object]:
    thresholds = [float(t) for t in thresholds]
    return {
        "num_examples": 0,
        "sum_mae": 0.0,
        "sum_mse": 0.0,
        "num_values": 0,
        "thresholds": thresholds,
        "counts": {
            str(t): {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
            for t in thresholds
        },
    }


def update_metric_state(
    state: Dict[str, object],
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    eval_mask: Optional[torch.Tensor] = None,
):
    if pred_raw.shape != target_raw.shape:
        raise ValueError(f"Prediction and target shape mismatch: {pred_raw.shape} vs {target_raw.shape}")
    if eval_mask is not None:
        if eval_mask.ndim == 1:
            eval_mask = eval_mask.unsqueeze(0).expand(pred_raw.shape[0], -1)
        if eval_mask.ndim != 2 or eval_mask.shape != pred_raw.shape[:2]:
            raise ValueError(
                f"Expected eval_mask with shape {(pred_raw.shape[0], pred_raw.shape[1])}, got {tuple(eval_mask.shape)}"
            )
        eval_mask = eval_mask.to(device=pred_raw.device, dtype=torch.bool)
        eval_mask = eval_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        eval_mask = torch.ones_like(pred_raw[:, :, :1, :1, :1], dtype=torch.bool)

    diff = pred_raw - target_raw
    selected_diff = diff.masked_select(eval_mask)
    state["sum_mse"] += selected_diff.pow(2).sum().item()
    state["sum_mae"] += selected_diff.abs().sum().item()
    state["num_values"] += selected_diff.numel()
    state["num_examples"] += pred_raw.shape[0]

    for threshold in state["thresholds"]:
        key = str(threshold)
        pred_bin = pred_raw >= threshold
        target_bin = target_raw >= threshold
        pred_bin = pred_bin & eval_mask
        target_bin = target_bin & eval_mask
        counts = state["counts"][key]
        counts["tp"] += torch.logical_and(pred_bin, target_bin).sum().item()
        counts["fp"] += torch.logical_and(pred_bin, ~target_bin).sum().item()
        counts["fn"] += torch.logical_and(~pred_bin, target_bin).sum().item()
        counts["tn"] += torch.logical_and(~pred_bin, ~target_bin & eval_mask).sum().item()


def finalize_metric_state(state: Dict[str, object]) -> Dict[str, float]:
    if state["num_values"] == 0:
        raise ValueError("Cannot finalize empty metric state.")
    output = {
        "mae": state["sum_mae"] / state["num_values"],
        "mse": state["sum_mse"] / state["num_values"],
        "rmse": (state["sum_mse"] / state["num_values"]) ** 0.5,
        "num_examples": int(state["num_examples"]),
    }
    for threshold in state["thresholds"]:
        counts = state["counts"][str(threshold)]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        output[f"csi_{int(threshold)}"] = tp / (tp + fp + fn + EPS)
        output[f"pod_{int(threshold)}"] = tp / (tp + fn + EPS)
        output[f"far_{int(threshold)}"] = fp / (tp + fp + EPS)
        output[f"bias_{int(threshold)}"] = (tp + fp) / (tp + fn + EPS)
    return output


def sync_metric_state(state: Dict[str, object]):
    if not (dist.is_available() and dist.is_initialized()):
        return state

    backend = dist.get_backend()
    if backend == "nccl" and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    summary = torch.tensor(
        [
            float(state["num_examples"]),
            float(state["sum_mae"]),
            float(state["sum_mse"]),
            float(state["num_values"]),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(summary, op=dist.ReduceOp.SUM)
    state["num_examples"] = int(summary[0].item())
    state["sum_mae"] = float(summary[1].item())
    state["sum_mse"] = float(summary[2].item())
    state["num_values"] = int(summary[3].item())

    for threshold in state["thresholds"]:
        counts = state["counts"][str(threshold)]
        count_tensor = torch.tensor(
            [counts["tp"], counts["fp"], counts["fn"], counts["tn"]],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        counts["tp"] = float(count_tensor[0].item())
        counts["fp"] = float(count_tensor[1].item())
        counts["fn"] = float(count_tensor[2].item())
        counts["tn"] = float(count_tensor[3].item())
    return state
