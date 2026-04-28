from typing import Dict, Optional

import torch
from tqdm import tqdm

from non_markov_diffusion.sde import (
    CosineDecayingVolatilitySDE,
    DecayingVolatilitySDE,
    PeriodicVolatilitySDE,
    UniformVolatilitySDE,
)


VALID_METHODS = ("abc", "conditional_diffusion_bridge", "noise_to_data_diffusion")


def get_method_flags(method: str) -> Dict[str, bool]:
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    return {
        "aux_tau": method != "abc",
        "noise_to_data_diffusion": method == "noise_to_data_diffusion",
        "supports_brownian_bridge_residual": method == "abc",
    }


def validate_benchmark_config(args, method: str):
    flags = get_method_flags(method)
    if flags["noise_to_data_diffusion"] and not flags["aux_tau"]:
        raise ValueError("noise_to_data_diffusion requires aux_tau.")
    if getattr(args, "brownian_bridge_residual", False) and not flags["supports_brownian_bridge_residual"]:
        raise ValueError(
            "--brownian-bridge-residual is only supported for the physical-time "
            "ABC method."
        )
    if getattr(args, "l_sub", None) is not None and args.l_sub <= 0:
        raise ValueError("l_sub must be positive.")


def load_sde(args, model):
    if args.sde_type == "uniform_volatility":
        return UniformVolatilitySDE(
            A=args.uniform_sde_A,
            K=args.uniform_sde_K,
            score_network=model,
        )
    if args.sde_type == "decaying_volatility":
        if args.decaying_sde_A == args.decaying_sde_B:
            raise ValueError("decaying_volatility requires decaying_sde_A != decaying_sde_B.")
        return DecayingVolatilitySDE(
            A=args.decaying_sde_A,
            B=args.decaying_sde_B,
            K=args.decaying_sde_K,
            score_network=model,
        )
    if args.sde_type == "periodic_volatility":
        return PeriodicVolatilitySDE(
            alpha=args.periodic_sde_alpha,
            k=args.periodic_sde_k,
            eps=args.periodic_sde_eps,
            score_network=model,
        )
    if args.sde_type == "cosine_decaying_volatility":
        return CosineDecayingVolatilitySDE(
            alpha=args.periodic_sde_alpha,
            eps=args.periodic_sde_eps,
            score_network=model,
        )
    raise ValueError(f"Unsupported sde_type: {args.sde_type}")


def sanitize_timesteps(
    t: torch.Tensor,
    frame_times: torch.Tensor,
    min_t: float,
    margin_eps: float = 7.5e-4,
    adjust_eps: float = 7.6e-4,
):
    if not torch.allclose(frame_times[0], torch.zeros_like(frame_times[0])):
        raise ValueError("Expected the first frame time to be 0.0.")
    if not torch.allclose(frame_times[-1], torch.ones_like(frame_times[-1])):
        raise ValueError("Expected the last frame time to be 1.0.")
    if torch.min(frame_times[1:] - frame_times[:-1]).item() <= 2 * adjust_eps:
        raise ValueError("Frame spacing is too small for the requested timestep margin.")

    t = t.clamp(min=min_t, max=1.0 - 1e-6)

    def rand_between(a, b, tau):
        if not a < b:
            raise ValueError(f"Invalid timestep interval: [{a}, {b}]")
        return torch.rand_like(tau) * (b - a) + a

    for waypoint_idx, waypoint in enumerate(frame_times.tolist()):
        if waypoint_idx > 0:
            prev_waypoint = frame_times[waypoint_idx - 1].item()
            t = torch.where(
                (waypoint - margin_eps <= t) & (t < waypoint),
                rand_between(prev_waypoint + adjust_eps, waypoint - adjust_eps, t),
                t,
            )
        if waypoint_idx < len(frame_times) - 1:
            next_waypoint = frame_times[waypoint_idx + 1].item()
            t = torch.where(
                (waypoint <= t) & (t <= waypoint + margin_eps),
                rand_between(waypoint + adjust_eps, next_waypoint - adjust_eps, t),
                t,
            )
    t = t.clamp(min=min_t, max=1.0 - adjust_eps)
    return t


def generate_t_prev(cond_times: torch.Tensor, cond_masks: torch.Tensor, t: torch.Tensor):
    batch_size, seq_len = cond_times.shape
    if cond_masks.shape != (batch_size, seq_len):
        raise ValueError("cond_masks must match cond_times.")
    if not cond_masks[:, 0].all():
        raise ValueError("Expected the first conditioning frame to always be visible.")
    if not torch.allclose(cond_times[:, 0], torch.zeros_like(cond_times[:, 0])):
        raise ValueError("Expected the first conditioning time to be 0.0.")

    valid_prev = (cond_times <= t.unsqueeze(1)) & cond_masks
    masked_prev_times = torch.where(
        valid_prev,
        cond_times,
        torch.full_like(cond_times, float("-inf")),
    )
    t_prev = masked_prev_times.max(dim=1).values
    t_prev_idx = masked_prev_times.argmax(dim=1)
    return t_prev, t_prev_idx


def generate_t_next(cond_times: torch.Tensor, cond_masks: torch.Tensor, t: torch.Tensor):
    batch_size, seq_len = cond_times.shape
    if cond_masks.shape != (batch_size, seq_len):
        raise ValueError("cond_masks must match cond_times.")
    if not torch.allclose(cond_times[:, -1], torch.ones_like(cond_times[:, -1])):
        raise ValueError("Expected the last conditioning time to be 1.0.")

    valid_next_observed = (cond_times > t.unsqueeze(1)) & cond_masks
    valid_next_observed[:, -1] = True
    masked_next_obs_times = torch.where(
        valid_next_observed,
        cond_times,
        torch.full_like(cond_times, float("inf")),
    )
    upper_bound = masked_next_obs_times.min(dim=1).values

    candidates = (cond_times > t.unsqueeze(1)) & (cond_times <= upper_bound.unsqueeze(1))
    rand_scores = torch.where(
        candidates,
        torch.rand(batch_size, seq_len, device=t.device),
        torch.full((batch_size, seq_len), float("-inf"), device=t.device),
    )
    t_next_idx = rand_scores.argmax(dim=1)
    t_next = cond_times.gather(dim=1, index=t_next_idx.unsqueeze(1)).squeeze(1)
    return t_next, t_next_idx


def select_cond_images_times_masks(
    cond_images: torch.Tensor,
    cond_times: torch.Tensor,
    cond_masks: torch.Tensor,
):
    batch_size, seq_len, channels, height, width = cond_images.shape
    if cond_times.shape != (batch_size, seq_len):
        raise ValueError("cond_times must match cond_images.")
    if cond_masks.shape != (batch_size, seq_len):
        raise ValueError("cond_masks must match cond_images.")
    if cond_masks.dtype != torch.bool:
        raise ValueError("cond_masks must be boolean.")
    if not torch.all(cond_masks[:, 0]):
        raise ValueError("The first conditioning frame must always be visible.")

    mask = cond_masks[0]
    if not torch.all(cond_masks == mask.unsqueeze(0)):
        raise ValueError("Expected a shared causal visibility mask across the batch.")

    cond_images_sel = cond_images[:, mask]
    cond_times_sel = cond_times[:, mask]
    cond_masks_sel = cond_masks[:, mask]
    return cond_images_sel, cond_times_sel, cond_masks_sel


def sample_conditioning_masks(
    cond_times: torch.Tensor,
    t: torch.Tensor,
    l_sub: int,
    force_causal: bool,
):
    batch_size, seq_len = cond_times.shape
    if not 0 < l_sub < seq_len:
        raise ValueError(f"l_sub must be in [1, {seq_len - 1}], got {l_sub}")

    num_cond_frames = (torch.randperm(l_sub, device=t.device) + 1).repeat(batch_size // l_sub + 1)[:batch_size]
    raw_scores = torch.rand((batch_size, seq_len), device=t.device)
    raw_scores[:, 0] = -1.0
    ranks = raw_scores.argsort(dim=-1)
    cond_masks = ranks < num_cond_frames.unsqueeze(1)
    if force_causal:
        cond_masks[cond_times > t.unsqueeze(1)] = False
    cond_masks[:, 0] = True
    if cond_masks.all(dim=-1).any():
        raise ValueError("Conditioning mask unexpectedly exposed the full trajectory.")
    return cond_masks


def sample_training_tuple(
    videos,
    prefix_length: int,
    method: str,
    device: torch.device,
    l_sub: int,
    force_causal: bool,
):
    flags = get_method_flags(method)
    batch_size, total_frames, channels, height, width = videos.shape
    if channels != 1:
        raise ValueError(f"Expected a single VIL channel, got {channels}")
    if not 0 < prefix_length < total_frames:
        raise ValueError(f"prefix_length must be in (0, {total_frames}), got {prefix_length}")

    frame_times = torch.linspace(0, 1, total_frames, device=device)
    cond_times = frame_times.unsqueeze(0).expand(batch_size, -1).float()
    cond_images = videos.clone()

    t = torch.rand(batch_size, device=device)
    t = sanitize_timesteps(t=t.float(), frame_times=frame_times.float(), min_t=1e-6)
    cond_masks = sample_conditioning_masks(
        cond_times=cond_times,
        t=t,
        l_sub=l_sub,
        force_causal=force_causal,
    )
    t_prev, t_prev_idx = generate_t_prev(cond_times=cond_times, cond_masks=cond_masks, t=t)
    t_next, t_next_idx = generate_t_next(cond_times=cond_times, cond_masks=cond_masks, t=t)

    x_t_prev = videos[torch.arange(batch_size, device=device), t_prev_idx]
    x_t_next = videos[torch.arange(batch_size, device=device), t_next_idx]

    aux_tau = (t - t_prev) / (t_next - t_prev) if flags["aux_tau"] else None

    return {
        "cond_images": cond_images,
        "cond_times": cond_times,
        "cond_masks": cond_masks,
        "t_prev": t_prev,
        "t": t.float(),
        "t_next": t_next.float(),
        "t_prev_idx": t_prev_idx,
        "t_next_idx": t_next_idx,
        "x_t_prev": x_t_prev,
        "x_t_next": x_t_next,
        "aux_tau": aux_tau.float() if aux_tau is not None else None,
    }


@torch.no_grad()
def rollout_bridge_model(
    model,
    sde,
    videos,
    num_sampling_steps: int,
    method: str,
    causal: bool,
    prefix_length: int,
    pin_every: int,
    teacher_force_pinned: bool = False,
    clip_dt_eps: Optional[float] = None,
):
    flags = get_method_flags(method)
    device = videos.device
    batch_size, total_frames, channels, height, width = videos.shape
    if channels != 1:
        raise ValueError(f"SEVIR bridge rollout expects a single VIL channel, got {channels}")
    if not 0 < prefix_length < total_frames:
        raise ValueError(f"prefix_length must be in (0, {total_frames}), got {prefix_length}")
    if num_sampling_steps <= 0:
        raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
    if causal:
        if pin_every != 1:
            raise ValueError(f"Causal rollout expects pin_every=1, got {pin_every}")
    else:
        if prefix_length != 1:
            raise ValueError(f"Pinned rollout expects prefix_length=1, got {prefix_length}")
        if pin_every <= 1:
            raise ValueError(f"Pinned rollout expects pin_every > 1, got {pin_every}")
    if causal and teacher_force_pinned:
        raise ValueError("teacher_force_pinned is only valid in pinned mode.")

    frame_times = torch.linspace(0, 1, total_frames, device=device)
    min_frame_gap = torch.min(frame_times[1:] - frame_times[:-1]).item()
    if 1.0 / float(num_sampling_steps) >= min_frame_gap:
        raise ValueError(
            f"num_sampling_steps={num_sampling_steps} is too small for {total_frames} frames. "
            "Need 1 / num_sampling_steps < min frame gap, matching the Sky sampler."
        )
    cond_images = torch.zeros_like(videos)
    initial_cond_images = torch.zeros_like(videos)
    if causal:
        cond_images[:, :prefix_length] = videos[:, :prefix_length]
        initial_cond_images[:, :prefix_length] = videos[:, :prefix_length]
        eval_frame_mask = torch.ones(total_frames, device=device, dtype=torch.bool)
        eval_frame_mask[:prefix_length] = False
    else:
        cond_images[:, ::pin_every] = videos[:, ::pin_every]
        cond_images[:, -1] = videos[:, -1]
        initial_cond_images[:, ::pin_every] = videos[:, ::pin_every]
        initial_cond_images[:, -1] = videos[:, -1]
        eval_frame_mask = torch.ones(total_frames, device=device, dtype=torch.bool)
        eval_frame_mask[::pin_every] = False
        eval_frame_mask[-1] = False
    cond_times = frame_times.unsqueeze(0).expand(batch_size, -1)
    y = torch.zeros(batch_size, device=device, dtype=torch.long)

    snapshots = []
    step_idx = 0
    x_t = videos[:, 0].clone() if (not flags["noise_to_data_diffusion"]) else torch.randn_like(videos[:, 0])
    t = torch.zeros(batch_size, device=device, dtype=torch.float32)
    waypoint_idx = 1
    left_over_dt = 0.0

    progress = None
    if num_sampling_steps >= 32:
        progress = tqdm(total=num_sampling_steps, desc=f"{method} rollout")
    for _ in range(num_sampling_steps):
        if waypoint_idx >= total_frames:
            break
        cond_masks = t.unsqueeze(1) >= cond_times
        if not causal:
            cond_masks[:, ::pin_every] = True
            cond_masks[:, -1] = True
        cond_images_sel, cond_times_sel, cond_masks_sel = select_cond_images_times_masks(
            cond_images=cond_images,
            cond_times=cond_times,
            cond_masks=cond_masks,
        )

        next_idx = torch.bucketize(t, frame_times, right=True)
        if (next_idx >= total_frames).any():
            break
        t_next = frame_times[next_idx]
        dt = 1.0 / float(num_sampling_steps)
        if clip_dt_eps is not None:
            dt += left_over_dt
            if t[0].item() + dt > t_next[0].item() + clip_dt_eps:
                left_over_dt = (t + dt - t_next - clip_dt_eps)[0].item()
                dt = (t_next - t + clip_dt_eps)[0].item()
            else:
                left_over_dt = 0.0

        if flags["aux_tau"]:
            if (next_idx <= 0).any():
                raise ValueError("Auxiliary-time rollout requires a valid previous waypoint.")
            t_prev = frame_times[next_idx - 1]
            aux_tau = (t - t_prev) / (t_next - t_prev)
            model_t = aux_tau.clamp(min=1e-6, max=1.0 - 1e-6)
            solver_dt = dt / (t_next - t_prev)[0].item()
        else:
            model_t = t.clamp(min=1e-6, max=1.0 - 1e-6)
            solver_dt = dt

        d_x = sde.dX_t(
            x_t=x_t,
            t=model_t,
            t_next=t_next,
            x_t_history=cond_images_sel,
            t_history=cond_times_sel,
            cond_masks=cond_masks_sel,
            y=y,
            dt=solver_dt,
        )
        x_t = x_t + d_x
        t = torch.minimum(t + dt, torch.full_like(t, 1.0))
        step_idx += 1
        if progress is not None:
            progress.update()

        while waypoint_idx < total_frames and frame_times[waypoint_idx].item() <= t[0].item():
            if causal and waypoint_idx < prefix_length:
                x_t = cond_images[:, waypoint_idx].clone()
            elif (not causal) and teacher_force_pinned and (waypoint_idx % pin_every == 0 or waypoint_idx == total_frames - 1):
                x_t = cond_images[:, waypoint_idx].clone()
            else:
                cond_images[:, waypoint_idx] = x_t.clone()
            if flags["noise_to_data_diffusion"]:
                x_t = torch.randn_like(x_t)
            waypoint_idx += 1
    if progress is not None:
        progress.close()
    if waypoint_idx < total_frames and not teacher_force_pinned:
        cond_images[:, -1] = x_t.clone()

    return {
        "pred_frames": cond_images,
        "gt_frames": videos,
        "conditioning_frames": initial_cond_images,
        "eval_frame_mask": eval_frame_mask,
        "snapshot_samples": snapshots,
        "num_solver_steps": step_idx,
    }
