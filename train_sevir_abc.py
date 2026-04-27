import argparse
import os
from copy import deepcopy
from time import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import DiT_models
from non_markov_diffusion.loss import dsm_loss, sample_p_base_x_t_cond_x_t_prev_x_t_next
from sevir_benchmark.abc import get_method_flags, load_sde, sample_training_tuple, validate_benchmark_config
from sevir_benchmark.data import SevirVILDataset
from sevir_benchmark.utils import (
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    namespace_to_dict,
    save_json,
    set_seed,
)


def update_ema(ema_model, model, decay=0.999):
    ema_params = dict(ema_model.named_parameters())
    model = model.module if hasattr(model, "module") else model
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def _resume_compatibility_keys(ckpt_args):
    keys = [
        "method",
        "model",
        "image_size",
        "prefix_length",
        "target_length",
        "clip_starts",
        "val_fraction",
        "max_train_events",
        "normalize_mode",
        "sde_type",
        "l_sub",
        "force_causal",
        "brownian_bridge_residual",
    ]
    if ckpt_args.sde_type == "uniform_volatility":
        keys.extend(["uniform_sde_A", "uniform_sde_K"])
    elif ckpt_args.sde_type == "decaying_volatility":
        keys.extend(["decaying_sde_A", "decaying_sde_B", "decaying_sde_K"])
    elif ckpt_args.sde_type in ("periodic_volatility", "cosine_decaying_volatility"):
        keys.extend(["periodic_sde_alpha", "periodic_sde_eps"])
        if ckpt_args.sde_type == "periodic_volatility":
            keys.append("periodic_sde_k")
    return keys


def assert_resume_compatible(current_args, checkpoint_args):
    mismatches = []
    for key in _resume_compatibility_keys(checkpoint_args):
        current_value = getattr(current_args, key)
        checkpoint_value = getattr(checkpoint_args, key)
        if current_value != checkpoint_value:
            mismatches.append(f"{key}: current={current_value!r}, checkpoint={checkpoint_value!r}")
    if mismatches:
        mismatch_text = "\n".join(mismatches)
        raise ValueError(
            "Refusing to resume SEVIR training with a different benchmark/model configuration.\n"
            f"{mismatch_text}"
        )

def main():
    parser = argparse.ArgumentParser(description="Train bridge-based SEVIR VIL nowcasting models.")
    parser.add_argument("--data-root", type=str, default="datasets/sevir")
    parser.add_argument("--results-dir", type=str, default="results/sevir/abc")
    parser.add_argument(
        "--method",
        type=str,
        choices=["abc", "conditional_diffusion_bridge", "noise_to_data_diffusion"],
        default="abc",
    )
    parser.add_argument("--model", type=str, choices=[k for k in DiT_models.keys() if "XA" in k], default="DiTXA-B/2")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--prefix-length", type=int, default=13)
    parser.add_argument("--target-length", type=int, default=12)
    parser.add_argument("--clip-starts", type=int, nargs="*", default=[12])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--normalize-mode", type=str, choices=["zero_center", "unit"], default="zero_center")
    parser.add_argument("--l-sub", type=int, default=16)
    parser.add_argument("--force-causal", action="store_true")
    parser.add_argument("--brownian-bridge-residual", action="store_true")
    parser.add_argument("--sde-type", type=str, choices=["uniform_volatility", "decaying_volatility", "periodic_volatility", "cosine_decaying_volatility"], default="uniform_volatility")
    parser.add_argument("--uniform_sde_A", type=float, default=0.0)
    parser.add_argument("--uniform_sde_K", type=float, default=1.0)
    parser.add_argument("--decaying_sde_A", type=float, default=0.0)
    parser.add_argument("--decaying_sde_B", type=float, default=3.0)
    parser.add_argument("--decaying_sde_K", type=float, default=2.0)
    parser.add_argument("--periodic_sde_alpha", type=float, default=0.5)
    parser.add_argument("--periodic_sde_k", type=int, default=1)
    parser.add_argument("--periodic_sde_eps", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args()
    method_flags = get_method_flags(args.method)
    validate_benchmark_config(args=args, method=args.method)

    device = init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.global_seed * world_size + rank)
    if world_size > 1:
        print(f"Starting rank={rank}, world_size={world_size}, device={device}", flush=True)

    train_dataset = SevirVILDataset(
        root_dir=args.data_root,
        split="train",
        image_size=args.image_size,
        prefix_length=args.prefix_length,
        target_length=args.target_length,
        clip_starts=args.clip_starts,
        val_fraction=args.val_fraction,
        seed=args.global_seed,
        max_events=args.max_train_events,
        normalize_mode=args.normalize_mode,
    )
    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=1,
        num_classes=1,
        class_dropout_prob=0.0,
    )
    ema = deepcopy(model).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False

    if world_size > 1:
        model = DDP(
            model.to(device),
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    autocast_enabled = torch.cuda.is_available()
    autocast_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    start_epoch = 0
    train_steps = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        assert_resume_compatible(current_args=args, checkpoint_args=checkpoint["args"])
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        train_steps = checkpoint.get("train_steps", 0)
    elif hasattr(model, "module"):
        update_ema(ema, model.module, decay=0.0)

    sde = load_sde(args=args, model=model)
    if args.brownian_bridge_residual:
        wrapped_model = model.module if hasattr(model, "module") else model
        wrapped_model.sde = sde
        ema.sde = load_sde(args=args, model=ema)

    if is_main_process():
        save_json(namespace_to_dict(args), os.path.join(args.results_dir, "args.json"))
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    running_loss = 0.0
    running_steps = 0
    start_time = time()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for videos, _ in tqdm(loader, desc=f"Epoch {epoch}", disable=(not is_main_process())):
            videos = videos.to(device)
            batch_size = videos.shape[0]
            batch = sample_training_tuple(
                videos=videos,
                prefix_length=args.prefix_length,
                method=args.method,
                device=device,
                l_sub=args.l_sub,
                force_causal=args.force_causal,
            )
            model_t = batch["aux_tau"] if method_flags["aux_tau"] else batch["t"]
            bridge_t_prev = torch.zeros_like(batch["t_prev"]) if method_flags["aux_tau"] else batch["t_prev"]
            bridge_t_next = torch.ones_like(batch["t_next"]) if method_flags["aux_tau"] else batch["t_next"]
            x_t = sample_p_base_x_t_cond_x_t_prev_x_t_next(
                sde=sde,
                x_t_prev=(
                    torch.randn_like(batch["x_t_next"])
                    if method_flags["noise_to_data_diffusion"]
                    else batch["x_t_prev"]
                ),
                x_t_next=batch["x_t_next"],
                t=model_t,
                t_prev=bridge_t_prev,
                t_next=bridge_t_next,
            )
            y = torch.zeros(batch_size, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = dsm_loss(
                    model=model,
                    sde=sde,
                    x_t=x_t,
                    x_t_next=batch["x_t_next"],
                    x_t_history=batch["cond_images"],
                    t=model_t,
                    t_next=batch["t_next"],
                    t_history=batch["cond_times"],
                    cond_masks=batch["cond_masks"],
                    y=y,
                    logvar_net=None,
                    t_is_physical=(not method_flags["aux_tau"]),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            update_ema(ema, model, decay=args.ema_decay)

            running_loss += loss.item()
            running_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                elapsed = max(time() - start_time, 1e-6)
                local_stats = torch.tensor(
                    [running_loss, float(running_steps)],
                    device=device,
                    dtype=torch.float64,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
                if is_main_process():
                    print(
                        f"step={train_steps:07d} "
                        f"loss={local_stats[0].item() / max(local_stats[1].item(), 1.0):.5f} "
                        f"steps_per_sec={running_steps / elapsed:.3f}"
                    )
                running_loss = 0.0
                running_steps = 0
                start_time = time()

        if (epoch + 1) % args.ckpt_every == 0:
            barrier()
            if is_main_process():
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:04d}.pt")
                torch.save(
                    {
                        "model": (model.module if hasattr(model, "module") else model).state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "train_steps": train_steps,
                        "args": args,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")
            barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
