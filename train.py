# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import wandb

from models import DiT_models, LogvarNet
from diffusers.models import AutoencoderKL

from non_markov_diffusion.loss import dsm_loss, sample_p_base_x_t_cond_x_t_prev_x_t_next
from non_markov_diffusion.sde import DecayingVolatilitySDE, PeriodicVolatilitySDE, CosineDecayingVolatilitySDE
from custom_data_utils.checkerboard_dataset import CheckerboardDataset
from custom_data_utils.utils import unpack_batch, _ensure_latents_locally
from encode_latents import PrecomputedLatentDataset

import torchvision.transforms.functional as TF
import cv2

from functools import partial

from torch.amp import autocast
from tqdm import tqdm

from download import find_model
from sample_non_markov import run_eval_loop

import gc
import json
#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def initialize_from_pretrained_model(model, pretrained_model_name, logger):
    pretrained_state = find_model(pretrained_model_name)

    # Easy ones to copy over
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for k, v in pretrained_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible_state[k] = v
        else:
            skipped.append(k)
    if logger is not None:
        logger.info(f"Loading {len(compatible_state)}/{len(pretrained_state)} pretrained params")
        logger.info(f"Compatible keys: {compatible_state.keys()}")
        logger.info(f"Skipped: {skipped}")

    # Initialize adaLN modulation weights
    for i in range(len(model.blocks)):
        key_w = f"blocks.{i}.adaLN_modulation.1.weight"
        key_b = f"blocks.{i}.adaLN_modulation.1.bias"
        assert key_w not in compatible_state and key_b not in compatible_state

        pretrained_w = pretrained_state[key_w]  # shape: (6*D, D)
        pretrained_b = pretrained_state[key_b]  # shape: (6*D,)
        D = pretrained_w.shape[1]
        assert pretrained_w.shape == (6*D, D) and pretrained_b.shape == (6*D,)

        # New: shift_msa, scale_msa, gate_msa, shift_xa, scale_xa, gate_xa, shift_cond, scale_cond, gate_cond, shift_mlp, scale_mlp, gate_mlp
        # Old: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

        new_w = model.state_dict()[key_w].clone()  # (12*D, D), already initialized
        new_b = model.state_dict()[key_b].clone()   # (12*D,)

        # Copy self-attention modulation (first 3 chunks)
        new_w[:3*D] = pretrained_w[:3*D]
        new_b[:3*D] = pretrained_b[:3*D]
        # Copy MLP modulation (last 3 chunks)
        new_w[9*D:] = pretrained_w[3*D:]
        new_b[9*D:] = pretrained_b[3*D:]

        # Leave chunks 3-8 (cross-attn + cond modulation) at their initialized values (zeros)
        
        compatible_state[key_w] = new_w
        compatible_state[key_b] = new_b

    # Initialize final layer
    out_C = model.state_dict()["final_layer.linear.weight"].shape[0]  # p*p*C
    assert model.state_dict()["final_layer.linear.weight"].shape == pretrained_state["final_layer.linear.weight"][:out_C].shape
    assert model.state_dict()["final_layer.linear.bias"].shape == pretrained_state["final_layer.linear.bias"][:out_C].shape
    compatible_state["final_layer.linear.weight"] = pretrained_state["final_layer.linear.weight"][:out_C]
    compatible_state["final_layer.linear.bias"] = pretrained_state["final_layer.linear.bias"][:out_C]

    # 4. Load compatible weights (strict=False to allow missing keys)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if logger is not None:
        logger.info(f"Missing keys (will be randomly initialized): {missing}")

    # 5. Copy over t embedders and cond embedder
    model.cond_t_embedder.load_state_dict(model.t_embedder.state_dict())
    model.next_t_embedder.load_state_dict(model.t_embedder.state_dict())
    model.cond_embedder.load_state_dict(model.x_embedder.state_dict())

    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def sanitize_timesteps(t, frame_times, margin_eps=7.5e-4, adjust_eps=7.6e-4):
    assert torch.min(frame_times[1:] - frame_times[:-1]).item() > 2 * adjust_eps > 2 * margin_eps
    assert t.dtype == frame_times.dtype == torch.float32
    assert (not (t >= 1.0).any()) and (not (t < 0.0).any())
    assert torch.allclose(frame_times[-1], torch.full_like(frame_times[-1], 1.0))
    assert torch.allclose(frame_times[0], torch.full_like(frame_times[0], 0.0))
    def rand_between(a, b, tau):
        assert a < b
        return torch.rand_like(tau) * (b - a) + a
    for w_idx, waypoint in enumerate(frame_times.tolist()):
        if w_idx > 0: # doesn't make sense that we would have times less than first waypoint
            prev_waypoint = frame_times[w_idx - 1]
            t = torch.where(
                (waypoint - margin_eps <= t) & (t < waypoint), # lower part
                rand_between(a=prev_waypoint+adjust_eps, b=waypoint-adjust_eps, tau=t), # push down, but don't violate previous waypoint
                t
            )
        if w_idx < len(frame_times) - 1: # wouldn't have times greater than last waypoint
            next_waypoint = frame_times[w_idx + 1]
            t = torch.where(
                (waypoint <= t) & (t <= waypoint + margin_eps), 
                rand_between(a=waypoint+adjust_eps, b=next_waypoint-adjust_eps, tau=t), # push up, but don't violate next waypoint
                t
            )
    assert (t > 0.0).all() and (t < 1.0).all()
    assert (torch.abs(t.unsqueeze(1) - frame_times.unsqueeze(0)) >= margin_eps).all() # smallest diff is still outside the margin
    return t

def generate_t_prev(cond_times, cond_masks, t, device):
    assert cond_masks[:, 0].all() # should always make 0.0 visible
    assert torch.allclose(cond_times[:, 0], torch.full_like(cond_times[:, 0], 0.0)) # should always have 0.0 as the first frame
    assert t.shape == (cond_times.shape[0],)
    assert cond_times.shape == cond_masks.shape
    # (B, L): True where frame is before t AND is observed
    valid_prev = (cond_times <= t.unsqueeze(1)) & cond_masks
    # Replace invalid with -inf so max ignores them
    masked_prev_times = torch.where(valid_prev, cond_times, torch.tensor(float('-inf'), device=device))
    t_prev = masked_prev_times.max(dim=1).values          # (B,)
    t_prev_idx = masked_prev_times.argmax(dim=1)           # (B,)
    assert t_prev.shape == t_prev_idx.shape == (cond_times.shape[0],)
    assert (t_prev <= t).all()
    assert (torch.sum(valid_prev, dim=1).sum() >= 1).all(), "At least one valid frame must be before t"
    return t_prev, t_prev_idx

def generate_t_next(cond_times, cond_masks, t, B, L, device):
    assert torch.allclose(cond_times[:, -1], torch.full_like(cond_times[:, -1], 1.0)) # should always have 1.0 as the last frame
    assert cond_times.shape == cond_masks.shape == (B, L)
    # Step 1: upper bound = nearest future OBSERVED frame
    valid_next_observed = (cond_times > t.unsqueeze(1)) & cond_masks  # (B, L)
    valid_next_observed[:, -1] = True # it's always valid to try to predict the last frame
    masked_next_obs_times = torch.where(valid_next_observed, cond_times, torch.tensor(float('inf'), device=device))
    upper_bound = masked_next_obs_times.min(dim=1).values  # (B,)

    # Step 2: candidates are ALL frame times in (t[i], upper_bound[i]]
    candidates = (cond_times > t.unsqueeze(1)) & (cond_times <= upper_bound.unsqueeze(1))  # (B, L)

    # Step 3: randomly pick one candidate per batch item (Gumbel-max trick)
    rand_scores = torch.where(candidates, torch.rand(B, L, device=device), torch.tensor(float('-inf'), device=device))
    t_next_idx = rand_scores.argmax(dim=1)                 # (B,)
    t_next = cond_times.gather(dim=1, index=t_next_idx.unsqueeze(1)).squeeze(1)  # (B,)

    assert upper_bound.shape == (B,)
    assert (upper_bound > t).all()
    assert t_next.shape == t_next_idx.shape == t.shape == (cond_masks.shape[0],) == (B,)
    assert (t_next > t).all()
    return t_next, t_next_idx

def sanity_check_t_prev_t_next(t_prev, t_next, t_prev_idx, t_next_idx, t, cond_times, cond_masks, L, B, device):
    assert torch.all(t_prev <= t)
    assert torch.all(t_next > t)
    idx = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)
    range_mask = (idx > t_prev_idx.unsqueeze(1)) & (idx <= t_next_idx.unsqueeze(1))  # (B, L)
    obs_in_range = (range_mask & cond_masks).sum(dim=1)                        # (B,)

    assert torch.all(obs_in_range <= 1), \
        "At most one observed frame in (t_prev_idx, t_next_idx]"
    assert torch.all((obs_in_range == 0) ^ cond_masks[torch.arange(B, device=device), t_next_idx]), \
        "If one observed frame exists in range, it must be at t_next_idx"

    assert torch.allclose(cond_times[torch.arange(B, device=device), t_next_idx], t_next)
    assert torch.allclose(cond_times[torch.arange(B, device=device), t_prev_idx], t_prev)

    assert not (cond_masks.all(dim=-1)).any()

    return

def calculate_fvd(args, checkpoint_path, eval_dir):
    # Build eval args by stripping the "eval_" prefix
    eval_kwargs = {}
    for key, value in vars(args).items():
        EVAL_PREFIX = "eval_"
        if key.startswith(EVAL_PREFIX):
            eval_kwargs[key[len(EVAL_PREFIX):]] = value
    
    eval_kwargs["ckpt"] = checkpoint_path
    eval_kwargs["out_folder"] = eval_dir
    
    eval_args = argparse.Namespace(**eval_kwargs)
    return run_eval_loop(eval_args)

"""
def read_seq_len_from_json(args):
    # NOTE: do NOT call this in a loop; it's only a one-off operation (since we're reading from a file, which is a bottleneck)
    if args.dataset_name in ("Sky-timelapse", "CelebV-HQ"):
        with open(os.path.join(args.latents_folder, "args.json"), "r") as f:
            latents_args = json.load(f)
            if args.dataset_name == "Sky-timelapse":
                return latents_args["sky_timelapse_frames_per_clip"]
            elif args.dataset_name == "CelebV-HQ":
                return latents_args["celebv_hq_frames_per_clip"]
            else:
                raise ValueError(f"I no see that one: {args.dataset_name}")
    elif args.dataset_name == "Checkerboard":
        return 32
    else:
        raise ValueError(f"I no see that one: {args.dataset_name}")
"""

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(project="dit", name=os.path.basename(args.results_dir), config=vars(args))

        eval_dir = f"{experiment_dir}/eval"
        os.makedirs(eval_dir, exist_ok=True)
    else:
        logger = create_logger(None)

    # Create model:
    assert "XA" in args.model, "Model must be a DiT-XA model"

    if args.dataset_name in ("Sky-timelapse", "CelebV-HQ"):
        assert args.num_classes == 1, "Sky-timelapse has only one class"
    elif args.dataset_name == "Checkerboard":
        assert args.num_classes == 3, "Checkerboard has 3 classes"
    else:
        raise ValueError(f"I no see that one: {args.dataset_name}")
    model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=3 if args.dataset_name == "Checkerboard" else 4,
        num_classes=args.num_classes,
    )
    if args.pretrained_model_name is not None:
        assert args.model == "DiTXA-XL/2", "New model must be a DiTXA-XL/2 model"
        assert args.dataset_name != "Checkerboard", "Pixel-space does not support pretrained models"
        model = initialize_from_pretrained_model(model=model, pretrained_model_name=args.pretrained_model_name, logger=logger if rank == 0 else None)
        logger.info(f"Initialized model from pretrained model: {args.pretrained_model_name}")
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    # diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.logvar_hidden_size is not None and args.logvar_hidden_size > 0:
        raise ValueError("Logvar Net is not supported for now")
        """
        logvar_net = LogvarNet(
            seq_len=read_seq_len_from_json(args=args), 
            hidden_size=args.logvar_hidden_size,
        )
        logvar_net = DDP(logvar_net.to(device), device_ids=[rank])
        logger.info(f"Logvar Net Parameters: {sum(p.numel() for p in logvar_net.parameters()):,}")
        optimized_parameters = list(model.parameters()) + list(logvar_net.parameters())
        """
    else:
        logvar_net = None
        optimized_parameters = list(model.parameters())

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(optimized_parameters, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=0)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, 
        start_factor=0.001, 
        end_factor=1.0, 
        total_iters=args.warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, 
        T_max=args.cosine_decay_steps,  # total steps for the cosine phase
        eta_min=args.lr * 0.001,         # minimum LR (0.1% of peak)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[args.warmup_steps]
    )
    # Setup SDE:
    if args.sde_type == "decaying_volatility":
        assert args.decaying_sde_A != args.decaying_sde_B, "A and B must be different"
        sde = DecayingVolatilitySDE(A=args.decaying_sde_A, B=args.decaying_sde_B, K=args.decaying_sde_K, score_network=model)
    elif args.sde_type == "periodic_volatility":
        sde = PeriodicVolatilitySDE(alpha=args.periodic_sde_alpha, k=args.periodic_sde_k, eps=args.periodic_sde_eps, score_network=model)
    elif args.sde_type == "cosine_decaying_volatility":
        sde = CosineDecayingVolatilitySDE(alpha=args.periodic_sde_alpha, eps=args.periodic_sde_eps, score_network=model)
    else:
        raise ValueError(f"Invalid SDE type: {args.sde_type}")
    
    if args.brownian_bridge_residual:
        model.module.sde = sde
        assert not args.aux_tau, f"Brownian bridge residual is not supported with aux_tau"
    if args.noise_to_data_diffusion:
        assert args.aux_tau, f"Noise-to-data diffusion requires aux_tau"
        assert not args.brownian_bridge_residual, f"Noise-to-data diffusion does not support brownian bridge residual"

    # dataset = ImageFolder(args.data_path, transform=transform)
    if args.dataset_name in ("Sky-timelapse", "CelebV-HQ"): # we already computed them, just pull them
        assert args.latents_folder is not None, "Must use precomputed latents. Ain't no way we're computing them on the fly."
        if rank == 0:
            _ensure_latents_locally(latents_folder=args.latents_folder, dataset_name=args.dataset_name, mode="train")
        dist.barrier() # NO race condition
        dataset = PrecomputedLatentDataset(
            latents_path=os.path.join(args.latents_folder, "latents.npy"),
            labels_path=os.path.join(args.latents_folder, "labels.npy"),
            percent_train=args.latents_percent_train,
            train=True,
        )
        with open(os.path.join(args.latents_folder, "args.json"), "r") as f:
            latents_args = json.load(f)
        print(f"Loaded latents from {args.latents_folder}")
        print(f"Latents args: {latents_args}")
        assert latents_args["image_size"] // 8 == args.image_size
    elif args.dataset_name == "Checkerboard":
        dataset = CheckerboardDataset(
            num_samples=args.checkerboard_num_samples,
            image_size=args.image_size,
            zero_center=True,
            seed=args.global_seed,
        )
    else:
        raise ValueError(f"I no see that one: {args.dataset_name}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,}") #images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=f"cuda:{device}", weights_only=False)
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if logvar_net is not None:
            logvar_net.module.load_state_dict(checkpoint["logvar_net"])
        if "train_steps" in checkpoint:
            train_steps = checkpoint["train_steps"]
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
        logger.info(f"Resumed at epoch {start_epoch}, step {train_steps}")

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in tqdm(loader, desc=f"Epoch {epoch}", disable=rank != 0):
            x, y = unpack_batch(args=args, batch=batch)
            B, L, C, H, W = x.shape
            assert B == args.global_batch_size // dist.get_world_size(), "Batch size must be divisible by world size"
            if args.dataset_name == "Checkerboard":
                assert L == 32 and C == 3
            elif args.dataset_name in ("Sky-timelapse", "CelebV-HQ"):
                assert C == 4, f"{args.dataset_name} must have 4 channels"
            assert H == args.image_size and W == args.image_size
            x = x.to(device)
            y = y.to(device)

            frame_times = torch.linspace(0, 1, L, device=device)

            ###
            # 0. Construct the generic conditioning sequence
            ###
            cond_times = frame_times.clone().unsqueeze(0).repeat(B, 1)
            assert cond_times.shape == (B, L)
            assert torch.all(cond_times[torch.randint(low=0, high=B, size=(1,)).item()] == frame_times)
            cond_images = x.clone()
            assert cond_images.shape == (B, L, C, args.image_size, args.image_size)

            ###
            # 1. Randomly sample the time
            ###
            t = torch.rand(B, device=device)
            t = sanitize_timesteps(t=t, frame_times=frame_times, margin_eps=args.margin_eps, adjust_eps=args.adjust_eps)
            assert t.shape == (B,) and t.max() < 1.0 and t.min() > 0.0

            ###
            # 2. Construct the conditioning masks
            ###
            # num_cond_frames = torch.randint(low=1, high=args.L_sub+1, size=(B,), device=device)
            num_cond_frames = (torch.randperm(args.L_sub, device=device) + 1).repeat(B // args.L_sub + 1)[:B]
            assert num_cond_frames.shape == (B,)
            raw_scores = torch.rand((B, L), device=device)
            raw_scores[:, 0] = -1 # the first frame is always visible
            ranks = raw_scores.argsort(dim=-1)
            cond_masks = ranks < num_cond_frames.unsqueeze(1)
            if args.force_causal:
                cond_masks[cond_times > t.unsqueeze(1)] = False
            
            assert (ranks[:, 0] == 0).all()
            assert cond_masks.shape == (B, L)
            assert cond_masks[:, 0].all()
            assert (cond_masks.sum(dim=-1) <= num_cond_frames).all()
            assert (cond_masks.sum(dim=-1) <= L).all()
            assert (cond_masks.sum(dim=-1) > 0).all()

            ###
            # 3. Pick previous frame and next frame to predict (out of ALL possible frames)
            ###
            # GENERATE t_prev
            t_prev, t_prev_idx = generate_t_prev(cond_times=cond_times, cond_masks=cond_masks, t=t, device=device)
            # GENERATE t_next
            t_next, t_next_idx = generate_t_next(cond_times=cond_times, cond_masks=cond_masks, t=t, B=B, L=L, device=device)

            # SANITY CHECKS
            if args.dataset_name == "Sky-timelapse":
                assert L == latents_args["sky_timelapse_frames_per_clip"]
            elif args.dataset_name == "CelebV-HQ":
                assert L == latents_args["celebv_hq_frames_per_clip"]
            elif args.dataset_name == "Checkerboard":
                assert L == 32
            else:
                raise ValueError(f"I no see that one: {args.dataset_name}")
            sanity_check_t_prev_t_next(t_prev=t_prev.clone(), t_next=t_next.clone(), t_prev_idx=t_prev_idx.clone(), t_next_idx=t_next_idx.clone(), t=t.clone(), cond_times=cond_times.clone(), cond_masks=cond_masks.clone(), L=L, B=B, device=device)

            # Get corresponding frames
            x_t_prev = x[torch.arange(B, device=device), t_prev_idx]  # (B, 4, 32, 32)
            x_t_next = x[torch.arange(B, device=device), t_next_idx]  # (B, 4, 32, 32)
            assert x_t_prev.shape == x_t_next.shape == (B, C, args.image_size, args.image_size)

            ###
            # 3.5 (optional) Sample auxiliary tau (if naive diffusion bridge)
            ###
            if args.aux_tau:
                aux_tau = (t - t_prev) / (t_next - t_prev)
                # aux_tau = torch.rand(B, device=device) * (1 - args.margin_eps)
            else:
                aux_tau = None

            ###
            # 4. Sample noisy data
            ###

            x_t = sample_p_base_x_t_cond_x_t_prev_x_t_next(
                sde=sde,
                x_t_prev=x_t_prev if (not args.noise_to_data_diffusion) else torch.randn_like(x_t_prev),
                x_t_next=x_t_next,
                t=t if (aux_tau is None) else aux_tau,
                t_prev=t_prev if (aux_tau is None) else torch.zeros_like(t_prev),
                t_next=t_next if (aux_tau is None) else torch.ones_like(t_next),
            )
            assert x_t.isnan().sum() == 0, "x_t contains NaNs"

            if train_steps % args.log_every == 0 and rank == 0:
                message = f"t: {t[:4]}\n"
                message += f"t_prev: {t_prev[:4]}\n"
                message += f"t_next: {t_next[:4]}\n"
                message += f"cond_times: {cond_times[:4]}\n"
                message += f"cond_masks: {cond_masks[:4]}\n"
                print(f"\nDEBUGGING:\n{message}")
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = dsm_loss(
                    model=model,
                    sde=sde, 
                    x_t=x_t, # (N, C, H, W)
                    x_t_next=x_t_next, # (N, C, H, W)
                    x_t_history=cond_images, # (N, max_cond_images, C, H, W)
                    t=t if (aux_tau is None) else aux_tau, # (N,)
                    t_next=t_next, # (N,)
                    t_history=cond_times, # (N, max_cond_images)
                    cond_masks=cond_masks, # (N, max_cond_images)
                    y=y, # (N,) class labels
                    logvar_net=logvar_net,
                    t_is_physical=(aux_tau is None),
                )

            """ ORIGINAL CODE:
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            """

            opt.zero_grad()
            loss.backward()
            pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(optimized_parameters, max_norm=args.max_grad_norm) # don't explode gradients
            # TODO: check that grad norms did decrease ...
            opt.step()
            update_ema(ema, model.module)

            scheduler.step() # update lr

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank == 0:
                    wandb.log({
                        "train/loss": avg_loss, 
                        "train/steps_per_sec": steps_per_sec,
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "train/pre_clip_grad_norm": pre_clip_grad_norm.item(),
                    }, step=train_steps, commit=(train_steps % args.ckpt_every != 0))
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_steps": train_steps,
                        "epoch": epoch,
                        "args": args,
                        "logvar_net": logvar_net.module.state_dict() if logvar_net is not None else None,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    if args.force_causal:
                        assert args.eval_causal, "Eval must be causal if training is causal"
                    curr_eval_dir = os.path.join(eval_dir, f"step_{train_steps:07d}")
                    os.makedirs(curr_eval_dir, exist_ok=True)
                    # preserve RNG state
                    # TODO: ablate this
                    rng_state = torch.get_rng_state()
                    cuda_rng_state = torch.cuda.get_rng_state()
                    # evaluate FVD
                    fvd_score, video_paths = calculate_fvd(
                        args=args, 
                        checkpoint_path=checkpoint_path, 
                        eval_dir=curr_eval_dir,
                    )
                    # restore RNG state
                    torch.set_rng_state(rng_state)
                    torch.cuda.set_rng_state(cuda_rng_state)
                    gc.collect()
                    torch.cuda.empty_cache()
                    fvd_tags = \
                        (
                            f"causal_prefix_length_{args.eval_prefix_length}" if args.eval_causal \
                            else f"non_causal_pin_every_{args.eval_pin_every}"
                        ) + \
                        f"_subsample_every_{args.eval_subsample_every}" + \
                        f"_sim_steps_{args.eval_num_sampling_steps}" + \
                        f"_num_samples_{args.eval_max_num_batches * args.eval_batch_size}"
                    eval_log_dict = {
                        f"eval/fvd_{fvd_tags}": fvd_score,
                    }
                    for vid_idx, video_path in enumerate(video_paths):
                        eval_log_dict[f"eval/video_{vid_idx}_{fvd_tags}"] = wandb.Video(video_path)
                    wandb.log(eval_log_dict, step=train_steps)
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiTXA_B/2")
    parser.add_argument("--brownian-bridge-residual", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cosine-decay-steps", type=int, default=90000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    # SDE parameters
    parser.add_argument("--sde-type", type=str, choices=["decaying_volatility", "periodic_volatility", "cosine_decaying_volatility"], default="periodic_volatility")
    parser.add_argument("--decaying_sde_A", type=float, default=0.0)
    parser.add_argument("--decaying_sde_B", type=float, default=3.0)
    parser.add_argument("--decaying_sde_K", type=float, default=2.0)
    parser.add_argument("--periodic_sde_alpha", type=float, default=0.5)
    parser.add_argument("--periodic_sde_k", type=int, default=1)
    parser.add_argument("--periodic_sde_eps", type=float, default=0.01)
    # Timestep sanitization parameters
    parser.add_argument("--margin-eps", type=float, default=7.5e-4)
    parser.add_argument("--adjust-eps", type=float, default=7.6e-4)
    # Checkpoint resume parameters
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained-model-name", type=str, default=None, help="Name of pretrained DiT (non-cross attention) model to initialize from", choices=['DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'])
    # Dataset parameters
    parser.add_argument("--dataset-name", type=str, choices=["Sky-timelapse", "CelebV-HQ", "Checkerboard"], default="Sky-timelapse")
    # TODO: add subsampling parameters
    # Latent dataset parameters
    parser.add_argument("--latents-folder", type=str, default="/pscratch/sd/g/gabeguo/datasets/latents/sky_timelapse/res-256x256-fpc-64/train")
    parser.add_argument("--latents-percent-train", type=float, default=None)
    # Checkerboard parameters
    parser.add_argument("--checkerboard-num-samples", type=int, default=100000)
    # Conditioning parameters
    parser.add_argument("--L-sub", type=int, default=8)
    parser.add_argument("--force-causal", action="store_true")
    # Logvar net parameters
    parser.add_argument("--logvar-hidden-size", type=int, default=None)
    # Model mode parameters
    parser.add_argument("--aux-tau", action="store_true", help="Naive conditional diffusion bridge that uses auxiliary time variable.")
    parser.add_argument("--noise-to-data-diffusion", action="store_true", help="Use noise-to-data diffusion instead of conditional diffusion bridge.")
    # Eval parameters
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-num-batches", type=int, default=8)
    parser.add_argument("--eval-frames-decoded-per-call", type=int, default=2)
    parser.add_argument("--eval-stride-dataset", type=int, default=None)
    parser.add_argument("--eval-num-sampling-steps", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-snapshot-interval", type=int, default=1001)
    parser.add_argument("--eval-nrow", type=int, default=2)
    parser.add_argument("--eval-subsample-every", type=int, default=2)
    parser.add_argument("--eval-pin-every", type=int, default=8)
    parser.add_argument("--eval-causal", action="store_true")
    parser.add_argument("--eval-prefix-length", type=int, default=1)
    parser.add_argument("--eval-only-prefix-conditioning", action="store_true")
    parser.add_argument("--eval-teacher-force-pinned", action="store_true")
    parser.add_argument("--eval-latents-folder", type=str, default="/pscratch/sd/g/gabeguo/datasets/latents/sky_timelapse/res-256x256-fpc-64/test")
    parser.add_argument("--eval-save-interval", type=int, default=4)
    parser.add_argument("--eval-fvd-model", type=str, default="i3d")
    parser.add_argument("--eval-fvd-videomae-ckpt", type=str, default=None)
    parser.add_argument("--eval-ignore-last-frame", action="store_true")
    parser.add_argument("--eval-alt-subsample-every", type=int, default=None)
    parser.add_argument("--eval-clip-dt-eps", type=float, default=None)
    args = parser.parse_args()
    main(args)
