import argparse
import os
import math
import torch
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.transforms.functional as TF
from diffusers.models import AutoencoderKL

from models import DiT_models
from non_markov_diffusion.sde import DecayingVolatilitySDE, PeriodicVolatilitySDE, CosineDecayingVolatilitySDE
from download import find_model  # ok for local checkpoints

from tqdm import tqdm
import imageio
import json

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision
from functools import partial
from custom_data_utils.checkerboard_dataset import CheckerboardDataset
from encode_latents import PrecomputedLatentDataset
from custom_data_utils.utils import unpack_batch, _ensure_latents_locally

from cdfvd import fvd
from torchmetrics.image.fid import FrechetInceptionDistance

from huggingface_hub import hf_hub_download

def resolve_ckpt(ckpt: str, repo_id: str = "therealgabeguo/abc") -> str:
    """Return a local path, downloading from HF Hub if needed."""
    if os.path.isfile(ckpt):
        print(f"Using local checkpoint: {ckpt}")
        return ckpt
    print(f"Downloading checkpoint from HF Hub: {ckpt}")
    return hf_hub_download(repo_id=repo_id, filename=ckpt, repo_type="model")

def select_cond_images_times_masks(cond_images, cond_times, cond_masks):
    B, T, C, H, W = cond_images.shape
    assert C in (3, 4)
    assert H == W

    # mask is same across batch in this loop
    mask = cond_masks[0]  # (T,) bool
    assert torch.all(mask.unsqueeze(0).repeat(B, 1) == cond_masks)
    assert mask.shape == (T,) and mask.dtype == torch.bool

    # Option A: boolean indexing
    cond_images_sel = cond_images[:, mask, :, :, :]   # (B, L', C, H, W)
    cond_times_sel  = cond_times[:, mask]      # (B, L')
    cond_masks_sel  = cond_masks[:, mask]      # (B, L') (all True)

    assert cond_images_sel.shape == (B, mask.sum(), C, H, W)
    assert cond_times_sel.shape == (B, mask.sum())
    assert cond_masks_sel.shape == (B, mask.sum())
    assert torch.all(cond_masks_sel)

    # Option B: index_select (explicit indices)
    idx = mask.nonzero(as_tuple=True)[0]       # (L',) long
    alt_cond_images_sel = cond_images.index_select(1, idx)
    alt_cond_times_sel  = cond_times.index_select(1, idx)
    alt_cond_masks_sel  = cond_masks.index_select(1, idx)
    assert torch.all(alt_cond_images_sel == cond_images_sel)
    assert torch.all(alt_cond_times_sel == cond_times_sel)
    assert torch.all(alt_cond_masks_sel == cond_masks_sel)

    return cond_images_sel, cond_times_sel, cond_masks_sel

def decode(args, ckpt_args, latents_args, vae, the_frames, B, T, C):
    print(f"The frames shape before decoding: {the_frames.shape}")
    assert the_frames.shape == (B, T, C, ckpt_args.image_size, ckpt_args.image_size)

    decoded_frames = list()
    for i in tqdm(range(0, T, args.frames_decoded_per_call)):
        curr_frames = the_frames[:, i:i+args.frames_decoded_per_call, :, :, :]
        curr_num_frames = curr_frames.shape[1]
        curr_frames = vae.decode(
            curr_frames.reshape(B * curr_num_frames, C, ckpt_args.image_size, ckpt_args.image_size) / vae.config.scaling_factor
        ).sample
        assert curr_frames.shape == (B * curr_num_frames, 3, latents_args["image_size"], latents_args["image_size"])
        curr_frames = curr_frames.reshape(B, curr_num_frames, 3, latents_args["image_size"], latents_args["image_size"])
        decoded_frames.append(curr_frames)
    the_frames = torch.cat(decoded_frames, dim=1)
    assert the_frames.shape == (B, T, 3, latents_args["image_size"], latents_args["image_size"])
    
    """
    the_frames = vae.decode(
        the_frames.reshape(B * T, C, ckpt_args.image_size, ckpt_args.image_size) / vae.config.scaling_factor
    ).sample
    assert the_frames.shape == (B * T, 3, latents_args["image_size"], latents_args["image_size"])

    the_frames = the_frames.reshape(B, T, 3, latents_args["image_size"], latents_args["image_size"])
    """
    print(f"The frames shape after decoding: {the_frames.shape}")

    return the_frames

def convert_to_fvd_format(the_frames, B, T, img_size):
    assert the_frames.shape == (B, T, 3, img_size, img_size)
    # assert the_frames.min() > -1.25 and the_frames.max() < 1.25, f"the_frames.min(): {the_frames.min()}, the_frames.max(): {the_frames.max()}"

    the_frames = (the_frames + 1.0) / 2.0
    # assert the_frames.min() > -0.15 and the_frames.max() < 1.15

    the_frames = (the_frames * 255.0).clamp(min=0, max=255).to(torch.uint8) # [0, 255]
    the_frames = the_frames.permute(0, 1, 3, 4, 2) # (B, T, img_size, img_size, 3)
    assert the_frames.shape == (B, T, img_size, img_size, 3)

    return the_frames.cpu().numpy()

def convert_to_fid_format(the_frames, B, T, img_size):
    # input should be in [-1, 1]
    assert the_frames.shape == (B, T, 3, img_size, img_size)
    the_frames = (the_frames + 1.0) / 2.0
    the_frames = the_frames.clamp(min=0.0, max=1.0)
    the_frames = the_frames.reshape(B * T, 3, img_size, img_size)
    return the_frames

def load_sde(ckpt_args, model):
    if ckpt_args.sde_type == "decaying_volatility":
        sde = DecayingVolatilitySDE(A=ckpt_args.decaying_sde_A, B=ckpt_args.decaying_sde_B, K=ckpt_args.decaying_sde_K, score_network=model)
    elif ckpt_args.sde_type == "periodic_volatility":
        sde = PeriodicVolatilitySDE(alpha=ckpt_args.periodic_sde_alpha, k=ckpt_args.periodic_sde_k, eps=ckpt_args.periodic_sde_eps, score_network=model)
    elif ckpt_args.sde_type == "cosine_decaying_volatility":
        sde = CosineDecayingVolatilitySDE(alpha=ckpt_args.periodic_sde_alpha, eps=ckpt_args.periodic_sde_eps, score_network=model)
    else:
        raise ValueError(f"Invalid SDE type: {ckpt_args.sde_type}")
    return sde

def load_dataset(args, ckpt_args, device):
    if ckpt_args.dataset_name == "CelebV-HQ":
        assert ckpt_args.latents_percent_train < 1.0, "One dataset for CelebV-HQ, which we need to split"
    if ckpt_args.dataset_name in ("Sky-timelapse", "CelebV-HQ"):
        _ensure_latents_locally(
            latents_folder=args.latents_folder, # THIS latent folder (not the ckpt_args.latents_folder)
            dataset_name=ckpt_args.dataset_name, 
            mode="train" if ckpt_args.dataset_name == "CelebV-HQ" else "test", # only one dataset for CelebV-HQ
        )
        dataset = PrecomputedLatentDataset(
            latents_path=os.path.join(args.latents_folder, "latents.npy"),
            labels_path=os.path.join(args.latents_folder, "labels.npy"),
            percent_train=ckpt_args.latents_percent_train,
            train=False,
        )
        with open(os.path.join(args.latents_folder, "args.json"), "r") as f:
            latents_args = json.load(f)
        print(f"Loaded latents from {args.latents_folder}")
        print(f"Latents args: {latents_args}")
        assert latents_args["image_size"] // 8 == ckpt_args.image_size

        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{ckpt_args.vae}").to(device)
        assert vae.config.scaling_factor == 0.18215
    elif ckpt_args.dataset_name == "Checkerboard":
        dataset = CheckerboardDataset(
            num_samples=args.batch_size * args.max_num_batches,
            image_size=ckpt_args.image_size,
            zero_center=True,
            seed=args.seed + 1_000_000,
        )
        vae = None
        latents_args = None
        print(f"Loaded Checkerboard dataset")
    else:
        raise ValueError(f"I no see that one: {ckpt_args.dataset_name}")
    
    return dataset, latents_args, vae

def load_evaluator(args):
    assert args.fvd_model in ("i3d", "videomae")
    def construct_evaluator():
        return fvd.cdfvd(
            args.fvd_model, 
            n_fake=args.max_num_batches * args.batch_size,
            ckpt_path=args.fvd_videomae_ckpt if args.fvd_model == "videomae" else None,
        )
    evaluator = construct_evaluator()
    alt_evaluator = construct_evaluator() if (args.alt_subsample_every is not None) else None
    # TODO: can make the evaluators point to the same underlying model, if we're short on memory
    
    return evaluator, alt_evaluator

@torch.no_grad()
def run_eval_loop(args):
    os.makedirs(args.out_folder, exist_ok=True)
    ckpt_path = resolve_ckpt(args.ckpt)
    ckpt_args = torch.load(ckpt_path, weights_only=False)["args"]

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model (cross-attention)
    model = DiT_models[ckpt_args.model](
        input_size=ckpt_args.image_size,
        in_channels=3 if ckpt_args.dataset_name == "Checkerboard" else 4,
        num_classes=ckpt_args.num_classes,
    ).to(device)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    # SDE + VAE
    sde = load_sde(ckpt_args=ckpt_args, model=model)
    if ckpt_args.brownian_bridge_residual:
        assert not ckpt_args.aux_tau
        model.sde = sde
    dataset, latents_args, vae = load_dataset(args=args, ckpt_args=ckpt_args, device=device)

    if args.stride_dataset is not None:
        assert args.stride_dataset > 0
        dataset = Subset(dataset, indices=list(range(0, len(dataset), args.stride_dataset)))
        print(f"Loaded stride dataset with stride {args.stride_dataset}")
        print(f"len(dataset): {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=ckpt_args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    evaluator, alt_evaluator = load_evaluator(args=args)
    fid_evaluator = FrechetInceptionDistance(normalize=True, sync_on_compute=False).to(device) # only one device

    desired_image_size = ckpt_args.image_size if ckpt_args.dataset_name == "Checkerboard" else latents_args["image_size"]

    video_paths = list()
    print(f"len(loader): {len(loader)}")
    print(f"args.max_num_batches: {args.max_num_batches}")
    for idx, item in tqdm(enumerate(loader)):
        if idx >= args.max_num_batches:
            break
        videos, y = unpack_batch(args=ckpt_args, batch=item)  # videos: (B, T, C, H, W)
        curr_results = sample_non_markov(
            args=args, ckpt_args=ckpt_args, latents_args=latents_args, 
            model=model, sde=sde, vae=vae, 
            videos=videos, y=y, 
            device=device
        )
        if args.ignore_last_frame:
            curr_results["gt_frames"] = curr_results["gt_frames"][:, :-1, :, :, :]
            curr_results["pred_frames"] = curr_results["pred_frames"][:, :-1, :, :, :]
        B, T, C, H, W = curr_results["gt_frames"].shape
        assert curr_results["pred_frames"].shape == (B, T, C, H, W)

        # update FID
        fid_evaluator.update(
            convert_to_fid_format(
                curr_results["gt_frames"], 
                B=B, T=T, img_size=desired_image_size
            ), real=True
        )
        fid_evaluator.update(
            convert_to_fid_format(
                curr_results["pred_frames"], 
                B=B, T=T, img_size=desired_image_size
            ), real=False
        )

        # update FVD
        curr_real_videos = convert_to_fvd_format(
            the_frames=curr_results["gt_frames"], 
            B=B, T=T, img_size=desired_image_size,
        )
        evaluator.add_real_stats(curr_real_videos)
        curr_fake_videos = convert_to_fvd_format(
            the_frames=curr_results["pred_frames"], 
            B=B, T=T, img_size=desired_image_size,
        )
        evaluator.add_fake_stats(curr_fake_videos)
        if args.alt_subsample_every is not None:
            assert "alt_gt_latents" in curr_results and "alt_fake_latents" in curr_results
            alt_T = curr_results["alt_gt_latents"].shape[1]
            alt_real_videos = convert_to_fvd_format(
                the_frames=curr_results["alt_gt_latents"], 
                B=B, T=alt_T, img_size=desired_image_size,
            )
            alt_evaluator.add_real_stats(alt_real_videos)
            alt_fake_videos = convert_to_fvd_format(
                the_frames=curr_results["alt_fake_latents"], 
                B=B, T=alt_T, img_size=desired_image_size,
            )
            alt_evaluator.add_fake_stats(alt_fake_videos)
        if idx % args.save_interval == 0:
            # save predicted frames
            curr_video_path = save_images(
                args=args, ckpt_args=ckpt_args, latents_args=latents_args, vae=vae, 
                the_frames=curr_results["pred_frames"], 
                snapshot_samples=curr_results["snapshot_samples"], B=B, T=T, 
                out_folder=os.path.join(args.out_folder, f"batch_{idx}", "pred"),
            )
            video_paths.append(curr_video_path)
            # save ground truth frames for comparison
            save_images(
                args=args, ckpt_args=ckpt_args, latents_args=latents_args, vae=vae, 
                the_frames=curr_results["gt_frames"], 
                snapshot_samples=None, B=B, T=T, 
                out_folder=os.path.join(args.out_folder, f"batch_{idx}", "gt"),
            )
            if args.alt_subsample_every is not None:
                save_images(
                    args=args, ckpt_args=ckpt_args, latents_args=latents_args, vae=vae, 
                    the_frames=curr_results["alt_fake_latents"], 
                    snapshot_samples=None, B=B, T=alt_T, 
                    out_folder=os.path.join(args.out_folder, f"batch_{idx}", "alt", "pred"),
                )
                save_images(
                    args=args, ckpt_args=ckpt_args, latents_args=latents_args, vae=vae, 
                    the_frames=curr_results["alt_gt_latents"], 
                    snapshot_samples=None, B=B, T=alt_T, 
                    out_folder=os.path.join(args.out_folder, f"batch_{idx}", "alt", "gt"),
                )
    
    assert evaluator.n_fake == args.max_num_batches * args.batch_size
    fvd_score = evaluator.compute_fvd_from_stats()
    print(f"FVD score: {fvd_score}")
    os.makedirs(args.out_folder, exist_ok=True)
    with open(os.path.join(args.out_folder, "fvd_score.json"), "w") as f:
        json.dump({"fvd_score": fvd_score}, f, indent=4)

    if args.alt_subsample_every is not None:
        assert alt_evaluator.n_fake == args.max_num_batches * args.batch_size
        alt_fvd_score = alt_evaluator.compute_fvd_from_stats()
        print(f"Alt FVD score: {alt_fvd_score}")
        with open(os.path.join(args.out_folder, "alt_fvd_score.json"), "w") as f:
            json.dump({"alt_fvd_score": alt_fvd_score}, f, indent=4)
    
    fid_evaluator = fid_evaluator.cpu()
    fid_score = fid_evaluator.compute()
    print(f"FID score: {fid_score}")
    with open(os.path.join(args.out_folder, "fid_score.json"), "w") as f:
        json.dump({"fid_score": fid_score.item()}, f, indent=4)

    with open(os.path.join(args.out_folder, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    return fvd_score, video_paths

def sample_non_markov(args, ckpt_args, latents_args, model, sde, vae, videos, y, device):
    B, T, C, H, W = videos.shape
    latents = videos.to(device)
    y = y.to(device)

    orig_frame_times = torch.linspace(0, 1, T, device=device)

    latents = torch.cat([
        latents[:, :-1:args.subsample_every],
        latents[:, -1:]
    ], dim=1) # subsample frames, but force end time 1 (and start 0)
    frame_times = torch.cat([
        orig_frame_times[:-1:args.subsample_every],
        orig_frame_times[-1:]
    ], dim=0) # get the corresponding times

    if args.alt_subsample_every is not None:
        assert args.alt_subsample_every > 0
        # not necessarily include time 1
        alt_gt_latents = videos[:, ::args.alt_subsample_every].to(device)
        
        alt_fake_latents = torch.zeros_like(alt_gt_latents)
        alt_fake_latents[:, 0] = alt_gt_latents[:, 0].clone()

        alt_frame_times = orig_frame_times[::args.alt_subsample_every]

    print(f"T before: {T}")
    T = frame_times.shape[0] # reset the length for the future code
    print(f"T after: {T}")
    if ckpt_args.dataset_name in ("Sky-timelapse", "CelebV-HQ"):
        assert C == 4
    elif ckpt_args.dataset_name == "Checkerboard":
        assert C == 3
    else:
        raise ValueError(f"I no see that one: {ckpt_args.dataset_name}")

    assert latents.shape == (B, T, C, ckpt_args.image_size, ckpt_args.image_size)
    assert frame_times.shape == (T,)
    assert torch.allclose(frame_times[0], torch.tensor(0.0, device=device))
    assert torch.allclose(frame_times[-1], torch.tensor(1.0, device=device))

    # Start at t=0 from blurred latent
    x_t = latents[:, 0].clone() if (not ckpt_args.noise_to_data_diffusion) else torch.randn_like(latents[:, 0])
    num_steps = args.num_sampling_steps
    t = torch.zeros(B, device=device, dtype=x_t.dtype)

    assert 1 / num_steps < torch.min(frame_times[1:] - frame_times[:-1]).item()

    cond_images = torch.zeros_like(latents)
    if args.causal:
        assert (not args.teacher_force_pinned) # no such thing in causal rollout
        assert args.pin_every == 1
        assert args.prefix_length > 0
        assert args.prefix_length < T
        cond_images[:, :args.prefix_length] = latents[:, :args.prefix_length].clone()
    else:
        assert args.prefix_length == 1
        assert args.pin_every > 1
        cond_images[:, ::args.pin_every] = latents[:, ::args.pin_every].clone()
        cond_images[:, -1] = latents[:, -1].clone() # pin the last frame
    cond_times = frame_times.clone().unsqueeze(0).repeat(B, 1)

    snapshot_samples = list()
    # next waypoint index to fill in
    waypoint_idx = 1 # set to 1, since first frame is ALWAYS filled
    alt_waypoint_idx = 1
    # adjustment when t + dt > t_next: truncate the step size and redistribute to next step
    if args.clip_dt_eps is not None:
        assert 1 / (2 * num_steps) > args.clip_dt_eps >= 1e-6 # avoid numerical issues
        left_over_dt = 0.0 # by default, no dt adjustment
    for step_idx in tqdm(range(num_steps)):
        cond_masks = (t[:, None] >= cond_times)
        if not args.causal:
            cond_masks[:, ::args.pin_every] = True # pin the frames
            cond_masks[:, -1] = True # pin the last frame
        next_idx = torch.bucketize(t, frame_times, right=True)
        t_next = frame_times[next_idx]

        assert (next_idx < T).all()
        assert t_next.shape == next_idx.shape == t.shape == (B,)
        assert torch.numel(torch.unique(t_next)) == 1
        assert torch.numel(torch.unique(t)) == 1

        cond_images_sel, cond_times_sel, cond_masks_sel = select_cond_images_times_masks(
            cond_images=cond_images, 
            cond_times=cond_times, 
            cond_masks=cond_masks
        )
        if args.only_prefix_conditioning: # only condition on the prefix frames (no future, no generated frames)
            cond_images_sel = cond_images_sel[:, :args.prefix_length, ...]
            cond_times_sel = cond_times_sel[:, :args.prefix_length]
            cond_masks_sel = cond_masks_sel[:, :args.prefix_length]
            assert torch.allclose(cond_times_sel[:, 0], torch.zeros_like(cond_times_sel[:, 0]))

        dt = 1 / num_steps # default step size
        if args.clip_dt_eps is not None: # prevent dt from going over waypoint
            # adjust remaining dt (if applicable) from the previous step, so we're back on track
            dt += left_over_dt
            # is the step too large?
            if t[0].item() + dt > t_next[0].item() + args.clip_dt_eps:
                left_over_dt = (t + dt - t_next - args.clip_dt_eps)[0].item() # adjustment for next time
                dt = (t_next - t + args.clip_dt_eps)[0].item() # adjust current step
                # print(f"dt adjusted to {dt:.3f} because t + dt > t_next. t: {t[0].item():.3f}, left_over_dt: {left_over_dt:.3f}")
            else:
                left_over_dt = 0.0 # make sure there's no t adjustment for normal iterations

        if ckpt_args.aux_tau:
            assert (next_idx > 0).all()
            t_prev = frame_times[next_idx - 1]
            aux_tau = (t - t_prev) / (t_next - t_prev)

            assert (aux_tau >= 0).all() and (aux_tau <= 1).all()
            assert torch.numel(torch.unique(aux_tau)) == 1
            assert torch.numel(torch.unique(t_prev)) == 1

            d_tau = dt / (t_next - t_prev)[0].item() # use the ADJUSTED dt, so d_tau won't overshoot
            assert 0 < dt < d_tau <= 1
        else:
            aux_tau = None

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
          dX = sde.dX_t(
            x_t=x_t,
            t=t if (aux_tau is None) else aux_tau,
            t_next=t_next,
            x_t_history=cond_images_sel,
            t_history=cond_times_sel,
            cond_masks=cond_masks_sel,
            y=y,
            dt=dt if (aux_tau is None) else d_tau,
          )
        x_t = x_t + dX
        t = t + dt # update with adjusted timestep

        if step_idx % args.snapshot_interval == 0:
            snapshot_samples.append(x_t.clone())
            # print(f"dX mean: {dX.mean():.3e}, dX std: {dX.std():.3e}")
            if False: #aux_tau is not None:
                print(f"aux_tau: {aux_tau[0].item():.3f}, d_tau: {d_tau:.3f}")

        # record frames
        assert torch.allclose(t, torch.full_like(t, fill_value=(step_idx + 1) * dt)) or (args.clip_dt_eps is not None)
        assert frame_times.shape == (T,)
        if (args.clip_dt_eps is not None) and (left_over_dt > 0):
            assert frame_times[waypoint_idx].item() <= t[0].item() # supposed to trigger writing
        # want to test how the intermediate frames look (super-resolution)
        if args.alt_subsample_every is not None:
            # NOTE: will not take the teacher forcing, since intent is to test the trajectory at times that were not fed
            while alt_waypoint_idx < alt_fake_latents.shape[1] and alt_frame_times[alt_waypoint_idx].item() <= t[0].item():
                alt_fake_latents[:, alt_waypoint_idx] = x_t.clone()
                alt_waypoint_idx += 1
        # test the actual waypoint frames
        while waypoint_idx < T and frame_times[waypoint_idx].item() <= t[0].item():
            if waypoint_idx < args.prefix_length: # these images were fed to us
                assert args.causal
                # print(f"Teacher forcing waypoint {waypoint_idx} at time {frame_times[waypoint_idx].item():.3f}")
                x_t = cond_images[:, waypoint_idx].clone() # teacher forcing
            elif (not args.causal) and args.teacher_force_pinned and (waypoint_idx % args.pin_every == 0 or waypoint_idx == T - 1): # non-causal teacher forcing
                # print(f"Teacher forcing future pinned waypoint {waypoint_idx} at time {frame_times[waypoint_idx].item():.3f}")
                x_t = cond_images[:, waypoint_idx].clone() # teacher forcing
            else: # fill in the images
                # print(f"Filling in waypoint {waypoint_idx} at time {frame_times[waypoint_idx].item():.3f}")
                cond_images[:, waypoint_idx] = x_t.clone()
            # reset to noise, now that we filled in (or maybe didn't) everything
            if ckpt_args.noise_to_data_diffusion:
                assert ckpt_args.aux_tau and (aux_tau is not None)
                assert not ckpt_args.brownian_bridge_residual
                # assert not args.teacher_force_pinned # commented out, so that we have option to prevent overwriting conditioning
                # print(f"Hit waypoint {waypoint_idx} at time {frame_times[waypoint_idx].item():.3f}: reset to noise")
                x_t = torch.randn_like(x_t)
            waypoint_idx += 1 # advance to the next waypoint
    # we reached the end, but didn't fill in the last frame, so need to make up (unless we want teacher forcing)
    if waypoint_idx < T and (not args.teacher_force_pinned): # make sure no teacher force
        assert waypoint_idx == T - 1
        cond_images[:, -1] = x_t.clone() # also fill in the last frame, since the loop terminates
    if (args.alt_subsample_every is not None) and (alt_waypoint_idx < alt_fake_latents.shape[1]):
        assert args.alt_subsample_every == 1, f"Should have already filled this in"
        assert alt_waypoint_idx == alt_fake_latents.shape[1] - 1
        alt_fake_latents[:, -1] = x_t.clone() # also fill in the last frame, since the loop terminates
    # Decode
    assert cond_images.shape == (B, T, C, ckpt_args.image_size, ckpt_args.image_size)
    the_frames = cond_images.clone()
    if ckpt_args.dataset_name in ("Sky-timelapse", "CelebV-HQ"):
        the_frames = decode(
            args=args, ckpt_args=ckpt_args, latents_args=latents_args, vae=vae, 
            the_frames=the_frames, # generated frames
            B=B, T=T, C=C,
        )
        gt_frames = decode(
            args=args, ckpt_args=ckpt_args, latents_args=latents_args, vae=vae, 
            the_frames=latents, # ground truth frames
            B=B, T=T, C=C,
        )
        alt_retvals = dict()
        if args.alt_subsample_every is not None:
            for key_name, alt_latents in [("alt_gt_latents", alt_gt_latents), ("alt_fake_latents", alt_fake_latents)]:
                alt_frames = decode(
                    args=args, ckpt_args=ckpt_args, latents_args=latents_args, vae=vae, 
                    the_frames=alt_latents, # ground truth frames
                    B=B, T=alt_latents.shape[1], C=C,
                )
                alt_retvals[key_name] = alt_frames

    else:
        gt_frames = latents.clone()
        alt_retvals = {
            "alt_gt_latents": alt_gt_latents.clone(),
            "alt_fake_latents": alt_fake_latents.clone(),
        } if args.alt_subsample_every is not None else dict()

    return {
        "pred_frames": the_frames,
        "gt_frames": gt_frames,
        "snapshot_samples": snapshot_samples,
    } | alt_retvals

def save_images(args, ckpt_args, latents_args, vae, the_frames, snapshot_samples, B, T,  out_folder):
    os.makedirs(out_folder, exist_ok=True)

    if (snapshot_samples is not None):
        snapshot_folder = os.path.join(out_folder, "snapshots")
        os.makedirs(snapshot_folder, exist_ok=True)
        for i, snapshot_sample in tqdm(enumerate(snapshot_samples)):
            if ckpt_args.dataset_name in ("Sky-timelapse", "CelebV-HQ"):
                snapshot_sample = vae.decode(
                    snapshot_sample / vae.config.scaling_factor
                ).sample
                assert snapshot_sample.shape == (B, 3, latents_args["image_size"], latents_args["image_size"])
            save_image(snapshot_sample, os.path.join(snapshot_folder, f"snapshot_{i}.png"), nrow=args.nrow, normalize=True, value_range=(-1, 1))
        # Create GIF from snapshots
        gif_frames = [imageio.imread(os.path.join(snapshot_folder, f"snapshot_{i}.png")) 
                    for i in range(len(snapshot_samples))]
        imageio.mimsave(os.path.join(out_folder, "diffusion.gif"), gif_frames, duration=0.1, loop=0)
        imageio.mimsave(os.path.join(out_folder, "diffusion.mp4"), gif_frames, fps=10)

    waypoint_frame_folder = os.path.join(out_folder, "waypoint_frames")
    os.makedirs(waypoint_frame_folder, exist_ok=True)
    for i in tqdm(range(T)):
        waypoint_frame = the_frames[:, i, :, :, :]
        save_image(waypoint_frame, os.path.join(waypoint_frame_folder, f"waypoint_frame_{i}.png"), nrow=args.nrow, normalize=True, value_range=(-1, 1))
    # Create GIF from waypoint frames
    gif_frames = [imageio.imread(os.path.join(waypoint_frame_folder, f"waypoint_frame_{i}.png")) 
                for i in range(T)]
    gif_path = os.path.join(out_folder, "waypoint_frames.gif")
    imageio.mimsave(gif_path, gif_frames, duration=args.subsample_every/30, loop=0)
    video_path = os.path.join(out_folder, "waypoint_frames.mp4")
    imageio.mimsave(video_path, gif_frames, fps=30/args.subsample_every)

    return video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride-dataset", type=int, default=None)
    parser.add_argument("--frames-decoded-per-call", type=int, default=2)
    parser.add_argument("--max-num-batches", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_folder", type=str, default="sample_sde")
    parser.add_argument("--snapshot-interval", type=int, default=20)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--subsample-every", type=int, default=1)
    parser.add_argument("--pin-every", type=int, default=4)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--prefix-length", type=int, default=1)
    parser.add_argument("--only-prefix-conditioning", action="store_true")
    parser.add_argument("--teacher-force-pinned", action="store_true")
    parser.add_argument("--latents-folder", type=str, default="/pscratch/sd/g/gabeguo/datasets/latents/Sky-timelapse/res-256x256-fpc-64/test")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--fvd-model", type=str, choices=["i3d", "videomae"], default="i3d")
    parser.add_argument("--fvd-videomae-ckpt", type=str, default=None) #"/pscratch/sd/g/gabeguo/DiT/pretrained_models/vit_g_hybrid_pt_1200e_ssv2_ft.pth")
    parser.add_argument("--ignore-last-frame", action="store_true")
    parser.add_argument("--alt-subsample-every", type=int, default=None)
    parser.add_argument("--clip-dt-eps", type=float, default=None)
    args = parser.parse_args()
    run_eval_loop(args)