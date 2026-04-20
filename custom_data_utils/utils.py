import torch
import torch.nn.functional as F
import os

def unpack_batch(args, batch):
    if args.dataset_name in ("UCF-101", "Sky-timelapse", "CelebV-HQ"):
        x, y = batch
    elif args.dataset_name == "BAIR":
        x = batch['video']
        B, L, C, H, W = x.shape
        assert C == 3, "BAIR must have 3 channels"
        assert H == args.image_size and W == args.image_size
        y = torch.zeros(B, device=x.device, dtype=torch.long)
    elif args.dataset_name == "Checkerboard":
        x = batch['video']
        B, L, C, H, W = x.shape
        assert C == 3, "Checkerboard must have 3 channels"
        assert H == args.image_size and W == args.image_size
        y = batch['label'].long()
    else:
        raise ValueError(f"I no see that one: {args.dataset_name}")
    return x, y

def center_crop_transform(video, image_size=256):
    # video: (T, C, H, W) uint8 tensor from UCF101
    T, C, H, W = video.shape
    assert C == 3, "Number of channels must be 3"
    # Resize so the shortest side equals image_size
    if H < W:
        new_h, new_w = image_size, int(image_size * W / H)
    else:
        new_h, new_w = int(image_size * H / W), image_size
    video = F.interpolate(
        video.float(), size=(new_h, new_w), mode='bilinear', align_corners=False
    )
    # Center crop to image_size x image_size
    crop_y = (new_h - image_size) // 2
    crop_x = (new_w - image_size) // 2
    video = video[:, :, crop_y:crop_y + image_size, crop_x:crop_x + image_size]
    return video

def normalize_transform(video):
    assert video.max() > 1.01 # otherwise the video is already normalized
    assert video.min() >= 0.0
    if video.max() > 255.0:
        print(f"Video max: {video.max()}")
    assert video.max() <= 255.0 + 1e-3, f"Video max: {video.max()}"
    video = video.clamp(0.0, 255.0)
    video = video.float() / 255.0
    assert video.min() >= 0.0 and video.max() <= 1.0
    return video * 2 - 1

def collate_no_audio(batch):
    videos, _, labels = zip(*batch) # TODO: check this if we have problems
    return torch.stack(videos, 0), torch.tensor(labels)


def _ensure_latents_locally(latents_folder: str, dataset_name: str, mode: str = "train") -> None:
    """If latents/labels/args aren't on disk, download them from HF.

    Mirrors the layout in https://huggingface.co/datasets/therealgabeguo/abc_data
    e.g. sky_timelapse/res-256x256-fpc-64/train/{latents.npy, labels.npy, args.json}

    Forces the latents_folder to follow the layout in https://huggingface.co/datasets/therealgabeguo/abc_data (unless it's already there, then it's not my problem)
    """
    required = ("latents.npy", "labels.npy", "args.json")
    missing = [f for f in required if not os.path.exists(os.path.join(latents_folder, f))]
    if len(missing) == 0:
        print("All latents/labels/args are already on disk")
        return

    if dataset_name == "CelebV-HQ":
        assert mode == "train" # Only one split
    assert mode in ("train", "test"), f"Invalid mode: {mode}"

    hf_dataset_subdir = { # both have 256x256
        "Sky-timelapse": f"sky_timelapse/res-256x256-fpc-64/{mode}", # 64 frames
        "CelebV-HQ": f"celebv_hq/res-256x256-fpc-32/{mode}", # 32 frames
    }.get(dataset_name)
    if hf_dataset_subdir is None:
        raise FileNotFoundError(
            f"Missing {missing} in {latents_folder} and no HF fallback is "
            f"configured for dataset '{dataset_name}'."
        )
    latents_folder = latents_folder.rstrip("/")
    assert latents_folder.endswith(hf_dataset_subdir), f"Latents folder {latents_folder} does not match the expected layout: {hf_dataset_subdir}"
    assert len(latents_folder) > len(hf_dataset_subdir) + 1, f"Folder is NOT long enough (no saving in root folders)"

    from huggingface_hub import hf_hub_download  # lazy import
    os.makedirs(latents_folder, exist_ok=True)
    for fname in missing:
        print(f"[HF] Downloading {hf_dataset_subdir}/{fname} -> {latents_folder}")
        hf_hub_download(
            repo_id="therealgabeguo/abc_data",
            repo_type="dataset",
            subfolder=hf_dataset_subdir,
            filename=fname,
            local_dir=latents_folder[:-len(hf_dataset_subdir)],
        )
    return
