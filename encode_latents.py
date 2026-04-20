import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import os
import argparse
from custom_data_utils.utils import center_crop_transform, normalize_transform, collate_no_audio
import json
from custom_data_utils.sky_timelapse_dataset import VideoFolder
from custom_data_utils.mp4_dataset import Mp4FolderDataset

# TODO: refactor to not have same arguments for each dataset
def main():
    parser = argparse.ArgumentParser()
    # UCF-101 arguments
    parser.add_argument("--ucf101-data-path", type=str, default="/pscratch/sd/g/gabeguo/datasets/UCF-101")
    parser.add_argument("--ucf101-annotation-path", type=str, default="/pscratch/sd/g/gabeguo/datasets/ucfTrainTestlist")
    parser.add_argument("--ucf101-fold", type=int, default=1)
    parser.add_argument("--ucf101-frames-per-clip", type=int, default=32)
    parser.add_argument("--ucf101-step-between-clips", type=int, default=4)
    # Sky-timelapse arguments
    parser.add_argument("--sky-timelapse-data-path", type=str, default="/pscratch/sd/g/gabeguo/datasets/sky_timelapse/sky_timelapse/sky_test")
    parser.add_argument("--sky-timelapse-frames-per-clip", type=int, default=32)
    # CelebV-HQ arguments
    parser.add_argument("--celebv-hq-data-path", type=str, default="/pscratch/sd/g/gabeguo/CelebV-HQ/downloaded_celebvhq/processed")
    parser.add_argument("--celebv-hq-frames-per-clip", type=int, default=32)
    # Dataset options
    parser.add_argument("--dataset", type=str, default="UCF-101", choices=["UCF-101", "Sky-timelapse", "CelebV-HQ"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae", type=str, default="ema")
    parser.add_argument("--output-dir", type=str, default="/pscratch/sd/g/gabeguo/datasets/latents/UCF-101")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--save-float32", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    args.output_dir = os.path.join(args.output_dir, f"{'test' if args.test else 'train'}")
    os.makedirs(args.output_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    if args.dataset == "UCF-101":
        data_transform = transforms.Compose([
            partial(center_crop_transform, image_size=args.image_size),
            normalize_transform,
        ])
        dataset = torchvision.datasets.UCF101(
            root=args.ucf101_data_path,
            annotation_path=args.ucf101_annotation_path,
            frames_per_clip=args.ucf101_frames_per_clip,
            step_between_clips=args.ucf101_step_between_clips,
            fold=args.ucf101_fold, train=not args.test,
            transform=data_transform,
            num_workers=args.num_workers,
            output_format="TCHW",
        )
    elif args.dataset == "Sky-timelapse":
        dataset = VideoFolder(
            root=args.sky_timelapse_data_path, 
            nframes=args.sky_timelapse_frames_per_clip, 
            transform=transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
        )
    elif args.dataset == "CelebV-HQ":
        dataset = Mp4FolderDataset(
            root=args.celebv_hq_data_path,
            nframes=args.celebv_hq_frames_per_clip,
            transform=transforms.Compose([
                partial(center_crop_transform, image_size=args.image_size),
                normalize_transform,
            ])
        )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=1,
        collate_fn=collate_no_audio if args.dataset == "UCF-101" else None
    )

    latent_size = args.image_size // 8  # 32 for 256x256

    # Pre-allocate a memory-mapped file for all latents
    N = len(dataset)
    if args.dataset == "UCF-101":
        L = args.ucf101_frames_per_clip
    elif args.dataset == "Sky-timelapse":
        L = args.sky_timelapse_frames_per_clip
    elif args.dataset == "CelebV-HQ":
        L = args.celebv_hq_frames_per_clip
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    shape = (N, L, 4, latent_size, latent_size)
    
    latents_path = os.path.join(args.output_dir, "latents.npy")
    labels_path = os.path.join(args.output_dir, "labels.npy")

    the_dtype = np.float32 if args.save_float32 else np.float16
    latents_mmap = np.lib.format.open_memmap(
        latents_path, mode='w+', dtype=the_dtype, shape=shape
    )
    all_labels = np.zeros(N, dtype=np.int64)

    idx = 0
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for videos, labels in tqdm(loader, desc="Encoding"):
            if args.dataset == "UCF-101":
                B_cur, L_cur, C, H, W = videos.shape
                assert L_cur == args.ucf101_frames_per_clip
            elif args.dataset == "Sky-timelapse":
                B_cur, C, L_cur, H, W = videos.shape
                assert C == 3
                assert L_cur == args.sky_timelapse_frames_per_clip
                videos = videos.permute(0, 2, 1, 3, 4)
                assert -1.01 <= videos.min() and videos.max() <= 1.01
                assert videos.shape == (B_cur, L_cur, C, H, W)
                labels = torch.zeros_like(labels) # labels are meaningless for sky-timelapse
            elif args.dataset == "CelebV-HQ":
                B_cur, L_cur, C, H, W = videos.shape
                assert L_cur == args.celebv_hq_frames_per_clip
                assert H == W
                assert C == 3
                assert -1.01 < videos.min()
                assert videos.max() < 1.01
            videos = videos.to(device)
            flat = videos.reshape(B_cur * L_cur, C, H, W)
            z = vae.encode(flat).latent_dist.sample().mul_(vae.config.scaling_factor)
            z = z.reshape(B_cur, L_cur, *z.shape[1:])
            
            latents_mmap[idx:idx+B_cur] = z.cpu().float().numpy().astype(the_dtype)
            all_labels[idx:idx+B_cur] = labels.numpy()
            idx += B_cur

            if (idx // B_cur) % 5000 == 0:
                latents_mmap.flush()

    latents_mmap.flush()
    np.save(labels_path, all_labels[:idx])
    print(f"Saved {idx} clips to {args.output_dir}")

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    return

class PrecomputedLatentDataset(torch.utils.data.Dataset):
    def __init__(self, latents_path, labels_path, percent_train=None, train=True):
        self.latents = np.load(latents_path, mmap_mode='r')  # zero-copy
        self.labels = np.load(labels_path)
        if percent_train is not None:
            assert 0 < percent_train < 1
            num_train = int(len(self.labels) * percent_train)
            if train:
                self.latents = self.latents[:num_train]
                self.labels = self.labels[:num_train]
            else:
                self.latents = self.latents[num_train:]
                self.labels = self.labels[num_train:]
        return
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        z = torch.from_numpy(self.latents[idx].copy()).float()  # (L, 4, 32, 32)
        y = int(self.labels[idx])
        return z, y

if __name__ == "__main__":
    main()