"""
Synthetic Checkerboard Video Dataset.

Generates on-the-fly videos of an 8x8 checkerboard whose even-indexed squares
(row-major) are progressively drawn in random order over 32 frames.
Each video uses a single base color (red, green, or blue), with each square
drawn in a different shade of that color.

Usage:
    dataset = CheckerboardDataset(num_samples=10000, image_size=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for batch in loader:
        videos = batch['video']   # (B, 32, 3, H, W) float32 in [0, 1] or [-1, 1]
        labels = batch['label']   # (B,) int64 in {0, 1, 2}
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CheckerboardDataset(Dataset):
    """
    Synthetic video dataset of progressively drawn checkerboard patterns.

    Each sample is a 32-frame video on a black canvas. At each frame, one
    even-indexed square (in row-major order on an 8x8 grid) is filled with
    a unique shade of a randomly chosen base color (red/green/blue). The
    drawing order is randomized per sample.

    Args:
        num_samples: Number of videos in the dataset.
        image_size: Spatial resolution of each frame (must be divisible by 8).
        zero_center: If True, rescale pixel values from [0, 1] to [-1, 1].
        seed: Base random seed for reproducibility. If None, uses idx directly.
    """

    BOARD_SIZE = 8
    NUM_SQUARES = 32  # only even-indexed squares out of 64
    NUM_FRAMES = 32
    NUM_CLASSES = 3

    def __init__(self, num_samples=10000, image_size=64, zero_center=False, seed=42):
        super().__init__()
        assert image_size % self.BOARD_SIZE == 0, \
            f"image_size ({image_size}) must be divisible by board size ({self.BOARD_SIZE})"
        self.num_samples = num_samples
        self.image_size = image_size
        self.square_px = image_size // self.BOARD_SIZE
        self.zero_center = zero_center
        self.seed = seed

        self.even_indices = [i for i in range(self.BOARD_SIZE ** 2) if sum(divmod(i, self.BOARD_SIZE)) % 2 == 0]
        assert len(self.even_indices) == self.NUM_SQUARES == self.NUM_FRAMES == 32

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx if self.seed is not None else idx)

        color_class = rng.randint(0, self.NUM_CLASSES)
        draw_order = rng.permutation(self.even_indices)
        shades = rng.uniform(0.2, 1.0, size=self.NUM_SQUARES).astype(np.float32)

        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        frames = []

        for frame_idx in range(self.NUM_FRAMES):
            sq = draw_order[frame_idx]
            row, col = divmod(sq, self.BOARD_SIZE)
            y0 = row * self.square_px
            x0 = col * self.square_px

            color = np.zeros(3, dtype=np.float32)
            color[color_class] = shades[frame_idx]
            canvas[y0:y0 + self.square_px, x0:x0 + self.square_px] = color
            frames.append(canvas.copy())

        video = torch.from_numpy(np.stack(frames))   # (T, H, W, C)
        video = video.permute(0, 3, 1, 2)            # (T, C, H, W)
        assert video.shape == (self.NUM_FRAMES, 3, self.image_size, self.image_size)

        if self.zero_center:
            video = video * 2.0 - 1.0

        return {
            'video': video,           # (32, 3, H, W)
            'label': color_class,     # int in {0, 1, 2}
        }


if __name__ == '__main__':
    dataset = CheckerboardDataset(num_samples=100, image_size=64, zero_center=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"Video shape:  {batch['video'].shape}")   # (8, 32, 3, 64, 64)
    print(f"Label shape:  {batch['label'].shape}")    # (8,)
    print(f"Video range:  [{batch['video'].min():.2f}, {batch['video'].max():.2f}]")
    print(f"Labels:       {batch['label'].tolist()}")

    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    os.makedirs('dummy_renders/checkerboard', exist_ok=True)
    for vid_idx in range(2):
        label = batch['label'][vid_idx].item()
        color_name = ['red', 'green', 'blue'][label]
        pil_frames = []
        for t in range(32):
            frame = batch['video'][vid_idx, t].permute(1, 2, 0).numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))

            plt.imshow(frame)
            plt.title(f'{color_name} | frame {t}')
            plt.savefig(f'dummy_renders/checkerboard/vid{vid_idx}_{color_name}_f{t:02d}.png')
            plt.close()

        pil_frames[0].save(
            f'dummy_renders/checkerboard/vid{vid_idx}_{color_name}.gif',
            save_all=True,
            append_images=pil_frames[1:],
            duration=150,
            loop=0,
        )
    print("Saved sample renders to dummy_renders/checkerboard/")