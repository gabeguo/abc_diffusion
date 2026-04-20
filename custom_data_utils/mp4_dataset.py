import decord
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from tqdm import tqdm
from collections import OrderedDict

class Mp4FolderDataset(Dataset):
    def __init__(self, root, nframes, transform=None, max_open_readers=2):
        self.videos = sorted([
            os.path.join(root, f) for f in os.listdir(root) if f.endswith('.mp4')
        ])
        self.nframes = nframes
        self.transform = transform

        # Pre-scan all videos to build a flat chunk index
        self.chunks = []  # list of (video_idx, start_frame)
        for vid_idx, path in enumerate(tqdm(self.videos, desc="Scanning videos")):
            vr = decord.VideoReader(path)
            total = len(vr)
            for start in range(0, total - nframes + 1, nframes):
                self.chunks.append((vid_idx, start))
            # If you also want a partial tail chunk (padded or dropped), you could handle it here

        self.max_open_readers = max_open_readers
        self._readers = OrderedDict()
        return

    def _get_reader(self, vid_idx):
        # Reuse if already open
        if vid_idx in self._readers:
            vr = self._readers.pop(vid_idx)
            self._readers[vid_idx] = vr
            return vr
        # Open lazily
        vr = decord.VideoReader(self.videos[vid_idx])
        self._readers[vid_idx] = vr
        # LRU eviction so we don't keep too many files open
        if len(self._readers) > self.max_open_readers:
            self._readers.popitem(last=False)
        return vr

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        vid_idx, start = self.chunks[idx]
        vr = self._get_reader(vid_idx)
        indices = list(range(start, start + self.nframes))
        assert len(indices) == self.nframes
        frames = vr.get_batch(indices).asnumpy()          # (T, H, W, C), uint8
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        T, C, H, W = frames.shape
        assert T == self.nframes and C == 3 and H > C and W > C
        assert torch.max(frames) <= 255 and torch.min(frames) >= 0
        assert torch.max(frames) > 1.01 # make sure we're not already normalized
        if self.transform:
            frames = self.transform(frames)
        label = 0
        return frames, label