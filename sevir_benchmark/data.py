import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_CUTOFF_DATE = "2019-06-01"
DEFAULT_CLIP_STARTS = (12,)
EXPECTED_TOTAL_FRAMES = 25
DEFAULT_PREFIX_LENGTH = 13
VIL_DATASET_KEYS = ("vil", "VIL")
TIMESTAMP_COLUMNS = (
    "time_utc",
    "event_start_time",
    "start_time",
    "datetime_utc",
    "timestamp",
    "time",
)


@dataclass(frozen=True)
class SevirSample:
    event_id: str
    file_path: str
    file_index: int
    clip_start: int
    timestamp: pd.Timestamp


def _resolve_catalog_path(root_dir: str) -> str:
    catalog_path = os.path.join(root_dir, "CATALOG.csv")
    if not os.path.isfile(catalog_path):
        raise FileNotFoundError(
            f"Could not find CATALOG.csv under {root_dir}. "
            "Run download_sevir_vil.py first or pass the correct --data-root."
        )
    return catalog_path


def _load_catalog(root_dir: str) -> pd.DataFrame:
    catalog = pd.read_csv(_resolve_catalog_path(root_dir), low_memory=False)
    lower_map = {col.lower(): col for col in catalog.columns}

    img_type_col = lower_map.get("img_type")
    if img_type_col is None:
        raise ValueError(f"Expected img_type column in catalog. Columns: {list(catalog.columns)}")
    catalog = catalog[catalog[img_type_col].astype(str).str.lower() == "vil"].copy()

    required_lower = ("event_id", "file_name", "file_index")
    missing = [col for col in required_lower if col not in lower_map]
    if missing:
        raise ValueError(f"Missing required catalog columns: {missing}. Columns: {list(catalog.columns)}")

    timestamp_col = None
    for candidate in TIMESTAMP_COLUMNS:
        actual = lower_map.get(candidate.lower())
        if actual is not None:
            timestamp_col = actual
            break
    if timestamp_col is None:
        raise ValueError(
            "Could not find a timestamp column in the SEVIR catalog. "
            f"Tried {TIMESTAMP_COLUMNS}, available columns: {list(catalog.columns)}"
        )

    catalog = catalog.rename(
        columns={
            lower_map["event_id"]: "event_id",
            lower_map["file_name"]: "file_name",
            lower_map["file_index"]: "file_index",
            timestamp_col: "timestamp",
        }
    )
    catalog["timestamp"] = pd.to_datetime(catalog["timestamp"], utc=True, errors="coerce")
    if catalog["timestamp"].isna().any():
        bad_rows = int(catalog["timestamp"].isna().sum())
        raise ValueError(f"Failed to parse {bad_rows} SEVIR timestamps from catalog.")
    return catalog[["event_id", "file_name", "file_index", "timestamp"]]


def _event_hash_bucket(event_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{event_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16)


def _resolve_file_path(root_dir: str, file_name: str) -> str:
    candidates = [
        os.path.join(root_dir, "data", "vil", file_name),
        os.path.join(root_dir, "data", file_name),
        os.path.join(root_dir, file_name),
    ]
    if file_name.startswith("vil/"):
        candidates.append(os.path.join(root_dir, "data", file_name[len("vil/"):]))
    if len(file_name.split("/")) == 1:
        year_guess = next((token for token in file_name.split("_") if token.isdigit() and len(token) == 4), None)
        if year_guess is not None:
            candidates.append(os.path.join(root_dir, "data", "vil", year_guess, file_name))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    vil_root = os.path.join(root_dir, "data", "vil")
    if os.path.isdir(vil_root):
        for current_root, _, files in os.walk(vil_root):
            if file_name in files:
                return os.path.join(current_root, file_name)

    raise FileNotFoundError(
        f"Could not locate SEVIR VIL file '{file_name}' under {root_dir}. "
        "Make sure the VIL modality has been downloaded."
    )


def _select_split_events(
    event_records: pd.DataFrame,
    split: str,
    cutoff_date: str,
    val_fraction: float,
    seed: int,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    pre_cutoff = event_records[event_records["timestamp"] < cutoff]
    post_cutoff = event_records[event_records["timestamp"] >= cutoff]

    if split == "test":
        return post_cutoff

    if split not in ("train", "val"):
        raise ValueError(f"Unsupported split: {split}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    if val_fraction == 0.0:
        return pre_cutoff if split == "train" else pre_cutoff.iloc[:0]

    event_records = pre_cutoff.copy()
    event_records["bucket"] = event_records["event_id"].map(lambda x: _event_hash_bucket(str(x), seed))
    is_val = event_records["bucket"] < val_fraction
    selected = event_records[is_val] if split == "val" else event_records[~is_val]
    return selected.drop(columns=["bucket"])


def build_sevir_samples(
    root_dir: str,
    split: str,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
    val_fraction: float = 0.1,
    seed: int = 0,
    clip_starts: Sequence[int] = DEFAULT_CLIP_STARTS,
    total_frames: int = EXPECTED_TOTAL_FRAMES,
    max_events: Optional[int] = None,
) -> List[SevirSample]:
    catalog = _load_catalog(root_dir)
    grouped = (
        catalog.sort_values(["event_id", "timestamp"])
        .groupby("event_id", as_index=False)
        .first()
    )
    event_index = _select_split_events(
        event_records=grouped,
        split=split,
        cutoff_date=cutoff_date,
        val_fraction=val_fraction,
        seed=seed,
    )
    if max_events is not None:
        event_index = event_index.iloc[:max_events]

    samples: List[SevirSample] = []
    for row in event_index.itertuples(index=False):
        file_path = _resolve_file_path(root_dir, row.file_name)
        for clip_start in clip_starts:
            if clip_start < 0:
                raise ValueError(f"clip_start must be non-negative, got {clip_start}")
            samples.append(
                SevirSample(
                    event_id=str(row.event_id),
                    file_path=file_path,
                    file_index=int(row.file_index),
                    clip_start=int(clip_start),
                    timestamp=row.timestamp,
                )
            )
    if not samples:
        raise ValueError(
            f"No SEVIR VIL samples found for split='{split}'. "
            "Check that the requested data years are present locally."
        )
    return samples


def _find_h5_dataset(h5_file: h5py.File):
    for key in VIL_DATASET_KEYS:
        if key in h5_file:
            return h5_file[key]
    for key in h5_file.keys():
        value = h5_file[key]
        if isinstance(value, h5py.Dataset):
            return value
    raise ValueError(f"Could not find any dataset in H5 file: {h5_file.filename}")


def _canonicalize_frames(array: torch.Tensor) -> torch.Tensor:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D VIL tensor, got shape {tuple(array.shape)}")
    if array.shape[0] in (25, 49):
        frames = array
    elif array.shape[-1] in (25, 49):
        frames = array.permute(2, 0, 1)
    else:
        time_axis = min(range(3), key=lambda idx: abs(array.shape[idx] - 49))
        if array.shape[time_axis] < EXPECTED_TOTAL_FRAMES:
            raise ValueError(f"Could not infer time axis for shape {tuple(array.shape)}")
        frames = array.permute(time_axis, *[i for i in range(3) if i != time_axis])
    return frames.unsqueeze(1).float()


class SevirVILDataset(Dataset):
    """
    SEVIR VIL loader for precipitation nowcasting.

    By default this uses the common 25-frame center clip with 13 context frames
    and 12 forecast frames.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 128,
        prefix_length: int = DEFAULT_PREFIX_LENGTH,
        target_length: int = EXPECTED_TOTAL_FRAMES - DEFAULT_PREFIX_LENGTH,
        clip_starts: Sequence[int] = DEFAULT_CLIP_STARTS,
        cutoff_date: str = DEFAULT_CUTOFF_DATE,
        val_fraction: float = 0.1,
        seed: int = 0,
        max_events: Optional[int] = None,
        normalize_mode: str = "zero_center",
        return_native: bool = False,
        return_metadata: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.prefix_length = prefix_length
        self.target_length = target_length
        self.total_frames = prefix_length + target_length
        self.normalize_mode = normalize_mode
        self.return_native = return_native
        self.return_metadata = return_metadata
        self._file_handles: Dict[str, h5py.File] = {}
        self.samples = build_sevir_samples(
            root_dir=root_dir,
            split=split,
            cutoff_date=cutoff_date,
            val_fraction=val_fraction,
            seed=seed,
            clip_starts=clip_starts,
            total_frames=self.total_frames,
            max_events=max_events,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_dataset(self, file_path: str):
        handle = self._file_handles.get(file_path)
        if handle is None:
            handle = h5py.File(file_path, "r")
            self._file_handles[file_path] = handle
        return _find_h5_dataset(handle)

    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = frames.clamp(0.0, 1.0)
        if self.normalize_mode == "zero_center":
            frames = frames * 2.0 - 1.0
        elif self.normalize_mode != "unit":
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")
        return frames

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        dataset = self._get_dataset(sample.file_path)
        raw = torch.from_numpy(dataset[sample.file_index][...])
        frames = _canonicalize_frames(raw)

        clip_end = sample.clip_start + self.total_frames
        if frames.shape[0] < clip_end:
            raise ValueError(
                f"SEVIR sample has {frames.shape[0]} frames, but requested clip "
                f"[{sample.clip_start}:{clip_end}]"
            )
        frames = frames[sample.clip_start:clip_end]
        native_frames = self._normalize(frames.clone())
        resized_frames = native_frames
        if self.image_size is not None and native_frames.shape[-1] != self.image_size:
            resized_frames = F.interpolate(
                native_frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        label = torch.zeros((), dtype=torch.long)
        metadata = {
            "dataset_index": idx,
            "event_id": sample.event_id,
            "timestamp": sample.timestamp.isoformat(),
            "clip_start": sample.clip_start,
        }
        if self.return_native and self.return_metadata:
            return resized_frames, label, native_frames, metadata
        if self.return_native:
            return resized_frames, label, native_frames
        if self.return_metadata:
            return resized_frames, label, metadata
        return resized_frames, label

    def __del__(self):
        for handle in getattr(self, "_file_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def inverse_normalize_vil(x: torch.Tensor, normalize_mode: str = "zero_center") -> torch.Tensor:
    if normalize_mode == "zero_center":
        x = (x + 1.0) / 2.0
    elif normalize_mode != "unit":
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
    return x.clamp(0.0, 1.0) * 255.0
