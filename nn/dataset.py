"""PyTorch dataset over the prepared G-buffer npz frames."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HairGBufferDataset(Dataset):
    """Random 512x512 patches (or full frames) from prepared npz dumps.

    train=True samples random crops with random horizontal flips
    (tangent.x is negated accordingly). train=False yields full frames.
    """

    def __init__(self, prepared_root: str | Path, split: str = "train",
                 patch_size: int | None = 512, frames_per_item: int = 1,
                 val_stride: int = 16, preload: bool = False):
        self.root = Path(prepared_root)
        self.patch_size = patch_size
        self.split = split
        self.samples = []
        self.cache: dict = {}

        for run_dir in sorted(p for p in self.root.iterdir()
                              if p.is_dir() and p.name.endswith(f"_{split}")):
            style = run_dir.name.rsplit("_", 1)[0]
            frames = sorted(run_dir.glob("frame_*.npz"))
            if split == "train" and val_stride > 1:
                # Hold out every val_stride-th frame for validation
                # (val_stride <= 1 keeps everything, used by ValDataset).
                frames = [f for i, f in enumerate(frames) if i % val_stride != 0]
            for frame in frames:
                self.samples.append((style, frame, frames_per_item))

        if preload:
            # Preload the training channels into RAM as float16 (~16 MB per
            # 720p frame). This removes the DataLoader worker processes
            # entirely - their shared-memory IPC kept exhausting the Windows
            # page file - and makes crops pure memory reads.
            for i, (style, frame, _) in enumerate(self.samples):
                data = self._load(frame)
                self.cache[frame] = {
                    "input": np.concatenate([data["input/coverage"],
                                             data["input/tangent"]], axis=-1).astype(np.float16),
                    "gt_coverage": data["gt/coverage"].astype(np.float16),
                    "gt_tangent": data["gt/tangent"].astype(np.float16),
                    "gt_mask": data["gt/hair_mask"].astype(np.float16),
                }
                if (i + 1) % 100 == 0:
                    print(f"preloaded {i + 1} / {len(self.samples)} frames into RAM", flush=True)
            cache_gb = sum(a["input"].nbytes + a["gt_coverage"].nbytes
                           + a["gt_tangent"].nbytes + a["gt_mask"].nbytes
                           for a in self.cache.values()) / 1e9
            print(f"RAM cache: {cache_gb:.1f} GB over {len(self.cache)} frames", flush=True)

    def __len__(self) -> int:
        return sum(count for _, _, count in self.samples)

    def _load(self, path: Path) -> dict:
        data = np.load(path)
        out = {}
        for key in data.files:
            array = data[key]
            if array.dtype == np.float32:
                array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            out[key] = array
        return out

    def __getitem__(self, index: int):
        # Map the flat index onto (style, frame) samples.
        for style, frame, count in self.samples:
            if index < count:
                break
            index -= count

        if frame in self.cache:
            data = self.cache[frame]
            x = data["input"]                     # already [C, T] concatenated
            y_coverage = data["gt_coverage"]
            y_tangent = data["gt_tangent"]
            y_mask = data["gt_mask"]
        else:
            data = self._load(frame)
            x = np.concatenate([data["input/coverage"],
                                data["input/tangent"]], axis=-1)  # (H, W, 4)
            y_coverage = data["gt/coverage"]
            y_tangent = data["gt/tangent"]
            y_mask = data["gt/hair_mask"]

        h, w = x.shape[:2]
        ps = self.patch_size

        if self.split == "train" and ps is not None and ps < min(h, w):
            top = np.random.randint(0, h - ps + 1)
            left = np.random.randint(0, w - ps + 1)
            x = x[top:top + ps, left:left + ps]
            y_coverage = y_coverage[top:top + ps, left:left + ps]
            y_tangent = y_tangent[top:top + ps, left:left + ps]
            y_mask = y_mask[top:top + ps, left:left + ps]

        if self.split == "train" and np.random.rand() < 0.5:
            x = x[:, ::-1].copy()
            y_coverage = y_coverage[:, ::-1].copy()
            y_tangent = y_tangent[:, ::-1].copy()
            y_mask = y_mask[:, ::-1].copy()
            x[..., 1] = -x[..., 1]  # tangent.x flips with the image.
            y_tangent[..., 0] = -y_tangent[..., 0]

        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()
        return {
            "style": style,
            "input": to_tensor(x),
            "gt_coverage": to_tensor(y_coverage),
            "gt_tangent": to_tensor(y_tangent),
            "gt_mask": to_tensor(y_mask),
        }


class ValDataset(HairGBufferDataset):
    """Full-frame validation dataset (no crops, no augmentation)."""

    def __init__(self, prepared_root: str | Path, split: str = "train",
                 val_stride: int = 16):
        # Keep ALL frames in the parent (val_stride=1), then pick the
        # held-out ones here - the parent would otherwise filter them out.
        super().__init__(prepared_root, split, patch_size=None,
                         frames_per_item=1, val_stride=1)
        self.val_stride = val_stride
        # Validation uses exactly the held-out frames.
        held_out = []
        for style, frame, _ in self.samples:
            index = int(frame.stem.split("_")[1])
            if index % self.val_stride == 0:
                held_out.append((style, frame, 1))
        self.samples = held_out

    def __getitem__(self, index: int):
        style, frame, _ = self.samples[index]
        data = self._load(frame)
        x = np.concatenate([data["input/coverage"], data["input/tangent"]], axis=-1)
        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()
        return {
            "style": style,
            "input": to_tensor(x),
            "gt_coverage": to_tensor(data["gt/coverage"]),
            "gt_tangent": to_tensor(data["gt/tangent"]),
            "gt_mask": to_tensor(data["gt/hair_mask"]),
        }
