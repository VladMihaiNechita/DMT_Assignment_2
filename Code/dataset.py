"""Grouped search dataset and packed collate function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .features import FeatureSpec
    from .preprocessing import make_relevance_labels
except ImportError:  # pragma: no cover - supports running from inside Code/
    from features import FeatureSpec
    from preprocessing import make_relevance_labels


@dataclass
class PackedBatch:
    search_numeric: torch.Tensor
    result_numeric: torch.Tensor
    categorical: torch.Tensor
    labels: torch.Tensor | None
    sample_weights: torch.Tensor | None
    cu_seqlens: torch.Tensor
    group_lengths: torch.Tensor
    srch_ids: torch.Tensor
    prop_ids: torch.Tensor

    @property
    def max_seqlen(self) -> int:
        return int(self.group_lengths.max().item()) if self.group_lengths.numel() else 0

    @property
    def num_tokens(self) -> int:
        return int(self.search_numeric.shape[0])

    @property
    def num_searches(self) -> int:
        return int(self.group_lengths.shape[0])

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "PackedBatch":
        return PackedBatch(
            search_numeric=self.search_numeric.to(device, non_blocking=non_blocking),
            result_numeric=self.result_numeric.to(device, non_blocking=non_blocking),
            categorical=self.categorical.to(device, non_blocking=non_blocking),
            labels=None if self.labels is None else self.labels.to(device, non_blocking=non_blocking),
            sample_weights=None if self.sample_weights is None else self.sample_weights.to(device, non_blocking=non_blocking),
            cu_seqlens=self.cu_seqlens.to(device, non_blocking=non_blocking),
            group_lengths=self.group_lengths.to(device, non_blocking=non_blocking),
            srch_ids=self.srch_ids.to(device, non_blocking=non_blocking),
            prop_ids=self.prop_ids.to(device, non_blocking=non_blocking),
        )


class ExpediaSearchDataset(Dataset):
    """A dataset where each item is one ``srch_id`` list."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_spec: FeatureSpec,
        has_labels: bool = True,
        sort_by_search: bool = True,
    ) -> None:
        if sort_by_search:
            frame = frame.sort_values("srch_id", kind="mergesort").reset_index(drop=True)

        self.feature_spec = feature_spec
        self.has_labels = bool(has_labels and {"click_bool", "booking_bool"}.intersection(frame.columns))

        self.search_numeric = frame[feature_spec.search_numeric].to_numpy(dtype=np.float32, copy=True)
        self.result_numeric = frame[feature_spec.result_numeric].to_numpy(dtype=np.float32, copy=True)
        self.categorical = frame[feature_spec.categorical].to_numpy(dtype=np.int64, copy=True)
        self.srch_ids = frame["srch_id"].to_numpy(dtype=np.int64, copy=True)
        if "_submission_prop_id" in frame:
            self.prop_ids = frame["_submission_prop_id"].to_numpy(dtype=np.int64, copy=True)
        elif "prop_id" in frame:
            self.prop_ids = frame["prop_id"].to_numpy(dtype=np.int64, copy=True)
        else:
            self.prop_ids = np.arange(len(frame), dtype=np.int64)
        self.labels = make_relevance_labels(frame) if self.has_labels else None
        self.sample_weights = self._make_sample_weights(frame) if self.has_labels else None

        if len(self.srch_ids) == 0:
            self.starts = np.array([], dtype=np.int64)
            self.ends = np.array([], dtype=np.int64)
        else:
            boundaries = np.flatnonzero(self.srch_ids[1:] != self.srch_ids[:-1]) + 1
            self.starts = np.r_[0, boundaries].astype(np.int64)
            self.ends = np.r_[boundaries, len(self.srch_ids)].astype(np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        start = self.starts[idx]
        end = self.ends[idx]
        item = {
            "search_numeric": self.search_numeric[start:end],
            "result_numeric": self.result_numeric[start:end],
            "categorical": self.categorical[start:end],
            "srch_ids": self.srch_ids[start:end],
            "prop_ids": self.prop_ids[start:end],
        }
        if self.labels is not None:
            item["labels"] = self.labels[start:end]
        if self.sample_weights is not None:
            item["sample_weights"] = self.sample_weights[start:end]
        return item

    def iter_group_slices(self) -> Iterator[tuple[int, int]]:
        yield from zip(self.starts.tolist(), self.ends.tolist())

    @staticmethod
    def _make_sample_weights(frame: pd.DataFrame) -> np.ndarray:
        if "position" not in frame.columns:
            return np.ones(len(frame), dtype=np.float32)
        labels = make_relevance_labels(frame)
        position = pd.to_numeric(frame["position"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        weights = np.ones(len(frame), dtype=np.float32)
        relevant = labels > 0
        weights[relevant] = np.log2(1.0 + np.clip(position[relevant], 1.0, None))
        return weights


class CachedExpediaSearchDataset(Dataset):
    """Dataset backed by tensors saved in the same packed format."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.search_numeric = tensors["search_numeric"]
        self.result_numeric = tensors["result_numeric"]
        self.categorical = tensors["categorical"]
        self.labels = tensors.get("labels")
        self.sample_weights = tensors.get("sample_weights")
        self.srch_ids = tensors["srch_ids"]
        self.prop_ids = tensors["prop_ids"]
        self.starts = tensors["starts"].to(dtype=torch.long)
        self.ends = tensors["ends"].to(dtype=torch.long)

    def __len__(self) -> int:
        return int(self.starts.numel())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx].item())
        end = int(self.ends[idx].item())
        item = {
            "search_numeric": self.search_numeric[start:end],
            "result_numeric": self.result_numeric[start:end],
            "categorical": self.categorical[start:end],
            "srch_ids": self.srch_ids[start:end],
            "prop_ids": self.prop_ids[start:end],
        }
        if self.labels is not None:
            item["labels"] = self.labels[start:end]
        if self.sample_weights is not None:
            item["sample_weights"] = self.sample_weights[start:end]
        return item


def _concat_values(values: list[np.ndarray | torch.Tensor]) -> torch.Tensor:
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values)
    return torch.from_numpy(np.concatenate(values))


def pack_collate_fn(items: list[dict[str, np.ndarray | torch.Tensor]]) -> PackedBatch:
    lengths = np.asarray([len(item["srch_ids"]) for item in items], dtype=np.int32)
    cu_seqlens = np.empty(len(lengths) + 1, dtype=np.int32)
    cu_seqlens[0] = 0
    np.cumsum(lengths, out=cu_seqlens[1:])

    has_labels = "labels" in items[0]
    labels = _concat_values([item["labels"] for item in items]).float() if has_labels else None
    has_weights = "sample_weights" in items[0]
    sample_weights = _concat_values([item["sample_weights"] for item in items]).float() if has_weights else None

    return PackedBatch(
        search_numeric=_concat_values([item["search_numeric"] for item in items]).float(),
        result_numeric=_concat_values([item["result_numeric"] for item in items]).float(),
        categorical=_concat_values([item["categorical"] for item in items]).long(),
        labels=labels,
        sample_weights=sample_weights,
        cu_seqlens=torch.from_numpy(cu_seqlens),
        group_lengths=torch.from_numpy(lengths),
        srch_ids=_concat_values([item["srch_ids"] for item in items]).long(),
        prop_ids=_concat_values([item["prop_ids"] for item in items]).long(),
    )
