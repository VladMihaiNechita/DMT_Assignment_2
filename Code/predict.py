"""Prediction and submission writing."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch

from .config import PREDICT_BATCH_SEARCHES, SUBMISSION_PATH
from .dataset import ExpediaSearchDataset
from .model import ExpediaTransformerRanker
from .train_model import make_loader


@torch.inference_mode()
def predict_scores(model: ExpediaTransformerRanker, test_dataset: ExpediaSearchDataset, device: torch.device) -> np.ndarray:
    model.to(device)
    model.eval()
    loader = make_loader(test_dataset, PREDICT_BATCH_SEARCHES, shuffle=False)

    scores = []
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(batch.search_numeric, batch.result_numeric, batch.categorical, batch.cu_seqlens, batch.max_seqlen)
        scores.append(pred.float().cpu().numpy())

    model.to("cpu")
    out = np.concatenate(scores)
    del loader
    gc.collect()
    return out


def write_submission(test_dataset: ExpediaSearchDataset, scores: np.ndarray) -> None:
    sub = pd.DataFrame({"srch_id": test_dataset.srch_ids, "prop_id": test_dataset.prop_ids, "score": scores})
    sub = sub.sort_values(["srch_id", "score"], ascending=[True, False], kind="mergesort")
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub[["srch_id", "prop_id"]].to_csv(SUBMISSION_PATH, index=False)
    print(f"\nWrote {SUBMISSION_PATH}")
