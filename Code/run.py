"""Run the final fixed transformer bag and create submission.csv."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Code"

from .config import DEVICE, MODEL_CONFIGS
from .data import fit_train_data, make_test_data
from .model import configure_torch_for_sdpa
from .predict import predict_scores, write_submission
from .train_model import train_model


def main() -> None:
    configure_torch_for_sdpa()
    device = torch.device(DEVICE)
    print("Using device:", device)

    train_dataset, feature_spec, preprocessor = fit_train_data()
    models = []

    for cfg in MODEL_CONFIGS:
        models.append(train_model(cfg, train_dataset, feature_spec, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The test file is deliberately read only after every model has finished
    # training, so the final run is train-only until prediction time.
    del train_dataset
    gc.collect()
    test_dataset = make_test_data(preprocessor, feature_spec)

    ensemble_scores = np.zeros(len(test_dataset.srch_ids), dtype=np.float64)
    for model, cfg in zip(models, MODEL_CONFIGS):
        print(f"Scoring {cfg['name']}")
        ensemble_scores += predict_scores(model, test_dataset, device) / len(MODEL_CONFIGS)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_submission(test_dataset, ensemble_scores)


if __name__ == "__main__":
    main()
