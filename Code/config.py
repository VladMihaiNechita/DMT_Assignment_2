"""Fixed settings for the final submission run."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "Datasets" / "training_set_VU_DM.parquet"
TEST_PATH = ROOT / "Datasets" / "test_set_VU_DM.parquet"
SUBMISSION_PATH = ROOT / "submission.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PREDICT_BATCH_SEARCHES = 1024

# These columns only exist in labeled data. If they are present in the final
# test file then we are accidentally using the wrong file.
FORBIDDEN_TEST_COLUMNS = {"click_bool", "booking_bool", "gross_bookings_usd", "position"}

# All six models use the same features. The differences are the seed,
# pre-score weight, epoch count, and two small diversity choices: seed 1 uses
# lower dropout, and seed 6 removes the auxiliary click/booking heads.
MODEL_CONFIGS = [
    {
        "name": "s1_pre04_drop15_e07_raw",
        "epochs": 7,
        "prediction_source": "model",
        "batch_searches": 512,
        "seed": 1,
        "pre_score_weight": 0.4,
        "dropout": 0.15,
        "lr": 0.00008,
    },
    {
        "name": "s2_pre05_e10",
        "epochs": 10,
        "prediction_source": "ema",
        "batch_searches": 512,
        "seed": 2,
        "pre_score_weight": 0.5,
        "lr": 0.00008,
    },
    {
        "name": "s3_pre06_e09",
        "epochs": 9,
        "prediction_source": "ema",
        "batch_searches": 512,
        "seed": 3,
        "pre_score_weight": 0.6,
        "lr": 0.00008,
    },
    {
        "name": "s4_pre07_e09",
        "epochs": 9,
        "prediction_source": "ema",
        "batch_searches": 512,
        "seed": 4,
        "pre_score_weight": 0.7,
        "lr": 0.00008,
    },
    {
        "name": "s5_pre08_e09_raw",
        "epochs": 9,
        "prediction_source": "model",
        "batch_searches": 512,
        "seed": 5,
        "pre_score_weight": 0.8,
        "lr": 0.00008,
    },
    {
        "name": "s6_pre06_noaux_e09_raw",
        "epochs": 9,
        "prediction_source": "model",
        "batch_searches": 512,
        "seed": 6,
        "pre_score_weight": 0.6,
        "click_loss_weight": 0.0,
        "booking_loss_weight": 0.0,
        "lr": 0.00008,
    },
]

# These are shared hyperparameters. They came from the validation experiments,
# but they are now fixed numbers, so this final code never needs validation
# labels or early stopping.
COMMON_MODEL_SETTINGS = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 6,
    "ffn_multiplier": 4,
    "dropout": 0.20,
    "weight_decay": 0.040,
    "embedding_weight_decay": 0.090,
    "warmup_ratio": 0.12,
    "loss_top_k": 10,
    "lambda_sigma": 1.2,
    "full_list_loss_weight": 0.030,
    "click_loss_weight": 0.050,
    "booking_loss_weight": 0.100,
    "use_position_weights": True,
    "head_hidden_dim": 256,
    "ema_decay": 0.995,
    "force_flash_attention": True,
    "grad_clip": 1.0,
}
