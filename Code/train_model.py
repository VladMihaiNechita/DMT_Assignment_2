"""Training code for one final transformer model."""

from __future__ import annotations

import gc
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .config import COMMON_MODEL_SETTINGS, NUM_WORKERS
    from .dataset import ExpediaSearchDataset, pack_collate_fn
    from .features import FeatureSpec
    from .losses import LambdaNDCGLoss, pairwise_logistic_loss
    from .model import ExpediaTransformerRanker, ModelConfig
except ImportError:  # pragma: no cover - supports running from inside Code/
    from config import COMMON_MODEL_SETTINGS, NUM_WORKERS
    from dataset import ExpediaSearchDataset, pack_collate_fn
    from features import FeatureSpec
    from losses import LambdaNDCGLoss, pairwise_logistic_loss
    from model import ExpediaTransformerRanker, ModelConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: ExpediaSearchDataset, batch_searches: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    kwargs = {
        "batch_size": batch_searches,
        "shuffle": shuffle,
        "collate_fn": pack_collate_fn,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": generator,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    embedding_ids = {id(p) for p in model.embeddings.parameters()}
    normal_params = []
    embedding_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in embedding_ids:
            embedding_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            normal_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": normal_params, "weight_decay": cfg["weight_decay"]},
            {"params": embedding_params, "weight_decay": cfg["embedding_weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
    )


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.10, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def make_ema_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if torch.is_floating_point(tensor)
    }


@torch.no_grad()
def update_ema(model: nn.Module, ema: dict[str, torch.Tensor], decay: float) -> None:
    for name, tensor in model.state_dict().items():
        if name not in ema or not torch.is_floating_point(tensor):
            continue
        shadow = ema[name]
        if shadow.device != tensor.device:
            shadow = shadow.to(tensor.device)
            ema[name] = shadow
        shadow.mul_(decay).add_(tensor.detach(), alpha=1.0 - decay)


def load_ema(model: nn.Module, ema: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    for name, tensor in ema.items():
        if name in state:
            state[name] = tensor.to(device=state[name].device, dtype=state[name].dtype)
    model.load_state_dict(state, strict=False)


def train_one_epoch(
    model: ExpediaTransformerRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: LambdaNDCGLoss,
    device: torch.device,
    cfg: dict,
    ema: dict[str, torch.Tensor],
) -> float:
    model.train()
    losses = []

    for batch in loader:
        batch = batch.to(device)
        if batch.labels is None:
            raise RuntimeError("Training batch has no labels.")

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model.forward_outputs(
                batch.search_numeric,
                batch.result_numeric,
                batch.categorical,
                batch.cu_seqlens,
                batch.max_seqlen,
            )
            scores = outputs["score"]
            weights = batch.sample_weights if cfg["use_position_weights"] else None

            loss = criterion(scores, batch.labels, batch.cu_seqlens, sample_weights=weights)
            if cfg["full_list_loss_weight"] > 0:
                loss = loss + cfg["full_list_loss_weight"] * pairwise_logistic_loss(
                    scores,
                    batch.labels,
                    batch.cu_seqlens,
                    sample_weights=weights,
                    sigma=cfg["lambda_sigma"],
                )

            click_target = (batch.labels > 0).to(dtype=scores.dtype)
            booking_target = (batch.labels >= 5).to(dtype=scores.dtype)
            click_loss = F.binary_cross_entropy_with_logits(outputs["click_logit"], click_target, reduction="none")
            booking_loss = F.binary_cross_entropy_with_logits(outputs["booking_logit"], booking_target, reduction="none")

            if weights is not None:
                click_loss = (click_loss * weights).sum() / weights.sum().clamp_min(1.0)
                booking_loss = (booking_loss * weights).sum() / weights.sum().clamp_min(1.0)
            else:
                click_loss = click_loss.mean()
                booking_loss = booking_loss.mean()

            loss = loss + cfg["click_loss_weight"] * click_loss + cfg["booking_loss_weight"] * booking_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        update_ema(model, ema, cfg["ema_decay"])
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def train_model(cfg: dict, train_dataset: ExpediaSearchDataset, feature_spec: FeatureSpec, device: torch.device) -> ExpediaTransformerRanker:
    cfg = {**COMMON_MODEL_SETTINGS, **cfg}
    seed_everything(cfg["seed"])
    print(f"\nTraining {cfg['name']} for {cfg['epochs']} epochs")

    loader = make_loader(train_dataset, cfg["batch_searches"], shuffle=True, seed=cfg["seed"])
    model_cfg = ModelConfig(
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ffn_multiplier=cfg["ffn_multiplier"],
        dropout=cfg["dropout"],
        force_flash_attention=cfg["force_flash_attention"],
        head_hidden_dim=cfg["head_hidden_dim"],
        pre_score_weight=cfg["pre_score_weight"],
    )
    model = ExpediaTransformerRanker(feature_spec, model_cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg["epochs"] * len(loader), cfg["warmup_ratio"])
    criterion = LambdaNDCGLoss(top_k=cfg["loss_top_k"], sigma=cfg["lambda_sigma"])
    ema = make_ema_copy(model)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(model, loader, optimizer, scheduler, criterion, device, cfg, ema)
        minutes = (time.perf_counter() - t0) / 60.0
        print(f"{cfg['name']} epoch {epoch:02d}/{cfg['epochs']:02d} train_loss={loss:.6f} minutes={minutes:.1f}")

    if cfg.get("prediction_source", "ema") == "ema":
        load_ema(model, ema)

    model.to("cpu")
    del loader, optimizer, scheduler, ema
    gc.collect()
    return model
