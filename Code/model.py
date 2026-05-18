"""Permutation-invariant packed transformer reranker."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .features import FeatureSpec
except ImportError:  # pragma: no cover - supports running from inside Code/
    from features import FeatureSpec

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - depends on PyTorch version
    SDPBackend = None
    sdpa_kernel = None

try:
    from torch.nn.attention.varlen import varlen_attn
except Exception:  # pragma: no cover - depends on PyTorch version
    varlen_attn = None


@dataclass
class ModelConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.10
    force_flash_attention: bool = False
    head_hidden_dim: int = 0
    pre_score_weight: float = 0.0
    numeric_clip: float = 0.0
    embedding_dims: dict[str, int] = field(default_factory=dict)


def configure_torch_for_sdpa() -> None:
    """Enable high-performance CUDA attention backends when available."""

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _default_embedding_dim(name: str, cardinality: int) -> int:
    if name == "prop_id":
        return min(96, max(32, int(round(math.sqrt(cardinality)))))
    if name == "srch_destination_id":
        return min(64, max(24, int(round(math.sqrt(cardinality)))))
    return min(32, max(8, int(round(math.sqrt(cardinality)))))


class PackedMultiheadSelfAttention(nn.Module):
    """Self-attention over packed search lists using ``cu_seqlens`` boundaries."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, force_flash_attention: bool) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.force_flash_attention = force_flash_attention
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        total_tokens = x.shape[0]
        qkv = self.qkv(x).view(total_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)

        if varlen_attn is not None and x.is_cuda:
            try:
                out = varlen_attn(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    cu_seqlens.to(dtype=torch.int32),
                    cu_seqlens.to(dtype=torch.int32),
                    int(max_seqlen),
                    int(max_seqlen),
                    is_causal=False,
                )
            except RuntimeError:
                if self.force_flash_attention:
                    raise
                out = self._sdpa_segment_loop(q, k, v, cu_seqlens)
        else:
            out = self._sdpa_segment_loop(q, k, v, cu_seqlens)

        return self.out(out.reshape(total_tokens, self.d_model))

    def _sdpa_segment_loop(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        context = nullcontext()
        if self.force_flash_attention and sdpa_kernel is not None and SDPBackend is not None and q.is_cuda:
            context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)

        outputs: list[torch.Tensor] = []
        boundaries = cu_seqlens.detach().cpu().tolist()
        dropout_p = self.dropout if self.training else 0.0

        with context:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                qs = q[start:end].transpose(0, 1).unsqueeze(0)
                ks = k[start:end].transpose(0, 1).unsqueeze(0)
                vs = v[start:end].transpose(0, 1).unsqueeze(0)
                attended = F.scaled_dot_product_attention(
                    qs,
                    ks,
                    vs,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
                outputs.append(attended.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = PackedMultiheadSelfAttention(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            force_flash_attention=cfg.force_flash_attention,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.d_model * cfg.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, max_seqlen)
        x = x + self.ffn(self.norm2(x))
        return x


class ExpediaTransformerRanker(nn.Module):
    """Listwise reranker with no positional encodings.

    Search-level numeric features are already repeated per hotel row by the
    dataset. The model explicitly concatenates search features, result features,
    and the requested ID embeddings before the transformer stack.
    """

    def __init__(self, feature_spec: FeatureSpec, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.feature_spec = feature_spec
        self.cfg = cfg or ModelConfig()
        self.category_order = list(feature_spec.categorical)

        self.embeddings = nn.ModuleDict()
        embedding_width = 0
        for name in self.category_order:
            cardinality = feature_spec.categorical_cardinalities[name]
            dim = self.cfg.embedding_dims.get(name, _default_embedding_dim(name, cardinality))
            self.embeddings[name] = nn.Embedding(cardinality, dim)
            embedding_width += dim

        input_width = feature_spec.num_search_numeric + feature_spec.num_result_numeric + embedding_width
        self.input = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.blocks = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.num_layers)])
        self.norm = nn.LayerNorm(self.cfg.d_model)
        self.head = self._make_score_head()
        self.pre_head = self._make_score_head() if self.cfg.pre_score_weight > 0 else None
        self.click_head = nn.Linear(self.cfg.d_model, 1)
        self.booking_head = nn.Linear(self.cfg.d_model, 1)

    def _make_score_head(self) -> nn.Module:
        if self.cfg.head_hidden_dim <= 0:
            return nn.Linear(self.cfg.d_model, 1)
        return nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden_dim, 1),
        )

    def _encode(
        self,
        search_numeric: torch.Tensor,
        result_numeric: torch.Tensor,
        categorical: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.cfg.numeric_clip > 0:
            clip_value = float(self.cfg.numeric_clip)
            search_numeric = search_numeric.clamp(-clip_value, clip_value)
            result_numeric = result_numeric.clamp(-clip_value, clip_value)
        embedded = [
            self.embeddings[name](categorical[:, idx])
            for idx, name in enumerate(self.category_order)
        ]
        x = torch.cat([search_numeric, result_numeric] + embedded, dim=1)
        x = self.input(x)
        pre_score = None
        if self.pre_head is not None:
            pre_score = self.pre_head(x).squeeze(-1)
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen)
        x = self.norm(x)
        return x, pre_score

    def _score(self, encoded: torch.Tensor, pre_score: torch.Tensor | None) -> torch.Tensor:
        score = self.head(encoded).squeeze(-1)
        if pre_score is not None:
            score = score + self.cfg.pre_score_weight * pre_score
        return score

    def forward(
        self,
        search_numeric: torch.Tensor,
        result_numeric: torch.Tensor,
        categorical: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        encoded, pre_score = self._encode(search_numeric, result_numeric, categorical, cu_seqlens, max_seqlen)
        return self._score(encoded, pre_score)

    def forward_outputs(
        self,
        search_numeric: torch.Tensor,
        result_numeric: torch.Tensor,
        categorical: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> dict[str, torch.Tensor]:
        x, pre_score = self._encode(search_numeric, result_numeric, categorical, cu_seqlens, max_seqlen)
        return {
            "score": self._score(x, pre_score),
            "click_logit": self.click_head(x).squeeze(-1),
            "booking_logit": self.booking_head(x).squeeze(-1),
        }
