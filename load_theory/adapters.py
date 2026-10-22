"""Adapters from vendor configs to the framework's Python model schema."""

import json
from pathlib import Path

from .core import AttentionSpec, AttentionKind, MLPSpec, MLPKind, Model


def load_deepseek_config(path: str | Path) -> tuple[Model, list[str]]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = raw.get("text_config", raw)
    attention = AttentionSpec(
        kind=AttentionKind.MLA,
        num_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"],
        q_lora_rank=cfg["q_lora_rank"],
        kv_lora_rank=cfg.get("kv_lora_rank", cfg["head_dim"]),
        o_lora_rank=cfg["o_lora_rank"],
        rope_dim=cfg.get("qk_rope_head_dim"),
    )
    mlp = MLPSpec(
        kind=MLPKind.MOE,
        intermediate_size=cfg["moe_intermediate_size"],
        num_experts=cfg["n_routed_experts"],
        topk=cfg["num_experts_per_tok"],
        num_shared_experts=cfg.get("n_shared_experts", 0),
        expert_precision_bytes=0.5,
    )
    model = Model(
        name=Path(path).parent.name,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        vocab_size=cfg["vocab_size"],
        attention=attention,
        mlp=mlp,
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        max_position_embeddings=cfg["max_position_embeddings"],
    )
    warnings = [
        "DeepSeek V4.1 sparse attention, compress_ratios, index, engram, DSpark/MTP "
        "and vision components are not fully modeled in v0.1; text backbone is approximated."
    ]
    return model, warnings
