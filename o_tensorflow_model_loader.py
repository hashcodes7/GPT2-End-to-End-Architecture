"""
weight_loader.py

Utility functions for loading pretrained GPT-2 weights into the custom
GPTModel implementation.

This module maps the pretrained parameter dictionary to the corresponding
layers of the custom GPT model, including:

- Token embeddings
- Positional embeddings
- Multi-head self-attention (Query, Key, Value, Output Projection)
- Feed-forward network
- Layer Normalization parameters
- Final Layer Normalization
- Output projection (weight tying)

Usage:
    from weight_loader import load_weights_into_gpt

    load_weights_into_gpt(gpt, params)
"""

import numpy as np
import torch


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch: {left.shape} != {right.shape}"
        )

    return torch.nn.Parameter(
        torch.as_tensor(
            right,
            dtype=left.dtype,
            device=left.device,
        )
    )


def load_weights_into_gpt(gpt, params):
    """
    Loads pretrained GPT-2 weights into a custom GPTModel instance.

    Args:
        gpt: Custom GPTModel object.
        params: Dictionary containing pretrained GPT-2 parameters.
    """

    # Embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # Transformer Blocks
    for b in range(len(params["blocks"])):

        # ---------------- Attention ----------------

        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"],
            3,
            axis=-1
        )

        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"],
            3,
            axis=-1
        )

        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )

        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # ---------------- Feed Forward ----------------

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )

        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )

        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # ---------------- LayerNorm ----------------

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"],
        )

        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"],
        )

        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"],
        )

        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"],
        )

    # Final LayerNorm
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # Output projection (weight tying)
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])