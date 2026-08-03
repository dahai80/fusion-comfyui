"""IPAdapter-Flux MLX — cross-attention image prompt injection for Flux DiT.

Ports the InstantX IPAdapter-Flux architecture to pure MLX:
  SigLIP-SO400M vision encoder -> MLPProj -> IPAFluxAttnProcessor

Imported by: __init__.py (NODE_CLASS_MAPPINGS registration)
API: FusionIPAdapterLoader returns FUSION_IP_ADAPTER_MODEL (FluxIPAdapterPipeline)
     FusionIPAdapterApply returns FUSION_IP_ADAPTER_EMBED (dict with image_embeds, pipeline, weight)
User instruction: "继续IPAdapter-Flux MLX 移植"
"""

import glob
import io
import logging
import os
import re
import weakref
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

import core.async_utils

logger = logging.getLogger("fusion_comfyui.nodes.ip_adapter")

_IP_ADAPTER_MODEL_DIRS = []


class _RMSNormNoAffine(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        norm = mx.fast.rms_norm(x, mx.ones(x.shape[-1:], dtype=x.dtype), self.eps)
        return norm


class _SigLIPSelfAttention(nn.Module):
    def __init__(self, dim=1152, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x):
        B, L, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.q_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, n * d)
        return self.out_proj(out)


class _SigLIPMLP(nn.Module):
    def __init__(self, dim=1152, mid_dim=4304):
        super().__init__()
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class _SigLIPEncoderLayer(nn.Module):
    def __init__(self, dim=1152, num_heads=16, mid_dim=4304):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.self_attn = _SigLIPSelfAttention(dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = _SigLIPMLP(dim, mid_dim)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _SigLIPMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, dim=1152, num_heads=16, mid_dim=4304):
        super().__init__()
        self.probe = mx.zeros((1, 1, dim))
        self.attention = _SigLIPSelfAttention(dim, num_heads)
        self.layernorm = nn.LayerNorm(dim)
        self.mlp = _SigLIPMLP(dim, mid_dim)

    def __call__(self, x):
        B = x.shape[0]
        dim = x.shape[2]
        n = self.attention.num_heads
        d = self.attention.head_dim
        probe = mx.broadcast_to(self.probe, (B, 1, dim))
        q = self.attention.q_proj(probe).reshape(B, 1, n, d).transpose(0, 2, 1, 3)
        normed_x = self.layernorm(x)
        k = self.attention.k_proj(normed_x).reshape(B, -1, n, d).transpose(0, 2, 1, 3)
        v = self.attention.v_proj(normed_x).reshape(B, -1, n, d).transpose(0, 2, 1, 3)
        hidden = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
        hidden = hidden.transpose(0, 2, 1, 3).reshape(B, 1, dim)
        hidden = self.attention.out_proj(hidden)
        residual = hidden
        hidden = self.layernorm(hidden)
        hidden = residual + self.mlp(hidden)
        return hidden[:, 0]


class SigLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        image_size=384,
        patch_size=14,
        dim=1152,
        num_heads=16,
        num_layers=27,
        mid_dim=4304,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.position_embedding = mx.zeros((1, self.num_patches, dim))
        for i in range(num_layers):
            setattr(self, f"encoder_layer_{i}", _SigLIPEncoderLayer(dim, num_heads, mid_dim))
        self.post_layernorm = nn.LayerNorm(dim)
        self.head = _SigLIPMultiheadAttentionPoolingHead(dim, num_heads)

    def __call__(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        x = x.reshape(B, -1, self.dim)
        x = x + self.position_embedding
        for i in range(27):
            layer = getattr(self, f"encoder_layer_{i}")
            x = layer(x)
        x = self.post_layernorm(x)
        pooler_output = self.head(x)
        return pooler_output


class MLPProjModel(nn.Module):
    def __init__(self, id_embeddings_dim=1152, cross_attention_dim=4096, num_tokens=128):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.linear1 = nn.Linear(id_embeddings_dim, id_embeddings_dim * 2)
        self.linear2 = nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def __call__(self, id_embeds):
        x = nn.gelu(self.linear1(id_embeds))
        x = self.linear2(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class IPAFluxAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size=3072,
        cross_attention_dim=4096,
        num_tokens=128,
        scale=1.0,
        timestep_range=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.timestep_range = timestep_range

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.norm_added_k = _RMSNormNoAffine(eps=1e-5)
        self.norm_added_v = _RMSNormNoAffine(eps=1e-5)

        self._image_embeds = None

    def set_image_embeds(self, embeds):
        self._image_embeds = embeds

    def clear_image_embeds(self):
        self._image_embeds = None

    def __call__(self, hidden_states, t_sigma=None):
        """Compute IP cross-attention output given hidden states.

        Args:
            hidden_states: (B, S, hidden_size) block-level hidden states used as Q
            t_sigma: current timestep sigma for range filtering
        Returns:
            (B, S, hidden_size) IP cross-attention output, or None if disabled
        """
        if self._image_embeds is None:
            return None

        range_mask = None
        if self.timestep_range is not None and t_sigma is not None:
            if not isinstance(t_sigma, _mx.array):
                t_sigma = _mx.array(t_sigma, dtype=hidden_states.dtype)
            t_start = _mx.array(self.timestep_range[0], dtype=hidden_states.dtype)
            t_end = _mx.array(self.timestep_range[1], dtype=hidden_states.dtype)
            range_mask = ((t_sigma <= t_start) & (t_sigma >= t_end)).astype(hidden_states.dtype)

        num_heads = self.hidden_size // 128
        head_dim = 128

        B, seq_len, _ = hidden_states.shape
        query = hidden_states.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        ip_k = self.to_k_ip(self._image_embeds)
        ip_v = self.to_v_ip(self._image_embeds)

        L_ip = ip_k.shape[1]
        ip_k = ip_k.reshape(B, L_ip, num_heads, head_dim).transpose(0, 2, 1, 3)
        ip_v = ip_v.reshape(B, L_ip, num_heads, head_dim).transpose(0, 2, 1, 3)

        ip_k = self.norm_added_k(ip_k)
        ip_v = self.norm_added_v(ip_v)

        scale = head_dim**-0.5
        attn_out = mx.fast.scaled_dot_product_attention(query, ip_k, ip_v, scale=scale)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, seq_len, self.hidden_size)

        result = attn_out * self.scale
        if range_mask is not None:
            result = result * range_mask
        return result


def _remap_siglip_weights(weights):
    remapped = {}
    for key, value in weights.items():
        if not key.startswith("vision_model."):
            continue

        if key == "vision_model.embeddings.patch_embedding.weight":
            if value.ndim == 4:
                value = mx.transpose(value, (0, 2, 3, 1))
            remapped["patch_embedding.weight"] = value
            continue

        if key == "vision_model.embeddings.position_embedding.weight":
            if value.ndim == 2:
                value = value.reshape(1, value.shape[0], value.shape[1])
            remapped["position_embedding"] = value
            continue

        if key.startswith("vision_model.post_layernorm."):
            param = key.split(".")[-1]
            remapped[f"post_layernorm.{param}"] = value
            continue

        m = re.match(r"vision_model\.encoder\.layers\.(\d+)\.(.*)", key)
        if m:
            block_idx = m.group(1)
            rest = m.group(2)
            for attn_name, mlx_name in [
                ("self_attn.q_proj.", "q_proj."),
                ("self_attn.k_proj.", "k_proj."),
                ("self_attn.v_proj.", "v_proj."),
                ("self_attn.out_proj.", "out_proj."),
            ]:
                if rest.startswith(attn_name):
                    param = rest.split(".")[-1]
                    remapped[f"encoder_layer_{block_idx}.self_attn.{mlx_name}{param}"] = value
                    break
            else:
                for old, new in [("layer_norm1.", "layer_norm1."), ("layer_norm2.", "layer_norm2.")]:
                    if rest.startswith(old):
                        param = rest.split(".")[-1]
                        remapped[f"encoder_layer_{block_idx}.{new}{param}"] = value
                        break
                else:
                    for old, new in [("mlp.fc1.", "mlp.fc1."), ("mlp.fc2.", "mlp.fc2.")]:
                        if rest.startswith(old):
                            param = rest.split(".")[-1]
                            remapped[f"encoder_layer_{block_idx}.{new}{param}"] = value
                            break

        m_head = re.match(r"vision_model\.head\.(.*)", key)
        if m_head:
            rest = m_head.group(1)
            if rest == "probe":
                remapped["head.probe"] = value.reshape(1, 1, -1)
            elif rest.startswith("attention."):
                param_name = rest[len("attention."):]
                if param_name in ("in_proj_weight", "in_proj_bias"):
                    _dim = value.shape[0] // 3
                    is_weight = param_name == "in_proj_weight"
                    slices = mx.split(value, 3, axis=0)
                    for name, idx in [("q_proj", 0), ("k_proj", 1), ("v_proj", 2)]:
                        suffix = "weight" if is_weight else "bias"
                        remapped[f"head.attention.{name}.{suffix}"] = slices[idx]
                else:
                    for attn_name, mlx_name in [
                        ("q_proj.", "q_proj."),
                        ("k_proj.", "k_proj."),
                        ("v_proj.", "v_proj."),
                        ("out_proj.", "out_proj."),
                    ]:
                        if param_name.startswith(attn_name):
                            param = param_name.split(".")[-1]
                            remapped[f"head.attention.{mlx_name}{param}"] = value
                            break
            elif rest.startswith("layernorm."):
                param = rest.split(".")[-1]
                remapped[f"head.layernorm.{param}"] = value
            elif rest.startswith("mlp."):
                mlp_rest = rest[len("mlp."):]
                for old, new in [("fc1.", "mlp.fc1."), ("fc2.", "mlp.fc2.")]:
                    if mlp_rest.startswith(old):
                        param = mlp_rest.split(".")[-1]
                        remapped[f"head.{new}{param}"] = value
                        break

    return remapped


def _remap_ip_adapter_weights(weights, num_tokens=128):
    proj_weights = {}
    attn_weights = {}

    for key, value in weights.items():
        if key.startswith("image_proj"):
            clean = key[len("image_proj."):]
            if clean == "proj.0.weight":
                proj_weights["linear1.weight"] = value
            elif clean == "proj.0.bias":
                proj_weights["linear1.bias"] = value
            elif clean == "proj.2.weight":
                proj_weights["linear2.weight"] = value
            elif clean == "proj.2.bias":
                proj_weights["linear2.bias"] = value
            elif clean.startswith("norm."):
                param = clean.split(".")[-1]
                proj_weights[f"norm.{param}"] = value
        elif key.startswith("ip_adapter"):
            clean = key[len("ip_adapter."):]
            attn_weights[clean] = value

    return proj_weights, attn_weights


def _preprocess_siglip_image(image_np):
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    if image_np.ndim == 4:
        image_np = image_np[0]
    if image_np.max() > 1.0:
        image_np = image_np.astype(np.float32) / 255.0
    else:
        image_np = image_np.astype(np.float32)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    import cv2
    resized = cv2.resize(image_np, (384, 384), interpolation=cv2.INTER_CUBIC)
    arr = (resized - mean) / std
    return mx.array(arr[np.newaxis])


def _load_ip_adapter_file(ip_path):
    """Load IP-Adapter weights from safetensors or PyTorch .bin file.

    Handles two weight formats:
    1. Safetensors with remapped keys (ip_adapter.double_blocks.N / single_blocks.N)
    2. PyTorch .bin with flat numeric keys (ip_adapter.N) — auto-remapped
    """
    if ip_path.is_file():
        ext = ip_path.suffix.lower()
        if ext in (".bin", ".pt", ".ckpt"):
            return _load_torch_ip_adapter(ip_path)
        try:
            return mx.load(str(ip_path))
        except Exception as e:
            logger.warning("Failed to load %s as safetensors: %s", ip_path, e)
            return None
    elif ip_path.is_dir():
        safetensors = sorted(glob.glob(str(ip_path / "*.safetensors")))
        if not safetensors:
            bins = sorted(glob.glob(str(ip_path / "*.bin")))
            if bins:
                return _load_torch_ip_adapter(Path(bins[0]))
            logger.warning("IP-Adapter weights not found at %s", ip_path)
            return None
        raw = {}
        for sf in safetensors:
            raw.update(mx.load(sf))
        return raw
    else:
        logger.warning("IP-Adapter path not found: %s", ip_path)
        return None


def _load_torch_ip_adapter(path):
    """Load PyTorch IP-Adapter .bin file, flatten and remap keys to MLX format."""
    try:
        import torch
    except ImportError:
        logger.error("Cannot load .bin IP-Adapter: torch not installed")
        return None

    logger.info("Loading PyTorch IP-Adapter from %s", path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)

    if not isinstance(ckpt, dict):
        logger.error("Unexpected IP-Adapter format in %s", path)
        return None

    flat = {}
    for top_key, subdict in ckpt.items():
        if isinstance(subdict, dict):
            for k, v in subdict.items():
                flat[f"{top_key}.{k}"] = v.detach().cpu().float().numpy()
        elif hasattr(subdict, "detach"):
            flat[top_key] = subdict.detach().cpu().float().numpy()

    remapped = {}
    for k, v in flat.items():
        if k.startswith("ip_adapter."):
            rest = k[len("ip_adapter."):]
            parts = rest.split(".", 1)
            try:
                idx = int(parts[0])
            except ValueError:
                remapped[k] = mx.array(v)
                continue
            suffix = parts[1]
            num_double = FluxIPAdapterPipeline._NUM_DOUBLE_BLOCKS
            if idx < num_double:
                new_key = f"ip_adapter.double_blocks.{idx}.{suffix}"
            else:
                new_key = f"ip_adapter.single_blocks.{idx - num_double}.{suffix}"
            remapped[new_key] = mx.array(v)
        else:
            remapped[k] = mx.array(v)

    logger.info("Loaded %d params from PyTorch IP-Adapter, remapped keys", len(remapped))
    return remapped


class FluxIPAdapterPipeline:
    def __init__(self, num_tokens=128, dtype=mx.float16):
        self.num_tokens = num_tokens
        self.dtype = dtype
        self.vision_encoder = None
        self.image_proj_model = None
        self.attn_processors = {}
        self._loaded = False

    @classmethod
    def from_pretrained(cls, siglip_dir, ip_ckpt_path, num_tokens=128, dtype=mx.float16):
        pipeline = cls(num_tokens=num_tokens, dtype=dtype)
        pipeline._load_siglip(siglip_dir, dtype)
        pipeline._load_ip_adapter(ip_ckpt_path, num_tokens, dtype)
        pipeline._loaded = True
        return pipeline

    def _load_siglip(self, siglip_dir, dtype):
        self.vision_encoder = SigLIPVisionEncoder()
        siglip_path = Path(siglip_dir)
        safetensors = sorted(glob.glob(str(siglip_path / "*.safetensors")))
        if not safetensors:
            logger.warning("SigLIP weights not found at %s, random init", siglip_dir)
            return
        raw = {}
        for sf in safetensors:
            raw.update(mx.load(sf))
        remapped = _remap_siglip_weights(raw)
        if remapped:
            self.vision_encoder.load_weights(list(remapped.items()))
            logger.info("SigLIP loaded %d params from %s", len(remapped), siglip_dir)
        else:
            logger.warning("SigLIP weight remapping returned empty from %s", siglip_dir)

    def _load_ip_adapter(self, ip_ckpt_path, num_tokens, dtype):
        self.image_proj_model = MLPProjModel(
            id_embeddings_dim=1152,
            cross_attention_dim=4096,
            num_tokens=num_tokens,
        )
        ip_path = Path(ip_ckpt_path)
        raw = _load_ip_adapter_file(ip_path)
        if raw is None:
            self._init_default_processors()
            return

        proj_weights, attn_weights = _remap_ip_adapter_weights(raw, num_tokens)
        if proj_weights:
            self.image_proj_model.load_weights(list(proj_weights.items()))
            logger.info("IP-Adapter projection loaded %d params", len(proj_weights))

        self._init_processors_from_weights(attn_weights, num_tokens, dtype)

    _NUM_DOUBLE_BLOCKS = 19
    _NUM_SINGLE_BLOCKS = 38

    def _init_default_processors(self):
        for i in range(self._NUM_DOUBLE_BLOCKS):
            name = f"double_blocks.{i}"
            self.attn_processors[name] = IPAFluxAttnProcessor(
                hidden_size=3072, cross_attention_dim=4096,
                num_tokens=self.num_tokens, scale=1.0,
            )
        for i in range(self._NUM_SINGLE_BLOCKS):
            name = f"single_blocks.{i}"
            self.attn_processors[name] = IPAFluxAttnProcessor(
                hidden_size=3072, cross_attention_dim=4096,
                num_tokens=self.num_tokens, scale=1.0,
            )
        logger.info("Initialized %d IP-Adapter attention processors (random weights)", len(self.attn_processors))

    def _init_processors_from_weights(self, attn_weights, num_tokens, dtype):
        double_keys = {k for k in attn_weights if k.startswith("double_blocks.")}
        single_keys = {k for k in attn_weights if k.startswith("single_blocks.")}
        num_double = max((int(k.split(".")[1]) for k in double_keys), default=-1) + 1
        num_single = max((int(k.split(".")[1]) for k in single_keys), default=-1) + 1
        if num_double <= 0:
            num_double = self._NUM_DOUBLE_BLOCKS
        if num_single <= 0:
            num_single = self._NUM_SINGLE_BLOCKS

        for i in range(num_double):
            name = f"double_blocks.{i}"
            proc = IPAFluxAttnProcessor(
                hidden_size=3072, cross_attention_dim=4096,
                num_tokens=num_tokens, scale=1.0,
            )
            k_key = f"double_blocks.{i}.to_k_ip.weight"
            v_key = f"double_blocks.{i}.to_v_ip.weight"
            w = {}
            if k_key in attn_weights:
                w["to_k_ip.weight"] = attn_weights[k_key]
            if v_key in attn_weights:
                w["to_v_ip.weight"] = attn_weights[v_key]
            if w:
                proc.load_weights(list(w.items()))
            self.attn_processors[name] = proc

        for i in range(num_single):
            name = f"single_blocks.{i}"
            proc = IPAFluxAttnProcessor(
                hidden_size=3072, cross_attention_dim=4096,
                num_tokens=num_tokens, scale=1.0,
            )
            k_key = f"single_blocks.{i}.to_k_ip.weight"
            v_key = f"single_blocks.{i}.to_v_ip.weight"
            w = {}
            if k_key in attn_weights:
                w["to_k_ip.weight"] = attn_weights[k_key]
            if v_key in attn_weights:
                w["to_v_ip.weight"] = attn_weights[v_key]
            if w:
                proc.load_weights(list(w.items()))
            self.attn_processors[name] = proc

        logger.info("Initialized %d IP-Adapter attention processors from weights", len(self.attn_processors))

    def get_image_embeds(self, image_np):
        if self.vision_encoder is None or self.image_proj_model is None:
            raise RuntimeError("IP-Adapter pipeline not loaded")

        pixel_values = _preprocess_siglip_image(image_np).astype(self.dtype)
        clip_embeds = self.vision_encoder(pixel_values)
        logger.debug("SigLIP pooler_output shape=%s", clip_embeds.shape)
        image_prompt_embeds = self.image_proj_model(clip_embeds)
        logger.debug("IP-Adapter image_embeds shape=%s", image_prompt_embeds.shape)
        return image_prompt_embeds

    def set_image_embeds(self, image_embeds):
        for proc in self.attn_processors.values():
            proc.set_image_embeds(image_embeds)

    def clear_image_embeds(self):
        for proc in self.attn_processors.values():
            proc.clear_image_embeds()

    def update_scale_and_range(self, weight, start_percent, end_percent, sigma_fn=None):
        for proc in self.attn_processors.values():
            proc.scale = weight
            if sigma_fn is not None and (start_percent > 0.0 or end_percent < 1.0):
                proc.timestep_range = (
                    sigma_fn(start_percent),
                    sigma_fn(end_percent),
                )
            else:
                proc.timestep_range = None

    def inject_into_transformer(self, transformer):
        """Install IP-Adapter attention processors into mflux Flux2 Transformer.

        Patches Flux2Transformer.__call__ at the class level to inject IP
        cross-attention after each double-stream block (on image hidden_states)
        and after each single-stream block (on the image portion of concatenated
        states).

        Args:
            transformer: Flux2Transformer instance
        """
        if getattr(self, "_patched_transformer_ref", None) is not None and self._patched_transformer_ref() is transformer:
            logger.info("IP-Adapter: already injected into this transformer")
            return

        self._patched_transformer_ref = weakref.ref(transformer)
        pipeline_ref = weakref.ref(self)

        cls = type(transformer)
        if hasattr(cls, "_ip_adapter_original_call"):
            self._original_transformer_call = cls._ip_adapter_original_call
        else:
            self._original_transformer_call = cls.__call__
            cls._ip_adapter_original_call = cls.__call__
        self._patched_class = cls

        num_double = len(transformer.transformer_blocks)
        num_single = len(transformer.single_transformer_blocks)

        _original_call = self._original_transformer_call

        def _patched_call(self_trans, hidden_states, encoder_hidden_states,
                          timestep, img_ids, txt_ids, guidance=None, kv_cache=None):
            _mx = mx
            from mflux.models.common.config.model_config import ModelConfig as _MC

            if not isinstance(timestep, _mx.array):
                timestep = _mx.array(timestep, dtype=hidden_states.dtype)
            if timestep.ndim == 0:
                timestep = _mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
            timestep = timestep.astype(hidden_states.dtype)
            _t_sigma = _mx.max(timestep)
            timestep_scale = _mx.where(_mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
            timestep = timestep * timestep_scale
            if guidance is not None:
                if not isinstance(guidance, _mx.array):
                    guidance = _mx.array(guidance, dtype=hidden_states.dtype)
                if guidance.ndim == 0:
                    guidance = _mx.full((hidden_states.shape[0],), guidance, dtype=hidden_states.dtype)
                guidance = guidance.astype(hidden_states.dtype)
                guidance_scale = _mx.where(_mx.max(guidance) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
                guidance = guidance * guidance_scale
            temb = self_trans.time_guidance_embed(timestep, guidance)
            temb = temb.astype(_MC.precision)
            ref_temb = None
            if kv_cache is not None and kv_cache.mode == "extract" and kv_cache.num_ref_tokens > 0:
                ref_temb = self_trans.time_guidance_embed(_mx.zeros_like(timestep), guidance)
                ref_temb = ref_temb.astype(_MC.precision)

            hidden_states = self_trans.x_embedder(hidden_states)
            encoder_hidden_states = self_trans.context_embedder(encoder_hidden_states)
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]

            image_rotary_emb = self_trans.pos_embed(img_ids)
            text_rotary_emb = self_trans.pos_embed(txt_ids)
            concat_rotary_emb = (
                _mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
                _mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
            )

            temb_mod_params_img = self_trans.double_stream_modulation_img(temb)
            temb_mod_params_txt = self_trans.double_stream_modulation_txt(temb)
            if ref_temb is not None:
                ref_temb_mod_params_img = self_trans.double_stream_modulation_img(ref_temb)
                temb_mod_params_img = tuple(
                    self_trans._blend_trailing_ref_mod_params(
                        mod_params=mod_params,
                        ref_mod_params=ref_mod_params,
                        seq_len=hidden_states.shape[1],
                        num_ref_tokens=kv_cache.num_ref_tokens,
                    )
                    for mod_params, ref_mod_params in zip(temb_mod_params_img, ref_temb_mod_params_img, strict=True)
                )

            txt_len = encoder_hidden_states.shape[1]

            for idx, block in enumerate(self_trans.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_params_img=temb_mod_params_img,
                    temb_mod_params_txt=temb_mod_params_txt,
                    image_rotary_emb=concat_rotary_emb,
                    kv_cache=kv_cache,
                    kv_cache_layer_idx=idx,
                )
                name = f"double_blocks.{idx}"
                _pl = pipeline_ref()
                proc = _pl.attn_processors.get(name) if _pl is not None else None
                if proc is not None and proc._image_embeds is not None:
                    ip_out = proc(hidden_states, t_sigma=_t_sigma)
                    if ip_out is not None:
                        hidden_states = hidden_states + ip_out

            hidden_states = _mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

            temb_mod_params_single = self_trans.single_stream_modulation(temb)[0]
            if ref_temb is not None:
                ref_temb_mod_params_single = self_trans.single_stream_modulation(ref_temb)[0]
                temb_mod_params_single = self_trans._blend_trailing_ref_mod_params(
                    mod_params=temb_mod_params_single,
                    ref_mod_params=ref_temb_mod_params_single,
                    seq_len=hidden_states.shape[1],
                    num_ref_tokens=kv_cache.num_ref_tokens,
                )
            for idx, block in enumerate(self_trans.single_transformer_blocks):
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb_mod_params=temb_mod_params_single,
                    image_rotary_emb=concat_rotary_emb,
                    kv_cache=kv_cache,
                    kv_cache_layer_idx=idx,
                )
                name = f"single_blocks.{idx}"
                _pl = pipeline_ref()
                proc = _pl.attn_processors.get(name) if _pl is not None else None
                if proc is not None and proc._image_embeds is not None:
                    img_states = hidden_states[:, txt_len:]
                    ip_out = proc(img_states, t_sigma=_t_sigma)
                    if ip_out is not None:
                        img_states = img_states + ip_out
                        hidden_states = _mx.concatenate(
                            [hidden_states[:, :txt_len], img_states], axis=1,
                        )

            hidden_states = hidden_states[:, txt_len:]
            if kv_cache is not None and kv_cache.mode == "extract" and kv_cache.num_ref_tokens > 0:
                hidden_states = hidden_states[:, :-kv_cache.num_ref_tokens, ...]
            hidden_states = self_trans.norm_out(hidden_states, temb)
            hidden_states = self_trans.proj_out(hidden_states)
            return hidden_states

        cls.__call__ = _patched_call
        if not hasattr(cls, "_ip_adapter_active_count"):
            cls._ip_adapter_active_count = 0
        cls._ip_adapter_active_count += 1
        logger.info(
            "IP-Adapter: injected into Flux2 transformer (%d double + %d single blocks)",
            num_double, num_single,
        )

    def remove_from_transformer(self, transformer=None):
        if hasattr(self, "_patched_class") and hasattr(self, "_original_transformer_call"):
            self._patched_class.__call__ = self._patched_class._ip_adapter_original_call
            if not hasattr(self._patched_class, "_ip_adapter_active_count"):
                self._patched_class._ip_adapter_active_count = 0
            self._patched_class._ip_adapter_active_count = max(0, self._patched_class._ip_adapter_active_count - 1)
            if self._patched_class._ip_adapter_active_count <= 0:
                del self._patched_class._ip_adapter_original_call
                if hasattr(self._patched_class, "_ip_adapter_active_count"):
                    del self._patched_class._ip_adapter_active_count
            del self._original_transformer_call
            del self._patched_class
        self._patched_transformer_ref = None
        logger.info("IP-Adapter: removed from Flux2 transformer (restored __call__)")


def _get_ip_adapter_models():
    if _IP_ADAPTER_MODEL_DIRS:
        return _IP_ADAPTER_MODEL_DIRS
    try:
        import folder_paths
        if "ipadapter-flux" in folder_paths.folder_names_and_paths:
            dirs, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
            for d in dirs:
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if f.endswith((".safetensors", ".pt", ".bin")):
                            _IP_ADAPTER_MODEL_DIRS.append(f)
    except Exception:
        pass
    if not _IP_ADAPTER_MODEL_DIRS:
        _IP_ADAPTER_MODEL_DIRS.append("ip_adapter_flux.safetensors")
    return _IP_ADAPTER_MODEL_DIRS


def _resolve_ip_adapter_path(filename):
    try:
        import folder_paths
        if "ipadapter-flux" in folder_paths.folder_names_and_paths:
            dirs, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
            for d in dirs:
                candidate = os.path.join(d, filename)
                if os.path.isfile(candidate):
                    return candidate
    except Exception:
        pass
    return filename


def _resolve_siglip_path(model_name="siglip-so400m-patch14-384"):
    try:
        import folder_paths
        clip_dir = os.path.join(folder_paths.models_dir, "clip_vision", model_name)
        if os.path.isdir(clip_dir):
            return clip_dir
    except Exception:
        pass
    hf_cache = os.path.expanduser(f"~/.cache/huggingface/hub/models--google--{model_name}")
    if os.path.isdir(hf_cache):
        return hf_cache
    fusion_cache = os.path.expanduser(f"~/.cache/fusion-mlx/siglip/{model_name}")
    if os.path.isdir(fusion_cache):
        return fusion_cache
    return model_name


def _image_to_np(image):
    arr = np.array(image, copy=False)
    if arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 4:
        arr = arr[0].copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3].copy()
    return arr


class FusionIPAdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter": (_get_ip_adapter_models(),),
                "siglip_model": (["siglip-so400m-patch14-384"], {"default": "siglip-so400m-patch14-384"}),
                "num_tokens": ("INT", {"default": 128, "min": 1, "max": 256}),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("FUSION_IP_ADAPTER_MODEL",)
    RETURN_NAMES = ("ip_adapter_model",)
    FUNCTION = "load_ip_adapter"
    CATEGORY = "Fusion-MLX/IP-Adapter"

    def load_ip_adapter(self, ipadapter, siglip_model="siglip-so400m-patch14-384",
                        num_tokens=128, dtype="float16"):
        from core.lifecycle import FusionMemoryGuardian
        FusionMemoryGuardian.maybe_purge()

        ip_ckpt_path = _resolve_ip_adapter_path(ipadapter)
        siglip_dir = _resolve_siglip_path(siglip_model)
        dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        mx_dtype = dtype_map.get(dtype, mx.float16)

        logger.info("FusionIPAdapterLoader: ipadapter=%s siglip=%s tokens=%d dtype=%s",
                     ipadapter, siglip_model, num_tokens, dtype)

        try:
            pipeline = FluxIPAdapterPipeline.from_pretrained(
                siglip_dir, ip_ckpt_path, num_tokens, mx_dtype,
            )
        except Exception as e:
            logger.error("FusionIPAdapterLoader: failed: %s", e)
            raise

        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionIPAdapterLoader: loaded IP-Adapter pipeline")
        return (pipeline,)


class FusionIPAdapterApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_adapter_model": ("FUSION_IP_ADAPTER_MODEL",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FUSION_IP_ADAPTER_EMBED",)
    RETURN_NAMES = ("ip_adapter_embed",)
    FUNCTION = "apply_ip_adapter"
    CATEGORY = "Fusion-MLX/IP-Adapter"

    def apply_ip_adapter(self, ip_adapter_model, image, weight=1.0,
                         start_percent=0.0, end_percent=1.0):
        image_np = _image_to_np(image)
        logger.info("FusionIPAdapterApply: image_shape=%s weight=%.2f range=[%.3f,%.3f]",
                     image_np.shape, weight, start_percent, end_percent)

        try:
            image_embeds = ip_adapter_model.get_image_embeds(image_np)
        except Exception as e:
            logger.error("FusionIPAdapterApply: image encoding failed: %s", e)
            raise

        sigma_fn = lambda percent: 1.0 - percent
        ip_adapter_model.update_scale_and_range(weight, start_percent, end_percent, sigma_fn=sigma_fn)

        result = {
            "image_embeds": image_embeds,
            "pipeline": ip_adapter_model,
            "weight": weight,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        logger.info("FusionIPAdapterApply: embeds_shape=%s", image_embeds.shape)
        return (result,)


class FusionIPAdapterInject:
    """Inject IP-Adapter image embeddings into a Fusion-MLX image pipeline.

    Workflow: FusionIPAdapterLoader -> FusionIPAdapterApply -> FusionIPAdapterInject
    The inject node patches the Flux DiT transformer to add IP cross-attention
    during denoising, then generates the image, then removes the patch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "ip_adapter_embed": ("FUSION_IP_ADAPTER_EMBED",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/IP-Adapter"

    def generate(self, pipeline, ip_adapter_embed, prompt, negative_prompt="",
                 width=1024, height=1024, steps=20, cfg=4.0, seed=42):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()

        ip_pipeline = ip_adapter_embed["pipeline"]
        image_embeds = ip_adapter_embed["image_embeds"]

        logger.info(
            "FusionIPAdapterInject: prompt_len=%d embeds_shape=%s size=%dx%d "
            "steps=%d cfg=%.1f seed=%d",
            len(prompt), image_embeds.shape, width, height, steps, cfg, seed,
        )

        ip_pipeline.set_image_embeds(image_embeds)

        try:
            result_raw = core.async_utils.run_async(
                self._generate_with_ip(
                    pipeline, ip_pipeline, prompt, negative_prompt,
                    width, height, steps, cfg, seed,
                ),
                timeout=600,
            )
        except Exception as e:
            logger.error("FusionIPAdapterInject: failed: %s", e)
            raise
        finally:
            try:
                ip_pipeline.clear_image_embeds()
                ip_pipeline.remove_from_transformer()
            except Exception as cleanup_err:
                logger.warning("FusionIPAdapterInject: cleanup error: %s", cleanup_err)

        raw_arr = result_raw[0]
        if isinstance(raw_arr, np.ndarray):
            image_np = raw_arr.astype(np.float32) / 255.0
        else:
            img = Image.open(io.BytesIO(raw_arr))
            image_np = np.array(img).astype(np.float32) / 255.0
        if image_np.ndim == 3:
            image_np = image_np[np.newaxis, ...]
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :, :3]

        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionIPAdapterInject: output shape=%s", image_np.shape)
        return (image_np,)

    async def _generate_with_ip(self, pipeline, ip_pipeline, prompt,
                                 negative_prompt, width, height, steps, cfg, seed):
        await pipeline.ensure_started()

        engine = pipeline._engine
        if engine._flux is None:
            raise RuntimeError("ImageGen engine not started")

        transformer = engine._flux.transformer
        if transformer is None:
            raise RuntimeError("Flux transformer not loaded")

        ip_pipeline.inject_into_transformer(transformer)

        neg = negative_prompt if negative_prompt else None
        result_raw = await engine.generate(
            prompt=prompt, width=width, height=height,
            steps=steps, seed=seed, guidance=cfg, n_images=1,
            negative_prompt=neg, output_format="raw",
        )

        return result_raw
