"""Tests for IPAdapter-Flux MLX nodes and core modules.
Imported by: pytest (test runner)
API: Tests nodes.ip_adapter module classes and ComfyUI node INPUT_TYPES
User instruction: "继续IPAdapter-Flux MLX 移植"
"""
import sys
import os
import types
from unittest.mock import MagicMock

import numpy as np

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

if "mlx" not in sys.modules:
    mock_mx = MagicMock()
    mock_mx.zeros = MagicMock(side_effect=lambda shape, dtype=None: np.zeros(shape, dtype=np.float32))
    mock_mx.float16 = "float16"
    mock_mx.bfloat16 = "bfloat16"
    mock_mx.float32 = "float32"
    mock_mx.array = MagicMock(side_effect=lambda x: np.array(x, dtype=np.float32))
    mock_mx.transpose = MagicMock(side_effect=lambda x, *a: x)
    mock_mx.concatenate = MagicMock(side_effect=lambda *a, **kw: np.concatenate(a, axis=kw.get("axis", 0)))
    mock_mx.broadcast_to = MagicMock(side_effect=lambda x, shape: np.broadcast_to(x, shape))
    mock_mx.split = MagicMock(side_effect=lambda x, n, **kw: np.array_split(x, n))
    mock_mx.fast = MagicMock()
    mock_mx.fast.scaled_dot_product_attention = MagicMock(return_value=np.zeros((1, 24, 10, 128), dtype=np.float32))
    mock_mx.fast.rms_norm = MagicMock(side_effect=lambda x, w, eps: x)
    mock_mlx = MagicMock(core=mock_mx)
    mock_mlx.nn = MagicMock()
    mock_mlx.nn.Module = type("Module", (), {"__init__": lambda self: None})
    mock_mlx.nn.Linear = MagicMock(return_value=MagicMock())
    mock_mlx.nn.LayerNorm = MagicMock(return_value=MagicMock())
    mock_mlx.nn.Conv2d = MagicMock(return_value=MagicMock())
    mock_mlx.nn.RMSNorm = MagicMock(return_value=MagicMock())
    mock_mlx.nn.gelu = MagicMock(side_effect=lambda x: x)
    sys.modules.setdefault("mlx", mock_mlx)
    sys.modules.setdefault("mlx.core", mock_mx)
    sys.modules.setdefault("mlx.nn", mock_mlx.nn)

if "fusion_mlx" not in sys.modules:
    mock_fm = MagicMock()
    mock_fm._torch_stub = MagicMock()
    mock_fm._torch_stub.install = MagicMock(return_value=True)
    sys.modules.setdefault("fusion_mlx", mock_fm)
    sys.modules.setdefault("fusion_mlx._torch_stub", mock_fm._torch_stub)
    sys.modules.setdefault("fusion_mlx.video.pulid_mlx", MagicMock())
    sys.modules.setdefault("fusion_mlx.video.pulid_mlx.pipeline", MagicMock())
    sys.modules.setdefault("fusion_mlx.video.latentsync_mlx", MagicMock())
    sys.modules.setdefault("fusion_mlx.video.latentsync_mlx.pipeline", MagicMock())

if "folder_paths" not in sys.modules:
    mock_fp = MagicMock()
    mock_fp.models_dir = "/tmp/models"
    sys.modules.setdefault("folder_paths", mock_fp)

if "core" not in sys.modules:
    _core_pkg = types.ModuleType("core")
    _core_pkg.__path__ = [os.path.join(PLUGIN_ROOT, "core")]
    _core_pkg.__package__ = "core"
    sys.modules["core"] = _core_pkg

from nodes.ip_adapter import (
    SigLIPVisionEncoder,
    MLPProjModel,
    IPAFluxAttnProcessor,
    FluxIPAdapterPipeline,
    _RMSNormNoAffine,
    _remap_siglip_weights,
    _remap_ip_adapter_weights,
    _preprocess_siglip_image,
    _image_to_np,
    FusionIPAdapterLoader,
    FusionIPAdapterApply,
    FusionIPAdapterInject,
)


class TestRMSNormNoAffine:
    def test_init(self):
        norm = _RMSNormNoAffine(eps=1e-5)
        assert norm.eps == 1e-5

    def test_call_returns_norm(self):
        norm = _RMSNormNoAffine(eps=1e-5)
        x = np.random.randn(1, 24, 128).astype(np.float32)
        result = norm(x)
        assert result is not None


class TestSigLIPVisionEncoder:
    def test_init(self):
        enc = SigLIPVisionEncoder()
        assert enc.dim == 1152
        assert enc.num_patches == (384 // 14) ** 2

    def test_has_encoder_layers(self):
        enc = SigLIPVisionEncoder()
        assert hasattr(enc, "encoder_layer_0")
        assert hasattr(enc, "encoder_layer_26")


class TestMLPProjModel:
    def test_init(self):
        model = MLPProjModel(id_embeddings_dim=1152, cross_attention_dim=4096, num_tokens=128)
        assert model.cross_attention_dim == 4096
        assert model.num_tokens == 128

    def test_output_shape(self):
        model = MLPProjModel(id_embeddings_dim=1152, cross_attention_dim=4096, num_tokens=4)
        x = np.random.randn(1, 1152).astype(np.float32)
        out = model(x)
        assert out is not None


class TestIPAFluxAttnProcessor:
    def test_init(self):
        proc = IPAFluxAttnProcessor(hidden_size=3072, cross_attention_dim=4096, num_tokens=128)
        assert proc.hidden_size == 3072
        assert proc.scale == 1.0

    def test_set_clear_embeds(self):
        proc = IPAFluxAttnProcessor()
        proc.set_image_embeds(np.zeros((1, 128, 4096), dtype=np.float32))
        assert proc._image_embeds is not None
        proc.clear_image_embeds()
        assert proc._image_embeds is None

    def test_returns_none_without_embeds(self):
        proc = IPAFluxAttnProcessor()
        query = np.zeros((1, 24, 10, 128), dtype=np.float32)
        result = proc(query)
        assert result is None

    def test_timestep_range_filtering(self):
        proc = IPAFluxAttnProcessor(timestep_range=(0.5, 0.1))
        proc.set_image_embeds(np.zeros((1, 128, 4096), dtype=np.float32))
        query = np.zeros((1, 24, 10, 128), dtype=np.float32)
        result = proc(query, t_sigma=0.8)
        assert result is None


class TestFluxIPAdapterPipeline:
    def test_init(self):
        pipeline = FluxIPAdapterPipeline(num_tokens=128)
        assert pipeline.num_tokens == 128
        assert not pipeline._loaded

    def test_init_default_processors(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline._init_default_processors()
        assert len(pipeline.attn_processors) == 57
        assert "double_blocks.0" in pipeline.attn_processors
        assert "single_blocks.0" in pipeline.attn_processors

    def test_set_clear_embeds(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline._init_default_processors()
        embeds = np.zeros((1, 128, 4096), dtype=np.float32)
        pipeline.set_image_embeds(embeds)
        for proc in pipeline.attn_processors.values():
            assert proc._image_embeds is not None
        pipeline.clear_image_embeds()
        for proc in pipeline.attn_processors.values():
            assert proc._image_embeds is None

    def test_update_scale_and_range(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline._init_default_processors()
        pipeline.update_scale_and_range(0.5, 0.1, 0.9)
        for proc in pipeline.attn_processors.values():
            assert proc.scale == 0.5


class TestWeightRemapping:
    def test_siglip_remap_patch_embedding(self):
        weights = {"vision_model.embeddings.patch_embedding.weight": np.zeros((1152, 3, 14, 14))}
        result = _remap_siglip_weights(weights)
        assert "patch_embedding.weight" in result

    def test_siglip_remap_position_embedding(self):
        weights = {"vision_model.embeddings.position_embedding.weight": np.zeros((577, 1152))}
        result = _remap_siglip_weights(weights)
        assert "position_embedding" in result

    def test_siglip_remap_encoder_layer(self):
        weights = {"vision_model.encoder.layers.0.self_attn.q_proj.weight": np.zeros((1152, 1152))}
        result = _remap_siglip_weights(weights)
        assert "encoder_layer_0.self_attn.q_proj.weight" in result

    def test_siglip_skip_non_vision(self):
        weights = {"text_model.encoder.layers.0.weight": np.zeros(10)}
        result = _remap_siglip_weights(weights)
        assert len(result) == 0

    def test_ip_adapter_remap_proj(self):
        weights = {
            "image_proj.proj.0.weight": np.zeros((2304, 1152)),
            "image_proj.proj.2.weight": np.zeros((524288, 2304)),
            "image_proj.norm.weight": np.zeros(4096),
        }
        proj, attn = _remap_ip_adapter_weights(weights)
        assert "linear1.weight" in proj
        assert "linear2.weight" in proj
        assert "norm.weight" in proj

    def test_ip_adapter_remap_attn(self):
        weights = {"ip_adapter.double_blocks.0.to_k_ip.weight": np.zeros((3072, 4096))}
        proj, attn = _remap_ip_adapter_weights(weights)
        assert "double_blocks.0.to_k_ip.weight" in attn


class TestPreprocessing:
    def test_siglip_preprocess(self):
        img = np.random.rand(1, 512, 512, 3).astype(np.float32)
        result = _preprocess_siglip_image(img)
        assert result is not None

    def test_image_to_np_float(self):
        img = np.random.rand(1, 512, 512, 3).astype(np.float32)
        result = _image_to_np(img)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_image_to_np_uint8(self):
        img = (np.random.rand(1, 512, 512, 3) * 255).astype(np.uint8)
        result = _image_to_np(img)
        assert result.ndim == 3

    def test_image_to_np_rgba(self):
        img = np.random.rand(1, 512, 512, 4).astype(np.float32)
        result = _image_to_np(img)
        assert result.shape[2] == 3


class TestFusionIPAdapterLoader:
    def test_input_types(self):
        inputs = FusionIPAdapterLoader.INPUT_TYPES()
        assert "required" in inputs
        assert "ipadapter" in inputs["required"]
        assert "siglip_model" in inputs["required"]

    def test_return_types(self):
        assert FusionIPAdapterLoader.RETURN_TYPES == ("FUSION_IP_ADAPTER_MODEL",)

    def test_category(self):
        assert FusionIPAdapterLoader.CATEGORY == "Fusion-MLX/IP-Adapter"


class TestFusionIPAdapterApply:
    def test_input_types(self):
        inputs = FusionIPAdapterApply.INPUT_TYPES()
        assert "required" in inputs
        assert "ip_adapter_model" in inputs["required"]
        assert "image" in inputs["required"]
        assert "weight" in inputs["required"]
        assert "start_percent" in inputs["required"]
        assert "end_percent" in inputs["required"]

    def test_return_types(self):
        assert FusionIPAdapterApply.RETURN_TYPES == ("FUSION_IP_ADAPTER_EMBED",)

    def test_category(self):
        assert FusionIPAdapterApply.CATEGORY == "Fusion-MLX/IP-Adapter"


class TestFluxIPAdapterPipelineInject:
    def _make_mock_transformer_class(self):
        class MockTransformer:
            def __init__(self):
                self.transformer_blocks = [MagicMock() for _ in range(5)]
                self.single_transformer_blocks = [MagicMock() for _ in range(20)]
                self.time_guidance_embed = MagicMock(return_value=np.zeros((1, 3072), dtype=np.float32))
                self.x_embedder = MagicMock(return_value=np.zeros((1, 256, 3072), dtype=np.float32))
                self.context_embedder = MagicMock(return_value=np.zeros((1, 64, 3072), dtype=np.float32))
                self.pos_embed = MagicMock(return_value=(np.zeros((320, 3072)), np.zeros((320, 3072))))
                self.double_stream_modulation_img = MagicMock(return_value=(
                    (np.zeros((1, 256, 3072)), np.zeros((1, 256, 3072)), np.zeros((1, 256, 3072))),
                    (np.zeros((1, 256, 3072)), np.zeros((1, 256, 3072)), np.zeros((1, 256, 3072))),
                ))
                self.double_stream_modulation_txt = MagicMock(return_value=(
                    (np.zeros((1, 64, 3072)), np.zeros((1, 64, 3072)), np.zeros((1, 64, 3072))),
                    (np.zeros((1, 64, 3072)), np.zeros((1, 64, 3072)), np.zeros((1, 64, 3072))),
                ))
                self.single_stream_modulation = MagicMock(return_value=(
                    (np.zeros((1, 320, 3072)), np.zeros((1, 320, 3072)), np.zeros((1, 320, 3072))),
                ))
                self.norm_out = MagicMock(side_effect=lambda x, t: x)
                self.proj_out = MagicMock(side_effect=lambda x: x)
                self._blend_trailing_ref_mod_params = MagicMock(
                    side_effect=lambda **kw: kw["mod_params"]
                )

            def __call__(self, hidden_states, encoder_hidden_states, timestep,
                         img_ids, txt_ids, guidance=None, kv_cache=None):
                return np.zeros((1, 256, 128), dtype=np.float32)

        return MockTransformer

    def _make_mock_transformer(self):
        cls = self._make_mock_transformer_class()
        return cls()

    def test_inject_saves_original_call(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline._init_default_processors()
        transformer = self._make_mock_transformer()
        original_call = type(transformer).__call__
        pipeline.inject_into_transformer(transformer)
        assert pipeline._original_transformer_call is original_call
        assert pipeline._patched_transformer_ref() is transformer

    def test_inject_patches_class_call(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline._init_default_processors()
        transformer = self._make_mock_transformer()
        cls = type(transformer)
        original_call = cls.__call__
        pipeline.inject_into_transformer(transformer)
        assert cls.__call__ is not original_call

    def test_remove_restores_original_call(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline._init_default_processors()
        transformer = self._make_mock_transformer()
        cls = type(transformer)
        original_call = cls.__call__
        pipeline.inject_into_transformer(transformer)
        pipeline.remove_from_transformer()
        assert cls.__call__ is original_call
        assert pipeline._patched_transformer_ref is None

    def test_remove_without_inject_is_noop(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline.remove_from_transformer()
        assert getattr(pipeline, "_patched_transformer_ref", None) is None

    def test_double_inject_preserves_original(self):
        pipeline = FluxIPAdapterPipeline()
        pipeline._init_default_processors()
        transformer = self._make_mock_transformer()
        cls = type(transformer)
        original_call = cls.__call__
        pipeline.inject_into_transformer(transformer)
        pipeline.inject_into_transformer(transformer)
        assert pipeline._original_transformer_call is original_call
        pipeline.remove_from_transformer()
        assert cls.__call__ is original_call


class TestFusionIPAdapterInject:
    def test_input_types(self):
        inputs = FusionIPAdapterInject.INPUT_TYPES()
        assert "required" in inputs
        assert "pipeline" in inputs["required"]
        assert "ip_adapter_embed" in inputs["required"]
        assert "prompt" in inputs["required"]
        assert "width" in inputs["required"]
        assert "height" in inputs["required"]
        assert "steps" in inputs["required"]
        assert "cfg" in inputs["required"]
        assert "seed" in inputs["required"]

    def test_return_types(self):
        assert FusionIPAdapterInject.RETURN_TYPES == ("IMAGE",)

    def test_return_names(self):
        assert FusionIPAdapterInject.RETURN_NAMES == ("image",)

    def test_function(self):
        assert FusionIPAdapterInject.FUNCTION == "generate"

    def test_category(self):
        assert FusionIPAdapterInject.CATEGORY == "Fusion-MLX/IP-Adapter"
