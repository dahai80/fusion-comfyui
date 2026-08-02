import os
import numpy as np
from unittest.mock import MagicMock, patch


class TestModelSamplingSD3:
    def test_input_types(self):
        from nodes.passthrough import ModelSamplingSD3
        inputs = ModelSamplingSD3.INPUT_TYPES()
        assert "model" in inputs["required"]
        assert "shift" in inputs["required"]

    def test_patch_returns_model(self):
        from nodes.passthrough import ModelSamplingSD3
        node = ModelSamplingSD3()
        mock_model = MagicMock()
        result = node.patch(mock_model, shift=3.0)
        assert result == (mock_model,)


class TestModelSamplingContinuousEDM:
    def test_input_types(self):
        from nodes.passthrough import ModelSamplingContinuousEDM
        inputs = ModelSamplingContinuousEDM.INPUT_TYPES()
        assert "model" in inputs["required"]
        assert "sampling" in inputs["required"]

    def test_patch_returns_model(self):
        from nodes.passthrough import ModelSamplingContinuousEDM
        node = ModelSamplingContinuousEDM()
        mock_model = MagicMock()
        result = node.patch(mock_model, "v_prediction", 120.0, 0.002)
        assert result == (mock_model,)


class TestModelSamplingFlux:
    def test_input_types(self):
        from nodes.passthrough import ModelSamplingFlux
        inputs = ModelSamplingFlux.INPUT_TYPES()
        assert "model" in inputs["required"]

    def test_patch_returns_model(self):
        from nodes.passthrough import ModelSamplingFlux
        node = ModelSamplingFlux()
        mock_model = MagicMock()
        result = node.patch(mock_model, 1024, 1024, 1.15, 0.5, 1.0)
        assert result == (mock_model,)


class TestBasicGuider:
    def test_input_types(self):
        from nodes.passthrough import BasicGuider
        inputs = BasicGuider.INPUT_TYPES()
        assert "model" in inputs["required"]
        assert "conditioning" in inputs["required"]

    def test_get_guider(self):
        from nodes.passthrough import BasicGuider
        node = BasicGuider()
        mock_model = MagicMock()
        mock_cond = MagicMock()
        result = node.get_guider(mock_model, mock_cond)
        assert len(result) == 1
        assert result[0]["model"] is mock_model
        assert result[0]["conditioning"] is mock_cond


class TestBasicScheduler:
    def test_input_types(self):
        from nodes.passthrough import BasicScheduler
        inputs = BasicScheduler.INPUT_TYPES()
        assert "model" in inputs["required"]
        assert "scheduler" in inputs["required"]
        assert "steps" in inputs["required"]

    def test_get_sigmas(self):
        from nodes.passthrough import BasicScheduler
        node = BasicScheduler()
        mock_model = MagicMock()
        result = node.get_sigmas(mock_model, "normal", 20, 1.0)
        assert len(result) == 1
        sigmas = result[0]
        assert len(sigmas) == 21
        assert sigmas[0] == 1.0
        assert abs(sigmas[-1]) < 1e-6

    def test_get_sigmas_partial_denoise(self):
        from nodes.passthrough import BasicScheduler
        node = BasicScheduler()
        result = node.get_sigmas(MagicMock(), "normal", 10, 0.5)
        sigmas = result[0]
        assert len(sigmas) == 11


class TestKSamplerSelect:
    def test_input_types(self):
        from nodes.passthrough import KSamplerSelect
        inputs = KSamplerSelect.INPUT_TYPES()
        assert "sampler_name" in inputs["required"]

    def test_get_sampler(self):
        from nodes.passthrough import KSamplerSelect
        node = KSamplerSelect()
        result = node.get_sampler("euler")
        assert result[0]["sampler_name"] == "euler"


class TestRandomNoise:
    def test_input_types(self):
        from nodes.passthrough import RandomNoise
        inputs = RandomNoise.INPUT_TYPES()
        assert "noise_seed" in inputs["required"]

    def test_get_noise(self):
        from nodes.passthrough import RandomNoise
        node = RandomNoise()
        result = node.get_noise(12345)
        assert result[0]["noise_seed"] == 12345


class TestFluxGuidance:
    def test_input_types(self):
        from nodes.passthrough import FluxGuidance
        inputs = FluxGuidance.INPUT_TYPES()
        assert "conditioning" in inputs["required"]
        assert "guidance" in inputs["required"]

    def test_append_dict(self):
        from nodes.passthrough import FluxGuidance
        node = FluxGuidance()
        cond = {"prompt": "test"}
        result = node.append(cond, 3.5)
        assert result[0]["guidance"] == 3.5
        assert result[0]["prompt"] == "test"

    def test_append_non_dict(self):
        from nodes.passthrough import FluxGuidance
        node = FluxGuidance()
        cond = "some_string"
        result = node.append(cond, 3.5)
        # Non-dict should be returned as-is in tuple
        assert result == (cond,)


class TestCLIPVisionLoader:
    def test_input_types(self):
        from nodes.passthrough import CLIPVisionLoader
        inputs = CLIPVisionLoader.INPUT_TYPES()
        assert "clip_name" in inputs["required"]

    def test_load_clip(self):
        from nodes.passthrough import CLIPVisionLoader
        node = CLIPVisionLoader()
        result = node.load_clip("clip_vision_h.safetensors")
        assert result[0]["clip_name"] == "clip_vision_h.safetensors"


class TestCLIPVisionEncode:
    def test_input_types(self):
        from nodes.passthrough import CLIPVisionEncode
        inputs = CLIPVisionEncode.INPUT_TYPES()
        assert "clip_vision" in inputs["required"]
        assert "image" in inputs["required"]

    def test_encode(self):
        from nodes.passthrough import CLIPVisionEncode
        node = CLIPVisionEncode()
        mock_image = MagicMock()
        result = node.encode({"clip_name": "test"}, mock_image)
        assert result[0]["image"] is mock_image


class TestLTXVConditioning:
    def test_input_types(self):
        from nodes.passthrough import LTXVConditioning
        inputs = LTXVConditioning.INPUT_TYPES()
        assert "positive" in inputs["required"]
        assert "negative" in inputs["required"]
        assert "frame_rate" in inputs["required"]

    def test_append_dict(self):
        from nodes.passthrough import LTXVConditioning
        node = LTXVConditioning()
        pos = {"prompt": "test"}
        neg = {"prompt": ""}
        result = node.append(pos, neg, 25.0)
        assert result[0]["frame_rate"] == 25.0
        assert result[1]["frame_rate"] == 25.0

    def test_append_non_dict(self):
        from nodes.passthrough import LTXVConditioning
        node = LTXVConditioning()
        pos = "positive_str"
        neg = "negative_str"
        result = node.append(pos, neg, 25.0)
        assert result == (pos, neg)


class TestLTXVScheduler:
    def test_input_types(self):
        from nodes.passthrough import LTXVScheduler
        inputs = LTXVScheduler.INPUT_TYPES()
        assert "steps" in inputs["required"]
        assert "stretch" in inputs["required"]

    def test_get_sigmas_basic(self):
        from nodes.passthrough import LTXVScheduler
        node = LTXVScheduler()
        result = node.get_sigmas(20, 2.05, 0.95, "enable", 0.1)
        sigmas = result[0]
        assert len(sigmas) == 21
        assert sigmas[0] > 0

    def test_get_sigmas_with_latent(self):
        from nodes.passthrough import LTXVScheduler
        node = LTXVScheduler()
        latent = {"samples": np.zeros((1, 128, 13, 16, 24))}
        result = node.get_sigmas(20, 2.05, 0.95, "enable", 0.1, latent=latent)
        sigmas = result[0]
        assert len(sigmas) == 21

    def test_get_sigmas_stretch_disabled(self):
        from nodes.passthrough import LTXVScheduler
        node = LTXVScheduler()
        result = node.get_sigmas(20, 2.05, 0.95, "disable", 0.1)
        sigmas = result[0]
        assert len(sigmas) == 21

    def test_get_sigmas_bool_stretch(self):
        from nodes.passthrough import LTXVScheduler
        node = LTXVScheduler()
        result = node.get_sigmas(20, 2.05, 0.95, True, 0.1)
        sigmas = result[0]
        assert len(sigmas) == 21

    def test_get_sigmas_no_latent(self):
        from nodes.passthrough import LTXVScheduler
        node = LTXVScheduler()
        result = node.get_sigmas(20, 2.05, 0.95, "enable", 0.1, latent=None)
        sigmas = result[0]
        assert len(sigmas) == 21

    def test_get_sigmas_latent_no_samples(self):
        from nodes.passthrough import LTXVScheduler
        node = LTXVScheduler()
        latent = {}
        result = node.get_sigmas(20, 2.05, 0.95, "enable", 0.1, latent=latent)
        sigmas = result[0]
        assert len(sigmas) == 21


class TestCosmosImageToVideoLatent:
    def test_input_types(self):
        from nodes.passthrough import CosmosImageToVideoLatent
        inputs = CosmosImageToVideoLatent.INPUT_TYPES()
        assert "vae" in inputs["required"]

    def test_generate(self):
        from nodes.passthrough import CosmosImageToVideoLatent
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 16, 16, 88, 160)))
            mock_mx.float32 = np.float32
            node = CosmosImageToVideoLatent()
            result = node.generate(MagicMock(), 1280, 704, 121, 1)
            assert "samples" in result[0]
            assert result[0]["num_frames"] == 121


class TestCosmosPredict2ImageToVideoLatent:
    def test_generate(self):
        from nodes.passthrough import CosmosPredict2ImageToVideoLatent
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 16, 24, 60, 106)))
            mock_mx.float32 = np.float32
            node = CosmosPredict2ImageToVideoLatent()
            result = node.generate(MagicMock(), 848, 480, 93, 1)
            assert "samples" in result[0]
            assert result[0]["num_frames"] == 93


class TestEmptyLTXVLatentVideo:
    def test_input_types(self):
        from nodes.passthrough import EmptyLTXVLatentVideo
        inputs = EmptyLTXVLatentVideo.INPUT_TYPES()
        assert "width" in inputs["required"]
        assert "height" in inputs["required"]
        assert "length" in inputs["required"]

    def test_generate(self):
        from nodes.passthrough import EmptyLTXVLatentVideo
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 128, 13, 16, 24)))
            mock_mx.float32 = np.float32
            node = EmptyLTXVLatentVideo()
            result = node.generate(768, 512, 97, 1)
            assert "samples" in result[0]
            assert result[0]["num_frames"] == 97
            assert result[0]["downscale_ratio_spacial"] == 32


class TestHunyuanImageToVideo:
    def test_generate(self):
        from nodes.passthrough import HunyuanImageToVideo
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 16, 14, 60, 106)))
            mock_mx.float32 = np.float32
            node = HunyuanImageToVideo()
            positive = {"prompt": "test"}
            result = node.generate(positive, MagicMock(), 848, 480, 53, 1)
            assert len(result) == 2
            assert "samples" in result[1]


class TestLTXVAddGuide:
    def test_append(self):
        from nodes.passthrough import LTXVAddGuide
        node = LTXVAddGuide()
        pos = {"prompt": "test"}
        neg = {"prompt": ""}
        vae = MagicMock()
        latent = MagicMock()
        image = MagicMock()
        result = node.append(pos, neg, vae, latent, image)
        assert result == (pos, neg, latent)


class TestLTXVCropGuides:
    def test_crop(self):
        from nodes.passthrough import LTXVCropGuides
        node = LTXVCropGuides()
        pos = {"prompt": "test"}
        neg = {"prompt": ""}
        latent = MagicMock()
        result = node.crop(pos, neg, latent)
        assert result == (pos, neg, latent)


class TestLTXVPreprocess:
    def test_preprocess(self):
        from nodes.passthrough import LTXVPreprocess
        node = LTXVPreprocess()
        images = MagicMock()
        result = node.preprocess(images)
        assert result == (images,)


class TestSVDImg2VidConditioning:
    def test_input_types(self):
        from nodes.passthrough import SVD_img2vid_Conditioning
        inputs = SVD_img2vid_Conditioning.INPUT_TYPES()
        assert "clip_vision" in inputs["required"]
        assert "init_image" in inputs["required"]

    def test_encode(self):
        from nodes.passthrough import SVD_img2vid_Conditioning
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 4, 14, 72, 128)))
            mock_mx.float32 = np.float32
            node = SVD_img2vid_Conditioning()
            result = node.encode(MagicMock(), MagicMock(), MagicMock(), 1024, 576, 14, 127, 6, 0.0)
            assert len(result) == 3
            positive, negative, latent = result
            assert "samples" in latent


class TestTextEncodeHunyuanVideoImageToVideo:
    def test_encode(self):
        from nodes.passthrough import TextEncodeHunyuanVideo_ImageToVideo
        node = TextEncodeHunyuanVideo_ImageToVideo()
        result = node.encode(MagicMock(), MagicMock(), "a cat", 2)
        assert result[0]["prompt"] == "a cat"


class TestTrimVideoLatent:
    def test_trim(self):
        from nodes.passthrough import TrimVideoLatent
        node = TrimVideoLatent()
        samples = {"samples": np.zeros((1, 4, 10, 64, 64))}
        result = node.trim(samples, 0, -1)
        assert result == (samples,)


class TestVideoLinearCFGGuidance:
    def test_patch(self):
        from nodes.passthrough import VideoLinearCFGGuidance
        node = VideoLinearCFGGuidance()
        mock_model = MagicMock()
        result = node.patch(mock_model, 1.0)
        assert result == (mock_model,)


class TestWanCameraEmbedding:
    def test_append(self):
        from nodes.passthrough import WanCameraEmbedding
        node = WanCameraEmbedding()
        result = node.append("Static", 832, 480, 81, 1.0)
        assert len(result) == 4
        assert result[1] == 832
        assert result[2] == 480
        assert result[3] == 81


class TestWanCameraImageToVideo:
    def test_generate(self):
        from nodes.passthrough import WanCameraImageToVideo
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 16, 21, 60, 104)))
            mock_mx.float32 = np.float32
            node = WanCameraImageToVideo()
            result = node.generate(
                {"prompt": "test"}, {"prompt": ""}, MagicMock(),
                832, 480, 81, 1,
            )
            assert len(result) == 3
            assert "samples" in result[2]
            assert "_i2v_image_path" not in result[2]

    def test_generate_with_start_image(self):
        from nodes.passthrough import WanCameraImageToVideo
        start_image = np.zeros((1, 480, 832, 3), dtype=np.float32)
        node = WanCameraImageToVideo()
        result = node.generate(
            {"prompt": "test"}, {"prompt": ""}, MagicMock(),
            832, 480, 81, 1, start_image=start_image,
        )
        assert len(result) == 3
        assert "_i2v_image_path" in result[2]
        tmp_path = result[2]["_i2v_image_path"]
        assert os.path.exists(tmp_path)
        os.unlink(tmp_path)

    def test_generate_with_start_image_and_camera_warns(self):
        from nodes.passthrough import WanCameraImageToVideo
        start_image = np.zeros((1, 480, 832, 3), dtype=np.float32)
        camera_conditions = np.zeros((1, 16, 21, 60, 104), dtype=np.float32)
        node = WanCameraImageToVideo()
        result = node.generate(
            {"prompt": "test"}, {"prompt": ""}, MagicMock(),
            832, 480, 81, 1, start_image=start_image, camera_conditions=camera_conditions,
        )
        assert "_i2v_image_path" in result[2]
        os.unlink(result[2]["_i2v_image_path"])


class TestWanVaceToVideo:
    def test_generate(self):
        from nodes.passthrough import WanVaceToVideo
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 16, 21, 60, 104)))
            mock_mx.float32 = np.float32
            node = WanVaceToVideo()
            result = node.generate(
                {"prompt": "test"}, {"prompt": ""}, MagicMock(),
                832, 480, 81, 1, 1.0,
            )
            assert len(result) == 4
            assert result[3] == 0  # trim_latent_val

    def test_generate_with_all_optionals(self):
        from nodes.passthrough import WanVaceToVideo
        with patch("nodes.passthrough.mx") as mock_mx:
            mock_mx.zeros = MagicMock(return_value=np.zeros((1, 16, 21, 60, 104)))
            mock_mx.float32 = np.float32
            node = WanVaceToVideo()
            result = node.generate(
                {"prompt": "test"}, {"prompt": ""}, MagicMock(),
                832, 480, 81, 1, 1.0,
                control_video=MagicMock(), control_masks=MagicMock(),
                reference_image=MagicMock(),
            )
            assert len(result) == 4


class TestNote:
    def test_input_types(self):
        from nodes.passthrough import Note
        inputs = Note.INPUT_TYPES()
        assert "required" in inputs

    def test_annotate(self):
        from nodes.passthrough import Note
        node = Note()
        result = node.annotate()
        assert result == {}

    def test_output_node(self):
        from nodes.passthrough import Note
        assert Note.OUTPUT_NODE is True
