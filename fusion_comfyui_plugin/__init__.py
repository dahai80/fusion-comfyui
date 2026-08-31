import logging
import os

logger = logging.getLogger("fusion_comfyui")

_torch_stub_installed = False


def _install_torch_stub():
    global _torch_stub_installed
    if _torch_stub_installed:
        return
    try:
        # _torch_stub is meta-install plumbing, not a public consumer symbol;
        # intentionally stays on internal path (out of scope for #624).
        from fusion_mlx._torch_stub import install as stub_install
        if stub_install():
            logger.info("ComfyUI-Fusion-MLX: torch stub installed (zero-PyTorch mode)")
        else:
            logger.warning("ComfyUI-Fusion-MLX: real torch detected, stub NOT installed")
    except ImportError:
        logger.warning("ComfyUI-Fusion-MLX: fusion_mlx._torch_stub not available")
    _torch_stub_installed = True


if os.environ.get("FUSION_MLX_NO_STUB", "").strip().lower() not in ("1", "true", "yes"):
    _install_torch_stub()

from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
from .nodes.loaders import (
    FusionModelLoaderNode,
    UNETLoader,
    CLIPLoader,
    DualCLIPLoader,
    VAELoader,
    CheckpointLoaderSimple,
    ImageOnlyCheckpointLoader,
    StableCascade_EmptyLatentImage,
    StableCascade_StageC_VAEEncode,
    StableCascade_StageB_Conditioning,
    StableCascade_SuperResolutionControlnet,
)
from .nodes.conditioning import FusionTextEncoderNode, CLIPTextEncode
from .nodes.image_transform import (
    ImageScale as _ImageScale,
    ImageScaleBy as _ImageScaleBy,
    ImageBatch as _ImageBatch,
    EmptyImage as _EmptyImage,
    ImagePadForOutpaint as _ImagePadForOutpaint,
    LoadImageMask as _LoadImageMask,
    PainterNode,
)
from .nodes._deadpath_stubs import (
    ConditioningSetMaskStub,
    VAEEncodeForInpaintStub,
    InpaintModelConditioningStub,
    ControlNetApplyStub,
    ControlNetApplyAdvancedStub,
    QwenImageDiffsynthControlnetStub,
)
from .nodes.samplers import (
    FusionKSamplerNode,
    KSampler,
    KSamplerAdvanced,
    LatentUpscale,
    SamplerCustom,
    SamplerCustomAdvanced,
)
from .nodes.vae import FusionVAEDecoderNode, VAEDecode, VAEDecodeTiled
from .nodes.latent import (
    FusionEmptyLatentNode,
    EmptyLatentImage,
    EmptySD3LatentImage,
    EmptyHunyuanLatentVideo,
    EmptyCosmosLatentVideo,
    Wan22ImageToVideoLatent,
    WanImageToVideo,
    LTXVImgToVideo,
)
from .nodes.image import LoadImage, SaveImage, PreviewImage
from .nodes.passthrough import (
    ModelSamplingSD3,
    ModelSamplingStableCascade,
    ModelSamplingContinuousEDM,
    ModelSamplingFlux,
    ModelSamplingAuraFlow,
    CFGNorm,
    LoraLoaderModelOnly,
    UnetLoaderGGUF,
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    FluxGuidance,
    CLIPVisionLoader,
    CLIPVisionEncode,
    LTXVConditioning,
    LTXVScheduler,
    CosmosImageToVideoLatent,
    CosmosPredict2ImageToVideoLatent,
    EmptyLTXVLatentVideo,
    HunyuanImageToVideo,
    LTXVAddGuide,
    LTXVCropGuides,
    LTXVPreprocess,
    SVD_img2vid_Conditioning,
    TextEncodeHunyuanVideo_ImageToVideo,
    TrimVideoLatent,
    VideoLinearCFGGuidance,
    WanCameraEmbedding,
    WanCameraImageToVideo,
    WanVaceToVideo,
    Note,
    MarkdownNote,
)
from .nodes.shortcuts import FusionImageGenNode, FusionVideoGenNode, FusionImageToVideoNode, FusionIdentityPipelineNode
from .nodes.video_io import FusionSaveVideoNode, FusionVideoConcatNode, SaveWEBM, SaveAnimatedWEBP
from .nodes.postprocess import FusionSubtitleOverlayNode
from .nodes.voice import (
    FusionVoiceLoaderNode,
    FusionVoiceSynthesizeNode,
    FusionVoiceCloneNode,
    FusionSaveAudioNode,
)
from .nodes.identity import (
    FusionIdentityLoader,
    FusionIdentityApply,
    FusionIdentityGenerate,
)
from .nodes.talking_head import (
    FusionLipsyncLoader,
    FusionLipsyncApply,
)
from .nodes.ip_adapter import (
    FusionIPAdapterLoader,
    FusionIPAdapterApply,
    FusionIPAdapterInject,
)
from .nodes.stats import FusionDenoiseStatsNode
from .nodes.h3 import (
    MiniMaxH3SigmaShift,
    EmptyMiniMaxH3LatentAV,
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    VAEDecodeAudio as H3VAEDecodeAudio,
    CreateVideo as H3CreateVideo,
    SaveVideo as H3SaveVideo,
    ImageScaleToTotalPixels as H3ImageScaleToTotalPixels,
    PrimitiveFloat as H3PrimitiveFloat,
    ComfyMathExpression as H3ComfyMathExpression,
)

FusionMemoryGuardian.setup_environment()
logger.info("ComfyUI-Fusion-MLX: memory guardian initialized")

NODE_CLASS_MAPPINGS = {
    # === Native node overrides (key = ComfyUI native node name) ===
    # Loaders
    "UNETLoader": UNETLoader,
    "CLIPLoader": CLIPLoader,
    "DualCLIPLoader": DualCLIPLoader,
    "VAELoader": VAELoader,
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "ImageOnlyCheckpointLoader": ImageOnlyCheckpointLoader,
    # Stable Cascade (native comfy_extras overrides)
    "StableCascade_EmptyLatentImage": StableCascade_EmptyLatentImage,
    "StableCascade_StageC_VAEEncode": StableCascade_StageC_VAEEncode,
    "StableCascade_StageB_Conditioning": StableCascade_StageB_Conditioning,
    "StableCascade_SuperResolutionControlnet": StableCascade_SuperResolutionControlnet,
    # Image
    "LoadImage": LoadImage,
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "ImageScale": _ImageScale,
    "ImageScaleBy": _ImageScaleBy,
    "ImageBatch": _ImageBatch,
    "EmptyImage": _EmptyImage,
    "ImagePadForOutpaint": _ImagePadForOutpaint,
    "LoadImageMask": _LoadImageMask,
    # Dead-path stubs (native nodes routing into unported PyTorch layers)
    "ConditioningSetMask": ConditioningSetMaskStub,
    "VAEEncodeForInpaint": VAEEncodeForInpaintStub,
    "InpaintModelConditioning": InpaintModelConditioningStub,
    "ControlNetApply": ControlNetApplyStub,
    "ControlNetApplyAdvanced": ControlNetApplyAdvancedStub,
    "PainterNode": PainterNode,
    "QwenImageDiffsynthControlnet": QwenImageDiffsynthControlnetStub,
    # Conditioning
    "CLIPTextEncode": CLIPTextEncode,
    # Samplers
    "KSampler": KSampler,
    "KSamplerAdvanced": KSamplerAdvanced,
    "SamplerCustom": SamplerCustom,
    "SamplerCustomAdvanced": SamplerCustomAdvanced,
    "LatentUpscale": LatentUpscale,
    # VAE
    "VAEDecode": VAEDecode,
    "VAEDecodeTiled": VAEDecodeTiled,
    # Latent
    "EmptyLatentImage": EmptyLatentImage,
    "EmptySD3LatentImage": EmptySD3LatentImage,
    "EmptyHunyuanLatentVideo": EmptyHunyuanLatentVideo,
    "EmptyCosmosLatentVideo": EmptyCosmosLatentVideo,
    "Wan22ImageToVideoLatent": Wan22ImageToVideoLatent,
    "WanImageToVideo": WanImageToVideo,
    "LTXVImgToVideo": LTXVImgToVideo,
    # Passthrough (no-op nodes needed by workflows)
    "ModelSamplingSD3": ModelSamplingSD3,
    "ModelSamplingStableCascade": ModelSamplingStableCascade,
    "ModelSamplingContinuousEDM": ModelSamplingContinuousEDM,
    "ModelSamplingFlux": ModelSamplingFlux,
    "ModelSamplingAuraFlow": ModelSamplingAuraFlow,
    "CFGNorm": CFGNorm,
    "LoraLoaderModelOnly": LoraLoaderModelOnly,
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "BasicGuider": BasicGuider,
    "BasicScheduler": BasicScheduler,
    "KSamplerSelect": KSamplerSelect,
    "RandomNoise": RandomNoise,
    "FluxGuidance": FluxGuidance,
    "CLIPVisionLoader": CLIPVisionLoader,
    "CLIPVisionEncode": CLIPVisionEncode,
    "LTXVConditioning": LTXVConditioning,
    "LTXVScheduler": LTXVScheduler,
    "CosmosImageToVideoLatent": CosmosImageToVideoLatent,
    "CosmosPredict2ImageToVideoLatent": CosmosPredict2ImageToVideoLatent,
    "EmptyLTXVLatentVideo": EmptyLTXVLatentVideo,
    "HunyuanImageToVideo": HunyuanImageToVideo,
    "LTXVAddGuide": LTXVAddGuide,
    "LTXVCropGuides": LTXVCropGuides,
    "LTXVPreprocess": LTXVPreprocess,
    "SVD_img2vid_Conditioning": SVD_img2vid_Conditioning,
    "TextEncodeHunyuanVideo_ImageToVideo": TextEncodeHunyuanVideo_ImageToVideo,
    "TrimVideoLatent": TrimVideoLatent,
    "VideoLinearCFGGuidance": VideoLinearCFGGuidance,
    "WanCameraEmbedding": WanCameraEmbedding,
    "WanCameraImageToVideo": WanCameraImageToVideo,
    "WanVaceToVideo": WanVaceToVideo,
    "Note": Note,
    "MarkdownNote": MarkdownNote,
    # Output nodes (native uses torch, we use PyAV/PIL)
    "SaveWEBM": SaveWEBM,
    "SaveAnimatedWEBP": SaveAnimatedWEBP,
    # === Fusion-MLX native nodes (backward compat) ===
    "FusionModelLoader": FusionModelLoaderNode,
    "FusionTextEncoder": FusionTextEncoderNode,
    "FusionKSampler": FusionKSamplerNode,
    "FusionVAEDecoder": FusionVAEDecoderNode,
    "FusionImageGen": FusionImageGenNode,
    "FusionVideoGen": FusionVideoGenNode,
    "FusionImageToVideo": FusionImageToVideoNode,
    "FusionIdentityPipeline": FusionIdentityPipelineNode,
    "FusionEmptyLatent": FusionEmptyLatentNode,
    "FusionSaveVideo": FusionSaveVideoNode,
    "FusionVideoConcat": FusionVideoConcatNode,
    "FusionSubtitleOverlay": FusionSubtitleOverlayNode,
    # Voice / TTS
    "FusionVoiceLoader": FusionVoiceLoaderNode,
    "FusionVoiceSynthesize": FusionVoiceSynthesizeNode,
    "FusionVoiceClone": FusionVoiceCloneNode,
    "FusionSaveAudio": FusionSaveAudioNode,
    # Identity / PuLID
    "FusionIdentityLoader": FusionIdentityLoader,
    "FusionIdentityApply": FusionIdentityApply,
    "FusionIdentityGenerate": FusionIdentityGenerate,
    # Talking-Head / Lip-Sync
    "FusionLipsyncLoader": FusionLipsyncLoader,
    "FusionLipsyncApply": FusionLipsyncApply,
    # IP-Adapter / Flux
    "FusionIPAdapterLoader": FusionIPAdapterLoader,
    "FusionIPAdapterApply": FusionIPAdapterApply,
    "FusionIPAdapterInject": FusionIPAdapterInject,
    # Debug
    "FusionDenoiseStats": FusionDenoiseStatsNode,
    # H3 (MiniMax H3) sampling-pipe nodes for AICF workflows (h3-t2v/i2v/r2v)
    "MiniMaxH3SigmaShift": MiniMaxH3SigmaShift,
    "EmptyMiniMaxH3LatentAV": EmptyMiniMaxH3LatentAV,
    "MiniMaxH3ImageToVideo": MiniMaxH3ImageToVideo,
    "MiniMaxH3ReferenceToVideo": MiniMaxH3ReferenceToVideo,
    "VAEDecodeAudio": H3VAEDecodeAudio,
    "CreateVideo": H3CreateVideo,
    "SaveVideo": H3SaveVideo,
    "ImageScaleToTotalPixels": H3ImageScaleToTotalPixels,
    "PrimitiveFloat": H3PrimitiveFloat,
    "ComfyMathExpression": H3ComfyMathExpression,
}

# ComfyUI's load_custom_node() filters native node names via `ignore` param,
# preventing custom nodes from overriding native NODE_CLASS_MAPPINGS entries.
# We monkey-patch the native classes directly to work around this restriction.
_native_overrides = {
    "UNETLoader": UNETLoader,
    "CLIPLoader": CLIPLoader,
    "DualCLIPLoader": DualCLIPLoader,
    "VAELoader": VAELoader,
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "ImageOnlyCheckpointLoader": ImageOnlyCheckpointLoader,
    "LoadImage": LoadImage,
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "ImageScale": _ImageScale,
    "ImageScaleBy": _ImageScaleBy,
    "ImageBatch": _ImageBatch,
    "EmptyImage": _EmptyImage,
    "ImagePadForOutpaint": _ImagePadForOutpaint,
    "LoadImageMask": _LoadImageMask,
    "ConditioningSetMask": ConditioningSetMaskStub,
    "VAEEncodeForInpaint": VAEEncodeForInpaintStub,
    "InpaintModelConditioning": InpaintModelConditioningStub,
    "ControlNetApply": ControlNetApplyStub,
    "ControlNetApplyAdvanced": ControlNetApplyAdvancedStub,
    "PainterNode": PainterNode,
    "QwenImageDiffsynthControlnet": QwenImageDiffsynthControlnetStub,
    "CLIPTextEncode": CLIPTextEncode,
    "KSampler": KSampler,
    "KSamplerAdvanced": KSamplerAdvanced,
    "SamplerCustom": SamplerCustom,
    "SamplerCustomAdvanced": SamplerCustomAdvanced,
    "VAEDecode": VAEDecode,
    "VAEDecodeTiled": VAEDecodeTiled,
    "LatentUpscale": LatentUpscale,
    "EmptyLatentImage": EmptyLatentImage,
    "EmptySD3LatentImage": EmptySD3LatentImage,
    "EmptyHunyuanLatentVideo": EmptyHunyuanLatentVideo,
    "EmptyCosmosLatentVideo": EmptyCosmosLatentVideo,
    "Wan22ImageToVideoLatent": Wan22ImageToVideoLatent,
    "WanImageToVideo": WanImageToVideo,
    "LTXVImgToVideo": LTXVImgToVideo,
    "ModelSamplingSD3": ModelSamplingSD3,
    "ModelSamplingContinuousEDM": ModelSamplingContinuousEDM,
    "ModelSamplingFlux": ModelSamplingFlux,
    "ModelSamplingStableCascade": ModelSamplingStableCascade,
    "ModelSamplingAuraFlow": ModelSamplingAuraFlow,
    "CFGNorm": CFGNorm,
    "LoraLoaderModelOnly": LoraLoaderModelOnly,
    "BasicGuider": BasicGuider,
    "BasicScheduler": BasicScheduler,
    "KSamplerSelect": KSamplerSelect,
    "RandomNoise": RandomNoise,
    "FluxGuidance": FluxGuidance,
    "CLIPVisionLoader": CLIPVisionLoader,
    "CLIPVisionEncode": CLIPVisionEncode,
    "LTXVConditioning": LTXVConditioning,
    "LTXVScheduler": LTXVScheduler,
    "CosmosImageToVideoLatent": CosmosImageToVideoLatent,
    "CosmosPredict2ImageToVideoLatent": CosmosPredict2ImageToVideoLatent,
    "EmptyLTXVLatentVideo": EmptyLTXVLatentVideo,
    "HunyuanImageToVideo": HunyuanImageToVideo,
    "LTXVAddGuide": LTXVAddGuide,
    "LTXVCropGuides": LTXVCropGuides,
    "LTXVPreprocess": LTXVPreprocess,
    "SVD_img2vid_Conditioning": SVD_img2vid_Conditioning,
    "TextEncodeHunyuanVideo_ImageToVideo": TextEncodeHunyuanVideo_ImageToVideo,
    "TrimVideoLatent": TrimVideoLatent,
    "VideoLinearCFGGuidance": VideoLinearCFGGuidance,
    "WanCameraEmbedding": WanCameraEmbedding,
    "WanCameraImageToVideo": WanCameraImageToVideo,
    "WanVaceToVideo": WanVaceToVideo,
    "StableCascade_EmptyLatentImage": StableCascade_EmptyLatentImage,
    "StableCascade_StageC_VAEEncode": StableCascade_StageC_VAEEncode,
    "StableCascade_StageB_Conditioning": StableCascade_StageB_Conditioning,
    "StableCascade_SuperResolutionControlnet": StableCascade_SuperResolutionControlnet,
    "Note": Note,
    "MarkdownNote": MarkdownNote,
    "SaveWEBM": SaveWEBM,
    "SaveAnimatedWEBP": SaveAnimatedWEBP,
    # H3 nodes that override core comfy_extras nodes (CreateVideo/SaveVideo in
    # nodes_video, VAEDecodeAudio in nodes_audio, aux in nodes_post_processing/
    # nodes_math/nodes_primitive). Monkey-patched so AICF H3 workflows route here.
    "VAEDecodeAudio": H3VAEDecodeAudio,
    "CreateVideo": H3CreateVideo,
    "SaveVideo": H3SaveVideo,
    "ImageScaleToTotalPixels": H3ImageScaleToTotalPixels,
    "PrimitiveFloat": H3PrimitiveFloat,
    "ComfyMathExpression": H3ComfyMathExpression,
}

try:
    import nodes as _comfy_nodes
    _patched = []
    _not_found = []
    for _name, _override_cls in _native_overrides.items():
        _native_cls = _comfy_nodes.NODE_CLASS_MAPPINGS.get(_name)
        if _native_cls is None:
            _not_found.append(_name)
            continue
        if _native_cls.__module__ != _override_cls.__module__:
            _comfy_nodes.NODE_CLASS_MAPPINGS[_name] = _override_cls
            _override_cls.RELATIVE_PYTHON_MODULE = "custom_nodes.ComfyUI-Fusion-MLX"
            _patched.append(_name)
    if _patched:
        logger.info("Patched %d native node overrides: %s", len(_patched), ", ".join(_patched[:5]))
    if _not_found:
        logger.warning("Native node overrides NOT FOUND in NODE_CLASS_MAPPINGS: %s", ", ".join(_not_found))
except Exception as _e:
    logger.warning("Failed to patch native node overrides: %s", _e)

NODE_DISPLAY_NAME_MAPPINGS = {
    # Native overrides show they're fusion-mlx powered
    "UNETLoader": "UNET Loader (fusion-mlx)",
    "CLIPLoader": "CLIP Loader (fusion-mlx)",
    "DualCLIPLoader": "DualCLIP Loader (fusion-mlx)",
    "VAELoader": "VAE Loader (fusion-mlx)",
    "CheckpointLoaderSimple": "Checkpoint Loader Simple (fusion-mlx)",
    "ImageOnlyCheckpointLoader": "Image Only Checkpoint Loader (fusion-mlx)",
    "LoadImage": "Load Image (fusion-mlx)",
    "SaveImage": "Save Image (fusion-mlx)",
    "PreviewImage": "Preview Image (fusion-mlx)",
    "ImageScale": "Image Scale (fusion-mlx)",
    "ImageScaleBy": "Image Scale By (fusion-mlx)",
    "ImageBatch": "Image Batch (fusion-mlx)",
    "EmptyImage": "Empty Image (fusion-mlx)",
    "ImagePadForOutpaint": "Pad Image for Outpaint (fusion-mlx)",
    "LoadImageMask": "Load Image Mask (fusion-mlx)",
    "ConditioningSetMask": "Set Latent Noise Mask (not on MLX)",
    "VAEEncodeForInpaint": "VAE Encode for Inpaint (not on MLX)",
    "InpaintModelConditioning": "Inpaint Model Conditioning (not on MLX)",
    "ControlNetApply": "Apply ControlNet (not on MLX)",
    "ControlNetApplyAdvanced": "Apply ControlNet Advanced (not on MLX)",
    "PainterNode": "Painter (not on MLX)",
    "QwenImageDiffsynthControlnet": "Qwen Image ControlNet (not on MLX)",
    "CLIPTextEncode": "CLIP Text Encode (fusion-mlx)",
    "KSampler": "KSampler (fusion-mlx)",
    "KSamplerAdvanced": "KSampler Advanced (fusion-mlx)",
    "SamplerCustom": "Sampler Custom (fusion-mlx)",
    "SamplerCustomAdvanced": "Sampler Custom Advanced (fusion-mlx)",
    "VAEDecode": "VAE Decode (fusion-mlx)",
    "VAEDecodeTiled": "VAE Decode Tiled (fusion-mlx)",
    "EmptyLatentImage": "Empty Latent Image (fusion-mlx)",
    "EmptySD3LatentImage": "Empty SD3 Latent Image (fusion-mlx)",
    "EmptyHunyuanLatentVideo": "Empty Hunyuan Latent Video (fusion-mlx)",
    "EmptyCosmosLatentVideo": "Empty Cosmos Latent Video (fusion-mlx)",
    "Wan22ImageToVideoLatent": "Wan2.2 Image To Video Latent (fusion-mlx)",
    "WanImageToVideo": "Wan Image To Video (fusion-mlx)",
    "LTXVImgToVideo": "LTXV Img To Video (fusion-mlx)",
    "ModelSamplingSD3": "Model Sampling SD3 (fusion-mlx)",
    "ModelSamplingContinuousEDM": "Model Sampling Continuous EDM (fusion-mlx)",
    "ModelSamplingFlux": "Model Sampling Flux (fusion-mlx)",
    "ModelSamplingStableCascade": "Model Sampling Stable Cascade (fusion-mlx)",
    "ModelSamplingAuraFlow": "Model Sampling AuraFlow (fusion-mlx)",
    "CFGNorm": "CFG Norm (fusion-mlx)",
    "LoraLoaderModelOnly": "Load LoRA (fusion-mlx)",
    "UnetLoaderGGUF": "Unet Loader GGUF (fusion-mlx)",
    "BasicGuider": "Basic Guider (fusion-mlx)",
    "BasicScheduler": "Basic Scheduler (fusion-mlx)",
    "KSamplerSelect": "KSampler Select (fusion-mlx)",
    "RandomNoise": "Random Noise (fusion-mlx)",
    "FluxGuidance": "Flux Guidance (fusion-mlx)",
    "CLIPVisionLoader": "CLIP Vision Loader (fusion-mlx)",
    "CLIPVisionEncode": "CLIP Vision Encode (fusion-mlx)",
    "LTXVConditioning": "LTXV Conditioning (fusion-mlx)",
    "LTXVScheduler": "LTXV Scheduler (fusion-mlx)",
    "CosmosImageToVideoLatent": "Cosmos Image To Video Latent (fusion-mlx)",
    "CosmosPredict2ImageToVideoLatent": "Cosmos Predict2 Image To Video Latent (fusion-mlx)",
    "EmptyLTXVLatentVideo": "Empty LTXV Latent Video (fusion-mlx)",
    "HunyuanImageToVideo": "Hunyuan Image To Video (fusion-mlx)",
    "LTXVAddGuide": "LTXV Add Guide (fusion-mlx)",
    "LTXVCropGuides": "LTXV Crop Guides (fusion-mlx)",
    "LTXVPreprocess": "LTXV Preprocess (fusion-mlx)",
    "SVD_img2vid_Conditioning": "SVD img2vid Conditioning (fusion-mlx)",
    "TextEncodeHunyuanVideo_ImageToVideo": "Text Encode Hunyuan Video i2v (fusion-mlx)",
    "TrimVideoLatent": "Trim Video Latent (fusion-mlx)",
    "VideoLinearCFGGuidance": "Video Linear CFG Guidance (fusion-mlx)",
    "WanCameraEmbedding": "Wan Camera Embedding (fusion-mlx)",
    "WanCameraImageToVideo": "Wan Camera Image To Video (fusion-mlx)",
    "WanVaceToVideo": "Wan VACE To Video (fusion-mlx)",
    # Staged pipeline
    "FusionModelLoader": "⚡ Fusion-MLX Model Loader",
    "FusionTextEncoder": "⚡ Fusion-MLX Text Encoder",
    "FusionKSampler": "⚡ Fusion-MLX Sampler (dflash)",
    "FusionVAEDecoder": "⚡ Fusion-MLX VAE Decoder",
    # Shortcuts
    "FusionImageGen": "⚡ Fusion-MLX Image Gen",
    "FusionVideoGen": "⚡ Fusion-MLX Video Gen",
    "FusionImageToVideo": "⚡ Fusion-MLX Image-to-Video",
    "FusionIdentityPipeline": "⚡ Fusion-MLX Identity Pipeline",
    # Latent
    "FusionEmptyLatent": "⚡ Fusion-MLX Empty Latent",
    # Video I/O
    "FusionSaveVideo": "⚡ Fusion-MLX Save Video",
    "FusionVideoConcat": "⚡ Fusion-MLX Video Concat",
    # Post-process
    "FusionSubtitleOverlay": "⚡ Fusion-MLX Subtitle Overlay",
    # Voice / TTS
    "FusionVoiceLoader": "⚡ Fusion-MLX Voice Loader",
    "FusionVoiceSynthesize": "⚡ Fusion-MLX Voice Synthesize",
    "FusionVoiceClone": "⚡ Fusion-MLX Voice Clone",
    "FusionSaveAudio": "⚡ Fusion-MLX Save Audio",
    # Identity / PuLID
    "FusionIdentityLoader": "⚡ Fusion-MLX Identity Loader",
    "FusionIdentityApply": "⚡ Fusion-MLX Identity Apply",
    "FusionIdentityGenerate": "⚡ Fusion-MLX Identity Generate",
    # Talking-Head / Lip-Sync
    "FusionLipsyncLoader": "⚡ Fusion-MLX Lipsync Loader",
    "FusionLipsyncApply": "⚡ Fusion-MLX Lipsync Apply",
    # IP-Adapter / Flux
    "FusionIPAdapterLoader": "⚡ Fusion-MLX IP-Adapter Loader",
    "FusionIPAdapterApply": "⚡ Fusion-MLX IP-Adapter Apply",
    "FusionIPAdapterInject": "⚡ Fusion-MLX IP-Adapter Inject",
    # Debug
    "FusionDenoiseStats": "⚡ Fusion-MLX Denoise Stats",
    # H3 (MiniMax H3) sampling-pipe nodes for AICF workflows
    "MiniMaxH3SigmaShift": "⚡ MiniMax H3 Sigma Shift (fusion-mlx)",
    "EmptyMiniMaxH3LatentAV": "⚡ Empty MiniMax H3 Latent AV (fusion-mlx)",
    "MiniMaxH3ImageToVideo": "⚡ MiniMax H3 Image To Video (fusion-mlx)",
    "MiniMaxH3ReferenceToVideo": "⚡ MiniMax H3 Reference To Video (fusion-mlx)",
    "VAEDecodeAudio": "⚡ VAE Decode Audio (fusion-mlx)",
    "CreateVideo": "⚡ Create Video (fusion-mlx)",
    "SaveVideo": "⚡ Save Video (fusion-mlx)",
    "ImageScaleToTotalPixels": "⚡ Image Scale To Total Pixels (fusion-mlx)",
    "PrimitiveFloat": "⚡ Primitive Float (fusion-mlx)",
    "ComfyMathExpression": "⚡ ComfyMath Expression (fusion-mlx)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
