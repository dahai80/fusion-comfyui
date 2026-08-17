import logging
from pathlib import Path


logger = logging.getLogger("fusion_comfyui.wrappers")


class FusionModelWrapper:
    """Replaces native ComfyUI MODEL type.
    Holds model path and engine reference for fusion-mlx pipeline.
    """

    def __init__(self, model_path: str, model_name: str, model_type: str = "video"):
        self.model_path = model_path
        self.model_name = model_name
        self.model_type = model_type
        self._engine = None
        self._started = False

        # ComfyUI core compatibility stubs
        self.patcher = None
        self.model_diffusion = None
        self.is_cli = False
        self.memory_required = lambda *a, **kw: 0

        logger.info(
            "FusionModelWrapper: path=%s name=%s type=%s",
            model_path, model_name, model_type,
        )

    def get_engine(self):
        from .engine_wrapper import FusionEngineWrapper
        if self._engine is None:
            self._engine = FusionEngineWrapper(
                model_name=self.model_path,
                offload_strategy="sequential",
                quant_bit="fp8_e4m3",
            )
        return self._engine

    def __repr__(self):
        return f"<FusionModel path={self.model_name} type={self.model_type}>"


class FusionCLIPWrapper:
    """Replaces native ComfyUI CLIP type.
    Holds text encoder path and reference to the model wrapper.
    """

    def __init__(self, model_path: str, model_name: str, clip_type: str = "wan",
                 model_wrapper: FusionModelWrapper = None):
        self.model_path = model_path
        self.model_name = model_name
        self.clip_type = clip_type
        self.model_wrapper = model_wrapper

        # ComfyUI core compatibility stubs
        self.patcher = None
        self.tokenizer = None
        self.load_device = None
        self.offload_device = None

        logger.info(
            "FusionCLIPWrapper: path=%s type=%s", model_path, clip_type,
        )

    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        return {"text": tokens.get("text", ""), "clip": self}

    def __repr__(self):
        return f"<FusionCLIP path={self.model_name} type={self.clip_type}>"


class FusionVAEWrapper:
    """Replaces native ComfyUI VAE type.
    Holds VAE path and reference to the model wrapper.
    """

    def __init__(self, model_path: str, model_name: str,
                 model_wrapper: FusionModelWrapper = None):
        self.model_path = model_path
        self.model_name = model_name
        self.model_wrapper = model_wrapper
        self._engine = None

        # ComfyUI core compatibility stubs
        self.first_stage_model = None
        self.load_device = None
        self.offload_device = None

        # Stable Cascade Effnet VAE: 16-ch stage_c latent, downscale 4x.
        # Set when this wrapper backs a stable_cascade checkpoint so the
        # native StableCascade_StageC_VAEEncode / SuperResolutionControlnet
        # nodes (which read vae.downscale_ratio / vae.encode) don't crash.
        self._is_cascade = _is_cascade_name(model_name)
        self.downscale_ratio = 4 if self._is_cascade else 8

        logger.info("FusionVAEWrapper: path=%s cascade=%s", model_path, self._is_cascade)

    def get_engine(self):
        if self._engine is not None:
            return self._engine
        if self.model_wrapper is not None:
            self._engine = self.model_wrapper.get_engine()
            logger.debug("FusionVAEWrapper: using cached engine from model_wrapper")
            return self._engine
        from .engine_wrapper import FusionEngineWrapper
        self._engine = FusionEngineWrapper(
            model_name=self.model_path,
            offload_strategy="sequential",
            quant_bit="fp8_e4m3",
        )
        logger.debug("FusionVAEWrapper: created new engine for %s", self.model_name)
        return self._engine

    def encode(self, x):
        # Native StableCascade_StageC_VAEEncode / SuperResolutionControlnet
        # call vae.encode(image_nchw) to produce a stage_c latent (16-ch).
        # The bridge KSampler regenerates end-to-end from prompt+size, so
        # the latent content is not consumed — only its shape must be valid
        # for the graph to execute. Return a zero latent with the Effnet
        # output shape (N, 16, H//downscale_ratio, W//downscale_ratio).
        import numpy as np
        if hasattr(x, "shape"):
            shape = tuple(x.shape)
        else:
            shape = (1, 3, 512, 512)
        n = shape[0] if len(shape) >= 4 else 1
        h = shape[2] if len(shape) >= 3 else 512
        w = shape[3] if len(shape) >= 4 else 512
        if self._is_cascade:
            latent = np.zeros((n, 16, h // self.downscale_ratio, w // self.downscale_ratio),
                              dtype=np.float32)
            logger.info("FusionVAEWrapper.encode(cascade): %s -> stage_c %s", shape, latent.shape)
            return latent
        logger.warning("FusionVAEWrapper.encode: non-cascade VAE encode requested (%s), "
                       "returning zero latent", shape)
        return np.zeros((n, 4, h // self.downscale_ratio, w // self.downscale_ratio),
                        dtype=np.float32)

    def __repr__(self):
        return f"<FusionVAE path={self.model_name}>"


class FusionConditioning:
    """Stores conditioning data (text embeddings or prompt text).
    When stage API is available, embeds are mx.array.
    When falling back to monolithic generate(), stores prompt text.
    """

    def __init__(self, data: dict):
        self.data = data

    def __repr__(self):
        embed = self.data.get("embed")
        if embed is not None:
            shape = tuple(embed.shape) if hasattr(embed, "shape") else "N/A"
            return f"<FusionCond embed_shape={shape}>"
        return f"<FusionCond prompt='{self.data.get('prompt', '')[:30]}...'>"


def _infer_model_type(model_name: str) -> str:
    name = model_name.lower()
    if any(k in name for k in ("wan", "ltx", "skyreels", "cosmos", "hunyuan", "svd", "stable-video", "img2vid")):
        return "video"
    return "image"


def _is_cascade_name(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "cascade" in name or "wuerstchen" in name


_CASCADE_MODEL_DIR = "models--stabilityai--stable-cascade-prior"


def _available_cascade_model() -> str | None:
    import os
    base = os.path.expanduser("~/.fusion-mlx/models")
    candidate = os.path.join(base, _CASCADE_MODEL_DIR)
    if os.path.isdir(candidate):
        return _CASCADE_MODEL_DIR
    return None


def _available_video_models() -> list:
    import os
    base = os.path.expanduser("~/.fusion-mlx/models")
    known_video = [
        "Wan2.2-5B", "Wan2.2-14B", "Wan2.2-14B-T2V", "Wan2.1-1.3B", "Wan2.1-14B", "Wan2.1-VACE-14B",
        "LTX-Video", "SkyReels-V3-14B-mxfp8", "SkyReels-V3-A2V-19B-MLX",
        "SkyReels-V3-R2V-14B-MLX", "SkyReels-V3-V2V-14B-MLX",
        "Cosmos-7B", "Cosmos-Predict2", "Wan2.2-TI2V-5B-mlx-q8",
        "FLUX.2-klein-base-4B", "FLUX.2-klein-9b", "FLUX.2-dev-mxfp8",
        "stable-video-diffusion-img2vid-xt", "stable-video-diffusion-img2vid",
    ]
    found = []
    for m in known_video:
        if os.path.isdir(os.path.join(base, m)):
            found.append(m)
        elif os.path.isdir(os.path.join(base, "Skywork", m)):
            found.append(os.path.join("Skywork", m))
        elif os.path.isdir(os.path.join(base, "dgrauet", m)):
            found.append(os.path.join("dgrauet", m))
    return found


def _fallback_model(requested: str) -> str:
    import os
    base = os.path.expanduser("~/.fusion-mlx/models")
    if os.path.isdir(os.path.join(base, requested)):
        return requested
    name = requested.lower()
    # Stable Cascade must NOT fall through to video models (wan/flux).
    # Route any cascade/wuerstchen checkpoint to the self-contained
    # fusion-mlx stable_cascade pipeline, which loads prior+decoder+vqgan
    # from CascadeModelPaths regardless of the stage_b/stage_c filename.
    if "cascade" in name or "wuerstchen" in name:
        cascade_dir = _available_cascade_model()
        if cascade_dir is not None:
            logger.info("Falling back %s -> %s (stable_cascade pipeline)", requested, cascade_dir)
            return cascade_dir
        logger.warning("Stable Cascade checkpoint %s requested but no cascade model installed", requested)
    available = _available_video_models()
    if "wan" in name and "Wan2.2-5B" in available:
        logger.info("Falling back %s -> Wan2.2-5B (requested not available)", requested)
        return "Wan2.2-5B"
    if "ltx" in name and "LTX-Video" in available:
        logger.info("Falling back %s -> LTX-Video (requested not available)", requested)
        return "LTX-Video"
    if "flux" in name and "FLUX.2-klein-base-4B" in available:
        logger.info("Falling back %s -> FLUX.2-klein-base-4B (requested not available)", requested)
        return "FLUX.2-klein-base-4B"
    if "svd" in name or "stable-video" in name or "img2vid" in name:
        for svd_name in (
            "stable-video-diffusion-img2vid-xt",
            "stable-video-diffusion-img2vid",
        ):
            if svd_name in available:
                logger.info("Falling back %s -> %s (SVD i2v)", requested, svd_name)
                return svd_name
    if any(k in name for k in ("sd_xl", "sdxl", "stable_diffusion", "sd15")):
        if "FLUX.2-klein-base-4B" in available:
            logger.info("Falling back %s -> FLUX.2-klein-base-4B (no SD backend)", requested)
            return "FLUX.2-klein-base-4B"
    if available:
        logger.info("Falling back %s -> %s (requested not available)", requested, available[0])
        return available[0]
    return requested


def _resolve_model_path(model_name: str) -> str:
    try:
        from fusion_mlx.model_registry import get_registry
        reg = get_registry()
        info = reg.get(model_name)
        if info and hasattr(info, "get") and info.get("path"):
            return str(info["path"])
    except Exception:
        pass
    import os
    candidates = [
        os.path.expanduser(f"~/.fusion-mlx/models/{model_name}"),
        os.path.expanduser(f"~/.cache/huggingface/hub/{model_name}"),
        os.path.expanduser(f"~/models/{model_name}"),
        f"/mnt/models/{model_name}",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # HF hub cache layout: ~/.fusion-mlx/models/models--{org}--{name}/.
    # SD1.5 and other HF-cached image models live here as a snapshot tree,
    # not a plain directory, so the isdir() checks above miss them. Detect
    # the cache dir and return the literal repo id — the downstream engine
    # (e.g. SD15Pipeline) resolves components via hf_hub_download against
    # HUGGINGFACE_HUB_CACHE. Returning the repo id here also prevents
    # _fallback_model from mis-routing SD1.5 to an available video model.
    hf_cache = os.path.expanduser(
        os.environ.get("HUGGINGFACE_HUB_CACHE", "~/.fusion-mlx/models")
    )
    cache_slug = "models--" + model_name.replace("/", "--")
    if os.path.isdir(os.path.join(hf_cache, cache_slug)):
        logger.info(
            "_resolve_model_path: HF cache hit %s -> %s (repo id, no fallback)",
            model_name, cache_slug,
        )
        return model_name
    resolved = _fallback_model(model_name)
    if resolved != model_name:
        for c in [
            os.path.expanduser(f"~/.fusion-mlx/models/{resolved}"),
            os.path.expanduser(f"~/.cache/huggingface/hub/{resolved}"),
            os.path.expanduser(f"~/models/{resolved}"),
        ]:
            if os.path.isdir(c):
                return c
    return model_name


def _map_checkpoint_to_model_name(ckpt_name: str) -> str:
    name = ckpt_name.lower()
    # Stable Cascade: stage_b/stage_c checkpoints both route to the
    # self-contained cascade pipeline. Check before wan/flux/sd.
    if "cascade" in name or "wuerstchen" in name:
        cascade_dir = _available_cascade_model()
        if cascade_dir is not None:
            logger.info("Mapping cascade ckpt %s -> %s", ckpt_name, cascade_dir)
            return cascade_dir
        logger.warning("Cascade ckpt %s mapped but no cascade model installed", ckpt_name)
        return ckpt_name
    if "ltx" in name and "video" in name:
        return "LTX-Video"
    if "wan" in name:
        if "2.2" in name or "22" in name:
            return "Wan2.2-5B"
        return "Wan2.2-5B"
    if "flux" in name:
        if "4b" in name:
            return "FLUX.2-klein-base-4B"
        if "9b" in name:
            return "FLUX.2-klein-9b"
        return "FLUX.2-klein-base-4B"
    if "cosmos" in name:
        return "Cosmos-7B"
    if "hunyuan" in name:
        return "HunyuanVideo"
    if "svd" in name or "stable-video" in name or "img2vid" in name:
        if "xt" in name:
            logger.info("Mapping SVD-XT checkpoint %s -> stable-video-diffusion-img2vid-xt", ckpt_name)
            return "stable-video-diffusion-img2vid-xt"
        logger.info("Mapping SVD checkpoint %s -> stable-video-diffusion-img2vid", ckpt_name)
        return "stable-video-diffusion-img2vid"
    if any(
        k in name
        for k in (
            "v2-1",
            "sd2-1",
            "sd2.1",
            "sd21",
            "stable-diffusion-2",
            "768-v",
            "768v",
            "768-ema",
        )
    ):
        logger.info(
            "Mapping SD2.1 checkpoint %s -> sd2-community/stable-diffusion-2-1",
            ckpt_name,
        )
        return "sd2-community/stable-diffusion-2-1"
    if any(k in name for k in ("sd15", "sd-v1-5", "stable-diffusion-v1-5", "v1-5-pruned", "v1-5")):
        logger.info("Mapping SD1.5 checkpoint %s -> runwayml/stable-diffusion-v1-5", ckpt_name)
        return "runwayml/stable-diffusion-v1-5"
    if any(k in name for k in ("sd_xl", "sdxl", "stable_diffusion_xl", "stable_diffusion")):
        logger.info("Mapping SDXL checkpoint %s -> FLUX.2-klein-base-4B (no SDXL backend)", ckpt_name)
        return "FLUX.2-klein-base-4B"
    return ckpt_name


def _map_clip_type_to_model_name(clip_type: str, clip_name: str) -> str:
    type_to_model = {
        "wan": "Wan2.2-5B",
        "ltxv": "LTX-Video",
        "cosmos": "Cosmos-7B",
        "hunyuan_image": "HunyuanVideo",
        "sd3": "FLUX.2-klein-base-4B",
        "flux2": "FLUX.2-klein-base-4B",
    }
    return type_to_model.get(clip_type, clip_name)


def _map_unet_name_to_model_name(unet_name: str) -> str:
    name = unet_name.lower()
    if "wan2.2" in name or "wan22" in name:
        if "14b" in name:
            if "t2v" in name:
                resolved = _fallback_model("Wan2.2-14B-T2V")
                logger.info(
                    "Map t2v 14B ckpt %s -> Wan2.2-14B-T2V (16-ch wan2.1 VAE)",
                    unet_name,
                )
                return resolved
            resolved = _fallback_model("Wan2.2-14B")
            return resolved
        if "ti2v" in name:
            return _fallback_model("Wan2.2-TI2V-5B-mlx-q8")
        return "Wan2.2-5B"
    if "wan2.1" in name or "wan21" in name:
        if "vace" in name:
            resolved = _fallback_model("Wan2.1-VACE-14B")
            logger.info("Map VACE ckpt %s -> Wan2.1-VACE-14B", unet_name)
            return resolved
        # i2v/ti2v/fun/camera must be checked BEFORE 14b, otherwise
        # "wan2.1_i2v_480p_14B" matches "14b" first and routes to
        # Wan2.1-14B (t2v-only) instead of the correct i2v fallback.
        if any(k in name for k in ("fun", "camera", "i2v", "ti2v")):
            # Prefer dedicated Wan2.1-14B I2V model if dit/ has weights
            resolved_i2v = _fallback_model("Wan2.1-14B")
            if resolved_i2v == "Wan2.1-14B":
                dit_dir = Path.home() / ".fusion-mlx" / "models" / "Wan2.1-14B" / "dit"
                has_dit = dit_dir.is_dir() and any(dit_dir.glob("*.safetensors"))
            else:
                has_dit = False
            if has_dit:
                logger.info(
                    "Mapping i2v 14B ckpt %s -> Wan2.1-14B (dit/ with i2v weights)",
                    unet_name,
                )
                return "Wan2.1-14B"
            # Fallback to Wan2.2-TI2V-5B which supports i2v
            resolved = _fallback_model("Wan2.2-TI2V-5B-mlx-q8")
            if resolved == "Wan2.2-TI2V-5B-mlx-q8":
                logger.info(
                    "Mapping i2v checkpoint %s -> Wan2.2-TI2V-5B (i2v-capable, "
                    "mask_blend-trained); t2v 1.3B cannot anchor image latent -> 花",
                    unet_name,
                )
                return resolved
            logger.warning(
                "i2v checkpoint %s requested but no i2v-capable fallback available "
                "(got %s); t2v fallback cannot anchor image -> output may be 花",
                unet_name,
                resolved,
            )
        if "14b" in name:
            resolved = _fallback_model("Wan2.1-14B")
            return resolved
        resolved = _fallback_model("Wan2.1-1.3B")
        return resolved
    if "wan" in name:
        return "Wan2.2-5B"
    if "ltx-2" in name or "ltx_2" in name or "ltx-2.3" in name:
        return _fallback_model("ltx-2.3-mlx-q8")
    if "ltx" in name:
        return "LTX-Video"
    if "skyreels" in name:
        if "a2v" in name or "19b" in name:
            return _fallback_model("SkyReels-V3-A2V-19B-MLX")
        if "r2v" in name:
            return _fallback_model("SkyReels-V3-R2V-14B-MLX")
        if "v2v" in name:
            return _fallback_model("SkyReels-V3-V2V-14B-MLX")
        return _fallback_model("SkyReels-V3-V2V-14B-MLX")
    if "flux" in name:
        if "4b" in name or "base-4" in name:
            return "FLUX.2-klein-base-4B"
        if "klein" in name:
            return "FLUX.2-klein-9b"
        return "FLUX.2-klein-base-4B"
    if "cosmos" in name:
        if "predict2" in name or "2b" in name:
            return "Cosmos-Predict2"
        return "Cosmos-7B"
    if "hunyuan" in name:
        return "HunyuanVideo"
    if "svd" in name or "stable-video" in name or "img2vid" in name:
        resolved = _fallback_model("stable-video-diffusion-img2vid-xt")
        logger.info("Map SVD unet %s -> %s (i2v)", unet_name, resolved)
        return resolved
    if "cogvideo" in name:
        return unet_name
    return unet_name


def _map_vae_name_to_model_name(vae_name: str) -> str:
    name = vae_name.lower()
    if "wan2.2" in name or "wan22" in name:
        return "Wan2.2-5B"
    if "wan2.1" in name or "wan21" in name or "wan_2.1" in name:
        if "14b" in name:
            return "Wan2.1-14B"
        return "Wan2.1-1.3B"
    if "wan" in name:
        return "Wan2.2-5B"
    if "cosmos" in name:
        return "Cosmos-7B"
    if "hunyuan" in name:
        return "HunyuanVideo"
    return vae_name
