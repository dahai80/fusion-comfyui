import logging

import mlx.core as mx
import numpy as np

import core.async_utils

logger = logging.getLogger("fusion_comfyui.nodes.vae")


class VAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "model/latent"

    def decode(self, vae, samples):
        from core.wrappers import FusionVAEWrapper
        from core.bridge import to_mlx_array, to_image_array
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()

        if "_decoded_frames" in samples:
            frames = samples.pop("_decoded_frames")
            logger.info("VAEDecode: monolithic path, passing through decoded frames shape=%s", frames.shape)
            if isinstance(frames, np.ndarray):
                if frames.ndim == 4:
                    return (frames,)
                elif frames.ndim == 3:
                    return (frames[np.newaxis, ...],)
                elif frames.ndim == 5:
                    return (frames[0],)
            elif isinstance(frames, mx.array):
                arr = np.array(frames)
                if arr.ndim == 4:
                    return (arr,)
                elif arr.ndim == 3:
                    return (arr[np.newaxis, ...],)
                elif arr.ndim == 5:
                    return (arr[0],)
            return (frames,)

        if "_decoded_frames_key" in samples:
            from .samplers import _decoded_frames_cache
            cache_key = samples.pop("_decoded_frames_key")
            frames = _decoded_frames_cache.pop(cache_key, None)
            if frames is not None:
                logger.info("VAEDecode: monolithic cache path, passing through decoded frames shape=%s", frames.shape)
                if isinstance(frames, np.ndarray):
                    if frames.ndim == 4:
                        return (frames,)
                    elif frames.ndim == 3:
                        return (frames[np.newaxis, ...],)
                    elif frames.ndim == 5:
                        return (frames[0],)
                elif isinstance(frames, mx.array):
                    arr = np.array(frames)
                    if arr.ndim == 4:
                        return (arr,)
                    elif arr.ndim == 3:
                        return (arr[np.newaxis, ...],)
                    elif arr.ndim == 5:
                        return (arr[0],)
            else:
                logger.warning("VAEDecode: _decoded_frames_key found but no cached frames")

        if not isinstance(vae, FusionVAEWrapper):
            logger.error("VAEDecode: received non-fusion VAE %s", type(vae))
            raise RuntimeError("VAEDecode override requires FusionVAEWrapper")

        logger.info("VAEDecode override: vae=%s", vae.model_name)

        raw = samples["samples"]
        mlx_latent = to_mlx_array(raw)

        engine = vae.get_engine()
        try:
            decoded = core.async_utils.run_async(
                self._decode_via_engine(engine, mlx_latent),
                timeout=300,
            )
        except Exception as e:
            logger.error("VAEDecode: decode failed: %s", e)
            raise

        if isinstance(decoded, mx.array):
            mx.eval(decoded)
            image_np = to_image_array(decoded)
        elif isinstance(decoded, np.ndarray):
            image_np = decoded
        else:
            logger.warning("VAEDecode: unexpected decode result type %s", type(decoded))
            image_np = np.zeros((1, 512, 512, 3), dtype=np.float32)

        logger.info("VAEDecode: output shape=%s dtype=%s", image_np.shape, image_np.dtype)
        return (image_np,)

    async def _decode_via_engine(self, engine, mlx_latent):
        await engine.ensure_started()
        try:
            return await engine._engine.decode(mlx_latent)
        except NotImplementedError:
            logger.warning("VAEDecode: backend decode not implemented, returning latent as-is")
            return mlx_latent


class VAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 64}),
            },
            "optional": {
                "temporal_size": ("INT", {"default": 64, "min": 1, "max": 4096}),
                "temporal_overlap": ("INT", {"default": 8, "min": 0, "max": 4096}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "model/latent"

    def decode(self, vae, samples, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8):
        from core.wrappers import FusionVAEWrapper
        from core.bridge import to_mlx_array, to_image_array
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()

        # Monolithic path: decoded frames already available from pipeline
        if "_decoded_frames" in samples or "_decoded_frames_key" in samples:
            vae_decode = VAEDecode()
            return vae_decode.decode(vae, samples)

        if not isinstance(vae, FusionVAEWrapper):
            logger.warning("VAEDecodeTiled: non-fusion VAE, falling back to VAEDecode")
            vae_decode = VAEDecode()
            return vae_decode.decode(vae, samples)

        logger.info("VAEDecodeTiled: vae=%s tile_size=%d temporal_size=%d", vae.model_name, tile_size, temporal_size)

        raw = samples["samples"]
        mlx_latent = to_mlx_array(raw)

        engine = vae.get_engine()
        try:
            decoded = core.async_utils.run_async(
                self._decode_tiled_via_engine(engine, mlx_latent, tile_size),
                timeout=600,
            )
        except Exception as e:
            logger.error("VAEDecodeTiled: tiled decode failed, falling back to non-tiled: %s", e)
            vae_decode = VAEDecode()
            return vae_decode.decode(vae, samples)

        if isinstance(decoded, mx.array):
            mx.eval(decoded)
            image_np = to_image_array(decoded)
        elif isinstance(decoded, np.ndarray):
            image_np = decoded
        else:
            logger.warning("VAEDecodeTiled: unexpected result type %s", type(decoded))
            image_np = np.zeros((1, 512, 512, 3), dtype=np.float32)

        logger.info("VAEDecodeTiled: output shape=%s dtype=%s", image_np.shape, image_np.dtype)
        return (image_np,)

    async def _decode_tiled_via_engine(self, engine, mlx_latent, tile_size):
        await engine.ensure_started()
        try:
            return await engine.decode_tiled(mlx_latent, tile_size=tile_size)
        except NotImplementedError:
            logger.warning("VAEDecodeTiled: backend decode_tiled not implemented, falling back to decode")
            return await engine._engine.decode(mlx_latent)


class FusionVAEDecoderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "latent": ("LATENT",),
                "tile_sample_min_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "Fusion-MLX/VAE"

    def decode(self, pipeline, latent, tile_sample_min_size=256):
        from core.bridge import to_mlx_array, to_image_array
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionVAEDecoder: tile_size=%d", tile_sample_min_size)

        raw_samples = latent["samples"]
        mlx_latent = to_mlx_array(raw_samples)

        try:
            decoded_mlx = core.async_utils.run_async(
                self._decode_staged(pipeline, mlx_latent, tile_sample_min_size),
                timeout=300,
            )
        except Exception as e:
            logger.error("FusionVAEDecoder: decode failed: %s", e)
            raise

        mx.eval(decoded_mlx)
        image_np = to_image_array(decoded_mlx)

        logger.info(
            "FusionVAEDecoder: output shape=%s dtype=%s",
            image_np.shape, image_np.dtype,
        )

        return (image_np,)

    async def _decode_staged(self, pipeline, mlx_latent, tile_sample_min_size):
        await pipeline.load_vae()
        try:
            result = await pipeline.decode_tiled(mlx_latent, tile_size=tile_sample_min_size)
        finally:
            await pipeline.unload_vae()
        return result
