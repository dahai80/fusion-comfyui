# H3 Sampling-Pipe Adaptation Design

**Goal:** Make AICF's native ComfyUI H3 workflows (h3-t2v/i2v/r2v) run on fusion-comfyui's MLX backend, so AICF can produce H3 video via fusion-comfyui instead of a separate CUDA ComfyUI.

**Architecture:** AICF submits ComfyUI API JSON (`POST /prompt`) using 7 H3-specific nodes absent from fusion-comfyui. MLX H3 generates one final muxed MP4 in a single `engine.generate()` call — incompatible with the native node flow (latents → split VAEDecode/VAEDecodeAudio → CreateVideo). So the H3 nodes are made **self-contained**: the conditioning node triggers generation and stashes the resulting MP4 path; downstream sampling/VAE/video nodes become passthroughs that forward the path; the SaveVideo override writes the final file AICF downloads via `/history`.

**Tech Stack:** Python, MLX, fusion-mlx engine (:11434), fusion-comfyui plugin (custom_nodes-style overrides), ComfyUI `io.ComfyNode` new-style node system.

**Spec:** this document.

## Global Constraints

- 4-space-multiple indent, no docstrings, default logging on every node
- Only modify `/Users/dahai/fusion/fusion-comfyui` (own project). Upstream fusion-mlx gaps filed as issues first, then PR.
- **H3 latent dims (verified `config.py` H3VAEConfig + `generate.py` `_latents_shape`): z_channels=24, spatial ÷16 (vae_ratio=16), temporal ÷4 causal (vae_ratio_t=4).** Latent shape `(1, 24, t, h//16, w//16)` where `t=(length-1)//4+1`. NOT 16 channels / ÷8.
- **H3 audio+image mutually exclusive** (`generate_video` raises): fl2va (i2va/l2va/keyframe) is video-only, only t2va (no image) may set `audio=True`. Conditioning nodes force `_h3_audio=False` when an image/last_frame is set.
- **H3 ref2va not wired upstream**: `MiniMaxH3Backend.generate()` does not forward `reference_images` to `generate_video()` (no ref2va branch). h3-r2v e2e is BLOCKED until fusion-mlx issue+PR. `_h3_ref_images` is staged by `MiniMaxH3ReferenceToVideo` but dropped at engine layer.
- fusion-mlx `VideoGenEngine.generate` missing `last_frame_image` forwarding — upstream fix needed (issue + PR)
- Real model load for any H3 test (33B minimax_h3), start/stop via `~/claude-home/fusion-mlx/start.sh start|stop`, download via `https://hf-mirror.com`
- Clean process data after verification, keep only final outputs + logs
- AICF reads final video via `/history` from the SaveVideo output node (meta.outputs node_id), keys `videos`/`images`

## Background — Verified Facts

1. **AICF submission model:** `src/lib/comfyui/client.ts` POSTs workflow JSON to `/prompt`, polls `/history/{id}`. Final video read from the `meta.outputs` node (SaveVideo, node 17 in t2v/i2v), under keys `videos` or `images` (animated). `OUTPUT_DIR=/tmp/aicf-out`. Downloaded, not streamed.

2. **AICF H3 workflow node graph (h3-t2v.json):**
   - `1 CLIPLoader` (clip_name=qwen3vl_32b...minimax_h3...awq, type=minimax)
   - `2 UNETLoader` (minimax_h3_fl2va_pruned_nvfp4)
   - `3 VAELoader` (minimax_h3_video_vae_fp16)
   - `4 VAELoader` (minimax_h3_audio_vae_fp32)
   - `5 MiniMaxH3SigmaShift` (model=[2,0], shift_video=12.0, shift_audio=3.0)
   - `6 CLIPTextEncode` (clip=[1,0], text=prompt)
   - `7 EmptyMiniMaxH3LatentAV` (width/height/length)
   - `8 MiniMaxH3ImageToVideo` (clip=[1,0], vae=[3,0], prompt, width/height/length, first_frame?, last_frame?)
   - `9 KSamplerSelect` (sampler_name=res_multistep)
   - `10 BasicScheduler` (model=[5,0], scheduler=simple, steps, denoise=1.0)
   - `11 RandomNoise` (noise_seed)
   - `12 BasicGuider` (model=[5,0], conditioning=[8,0])
   - `13 SamplerCustomAdvanced` (noise/guider/sampler/sigmas/latent_image=[8,1])
   - `14 VAEDecode` (samples=[13,0], vae=[3,0])
   - `15 VAEDecodeAudio` (samples=[13,0], vae=[4,0])
   - `16 CreateVideo` (images=[14,0], audio=[15,0], fps=24)
   - `17 SaveVideo` (video=[16,0], filename_prefix)

3. **h3-i2v.json** differs: node 8 `MiniMaxH3ImageToVideo` has `first_frame=[18,0]` + `last_frame=[20,0]` (LoadImage nodes), steps default 12. `h3-r2v` uses `MiniMaxH3ReferenceToVideo` with `ref_images`.

4. **fusion-comfyui existing overrides (work):** `CLIPLoader`, `UNETLoader`, `VAELoader`, `CLIPTextEncode`, `BasicGuider`, `BasicScheduler`, `KSamplerSelect`, `RandomNoise`, `SamplerCustomAdvanced`, `VAEDecode`. These are in `fusion_comfyui_plugin/nodes/{loaders,conditioning,passthrough,samplers,vae}.py` and registered in `__init__.py` NODE_CLASS_MAPPINGS.

5. **`SamplerCustomAdvanced` override (`samplers.py:529`):** extracts `model`+`conditioning` from guider dict, `noise_seed` from noise dict, calls `KSampler().sample()` → `_generate_monolithic()` → `engine._engine.generate(**gen_kwargs)` → returns decoded frames stashed via `_decoded_frames_key` → `VAEDecode` override passes them through as IMAGE. **This pipe works end-to-end for wan/hunyuan/cosmos.**

6. **fusion-mlx H3 backend (`engines/video_backends/minimax_h3.py:160`):** calls `generate_video(model_path, prompt, num_frames, width, height, fps, seed, num_inference_steps, output_path, quantize, audio, image, last_frame_image)`. Returns list of MP4 bytes. `audio=True` → `generate_t2va_av` joint denoise + ffmpeg mux into ONE mp4. `image`/`last_frame_image` mutually exclusive with `audio` (fl2va video-only; t2va audio+video).

7. **`VideoGenEngine.generate` (`engines/video.py:45`):** builds `VideoGenParams` from kwargs. Forwards `image`, `image_strength`, `audio`, `quantize`, `control_video`, `reference_images`. **Does NOT forward `last_frame_image`** — gap. `VideoGenParams` (base.py:82) HAS the field, backend (minimax_h3.py:154) reads `params.last_frame_image`, but engine.py drops it.

8. **New-style nodes:** `CreateVideo`/`SaveVideo` are `io.ComfyNode` (define_schema/execute), registered via `VideoExtension.get_node_list()` + `comfy_entrypoint()` in `comfy_extras/nodes_video.py`, NOT via NODE_CLASS_MAPPINGS. They consume `io.Video`/`io.Audio` types (`VideoFromComponents`, `VideoFromFile` in `comfy_api/latest/_input_impl/video_types.py`). `VAEDecodeAudio` does NOT exist in ComfyUI core — it's an AICF-required custom node (likely ComfyUI-VideoHelperSuite or similar) that the CUDA ComfyUI has installed.

9. **Audio architecture mismatch:** native flow = separate video latent (`VAEDecode`) + audio latent (`VAEDecodeAudio`) → `CreateVideo` muxes. MLX flow = single `engine.generate()` returns muxed MP4 (audio baked in when `audio=True`). There is no separate audio latent to feed `VAEDecodeAudio`.

## Gap Analysis — Nodes AICF Uses vs fusion-comfyui Overrides

| # | AICF node (h3-t2v/i2v/r2v) | fusion-comfyui status | Action |
|---|---|---|---|
| 1 | `CLIPLoader` (type=minimax) | OVERRIDE exists (loaders.py) | none — verify minimax type accepted |
| 2 | `UNETLoader` (minimax_h3_fl2va_pruned_nvfp4) | OVERRIDE exists (loaders.py) | none — verify h3 model id resolves |
| 3 | `VAELoader` (video_vae / audio_vae) | OVERRIDE exists (loaders.py) | none |
| 4 | `CLIPTextEncode` | OVERRIDE exists (conditioning.py) | none |
| 5 | `MiniMaxH3SigmaShift` | MISSING | create — passthrough returning model dict |
| 6 | `EmptyMiniMaxH3LatentAV` | MISSING | create — returns LATENT dict with dims |
| 7 | `MiniMaxH3ImageToVideo` (i2v) / `MiniMaxH3ReferenceToVideo` (r2v) | MISSING | create — conditioning nodes, carry first_frame/last_frame/ref into conditioning |
| 8 | `KSamplerSelect` | OVERRIDE exists (passthrough.py) | none |
| 9 | `BasicScheduler` | OVERRIDE exists (passthrough.py) | none |
| 10 | `RandomNoise` | OVERRIDE exists (passthrough.py) | none |
| 11 | `BasicGuider` | OVERRIDE exists (passthrough.py) | none |
| 12 | `SamplerCustomAdvanced` | OVERRIDE exists (samplers.py) — but routes via `engine.generate()` directly, NOT aware of H3's first_frame/last_frame/quantize/audio | extend — detect H3 conditioning keys, forward them |
| 13 | `VAEDecode` | OVERRIDE exists (vae.py) — passthrough of cached frames | none — will receive muxed path |
| 14 | `VAEDecodeAudio` | MISSING (not in ComfyUI core; custom node on CUDA box) | create — passthrough; audio already baked in MLX mp4 |
| 15 | `CreateVideo` | NEW-style `io.ComfyNode` in comfy_extras/nodes_video.py (ComfyUI core) | override — forward path; skip re-mux |
| 16 | `SaveVideo` | NEW-style `io.ComfyNode` in comfy_extras/nodes_video.py (ComfyUI core) | override — write final muxed MP4 to output dir so `/history` serves it |

**7 new/extended nodes:** `MiniMaxH3SigmaShift`, `EmptyMiniMaxH3LatentAV`, `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`, `VAEDecodeAudio`, `CreateVideo` (override), `SaveVideo` (override) + `SamplerCustomAdvanced` extension.

**Key architectural finding:** the existing `SamplerCustomAdvanced`/`KSampler`/`_generate_monolithic` pipe already calls `engine._engine.generate()`. For H3, the difference is the **kwargs set**: `quantize` (default `dit8_te4` for 33B to avoid jetsam OOM — verified in [[h3-drama-pipeline-landing]]), `audio=True` (t2va joint denoise), `image`/`last_frame_image` (i2v/fl2va). These must flow from the conditioning node → guider dict → SamplerCustomAdvanced → `_generate_monolithic` gen_kwargs.

**Why `SaveVideo` override is load-bearing:** AICF's `executor.ts` reads `meta.outputs` node_id (the SaveVideo node), pulls the file via `/history/{id}/output` under keys `videos`/`images`. ComfyUI's `/history` serves files saved to `output/`. The override must copy the MLX-generated muxed MP4 into `output/{filename_prefix}_00001.mp4` so the existing `/history` machinery serves it unchanged — AICF needs no changes.

## Design — Approach A (Self-Contained H3 Nodes)

**Principle:** reuse the existing `SamplerCustomAdvanced` → `KSampler.sample` → `_generate_monolithic` → `engine._engine.generate()` pipe as-is. The H3-specific differences are carried as **latent dict keys** (`_h3_*`) that `_generate_monolithic` already knows how to forward (mirroring the proven `_i2v_image_path` / `_vace_*` pattern). No separate generation path.

**Flow:**
1. `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` (conditioning node) builds the CONDITIONING dict (out 0) AND the LATENT dict (out 1). The LATENT dict carries `_h3_quantize`, `_h3_audio`, `_h3_first_frame_path`, `_h3_last_frame_path`, `_h3_ref_images` — paths written under `/tmp` from the incoming IMAGE tensors (same PIL-save pattern as `CosmosImageToVideoLatent:344`).
2. `BasicGuider` wraps `model` + `conditioning` (out 0) into the guider dict — unchanged.
3. `SamplerCustomAdvanced` receives guider + latent_image (out 1). It already extracts `model`/`conditioning`/`noise_seed` and calls `KSampler().sample()`. The H3 latent keys ride inside `latent_image` untouched.
4. `KSampler.sample` → `_generate_monolithic` detects `model_wrapper.model_type == "video"` + the `_h3_*` keys. It adds `quantize`, `audio`, `image`/`last_frame_image`, `reference_images` to `gen_kwargs` before `engine._engine.generate(**gen_kwargs)`.
5. MLX backend (`minimax_h3.py`) returns ONE muxed MP4 bytes array (audio already baked when `audio=True`).
6. `_generate_monolithic` decodes the MP4 via PyAV (existing path at line 203) → ndarray frames → stashed in `_decoded_frames_cache` with `_decoded_frames_key`.
7. `VAEDecode` override (vae.py) passes the cached frames through as IMAGE — unchanged.
8. `VAEDecodeAudio` override returns a **dummy audio** (single silent frame or empty) — audio is already in the MP4, the native split decode is not used. `CreateVideo` override forwards the IMAGE through, ignoring the dummy audio.
9. `SaveVideo` override receives the path. Since the muxed MP4 already exists as the engine output, the override **copies/writes the final MP4** to `output/{filename_prefix}_00001.mp4` and returns `{"ui": {"videos": [...]}}` so `/history` serves it — AICF downloads unchanged.

**Why not bypass sampling entirely in the conditioning node:** the conditioning node (out 0) feeds `BasicGuider`, and the sampler node is where AICF injects `seed`/`steps`/`cfg`/`sampler_name` from the meta inputs. Triggering generation in the conditioning node would lose those user-set params. Keeping generation in `_generate_monolithic` preserves the param flow and reuses battle-tested code. The `_h3_*` keys are the minimal glue.

**Quantize default:** 33B minimax_h3 needs `quantize="dit8_te4"` or jetsam OOM (verified [[h3-drama-pipeline-landing]]). The conditioning node sets `_h3_quantize="dit8_te4"` by default; override via a node input if a smaller model is used later.

**Upstream blocker (last_frame_image):** `VideoGenEngine.generate` does NOT forward `last_frame_image` to `VideoGenParams` even though the field + backend exist. For t2v/i2v without last_frame this is fine. For i2v WITH last_frame (AICF h3-i2v optional), file fusion-mlx issue + PR per project rules, gate behind a feature check locally until merged.

## Node Specs

All nodes live in a new file `fusion_comfyui_plugin/nodes/h3.py`, registered in `__init__.py` NODE_CLASS_MAPPINGS. Old-style classes (INPUT_TYPES/RETURN_TYPES/FUNCTION). 4-space indent, no docstrings, `logger = logging.getLogger("fusion_comfyui.nodes.h3")` at top.

### 1. `MiniMaxH3SigmaShift`

Passthrough — native node mutates model shift params; MLX backend hardcodes shift_video=12.0 / shift_audio=3.0 in `generate_t2va_av` (see [[h3-drama-pipeline-landing]]). Just forward the model so downstream `BasicGuider`/`BasicScheduler` get it.

```python
class MiniMaxH3SigmaShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "shift"
    CATEGORY = "Fusion-MLX/H3"

    def shift(self, model, shift_video=12.0, shift_audio=3.0):
        logger.info("MiniMaxH3SigmaShift: shift_video=%.1f shift_audio=%.1f (passthrough, MLX uses defaults)", shift_video, shift_audio)
        return (model,)
```

### 2. `EmptyMiniMaxH3LatentAV`

Returns LATENT dict with dims. H3 video VAE downsamples spatial 16x (vae_ratio=16), temporal 4x causal (vae_ratio_t=4), z_channels=24 (verified `config.py` H3VAEConfig + `generate.py` `_latents_shape`). Audio latent is separate in native flow but MLX bakes audio — so this just carries the video latent dims + the `_h3_audio=True` flag.

```python
class EmptyMiniMaxH3LatentAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 960, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 73, "min": 5, "max": 3600, "step": 17}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/H3"

    def generate(self, width=960, height=544, length=73, batch_size=1):
        # H3 video VAE: z_channels=24, spatial /16 (vae_ratio), temporal /4 causal (vae_ratio_t).
        # Verified in fusion_mlx/video/minimax_h3/config.py H3VAEConfig + generate.py _latents_shape.
        # t_latent=(length-1)//4+1. shape=(1, 24, t, h//16, w//16).
        t_latent = (length - 1) // 4 + 1
        latent = mx.zeros((batch_size, 24, t_latent, height // 16, width // 16), dtype=mx.float32)
        logger.info("EmptyMiniMaxH3LatentAV: shape=%s %dx%d frames=%d", latent.shape, width, height, length)
        return ({"samples": latent, "num_frames": length, "width": width, "height": height, "_h3_audio": True},)
```

### 3. `MiniMaxH3ImageToVideo`

Conditioning node. Two outputs: CONDITIONING (out 0, feeds BasicGuider) + LATENT (out 1, feeds SamplerCustomAdvanced latent_image). Carries `first_frame`/`last_frame` IMAGE → temp png paths under `/tmp` (PIL-save, same pattern as CosmosImageToVideoLatent). Sets `_h3_quantize`, `_h3_audio`, `_h3_first_frame_path`, `_h3_last_frame_path` on the LATENT dict.

```python
class MiniMaxH3ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 960, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 73, "min": 5, "max": 3600, "step": 17}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "quantize": (["dit8_te4", "dit8", "te4", "none"], {"default": "dit8_te4"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/H3"

    def generate(self, clip, vae, prompt="", width=960, height=544, length=73,
                 first_frame=None, last_frame=None, quantize="dit8_te4"):
        # H3 latent: z_channels=24, spatial /16, temporal /4. See EmptyMiniMaxH3LatentAV.
        # audio forced False: fl2va (image/last_frame) is video-only, audio+image mutually
        # exclusive (generate_video raises). Only t2va (no image) may set _h3_audio=True.
        t_latent = (length - 1) // 4 + 1
        latent = mx.zeros((1, 24, t_latent, height // 16, width // 16), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height,
                  "_h3_audio": False, "_h3_quantize": quantize}
        if first_frame is not None:
            result["_h3_first_frame_path"] = self._save_temp_image(first_frame, "h3_i2v_first")
        if last_frame is not None:
            result["_h3_last_frame_path"] = self._save_temp_image(last_frame, "h3_i2v_last")
        logger.info("MiniMaxH3ImageToVideo: %dx%d frames=%d quantize=%s first=%s last=%s",
                     width, height, length, quantize,
                     first_frame is not None, last_frame is not None)
        return ({"prompt": prompt}, result)

    @staticmethod
    def _save_temp_image(image, prefix):
        import tempfile
        from PIL import Image as PILImage
        from fusion_comfyui.core.bridge import to_numpy
        arr = to_numpy(image)
        if arr.ndim == 4:
            arr = arr[0]
        pil = PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, :3])
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"fusion_{prefix}_")
        pil.save(tmp.name)
        tmp.close()
        return tmp.name
```

### 4. `MiniMaxH3ReferenceToVideo`

Like ImageToVideo but for r2v (reference-to-video). Carries `ref_images` dict → temp paths list. `ref_image_size="match"` means keep ref resolution (MLX backend handles this). `audio_vae` input accepted but unused (audio baked in MLX).

```python
class MiniMaxH3ReferenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 960, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 73, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "512", "768", "960"], {"default": "match"}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "ref_images": ("IMAGE",),
                "quantize": (["dit8_te4", "dit8", "te4", "none"], {"default": "dit8_te4"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/H3"

    def generate(self, clip, vae, prompt="", width=960, height=544, length=73,
                 ref_image_size="match", audio_vae=None, ref_images=None, quantize="dit8_te4"):
        # H3 latent: z_channels=24, spatial /16, temporal /4. See EmptyMiniMaxH3LatentAV.
        # UPSTREAM GAP: MiniMaxH3Backend.generate() does NOT forward reference_images to
        # generate_video() — generate_video has no ref2va branch (only image/last_frame_image
        # keyframe fl2va path). _h3_ref_images is staged here but will be dropped at the engine
        # layer until fusion-mlx adds a ref2va branch (issue→PR→dep bump). h3-r2v e2e is BLOCKED.
        t_latent = (length - 1) // 4 + 1
        latent = mx.zeros((1, 24, t_latent, height // 16, width // 16), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height,
                  "_h3_audio": False, "_h3_quantize": quantize}
        if ref_images is not None:
            # ref_images may be a dict {ref_image_N: IMAGE} (r2v.json node 9) or a plain IMAGE
            refs = []
            if isinstance(ref_images, dict):
                for k in sorted(ref_images.keys()):
                    refs.append(MiniMaxH3ImageToVideo._save_temp_image(ref_images[k], "h3_r2v_ref"))
            else:
                arr = to_numpy(ref_images)
                if arr.ndim == 4:
                    for i in range(arr.shape[0]):
                        refs.append(MiniMaxH3ImageToVideo._save_temp_image(arr[i:i+1], "h3_r2v_ref"))
                else:
                    refs.append(MiniMaxH3ImageToVideo._save_temp_image(ref_images, "h3_r2v_ref"))
            result["_h3_ref_images"] = refs
        logger.info("MiniMaxH3ReferenceToVideo: %dx%d frames=%d quantize=%s refs=%d",
                     width, height, length, quantize, len(result.get("_h3_ref_images", [])))
        return ({"prompt": prompt}, result)
```

### 5. `VAEDecodeAudio`

Passthrough — audio already baked in the muxed MP4. Returns a dummy 1-frame silent audio tensor (shape `(1, 1, 2)` — 1 sample, 2 channels) so `CreateVideo`'s input contract holds. No real decode.

```python
class VAEDecodeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "Fusion-MLX/H3"

    def decode(self, samples, vae):
        # audio already muxed into the MLX-generated mp4; return silent dummy
        logger.info("VAEDecodeAudio: passthrough (audio baked in muxed mp4), waveformshape=1x2 silent")
        return ({"waveform": np.zeros((1, 2), dtype=np.float32), "sample_rate": 24000},)
```

### 6. `CreateVideo` override

Native `CreateVideo` (comfy_extras/nodes_video.py) muxes IMAGE + AUDIO → io.Video. Since MLX already produced the muxed MP4 and VAEDecode passes the frames through, this override forwards the IMAGE as a video object. Registered as OLD-style override in NODE_CLASS_MAPPINGS — this shadows the new-style core node (registration order: custom_nodes load after core, our `__init__.py` NODE_CLASS_MAPPINGS overrides win).

```python
class CreateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "create"
    CATEGORY = "Fusion-MLX/H3"

    def create(self, images, audio, fps=24.0):
        # frames already decoded from the muxed mp4; forward as-is
        logger.info("CreateVideo: passthrough frames=%s fps=%.1f (audio already in mp4)", images.shape, fps)
        return ({"images": images, "fps": fps, "audio": audio},)
```

### 7. `SaveVideo` override

Load-bearing for AICF. The native SaveVideo encodes io.Video → mp4 in output/. Since the MLX engine already produced the final muxed MP4 (stashed via `_decoded_frames_key` → but we need the raw bytes/path, not frames), the cleanest approach: **re-encode the passthrough IMAGE frames to mp4** using the existing `FusionSaveVideoNode._encode_video_av` helper, writing to `output/{filename_prefix}_00001.mp4`. This avoids needing the engine's raw MP4 path to reach SaveVideo. Returns `{"ui": {"videos": [{...}]}}` so `/history` serves the file.

```python
class SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "h3_t2v"}),
                "format": (["auto", "h264", "h265"], {"default": "auto"}),
                "codec": (["auto", "libx264", "libx265"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Fusion-MLX/H3"

    def save(self, video, filename_prefix="h3_t2v", format="auto", codec="auto"):
        import folder_paths
        from fusion_comfyui_plugin.nodes.video_io import FusionSaveVideoNode
        images = video["images"] if isinstance(video, dict) else video
        fps = video.get("fps", 24.0) if isinstance(video, dict) else 24.0
        out_codec = "libx264" if codec in ("auto", "libx264") else "libx265"
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, output_dir, images.shape[2], images.shape[1])
        file = f"{filename}_{counter:05d}_.mp4"
        filepath = os.path.join(full_output_folder, file)
        helper = FusionSaveVideoNode()
        helper._encode_video_av(images, filepath, int(fps), out_codec, 18)
        logger.info("SaveVideo(H3): saved %s frames=%s fps=%.1f", filepath, images.shape, fps)
        return {"ui": {"videos": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}
```

### 8. `SamplerCustomAdvanced` extension (modify `samplers.py`)

In `_generate_monolithic`, after the existing `gen_kwargs` block (line 187, before `engine._engine.generate`), add H3 key forwarding. Insert before the `try: result_raw = await engine._engine.generate`:

```python
        h3_quantize = latent_image.get("_h3_quantize")
        h3_audio = latent_image.get("_h3_audio", False)
        h3_first = latent_image.get("_h3_first_frame_path")
        h3_last = latent_image.get("_h3_last_frame_path")
        h3_refs = latent_image.get("_h3_ref_images")
        if h3_quantize:
            gen_kwargs["quantize"] = h3_quantize
        if h3_audio:
            gen_kwargs["audio"] = True
        if h3_first:
            gen_kwargs["image"] = h3_first
            if i2v_image:
                logger.warning("_generate_monolithic: H3 first_frame overrides _i2v_image_path")
        if h3_last:
            gen_kwargs["last_frame_image"] = h3_last
        if h3_refs:
            gen_kwargs["reference_images"] = h3_refs
        if h3_quantize or h3_audio or h3_first or h3_last or h3_refs:
            logger.info("_generate_monolithic: H3 mode quantize=%s audio=%s first=%s last=%s refs=%s",
                         h3_quantize, h3_audio, h3_first is not None, h3_last is not None,
                         len(h3_refs) if h3_refs else 0)
```

**`last_frame_image` forwarding caveat:** `VideoGenEngine.generate` does NOT pass `last_frame_image` to `VideoGenParams`. So `gen_kwargs["last_frame_image"]` will be silently dropped at the engine layer until the upstream PR lands. Gate: if `h3_last` is set and the engine doesn't accept it, the i2v will run video-only (last frame ignored) — acceptable degraded behavior, logged loudly. Upstream issue+PR is the real fix.

### Auxiliary nodes (AICF i2v/r2v also uses)

`LoadImage` — already registered (image.py). `ImageScaleToTotalPixels` — NOT registered; needed by h3-i2v.json node 18/20. `PrimitiveFloat` + `ComfyMathExpression` — needed by h3-r2v.json nodes 7/8 (length calc). Add minimal passthroughs:

```python
class ImageScaleToTotalPixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (["lanczos", "nearest-exact", "bilinear", "area"], {"default": "lanczos"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 32, "min": 1, "max": 1024}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Fusion-MLX/H3"

    def upscale(self, image, upscale_method="lanczos", megapixels=1.0, resolution_steps=32):
        from fusion_comfyui.core.bridge import to_numpy
        from PIL import Image as PILImage
        arr = to_numpy(image)
        if arr.ndim == 4:
            arr = arr[0]
        h, w = arr.shape[0], arr.shape[1]
        target = int(megapixels * 1024 * 1024)
        scale = (target / (h * w)) ** 0.5
        new_w = max(resolution_steps, int(round(w * scale / resolution_steps)) * resolution_steps)
        new_h = max(resolution_steps, int(round(h * scale / resolution_steps)) * resolution_steps)
        pil = PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, :3])
        pil = pil.resize((new_w, new_h), getattr(PILImage, upscale_method.upper().replace("-", ""), PILImage.LANCZOS))
        out = np.array(pil).astype(np.float32) / 255.0
        logger.info("ImageScaleToTotalPixels: %dx%d -> %dx%d (%.2fMP)", w, h, new_w, new_h, megapixels)
        return (out[np.newaxis, ...],)

class PrimitiveFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float": ("FLOAT", {"default": 5.0, "min": -1e9, "max": 1e9, "step": 0.01})}}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "emit"
    CATEGORY = "Fusion-MLX/H3"
    def emit(self, float=5.0):
        return (float,)

class ComfyMathExpression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"expression": ("STRING", {"default": "a", "multiline": False})},
            "optional": {"a": ("FLOAT",), "b": ("FLOAT",), "c": ("FLOAT",)},
        }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "eval_expr"
    CATEGORY = "Fusion-MLX/H3"
    def eval_expr(self, expression="a", a=None, b=None, c=None):
        import math
        env = {"a": a or 0.0, "b": b or 0.0, "c": c or 0.0, "max": max, "min": min, "round": round, "abs": abs, "math": math}
        val = float(eval(expression, {"__builtins__": {}}, env))
        logger.info("ComfyMathExpression: '%s' = %.4f", expression, val)
        return (val,)
```

### Registration (`__init__.py` NODE_CLASS_MAPPINGS)

```python
"MiniMaxH3SigmaShift": MiniMaxH3SigmaShift,
"EmptyMiniMaxH3LatentAV": EmptyMiniMaxH3LatentAV,
"MiniMaxH3ImageToVideo": MiniMaxH3ImageToVideo,
"MiniMaxH3ReferenceToVideo": MiniMaxH3ReferenceToVideo,
"VAEDecodeAudio": VAEDecodeAudio,
"CreateVideo": CreateVideo,
"SaveVideo": SaveVideo,
"ImageScaleToTotalPixels": ImageScaleToTotalPixels,
"PrimitiveFloat": PrimitiveFloat,
"ComfyMathExpression": ComfyMathExpression,
```

## Upstream fusion-mlx Gap

**`VideoGenEngine.generate` drops `last_frame_image`.** `VideoGenParams` (base.py:82) has `last_frame_image: str | None = None`; `minimax_h3.py:154` reads `params.last_frame_image` and passes it to `generate_video(last_frame_image=cond_last)`. But `engines/video.py:45` `VideoGenEngine.generate` builds `VideoGenParams` from a fixed kwarg set that omits `last_frame_image`. So any `last_frame_image` sent to `engine._engine.generate()` is silently ignored.

**Impact:** AICF h3-i2v with optional `last_frame` produces video-only without honoring the last frame (degraded, not broken). h3-t2v (no last_frame) unaffected. h3-r2v unaffected (uses reference_images, which IS forwarded).

**Process (project rule: 先提issue，再提PR，跟着提交落地code):**
1. File fusion-mlx issue: `VideoGenEngine.generate does not forward last_frame_image to VideoGenParams`. Cite base.py:82 + minimax_h3.py:154 + video.py:45.
2. Open fusion-mlx PR: add `last_frame_image=kwargs.get("last_frame_image")` to the `VideoGenParams(...)` construction in `engines/video.py`, mirroring the existing `image=` forwarding.
3. Land in fusion-comfyui: bump dependency floor in `pyproject.toml` once the PR version publishes.

**Local gate until merged:** `_generate_monolithic` sends `gen_kwargs["last_frame_image"]` regardless. If the installed fusion-mlx ignores it, behavior degrades to video-only-i2v (logged). No local code change needed to match the upstream — the kwarg is simply dropped upstream.

### Upstream Gap #2 — ref2va (reference_images) not wired

**`MiniMaxH3Backend.generate()` does not forward `reference_images` to `generate_video()`.** `VideoGenParams.reference_images` (base.py) is set from kwargs and `VideoGenEngine.generate` forwards it, but `MiniMaxH3Backend.generate` (minimax_h3.py ~line 124-202) constructs the `generate_video(...)` call with only `image`/`last_frame_image` — there is **no `reference_images=` argument and no ref2va branch** in `generate_video` (generate.py:645-746). `reference_audio` is explicitly rejected with `ValueError` (issue #589). `H3Partition.REF2VA` exists in config + backend `__init__` (path hint: model path containing `ref2va` auto-switches partition), but `generate()` is partition-agnostic at the call site — ref2va partition is scaffolding-only.

**Impact:** AICF h3-r2v (multi-reference-to-video) **cannot run** — `_h3_ref_images` staged by `MiniMaxH3ReferenceToVideo` reaches `_generate_monolithic` → `gen_kwargs["reference_images"]` → `engine._engine.generate(reference_images=...)` → `VideoGenParams.reference_images` set → `MiniMaxH3Backend.generate` **drops it** → `generate_video` runs t2va (no image/last_frame) ignoring refs. Output is a t2v video, not a ref-conditioned video.

**Process (project rule: 先提issue，再提PR，跟着提交落地code):**
1. File fusion-mlx issue: `MiniMaxH3Backend.generate drops reference_images — ref2va partition not wired to generate_video`. Cite minimax_h3.py generate() call + generate.py:645-746 (no ref2va branch) + config.py `H3Partition.REF2VA`.
2. Open fusion-mlx PR: add a ref2va branch in `generate_video` (new `generate_ref2va_video` or extend `generate_fl2va_video` with multi-ref conditioning) + forward `reference_images` in `MiniMaxH3Backend.generate`.
3. Land in fusion-comfyui: bump dep floor once published; flip `test_h3_r2v_e2e` from xfail to pass.

**Local gate until merged:** `MiniMaxH3ReferenceToVideo` still stages `_h3_ref_images` + `_generate_monolithic` still forwards `reference_images` so no local change is needed post-merge — the kwarg simply becomes effective once the engine consumes it. h3-r2v e2e stays `xfail(strict=True)` until then.

## Testing Plan

### Unit tests (no model load) — `tests/test_h3_nodes.py`

1. `test_minimax_h3_sigma_shift_passthrough` — node returns model unchanged, logs shift values.
2. `test_empty_h3_latent_av_shape` — `EmptyMiniMaxH3LatentAV(width=960,height=544,length=73)` → latent shape `(1,24,19,34,60)` (z_channels=24, h//16=34, w//16=60, t=(73-1)//4+1=19), `_h3_audio=True`.
3. `test_h3_image_to_video_conditioning_dict` — returns 2-tuple; out 0 is `{"prompt": ...}`; out 1 has `_h3_quantize`, `_h3_audio=False`, no `_h3_first_frame_path` when first_frame=None.
4. `test_h3_image_to_video_first_frame_temp_path` — pass a fake IMAGE (np zeros `(1,64,64,3)`), assert `_h3_first_frame_path` is a `/tmp/...png` that exists on disk.
5. `test_h3_reference_to_video_ref_images_dict` — pass `ref_images={"ref_image_1": <img>}`, assert `_h3_ref_images` is a 1-element list of existing png paths.
6. `test_vae_decode_audio_returns_silent` — returns dict with `waveform` shape `(1,2)` zeros.
7. `test_create_video_passthrough` — forwards images+fps+audio into a dict.
8. `test_save_video_writes_mp4` — monkeypatch `FusionSaveVideoNode._encode_video_av` to no-op, assert SaveVideo returns `{"ui": {"videos": [{"filename": ...}]}}` with `.mp4` extension and `type: output`.
9. `test_image_scale_to_total_pixels` — 64x64 @ 1MP → output 1024x1024 (resolution_steps=32).
10. `test_comfy_math_expression` — `"max(5, round(a*24))"` with a=5 → 120.
11. `test_generate_monolithic_h3_kwargs_forwarding` — call `_generate_monolithic` with a fake engine capturing `generate(**kwargs)`; latent_image with `_h3_quantize`/`_h3_audio`/`_h3_first_frame_path`/`_h3_ref_images`; assert kwargs has `quantize`, `audio=True`, `image`, `reference_images`.

### E2E (real model load, 33B minimax_h3) — `tests/test_h3_e2e.py` (marked `@pytest.mark.inference`)

**Setup:** `~/claude-home/fusion-mlx/start.sh start`; confirm 33B `minimax_h3_fl2va_pruned_nvfp4` present (download via `https://hf-mirror.com` if missing — see [[h3-drama-pipeline-landing]]).

1. `test_h3_t2v_e2e` — load h3-t2v.json, set prompt, POST /prompt, poll /history, assert output mp4 exists, `ffprobe` shows video stream + audio stream (audio baked), duration ~3s (73 frames @ 24fps). **Quality gate:** `ffmpeg -i out.mp4 -filter:v psnr` against a baseline OR `mean_volume -18dB` range (per [[h3-drama-pipeline-landing]] F1 verification).
2. `test_h3_i2v_e2e` — h3-i2v.json with first_frame only (last_frame gated on upstream PR), assert mp4 with **video stream only** (fl2va keyframe is video-only; audio+image are mutually exclusive in `generate_video`, no audio baked), frames show the first_frame content in frame 0 (pixel sample matches input image within tolerance).
3. `test_h3_r2v_e2e` — **BLOCKED upstream**: `MiniMaxH3Backend.generate()` does not forward `reference_images` to `generate_video()` (no ref2va branch). `pytest.mark.xfail(strict=True, reason="fusion-mlx ref2va not wired — issue→PR→dep bump")`. Re-enable once fusion-mlx PR lands + dep floor bumped. Until then `_h3_ref_images` is staged by the node but dropped at engine layer.

**Cleanup (project rule):** after each e2e, delete temp pngs (`/tmp/fusion_h3_*`), keep only the final mp4 + the test log. `~/claude-home/fusion-mlx/start.sh stop` at suite end.

### Smoke test — startup node count

After registration, start ComfyUI, hit `/object_info` — assert all 10 new node names appear. Assert existing node count (96+ mappings per [[p2-fork-image-numpy-2026-08-26]]) increases by 10.

## Rollout / Cleanup

**Build order:**
1. Create `fusion_comfyui_plugin/nodes/h3.py` with all 10 nodes + `_save_temp_image` helper.
2. Register 10 nodes in `__init__.py` NODE_CLASS_MAPPINGS (after existing video entries, ~line 232).
3. Extend `samplers.py:_generate_monolithic` with the H3 gen_kwargs block (before the `engine._engine.generate` try).
4. Run `ruff check fusion_comfyui_plugin/nodes/h3.py fusion_comfyui_plugin/__init__.py fusion_comfyui_plugin/nodes/samplers.py` — green.
5. Write + run unit tests `tests/test_h3_nodes.py` — green.
6. Commit: `feat: add H3 sampling-pipe nodes for AICF (MiniMaxH3* + video overrides)`.
7. Upstream fusion-mlx: file issue + PR for `last_frame_image` forwarding. Wait for merge/publish.
8. Bump `pyproject.toml` dep floor to the published fusion-mlx version with the fix.
9. E2E real-model test (cost-critical — 33B, schedule once): h3-t2v, h3-i2v (first_frame), h3-r2v.
10. Update `README.md` with H3 node list + AICF usage note.
11. Compact (project rule: after each Task).

**Files touched:**
- Create: `fusion_comfyui_plugin/nodes/h3.py`
- Modify: `fusion_comfyui_plugin/__init__.py` (imports + 10 mappings + display names)
- Modify: `fusion_comfyui_plugin/nodes/samplers.py` (`_generate_monolithic` H3 block)
- Create: `tests/test_h3_nodes.py`
- Create: `tests/test_h3_e2e.py` (inference-marked, auto-skip in CI)
- Modify: `README.md`
- Modify: `pyproject.toml` (dep floor bump, post-upstream-merge)

**What stays throwaway (cleaned post-verify):** temp pngs under `/tmp/fusion_h3_*`, any scratch e2e outputs not under `output/`, intermediate probe scripts. Final outputs kept: the e2e mp4s + test logs + the committed node code.

**Risk / fallback:** if `CreateVideo`/`SaveVideo` old-style override does NOT shadow the new-style core node (registration-order edge case), fallback = register a ComfyExtension from `fusion_comfyui_plugin` returning these names so they win. Verify at smoke-test step; if `/object_info` shows the core CreateVideo signature, escalate to ComfyExtension wiring before e2e.

**Out of scope (separate problems):**
- AICF Qwen2.5 image-edit nodes (UnetLoaderGGUF/FluxKontext*) — MLX-unsupported, not H3.
- AICF `last_frame` full e2e with upstream fix — gated, tested after PR lands.
- ref2va `reference_audio` — fusion-mlx issue #589 still open (not implemented upstream).
- ref2va ref IMAGES — **also upstream-blocked** (see Upstream Gap #2): `MiniMaxH3Backend.generate` does not forward `reference_images`, so h3-r2v with ref images is NOT runnable today despite the node staging them. Node + forwarding code land now (degrades to t2va); e2e flips from xfail to pass once fusion-mlx PR lands.
