# ComfyUI-Fusion-MLX

General-purpose ComfyUI nodes running on Apple Silicon via fusion-mlx / MLX. No PyTorch runtime dependency for inference — models run pure MLX/Metal.

## Node Overview

| Node | Category | Type | Description |
|------|----------|------|-------------|
| FusionModelLoader | Loaders | Staged | Load image/video model as FUSION_PIPELINE |
| FusionTextEncoder | Conditioning | Staged | Encode prompt → FUSION_COND |
| FusionKSampler | Sampling | Staged | Denoise latent with staged load/unload |
| FusionVAEDecoder | VAE | Staged | Decode latent → IMAGE |
| **FusionImageGen** | Shortcuts | One-shot | pipeline + prompt → IMAGE (txt2img) |
| **FusionVideoGen** | Shortcuts | One-shot | pipeline + prompt → IMAGE frames (txt2vid) |
| **FusionImageToVideo** | Shortcuts | One-shot | pipeline + image + prompt → IMAGE frames (img2vid) |
| **FusionEmptyLatent** | Latent | Utility | Create empty latent (image or video) |
| **FusionSaveVideo** | Video | Output | IMAGE frames → MP4/WEBM file |
| **FusionVideoConcat** | Video | Utility | Concatenate two video frame batches |
| **FusionSubtitleOverlay** | PostProcess | Utility | Burn subtitle text onto IMAGE/VIDEO |
| **FusionVoiceLoader** | Voice | Loader | Load TTS model (Kokoro/F5-TTS) as FUSION_TTS |
| **FusionVoiceSynthesize** | Voice | TTS | Text → AUDIO with optional voice/speed/ref_audio |
| **FusionVoiceClone** | Voice | TTS | Text + ref_audio → AUDIO (voice cloning) |
| **FusionSaveAudio** | Voice | Output | AUDIO → WAV file |
| **FusionIdentityLoader** | Identity | Loader | Load PuLID face identity model |
| **FusionIdentityApply** | Identity | Extract | Reference image → FUSION_IDENTITY_EMBED |
| **FusionIdentityGenerate** | Identity | Generate | Reference + prompt → identity-locked image |
| **FusionIdentityPipeline** | Shortcuts | One-shot | One-stop identity generation (load+extract+generate) |
| **FusionIPAdapterLoader** | IP-Adapter | Loader | Load IP-Adapter + SigLIP vision model |
| **FusionIPAdapterApply** | IP-Adapter | Extract | Image → FUSION_IP_ADAPTER_EMBED (vision embeddings) |
| **FusionIPAdapterInject** | IP-Adapter | Generate | Pipeline + embed → IP-Adapter conditioned image |
| **FusionLipsyncLoader** | Talking-Head | Loader | Load LatentSync lip-sync model |
| **FusionLipsyncApply** | Talking-Head | Generate | Video + audio → lip-synced video frames |

## Two Usage Patterns

### Pattern A: Staged Pipeline (Advanced)

Fine-grained control over each pipeline stage. Matches ComfyUI's native Load → Encode → Sample → Decode pattern.

```
FusionModelLoader → FusionTextEncoder → FusionKSampler → FusionVAEDecoder → SaveImage
                         ↓                    ↑                ↑
                   FUSION_COND          FusionEmptyLatent    FUSION_PIPELINE
```

### Pattern B: Shortcut Nodes (Simple)

One-shot generation with a single node. Uses the monolithic engine.generate() path internally.

```
FusionModelLoader → FusionImageGen → SaveImage       (txt2img)
FusionModelLoader → FusionVideoGen → FusionSaveVideo  (txt2vid)
FusionModelLoader → FusionImageToVideo → FusionSaveVideo  (img2vid)
```

## Node Details

### FusionModelLoader

Load a fusion-mlx model as a FUSION_PIPELINE wrapper.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model_name | combo | - | Model name from registry |
| offload_strategy | combo | sequential | Memory offload strategy |
| quant_bit | combo | fp8_e4m3 | Quantization (fp8_e4m3, 4bit, fp16) |

**Output**: FUSION_PIPELINE

### FusionTextEncoder

Encode text prompt into conditioning.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded pipeline |
| prompt | STRING | "" | Positive prompt |
| negative_prompt | STRING | "" | Negative prompt |

**Output**: FUSION_COND

### FusionKSampler

Denoise latent using staged DiT load/unload.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded pipeline |
| positive | FUSION_COND | - | Positive conditioning |
| negative | FUSION_COND | - | Negative conditioning |
| latent_image | LATENT | - | Input latent |
| steps | INT | 20 | Sampling steps |
| cfg | FLOAT | 6.0 | Classifier-free guidance |
| seed | INT | 42 | Random seed |
| width | INT | 1024 | Image width |
| height | INT | 1024 | Image height |
| num_frames | INT | 1 | Video frames (1=image) |

**Output**: LATENT

### FusionVAEDecoder

Decode latent to image using staged VAE load/unload.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded pipeline |
| latent | LATENT | - | Latent to decode |
| tile_sample_min_size | INT | 256 | Tile size for tiled decode |

**Output**: IMAGE

### FusionImageGen (Shortcut)

One-shot text-to-image generation.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded pipeline |
| prompt | STRING | "" | Positive prompt |
| negative_prompt | STRING | "" | Negative prompt |
| width | INT | 1024 | Image width |
| height | INT | 1024 | Image height |
| steps | INT | 20 | Sampling steps |
| cfg | FLOAT | 6.0 | CFG strength |
| seed | INT | 42 | Random seed |

**Output**: IMAGE

### FusionVideoGen (Shortcut)

One-shot text-to-video generation.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded pipeline |
| prompt | STRING | "" | Positive prompt |
| negative_prompt | STRING | "" | Negative prompt |
| width | INT | 768 | Video width |
| height | INT | 512 | Video height |
| num_frames | INT | 41 | Number of frames |
| fps | INT | 24 | Frames per second |
| steps | INT | 30 | Sampling steps |
| cfg | FLOAT | 5.0 | CFG strength |
| seed | INT | 42 | Random seed |

**Output**: IMAGE (frame batch)

### FusionImageToVideo (Shortcut)

One-shot image-to-video generation. Mirrors ComfyUI's WanImageToVideo / CosmosImageToVideoLatent pattern.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded pipeline |
| image | IMAGE | - | Start image |
| prompt | STRING | "" | Positive prompt |
| negative_prompt | STRING | "" | Negative prompt |
| width | INT | 768 | Video width |
| height | INT | 512 | Video height |
| num_frames | INT | 41 | Number of frames |
| fps | INT | 24 | Frames per second |
| steps | INT | 30 | Sampling steps |
| cfg | FLOAT | 5.0 | CFG strength |
| seed | INT | 42 | Random seed |

**Output**: IMAGE (frame batch)

### FusionEmptyLatent

Create empty latent for image or video generation. Mirrors EmptyLatentImage / EmptyHunyuanLatentVideo.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| width | INT | 1024 | Image/video width |
| height | INT | 1024 | Image/video height |
| batch_size | INT | 1 | Batch size |
| num_frames | INT | 1 | Frames (1=image, >1=video) |

**Output**: LATENT (16 channels, spatial /2, temporal /4 for video)

### FusionSaveVideo

Save IMAGE frame batch as video file. Mirrors SaveWEBM.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| images | IMAGE | - | Frame batch |
| filename_prefix | STRING | FusionVideo | Output filename prefix |
| fps | INT | 24 | Frames per second |
| codec | combo | libx264 | Video codec (libx264/libvpx-vp9/libx265) |
| crf | INT | 18 | Constant rate factor (quality) |
| audio_file | STRING | "" | Optional audio file path |

**Output**: None (OUTPUT_NODE)

### FusionVideoConcat

Concatenate two video frame batches.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| video_a | IMAGE | - | First video frames |
| video_b | IMAGE | - | Second video frames |

**Output**: IMAGE (concatenated frames)

### FusionSubtitleOverlay

Burn subtitle text onto image/video frames.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| images | IMAGE | - | Input frames |
| text | STRING | "" | Subtitle text |
| font_size | INT | 36 | Font size |
| position | combo | bottom | Position (bottom/top/center) |
| margin | INT | 40 | Margin from edge |
| font_color | STRING | white | Text color (name or #hex) |
| stroke_color | STRING | black | Stroke color |
| stroke_width | INT | 2 | Stroke width |
| bg_opacity | FLOAT | 0.5 | Background box opacity |
| max_width_ratio | FLOAT | 0.9 | Max text width ratio |
| font_path | STRING | "" | Custom font path (optional) |

**Output**: IMAGE

### FusionVoiceLoader

Load a TTS model as a FUSION_TTS engine wrapper. Uses fusion-mlx TTSEngine if available, falls back to direct mlx_audio.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model_name | STRING | mlx-community/kokoro-82m | TTS model name (HF repo or local path) |

**Output**: FUSION_TTS

### FusionVoiceSynthesize

Synthesize speech from text. Supports voice selection, speed control, and optional reference audio for voice cloning.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| tts_engine | FUSION_TTS | - | Loaded TTS engine |
| text | STRING | "" | Text to synthesize |
| voice | STRING | af_heart | Voice style |
| speed | FLOAT | 1.0 | Playback speed (0.5-3.0) |
| ref_audio | STRING | "" | Reference audio for voice cloning |
| ref_text | STRING | "" | Caption for reference audio |
| temperature | FLOAT | 0.7 | Sampling temperature |

**Output**: (AUDIO, STRING) - audio array + wav file path

### FusionVoiceClone

One-shot voice cloning: synthesize text in the voice of a reference audio.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| tts_engine | FUSION_TTS | - | Loaded TTS engine |
| text | STRING | "" | Text to synthesize |
| ref_audio | STRING | - | Reference audio path (required) |
| ref_text | STRING | "" | Caption for reference audio |
| speed | FLOAT | 1.0 | Playback speed |
| temperature | FLOAT | 0.7 | Sampling temperature |

**Output**: (AUDIO, STRING) - audio array + wav file path

### FusionSaveAudio

Save AUDIO array as WAV file.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| audio | AUDIO | - | Audio array |
| filename_prefix | STRING | FusionAudio | Output filename prefix |
| sample_rate | INT | 24000 | Sample rate |

**Output**: STRING (wav file path)

### FusionIdentityLoader

Load PuLID face identity model from fusion-mlx. Uses IDFormer + EVA-CLIP + InsightFace.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model_name | combo | pulid_flux_v0.9.1 | PuLID model name |
| dtype | combo | float16 | Model dtype (float16/bfloat16/float32) |

**Output**: FUSION_IDENTITY_MODEL

### FusionIdentityApply

Extract face identity embedding from a reference image using PuLID.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| identity_model | FUSION_IDENTITY_MODEL | - | Loaded PuLID model |
| image | IMAGE | - | Reference face image |
| weight | FLOAT | 1.0 | Identity strength (0.0-2.0) |
| start_at | FLOAT | 0.0 | Start denoising step ratio |
| end_at | FLOAT | 1.0 | End denoising step ratio |

**Output**: FUSION_IDENTITY_EMBED

### FusionIdentityGenerate

One-shot identity-locked generation: reference photo + text prompt → character image.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded generation pipeline |
| identity_model | FUSION_IDENTITY_MODEL | - | Loaded PuLID model |
| reference_image | IMAGE | - | Reference face photo |
| prompt | STRING | "" | Text prompt |
| negative_prompt | STRING | "" | Negative prompt |
| width | INT | 1024 | Image width |
| height | INT | 1024 | Image height |
| steps | INT | 20 | Sampling steps |
| cfg | FLOAT | 6.0 | CFG strength |
| identity_weight | FLOAT | 1.0 | Identity strength |
| seed | INT | 42 | Random seed |

**Output**: IMAGE

### FusionIdentityPipeline (Shortcut)

One-stop identity generation. Loads PuLID, extracts face, generates identity-locked image in a single node.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded generation pipeline |
| reference_image | IMAGE | - | Reference face photo |
| prompt | STRING | "" | Text prompt |
| negative_prompt | STRING | "" | Negative prompt |
| width | INT | 1024 | Image width |
| height | INT | 1024 | Image height |
| steps | INT | 20 | Sampling steps |
| cfg | FLOAT | 6.0 | CFG strength |
| identity_weight | FLOAT | 1.0 | Identity strength |
| seed | INT | 42 | Random seed |

**Output**: IMAGE

### FusionIPAdapterLoader

Load IP-Adapter-Flux model (InstantX architecture) with SigLIP vision encoder. Produces a pipeline that encodes reference images into cross-attention embeddings for injection into the Flux DiT.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| ipadapter | combo | ip_adapter_flux.safetensors | IP-Adapter checkpoint file |
| siglip_model | combo | siglip-so400m-patch14-384 | SigLIP vision model name |
| num_tokens | INT | 128 | Number of IP-Adapter tokens |
| dtype | combo | float16 | Model dtype |

**Output**: FUSION_IP_ADAPTER_MODEL

### FusionIPAdapterApply

Encode a reference image into IP-Adapter embeddings using the loaded pipeline. Controls weight and timestep range for the injection.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| ip_adapter_model | FUSION_IP_ADAPTER_MODEL | - | Loaded IP-Adapter pipeline |
| image | IMAGE | - | Reference image |
| weight | FLOAT | 1.0 | IP-Adapter strength (-1.0 to 5.0) |
| start_percent | FLOAT | 0.0 | Start timestep % (0.0-1.0) |
| end_percent | FLOAT | 1.0 | End timestep % (0.0-1.0) |

**Output**: FUSION_IP_ADAPTER_EMBED

### FusionIPAdapterInject

Inject IP-Adapter embeddings into a Fusion-MLX image pipeline and generate an image. Patches the Flux DiT transformer with IP cross-attention during denoising, then restores the original transformer after generation.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | FUSION_PIPELINE | - | Loaded generation pipeline |
| ip_adapter_embed | FUSION_IP_ADAPTER_EMBED | - | IP-Adapter embeddings from Apply node |
| prompt | STRING | "" | Text prompt |
| negative_prompt | STRING | "" | Negative prompt |
| width | INT | 1024 | Image width |
| height | INT | 1024 | Image height |
| steps | INT | 20 | Sampling steps |
| cfg | FLOAT | 4.0 | CFG strength |
| seed | INT | 42 | Random seed |

**Output**: IMAGE

### FusionLipsyncLoader

Load LatentSync MLX lip-sync pipeline.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model_name | combo | latentsync_unet | LatentSync model name |
| dtype | combo | float16 | Model dtype |

**Output**: FUSION_LIPSYNC_MODEL

### FusionLipsyncApply

Apply lip-sync to a video using an audio track. Outputs synchronized video frames.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| lipsync_model | FUSION_LIPSYNC_MODEL | - | Loaded LatentSync model |
| video_path | STRING | "" | Input video file path |
| audio_path | STRING | "" | Input audio file path |
| audio | AUDIO | (optional) | Audio from FusionVoiceSynthesize |
| output_fps | INT | 25 | Output video FPS |
| num_inference_steps | INT | 20 | Denoising steps |
| guidance_scale | FLOAT | 1.0 | Guidance strength |
| seed | INT | 42 | Random seed |

**Output**: IMAGE (video frames)

## Supported Models

| Model | Type | Quantization | Tested |
|-------|------|-------------|--------|
| FLUX.2-klein-base-4B | Image (T2I) | fp16 | ✅ |
| Wan2.2-5B | Video (T2V/I2V) | fp16 | ✅ |
| Wan2.2-14B | Video (T2V) | fp16 | pending model download |
| Wan2.2-TI2V-5B-mlx-q8 | Video (TI2V) | q8 | pending model download |
| Wan2.1-1.3B | Video (T2V) | fp16 | ✅ |
| Wan2.1-14B | Video (T2V) | fp16 | pending model download |
| LTX-Video | Video (T2V) | fp16 | ✅ |
| ltx-2.3-mlx-q8 | Video (T2V) | q8 | discoverable |
| SkyReels-V3-14B-mxfp8 | Video (V2V) | fp8 | discoverable |
| SkyReels-V3-A2V-19B-MLX | Video (A2V) | MLX | discoverable |
| SkyReels-V3-R2V-14B-MLX | Video (R2V) | MLX | discoverable |
| SkyReels-V3-V2V-14B-MLX | Video (V2V) | MLX | discoverable |
| Cosmos-7B | Video | - | ❌ no backend |
| HunyuanVideo | Video | - | ❌ no backend |
| stable-video-diffusion-img2vid-xt | Video (I2V) | fp16 | routed (pending model download) |
| stable-video-diffusion-img2vid | Video (I2V) | fp16 | routed (pending model download) |

### Native Node Overrides

The following ComfyUI native nodes are overridden to route through fusion-mlx:

- **CheckpointLoaderSimple** → loads model as FusionModelWrapper
- **UNETLoader** → maps unet name to fusion-mlx model (supports wan, ltx, ltx-2, skyreels subtypes, flux, cosmos, hunyuan, svd)
- **CLIPLoader** → maps clip type to fusion-mlx model
- **VAELoader** → maps vae name to fusion-mlx model
- **KSampler** → calls fusion-mlx monolithic generate()
- **EmptyLatentImage** → creates latent with fusion-mlx compatible dims
- **SaveWEBM / SaveAnimatedWEBP** → save video output

### Model Name Mapping

When using native ComfyUI nodes (e.g. UNETLoader), model names are auto-mapped:

| ComfyUI Name Pattern | Fusion-MLX Model |
|---------------------|------------------|
| wan2.2-14b* | Wan2.2-14B |
| wan2.2-ti2v* | Wan2.2-TI2V-5B-mlx-q8 |
| wan2.2* / wan22* | Wan2.2-5B |
| wan2.1-14b* | Wan2.1-14B |
| wan2.1* | Wan2.1-1.3B |
| ltx-2* / ltx_2* / ltx-2.3* | ltx-2.3-mlx-q8 |
| ltx* | LTX-Video |
| skyreels*a2v* / skyreels*19b* | SkyReels-V3-A2V-19B-MLX |
| skyreels*r2v* | SkyReels-V3-R2V-14B-MLX |
| skyreels*v2v* | SkyReels-V3-V2V-14B-MLX |
| skyreels* (default) | SkyReels-V3-V2V-14B-MLX |
| flux*4b* | FLUX.2-klein-base-4B |
| flux*klein* | FLUX.2-klein-9b |
| cosmos* | Cosmos-7B |
| hunyuan* | HunyuanVideo |

Models under `~/.fusion-mlx/models/Skywork/` and `~/.fusion-mlx/models/dgrauet/` subdirectories are also discovered automatically.

### Test Results (2026-07-25)

| Workflow | Model | Frames | Resolution | Time |
|----------|-------|--------|------------|------|
| LTX-Video T2V | LTX-Video | 97 | 768x512 | ~92s |
| Wan2.2 T2V | Wan2.2-5B | 49 | 832x480 | ~217s |
| Wan2.2 I2V | Wan2.2-5B | 49 | 832x480 | ~278s |
| Wan2.2 T2V (UNETLoader) | Wan2.2-5B | 41 | 832x480 | ~178s |
| Wan2.1-1.3B T2V | Wan2.1-1.3B | 33 | 832x480 | ~97s |
| Flux2 T2I | FLUX.2-klein-base-4B | 1 | 1024x1024 | ~45s |

### Unit Tests

315 tests, 0 failures, 3 skipped (as of 2026-07-25). Coverage includes:
- Model/CLIP/VAE/Conditioning wrappers
- Name mapping functions (checkpoint, unet, clip, vae) with SkyReels/LTX-2/TI2V subtypes
- Engine wrapper (staged pipeline, monolithic fallback)
- Video I/O (WEBM, animated WEBP, MP4)
- Voice/TTS nodes (loader, synthesize, clone, save)
- Identity/PuLID nodes (loader, apply, generate, image-to-bgr)
- IP-Adapter nodes (loader, apply, inject, DiT injection/removal, attention processors)
- Talking-Head/Lip-sync nodes (loader, apply, audio save)
- Native node overrides

## Architecture

```
ComfyUI-Fusion-MLX/
├── __init__.py           # Node registration + native overrides
├── core/
│   ├── engine_wrapper.py # FusionEngineWrapper (staged pipeline)
│   ├── wrappers.py       # FusionModelWrapper, FusionCLIPWrapper, FusionVAEWrapper, name mapping
│   ├── lifecycle.py      # FusionMemoryGuardian + PipelineStageContext
│   ├── async_utils.py    # Shared ThreadPoolExecutor + run_async helper
│   └── bridge.py         # UMABridge (torch↔mlx tensor conversion)
└── nodes/
    ├── loaders.py        # FusionModelLoaderNode, UNETLoader, CLIPLoader, VAELoader
    ├── conditioning.py   # FusionTextEncoderNode
    ├── samplers.py       # FusionKSamplerNode, KSampler override
    ├── vae.py            # FusionVAEDecoderNode
    ├── shortcuts.py      # FusionImageGenNode, FusionVideoGenNode, FusionImageToVideoNode, FusionIdentityPipelineNode
    ├── latent.py         # FusionEmptyLatentNode, Wan22ImageToVideoLatent, EmptyLatentImage override
    ├── video_io.py       # FusionSaveVideoNode, SaveWEBM, SaveAnimatedWEBP overrides
    ├── passthrough.py    # ~30 passthrough nodes (schedulers, guiders, etc.)
    ├── voice.py          # FusionVoiceLoader, FusionVoiceSynthesize, FusionVoiceClone, FusionSaveAudio
    ├── identity.py       # FusionIdentityLoader, FusionIdentityApply, FusionIdentityGenerate
    ├── ip_adapter.py     # FusionIPAdapterLoader, FusionIPAdapterApply, FusionIPAdapterInject
    ├── talking_head.py   # FusionLipsyncLoader, FusionLipsyncApply
    └── postprocess.py    # FusionSubtitleOverlayNode
```

## Memory Management

- FusionMemoryGuardian.maybe_purge() — threshold-based purge (skips gc+cache clear when active < 1GB)
- Staged load/unload: only one component (text_encoder / dit / vae) in memory at a time
- MPS suppression: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
- UMABridge handles MPS→CPU→numpy→MLX and reverse paths

## Performance Optimizations

| Optimization | Component | Impact |
|---|---|---|
| Threshold-based purge (maybe_purge) | lifecycle.py | Skips gc.collect()+mx.metal.clear_cache() when active memory < 1GB — avoids 50-200ms stalls per stage |
| Shared ThreadPoolExecutor | core/async_utils.py | Replaces per-call executor creation — eliminates thread spawn overhead on every node |
| mx.eval() at pipeline boundaries | engine_wrapper.py | Forces evaluation after denoise/decode — prevents lazy MLX sync stalls that block downstream ops |
| VAE engine caching | wrappers.py | `FusionVAEWrapper.get_engine()` reuses engine from model_wrapper — avoids re-creating engine on every decode |
| Raw array output (output_format="raw") | samplers.py, shortcuts.py, engine_wrapper.py, ip_adapter.py, identity.py | Eliminates PIL BytesIO→Image→numpy byte round-trip — direct np.ndarray from generate() |
| Dict comprehension for latent copy | samplers.py | Replaces latent_image.copy() — avoids copying large sample arrays unnecessarily |
| Null engine after stop | engine_wrapper.py | Releases model references on engine stop — prevents stale graph retention |
| Module-qualified run_async | all nodes | `core.async_utils.run_async()` — single patchable async entry point, testable and mockable |
| PuLIDPipeline caching | shortcuts.py | Caches PuLIDPipeline instance — avoids ~1GB weight reload on every identity generation call |
| In-memory video decode via av | samplers.py, engine_wrapper.py, shortcuts.py | `av.open(io.BytesIO(...))` replaces temp-file write + imageio subprocess read — eliminates disk I/O and ffmpeg subprocess overhead |
| imageio reader try/finally | samplers.py | Ensures file handles and temp files cleaned up on exception — prevents file descriptor leaks |
| Remove unnecessary .copy() | identity.py, ip_adapter.py | Eliminates double-allocation of large image arrays — halves peak memory for image conversion |
| Migrated per-call executors | identity.py, ip_adapter.py | All nodes now use shared `core.async_utils.run_async()` — no more per-call ThreadPoolExecutor |
| Safe transformer monkey-patch cleanup | ip_adapter.py | `finally` block with try/except around `remove_from_transformer()` — prevents class-level patch leak on errors |
| cv2.resize for SigLIP preprocessing | ip_adapter.py | Replaces float32→uint8→PIL→resize→numpy→float32 with direct cv2.resize on float32 — eliminates PIL serialization overhead |
| In-memory video encoding via av | video_io.py | `av.VideoFrame.from_ndarray()` replaces 97 PNG writes to disk + ffmpeg subprocess — eliminates all frame disk I/O |
| av for talking_head video decode | talking_head.py | Replaces imageio.get_reader (no try/finally, handle leak) with av.open + try/finally — prevents file handle leaks |
| Upstream: video output_format="raw" | fusion-mlx #217 | `VideoGenEngine.generate(output_format="raw")` returns numpy uint8 frames [T,H,W,3] — eliminates MP4 encode→av decode round-trip for video pipelines (samplers, shortcuts, engine_wrapper) |
| Temp file leak fix (talking_head) | talking_head.py | try/finally cleanup for temp MP4+WAV files + shared executor — prevents disk leaks on errors |
| Shared executor for talking_head | talking_head.py | Replaced per-call ThreadPoolExecutor with `core.async_utils.run_async()` |
| Upstream: T5 encoder caching | fusion-mlx #221 | Wan2Backend preloads T5 in `start()` + LRU text embedding cache (max 16) — eliminates 112s T5 reload on repeated calls, 10x+ speedup for cached prompts |

### Benchmark Results (Apple Silicon, MLX)

| Model | Task | Steps | Engine Start | Generate | Total |
|---|---|---|---|---|---|
| FLUX.2-klein-base-4B | Image 512×512 | 4 | 0.30s | 1.40s | 1.70s |
| Wan2.1-1.3B | Video 272×272×9f | 2 | 28.1s (cold) | — | 28.1s |
| Wan2.1-1.3B | Video 272×272×9f | 2 (cached T5) | 10.0s | — | **10.0s** |

With T5 encoder caching + output_format="raw": **2.8x speedup** on repeated video generation.

Run: `~/claude-home/fusion-mlx/.venv/bin/python3 bench_perf.py [image|video|both]`
