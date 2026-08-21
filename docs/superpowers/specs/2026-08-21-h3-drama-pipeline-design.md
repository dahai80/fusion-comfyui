# MiniMax-H3 Drama Pipeline — Design Spec

**Date:** 2026-08-21
**Status:** Approved (execution phase)
**Goal:** Enable minimax h3 to generate 西游记连续剧 (Journey-to-the-West serial drama) via the fusion-comfyui drama pipeline.

## Problem

Drama pipeline (`fusion_comfyui/nodes/drama/pipeline.py`) cannot use minimax h3 today.
Seven confirmed gaps block the path, split across this repo (local) and fusion-mlx (upstream).

## Gap Inventory

| # | Gap | Location | Owner |
|---|---|---|---|
| A | `_MODEL_TYPES` missing minimax/h3 → H3 routed to ImageGenEngine | `fusion_comfyui/core/engine_wrapper.py:15` | local |
| B | `_fallback_model` has no H3 branch | `fusion_comfyui_plugin/core/wrappers.py:211` | local |
| C | Drama pipeline has no video generation — only `generate_image` | `fusion_comfyui/nodes/drama/pipeline.py:74` | local |
| D | `DEFAULT_MODEL` hardcoded to FLUX.2-klein (image) | `fusion_comfyui/nodes/drama/pipeline.py:23` | local |
| E | quantize (dit8_te4) unreachable through engine API | `fusion_mlx/engines/video_backends/minimax_h3.py` + `engines/video.py` | upstream |
| F1 | H3 native audio path not wired (tokens discarded) | `fusion_mlx/video/minimax_h3/` | upstream |
| F2 | TTS+lipsync nodes exist but pipeline skips them | `fusion_comfyui/nodes/drama/pipeline.py:99` | local |
| G | H3 i2va/l2va/fl2va not implemented (only t2va) — no scene continuity | `fusion_mlx/video/minimax_h3/` | upstream |

### Evidence (per gap)

- **A** — `engine_wrapper.py:15` `_MODEL_TYPES = {"flux2":"image","flux":"image","wan2":"video","wan":"video","skyreels":"video","ltx":"video","cosmos":"video","hunyuan":"video","svd":"video"}`. No minimax/h3 key. `_infer_model_type()` loops this dict, returns `"image"` default → H3 request builds `ImageGenEngine`, never reaches `VideoGenEngine`. `_LATENT_CHANNELS` (line 28) same gap — H3 z_channels=24 absent.
- **B** — `wrappers.py:211` `_fallback_model(requested)`: checks isdir → cascade branch → `_available_video_models()` → wan/ltx/flux/svd/sdxl branches → else `available[0]`. `grep` for minimax/h3 in file = 0 matches. A requested H3 with no exact dir falls to `available[0]` (whatever video model loads first), silently wrong model.
- **C** — `pipeline.py:55-99` Phase 3: `model.generate_image(prompt=desc_en, ...)` → writes PNG → `scene_videos.append(frame_path)` (a still, not video) → `scene_audios.append("")`. No `generate_i2v`/video call anywhere in pipeline. Output is slideshow of images, not drama video.
- **D** — `pipeline.py:23` `DEFAULT_MODEL = os.environ.get("DRAMA_MODEL", "FLUX.2-klein-base-4B")`. Hardcoded image model. No `DRAMA_VIDEO_MODEL` env. Even with routing fixed (A/B), pipeline never asks for a video model.
- **E** — `engines/video_backends/minimax_h3.py:120` `generate(params)` calls `generate_video(..., output_path=...)` with NO `quantize` arg → defaults `"none"` in `generate.py`. `engines/video.py` `VideoGenEngine.generate()` builds `VideoGenParams(**kwargs)` but does not forward `quantize`. `base.py` `VideoGenParams` dataclass has no `quantize` field. So `quantize.py` (`quantize_dit`, `quantize_text_encoder`, supports none/te4/dit8/dit8_te4) is unreachable from any engine call. Without quantize, H3 FL2VA peak = TE 67G + DiT 66G + VAE 11G = 144G > M5 Max 137G → OOM.
- **F1** — `video/minimax_h3/transformer.py` `MiniMaxH3DiTModel.forward()` returns `(video_output, audio_output)` but caller in `generate.py` discards `audio_output`. `config.py` `H3AudioVAEConfig` defined, `grep` shows zero usages. H3 is an Omni-Transformer (generates video+audio tokens jointly) but the MLX port only decodes video. No native speech/music/SFX.
- **F2** — `nodes/drama/tts.py` `FusionTTS(BaseNode)` fully implemented (`TTSEngine.synthesize(text,speed,voice,ref_audio)`→wav). `nodes/drama/lipsync.py` `FusionLipSync(BaseNode)` fully implemented (latentsync+musetalk). `pipeline.py:99` sets `scene_audios.append("")` — TTS/lipsync explicitly skipped (comment cites "OOM"). Nodes exist, pipeline doesn't call them.
- **G** — `generate.py` `generate_video()` calls `generate_t2va_video(...)` — text-to-video-audio only. `config.py` `H3Config.tasks=("t2va","i2va","l2va","fl2va")` but only t2va implemented. i2va (image→video), l2va (last-frame→video), fl2va (first+last→video) = scene continuity primitives. Without them, each scene generates from text only → no visual continuity between scenes (孙悟空 looks different each cut).

## Architecture

```
drama pipeline (per chapter)
  │
  ├─ Phase 1  split_only  ─→ vlm.py DramaChapterParser → scenes[{desc_en, characters, ...}]
  │
  ├─ Phase 3  generate (CHANGED)
  │     for each scene:
  │       ├─ [image] first_frame = FLUX.generate_image(desc_en)   (keyframe, optional)
  │       ├─ [video] scene_video = H3.generate_video(             (was: generate_image → PNG)
  │       │       prompt=desc_en,
  │       │       image=first_frame,           (i2va/l2va — needs G upstream)
  │       │       quantize="dit8_te4",         (needs E upstream)
  │       │       num_frames, width, height, fps, seed)
  │       ├─ [tts]   scene_audio = FusionTTS.synthesize(line, voice)   (F2, env-gated)
  │       └─ [lipsync] final = FusionLipSync(scene_video, scene_audio) (F2, env-gated)
  │
  └─ Phase 4  assemble + concat → chapter.mp4 (+ audio track)
```

Routing layer (unchanged shape, new entries):
```
request "minimax-h3" → engine_wrapper._infer_model_type → "video" (A)
                  → VideoGenEngine → MiniMaxH3Backend
fallback "h3"/"minimax" → wrappers._fallback_model → "minimax-h3" (B)
```

Memory staging (per scene, reuses existing H3 staged-load):
```
load TE → encode prompt → release TE → load DiT(8bit)+VAE → denoise → release → next scene
```
quantize=dit8_te4 drops peak below 137G (DiT 8-bit ~33G, TE 4-bit ~17G, VAE 11G ≈ 61G).

## Local Fixes (A/B/C/D/F2)

### A — engine routing (`fusion_comfyui/core/engine_wrapper.py`)
- `_MODEL_TYPES` add: `"minimax":"video", "h3":"video", "fl2va":"video", "ref2va":"video"`
- `_LATENT_CHANNELS` add: `"minimax":24, "h3":24, "fl2va":24, "ref2va":24`
- Surgical: two dict literals, no logic change.

### B — fallback routing (`fusion_comfyui_plugin/core/wrappers.py`)
- `_fallback_model`: add branch before generic `available[0]`:
  `if ("minimax" in name or "h3" in name) and "minimax-h3" in available: return "minimax-h3"`
- Match existing branch style (wan/ltx/svd branches above it).

### C — pipeline video gen (`fusion_comfyui/nodes/drama/pipeline.py`)
- Phase 3: replace `generate_image`→PNG with `generate_video` per scene (when `DRAMA_VIDEO_MODEL` set).
- Keep image-only path as fallback (when no video model / env unset) — backward compat.
- `scene_videos.append(video_path)` (real mp4, not still).

### D — video model env (`fusion_comfyui/nodes/drama/pipeline.py`)
- Add `DRAMA_VIDEO_MODEL = os.environ.get("DRAMA_VIDEO_MODEL", "")`.
- When set → video path (C). When empty → existing image path (slideshow). Default empty = no behavior change for existing users.
- Pass `quantize=os.environ.get("DRAMA_QUANTIZE", "dit8_te4")` through to video gen.

### F2 — re-enable TTS+lipsync (`fusion_comfyui/nodes/drama/pipeline.py`)
- Gate on `DRAMA_TTS=1` (default off — memory + time cost).
- When on: call `FusionTTS.synthesize(scene_line, voice)` → wav; `FusionLipSync.run(video, wav)` → final.
- `scene_audios.append(wav_path)` instead of `""`.
- Voice per character via existing `char_map` heuristic in vlm.py.

## Upstream Fixes (E/F1/G)

Per global rule: file issue first, then PR, then follow to merge. All in `dahai80/fusion-mlx`.

### E — quantize reachable (blocks C/D full function)
- **Issue:** "quantize (dit8_te4) unreachable through VideoGenEngine API — H3 OOM on 137G RAM"
- **PR:** (1) `base.py` `VideoGenParams` add `quantize: str = "none"` field. (2) `video.py` `VideoGenEngine.generate()` forward `quantize=kwargs.get("quantize", "none")`. (3) `minimax_h3.py` `generate(params)` pass `quantize=params.quantize` to `generate_video()`. No generate.py change — it already accepts `quantize` kwarg.
- **Test:** unit test VideoGenParams carries quantize; integration test backend receives it.

### F1 — H3 native audio
- **Issue:** "MiniMax-H3 audio_output discarded — Omni-Transformer native audio not decoded (H3AudioVAEConfig unused)"
- **PR:** wire `audio_output` through `generate_video` → `H3AudioVAE` decode → write audio track. Larger; scope after E.
- **Honest:** this is substantial (audio VAE forward, tokenizer). May land as "experimental" flag.

### G — H3 i2va/l2va/fl2va (scene continuity)
- **Issue:** "MiniMax-H3 only t2va implemented — i2va/l2va/fl2va missing, no inter-scene visual continuity"
- **PR:** implement i2va (image-conditioned) first (most useful for keyframe→video), then l2va/fl2va. Depends on E (memory) — i2va loads image encoder too.
- **Honest:** i2va may need image-encoder weights not in current 71G bundle. Verify weight availability before PR.

## Execution Order

1. **Local A+B** (routing) — no deps. Tests + ruff. Unblocks H3 reaching VideoGenEngine.
2. **Upstream E** (issue→PR→merge) — quantize field. Unblocks memory-fit. Blocks #3 full function but not #3 code (can code against the field, test with mock).
3. **Local C+D+F2** (pipeline rewrite) — depends on A/B (routing) for e2e; E (quantize) for real run. Mock tests land now; real e2e after E merges + dep bump.
4. **Upstream F1+G** (issues→PRs→merge) — after E. Native audio + scene continuity. Largest scope, lands last.

## Testing

- **A/B** — unit: `_infer_model_type("minimax-h3")=="video"`; `_fallback_model("h3")=="minimax-h3"` (monkeypatch `_available_video_models`). ruff `fusion_comfyui tests`.
- **C/D/F2** — mock: pipeline with mocked `generate_video`/`FusionTTS`/`FusionLipSync` asserts video path taken when `DRAMA_VIDEO_MODEL` set, audio path when `DRAMA_TTS=1`, image fallback when unset. No real model load.
- **E** — upstream unit: `VideoGenParams(quantize="dit8_te4").quantize=="dit8_te4"`; backend spy receives `quantize` kwarg.
- **e2e** — real H3 t2v single scene (1.3B-equivalent frames, low res) after E merges. Assert mp4 exists, non-zero, sane duration. Clean process artifacts, keep only output + log.
- **CI** — `.github/workflows/ci.yml` macos-14 ruff+pytest; e2e auto-skip (no GPU in CI).

## Honest Limitations

- **Before E merges:** H3 t2v runs but OOMs at full FL2VA (144G > 137G). quantize unreachable. Local C/D code can land (mock-tested) but real e2e blocked.
- **Before F1 lands:** no native H3 audio. F2 (TTS+lipsync) covers speech but not music/SFX/ambient. Quality gap vs H3 omni output.
- **Before G lands:** no scene continuity — each scene independent t2v, characters inconsistent across cuts. Acceptable for proof-of-concept, not production serial.
- **Memory ceiling:** even with dit8_te4 (~61G peak), concurrent TE+DiT+VAE staging is tight on 137G. One scene at a time (serial), no parallel scenes.
- **H3 constraints:** max_n=1, dim_divisibility=16, max_frames=361 (15s@24fps), 768p/2k only. Drama scenes must fit these — long scenes split.
- **Scope of "能"**: after A+B+C+D+E+F2 land → t2v drama with TTS+lipsync **works** (serial, per-scene, no visual continuity). Full omni (F1) + continuity (G) = production-grade, lands last.
