# H3 Video Super-Resolution Post-Processing Design

**Date:** 2026-09-01
**Status:** Approved (pending spec review)
**Author:** dahai80

## Background

H3 (MiniMax-H3 FL2VA) i2v videos output by fusion-comfyui are perceptually blurry ("糊"). Root-cause investigation (2026-09-01, see memory `h3-vae-blur-rootcause-2026-08-30`) established this is **not a bug**:

- mp4 lap_var (3-frame mean) = 46.98 at 1344x768 — above the pure-VAE round-trip ceiling (22.5), so sampling adds detail, not at blurry floor.
- Blur is perceptual: comparing 16x-compressed VAE video frames against a sharp t2i keyframe (lap 213). H3 VAE = 16x spatial compression = inherent low-pass.
- **Raising sampler steps does NOT help**: isolated s12 vs s16 runs (identical keyframe/seed/768p/dit8_te4/anchors) gave lap_var mean 46.98 vs 47.26, ratio 1.006 (noise floor). H3 is guidance-distilled; steps refine the noise schedule, they do not add detail.
- Two non-viability walls for raising steps anyway: `fusion_comfyui/core/async_utils.py:33` `run_async(timeout=3600)` sampler cap (s25 died at 01:00:01) and AICF `ComfyUIProvider` 30min executor timeout.

Conclusion: H3 output sharpness is at the model's ceiling. The remaining lever is **post-processing super-resolution** — run a learning-based SR model on the decoded H3 video frames to recover high-frequency detail the VAE low-pass discarded.

## Goal & Success Criteria

**Goal:** Add a RealESRGAN x4plus MLX super-resolution stage to the H3 i2v pipeline, triggered by environment variable, with zero changes to the AICF orchestration shell (off-limits).

**Success criteria:**
1. fusion-mlx exposes a `/v1/images/super-resolution` endpoint running RealESRGAN x4plus in pure MLX/Metal.
2. fusion-comfyui SaveVideo node, when `FUSION_H3_SR=1`, super-resolves each H3 frame (1344x768 → 5376x3072, 4x) before re-encoding the mp4.
3. SR output is measurably sharper: lap_var of SR frames > lap_var of source frames (target: 2x+ improvement, calibrated against the sharp keyframe baseline of 213).
4. AICF end-to-end (h3-i2v workflow) produces a sharper mp4 with no AICF source/JSON changes — AICF downloads the same /history output, now SR'd.
5. Total pipeline (H3 generate + SR + re-encode) fits within acceptable bounds. If 4x (`FUSION_H3_SR_SCALE=4`) blows the time budget, the operator explicitly sets `FUSION_H3_SR_SCALE=2` — a deliberate documented config switch, never an automatic silent downgrade.

## Architecture Overview

```
AICF (AIComicWorkstation) — OFF-LIMITS, zero changes
  └─ submits static h3-i2v.json → ComfyUI :11445
       └─ H3 generate (~21min) → 76 frames, 1344x768, numpy float32
            └─ SaveVideo node (fusion_comfyui_plugin/nodes/h3.py)
                 ├─ FUSION_H3_SR=0 (default): re-encode frames → mp4 (current behavior)
                 └─ FUSION_H3_SR=1: per-frame HTTP POST → fusion-mlx SR endpoint
                      └─ fusion-mlx RealESRGAN x4plus (RRDB MLX, tile inference)
                           └─ 5376x3072 SR frames → back to SaveVideo
                                └─ re-encode SR frames → mp4 (crf=18, libx264)
                                     └─ /history serves mp4 → AICF downloads (AICF unaware of SR)
```

Key boundary: AICF consumes ComfyUI `/history` output only. SR happens entirely inside the ComfyUI graph (SaveVideo node) + fusion-mlx engine. AICF's static workflow JSON and TypeScript code are never touched — the SR toggle is a fusion-comfyui environment variable, invisible to AICF.

## PR-A: fusion-mlx RealESRGAN MLX Engine

**Upstream change.** Follow project rule: file issue → PR → release → PyPI dep bump.

**New module:** `fusion_mlx/image/sr/`
- `config.py` — `RealESRGANConfig`: num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32, res_scale=1.0. Mirrors the official RealESRGAN_x4plus architecture.
- `rrdb.py` — `RRDBNet` MLX port: residual-in-residual dense blocks + pixelshuffle upsampler. Pure `mlx.nn` / `mx` ops, no torch, no einops.
- `weights.py` — state-dict detection + load from `RealESRGAN_x4plus.pth` (downloaded via `https://hf-mirror.com` mirror into `~/.fusion-mlx/models/realesrgan/`). Handles key remapping if official .pth keys differ from MLX param names.
- `generate.py` — `super_resolve(images, scale=4, tile_size=512, tile_overlap=64)`: per-frame inference with tiled processing to avoid OOM on 5376x3072 outputs. Returns SR frames as numpy float32.

**New endpoint:** in `public_api.py`
- `POST /v1/images/super-resolution`
  - request: multipart/binary image bytes (PNG or raw), optional `scale` (2/4), `tile_size`
  - response: SR image bytes (PNG), plus JSON metadata (elapsed, in/out dims)
- Single-frame per request (per-frame POST pattern — see Open Questions). Stateless, no video context needed server-side.

**Model:** `RealESRGAN_x4plus.pth` (~64MB, standard RRDB). Download flow documented with hf-mirror. Detection from state-dict keys (not filename), per fusion-mlx model-loading convention.

**Alignment verification (TDD):** SR output compared against a torch reference (basicsr/diffusers RealESRGAN) on fixed test images — target PSNR > 35dB, SSIM > 0.95, to confirm the MLX port is faithful before release.

## PR-B: fusion-comfyui SR-aware SaveVideo

**In-repo change** (this project, modify allowed).

**File:** `fusion_comfyui_plugin/nodes/h3.py` — extend `SaveVideo.save()` (current lines 262-280).

**Environment variables:**
- `FUSION_H3_SR` — `0` (default, current behavior) | `1` (real SR via fusion-mlx) | `mock` (lanczos 4x placeholder, for env-trigger + AICF chain verification before PR-A ships)
- `FUSION_H3_SR_SCALE` — `4` (default) | `2` (fallback if 4x blows time budget)
- `FUSION_H3_SR_ENDPOINT` — `http://127.0.0.1:11434/v1/images/super-resolution` (default)

**Behavior in `save()`:**
1. Decode current behavior unchanged when `FUSION_H3_SR=0`.
2. When `=1` or `=mock`: for each frame in `images`, call SR (real HTTP POST or mock lanczos resize), collect SR frames into a new numpy array.
3. Re-encode the SR frame array to mp4 via existing `_encode_video_av` (reuse `FusionSaveVideoNode`). Output dims now 4x (or 2x).
4. Log: SR mode, frame count, per-frame elapsed, total SR elapsed, in/out resolution, endpoint used.

**Mock mode rationale:** PR-A (fusion-mlx) and PR-B ship in parallel. PR-B's mock mode lets us verify the env trigger, AICF end-to-end chain, and re-encode of large frames immediately — without waiting for PR-A release. Once PR-A dep bumps, flip `FUSION_H3_SR=1` for real inference.

**HTTP client:** reuse `fusion_comfyui.core` HTTP client patterns (httpx). Per-frame POST, binary image in / SR image out. Convert numpy frame ↔ PNG bytes via PIL (already a dependency in h3.py).

## Data Flow

1. H3 VAE decode produces 76 frames, 1344x768, numpy float32, arriving at `SaveVideo.save()` as `video["images"]`.
2. `FUSION_H3_SR` checked. If `0`: skip to step 6 (current path).
3. If `1`: for each frame i in [0..75]:
   a. Convert frame to PNG bytes (PIL).
   b. `POST /v1/images/super-resolution` with image bytes + scale.
   c. Receive SR PNG bytes, decode to numpy (5376x3072 for 4x).
   d. Append to SR frames array.
   Log per-frame elapsed; accumulate total.
4. If `mock`: PIL lanczos resize each frame by `FUSION_H3_SR_SCALE` (default 4x), no HTTP. Same output shape as real mode for the given scale.
5. SR frames array assembled: shape (76, 3072, 5376, 3) float32.
6. `FusionSaveVideoNode._encode_video_av(sr_frames, filepath, fps=24, codec="libx264", crf=18)`.
7. Return `{"ui": {"videos": [{"filename": ..., "type": "output"}]}}` — same contract, AICF downloads via /history unchanged.

## Error Handling

**Fail loud, no silent fallback (project Rule 12).** SR is a deliberate user opt-in (`FUSION_H3_SR=1`); if it fails, the user must know the output is NOT super-resolved, not be silently handed the blurry original believing it's SR'd.

- fusion-mlx SR endpoint unreachable / 5xx → `SaveVideo.save()` raises `RuntimeError` with the failing frame index + endpoint + HTTP status. Prompt fails; AICF sees the error. **No fallback to original frames.**
- SR endpoint returns 200 but corrupted/empty image → raise on decode, same loud failure.
- Single-frame SR timeout (configurable, default 30s) → raise; do not skip and continue (skipping produces a mixed-resolution video silently).
- `FUSION_H3_SR=mock` mode failures (PIL resize) → raise (programming error, should never happen).
- Model not downloaded on fusion-mlx side → endpoint returns a clear 503 with download instructions; SaveVideo surfaces it verbatim.
- `FUSION_H3_SR=0` path: unchanged, no new failure modes.

## Testing Strategy

**PR-A (fusion-mlx):**
- Unit: RRDBNet forward on a fixed input produces expected output shape; tile stitching matches full-image inference (tile vs whole, diff < 1e-4).
- Alignment: SR output vs torch basicsr/diffusers RealESRGAN_x4plus reference on 3 fixed test images — PSNR > 35dB, SSIM > 0.95. This is the gate before release.
- Endpoint: `/v1/images/super-resolution` round-trips a PNG, returns correct dims + metadata.

**PR-B (fusion-comfyui):**
- Unit (TDD, no model needed):
  - `FUSION_H3_SR=0` → unchanged behavior, mp4 dims == input dims.
  - `FUSION_H3_SR=mock` → mp4 dims == 4x input, lanczos, no HTTP call.
  - `FUSION_H3_SR=1` + endpoint stub (mock httpx) → per-frame POST called N times, SR frames re-encoded.
  - `FUSION_H3_SR=1` + endpoint 5xx → `save()` raises RuntimeError, no mp4 written.
  - `FUSION_H3_SR=1` + endpoint timeout → raises, no silent skip.
- Integration: `FUSION_H3_SR=mock` against live ComfyUI:11445 + AICF h3-i2v workflow → mp4 generated, AICF downloads, dims 4x. Verifies env trigger + AICF chain zero-change.
- Real-model e2e (after PR-A release): `FUSION_H3_SR=1`, real fusion-mlx SR, 76 frames. Measure lap_var SR-before vs SR-after. Target: SR-after lap_var approaches keyframe sharpness (213), not just 2x of 47.

**Cleanup:** all test artifacts (intermediate frames, stub mp4s) deleted after verification; only final output mp4 + logs kept (project rule).

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 4x SR on 76 frames blows AICF 30min deadline (H3=21min + SR 5-10min > 30min) | High | Measure real SR per-frame time first. If total > 9min, document `FUSION_H3_SR_SCALE=2` as the shippable config. x2 still a major sharpness gain over the VAE low-pass. Do NOT silently downgrade — log the scale used. |
| Per-frame independent SR → temporal flicker between frames | Medium | Phase 1 accepts per-frame SR (image SR, the chosen scope). If flicker unacceptable, Phase 3 upgrades to video temporal SR (RealBasicVSR/VRT) — out of this spec. Measure flicker; decide after seeing real output. |
| 5376x3072 single frame OOM (~190MB float32, more in intermediates) | Medium | Tile inference in PR-A (tile_size=512, overlap=64), reuse the VAE tile-split/blend pattern from `fusion_mlx/video/minimax_h3/vae.py`. |
| RealESRGAN .pth key remapping differs from MLX param names | Medium | `weights.py` handles remapping with explicit key map + logged unmatched keys (matches existing fusion-mlx model-load convention, e.g. the H3 VAE 559/560 pattern). |
| 76 HTTP round-trips latency overhead | Low | localhost, ~ms each; negligible vs inference. If measurable, batch endpoint is a documented Phase-2 option. |
| AICF / ComfyUI can't handle 5376x3072 mp4 (storage, playback) | Low | AICF just stores/downloads; playback is client-side. Verify mp4 file size reasonable (crf=18 on 4x is large but bounded). |

## Open Questions (resolved defaults)

1. **Endpoint form — per-frame POST vs batch.** Default: per-frame (76 requests). Simple, stateless, allows per-frame error isolation. Batch is a Phase-2 optimization if latency matters.
2. **SR failure — fallback vs fail-loud.** Default: **fail-loud** (raise, no fallback to original). SR is opt-in; silent fallback would deceive AICF into accepting a blurry video as SR'd, violating the end-to-end-verification principle.
3. **Check fusion-mlx repo for existing SR issues/PRs before filing.** Default: yes — search first to avoid duplicate issue, per the issue→PR flow rule.

## Out of Scope

- **Video temporal SR** (RealBasicVSR, VRT, frame-coherent super-resolution) — Phase 3, separate spec. Phase 1 is per-frame image SR.
- **AICF source/JSON changes** — off-limits by constraint. SR is invisible to AICF (env var + in-graph processing).
- **Changing H3 itself** (steps, guidance, VAE) — proven ineffective (ratio 1.006) and non-viable on timeouts. Closed.
- **SR of non-H3 video models** (wan2, ltx2, etc.) — out of scope; this spec targets the H3 "糊" problem specifically. Generalization is a later refactor.
