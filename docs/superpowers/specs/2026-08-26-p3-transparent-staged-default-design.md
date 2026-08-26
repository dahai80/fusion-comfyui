# P3 — Transparent Staged Default: Design Spec

> **Status:** DRAFT — awaiting approval. Spec authority for the implementation plan.
> **Phase:** P3 of the P1-P6 comprehensive ComfyUI fork (P1 core-unify ✅, P2 IMAGE/MASK→numpy ✅ merged via PR #58).
> **Branch base:** `main` (HEAD `e84214b`).

## Goal

Make the native ComfyUI `KSampler` route through fusion-mlx's **staged API** (text-encode → denoise → vae-decode, each stage loaded then strictly offloaded) instead of the monolithic `engine.generate()`. **Transparent** — user workflows unchanged; they keep dragging a single `KSampler` and get the staged memory-sawtooth benefit automatically. Cases with no upstream staged path (video I2V/VACE/camera, image img2img) auto-fallback to monolith. The I2V/VACE staged gap is filed as a fusion-mlx issue for a later phase.

## Context (verified from codebase)

- **Current default path:** `KSampler.sample` (samplers.py:269) → `_generate_monolithic` (samplers.py:58) → `engine._engine.generate(**kwargs)` (samplers.py:116/127/220). Bypasses staging entirely.
- **Staged path exists + e2e-tested:** `FusionKSamplerNode._sample_staged` (samplers.py:538), `FusionTextEncoderNode._encode_staged` (conditioning.py:79), `FusionVAEDecoderNode._decode_via_engine` (vae.py:106). `FusionEngineWrapper` exposes 10 stage methods (engine_wrapper.py:198-287).
- **fusion-mlx staged API** (stable via `public_api`): `VideoGenEngine`/`ImageGenEngine` each have `load_text_encoder`, `encode_text`, `unload_text_encoder`, `load_dit`, `denoise`, `unload_dit`, `load_vae`, `decode`, `decode_tiled`, `unload_vae` — all async. Implemented for Wan2 (T2V only — stage.py:7, wan2.py:350), SkyReels, FLUX.2 image. NOT for I2V/VACE/camera.
- **Verified safe:** `Wan2Backend.denoise(latent, pos_embed, neg_embed, steps, cfg, seed, num_frames)` takes embeds directly — **no reference to t5/text_encoder**. Strict unload of text_encoder after `encode_text` returns is correct.
- **Contract difference:** monolith `generate(output_format="raw")` returns numpy uint8 → caller does `/255.0`. Staged `decode` returns mx.array float (from `vae.decode_packed_latents`) → caller must `to_numpy` + ensure [0,1] float32. Result-handling differs per path.
- **Wrapper embed boundary:** `FusionEngineWrapper.denoise` (engine_wrapper.py:228) accepts **conditioning dicts** (the `positive`/`negative` from nodes) and extracts `pos_embed`/`neg_embed` itself (line 231-234). Staged pipeline passes conditioning dicts to `denoise`, NOT raw embeds. This matches the existing `_sample_staged` pattern (samplers.py:542).
- **`sample` needs NO change:** `KSampler.sample` (samplers.py:269) has two result contracts. Pixel frames (numpy `[T,H,W,3]`) → wrap as `[1,T,H,W,3]` + `_decoded_frames_cache` key (line 330-345). Latent (mx.array) → wrap as numpy latent (line 347-354). The new staged path decodes INSIDE KSampler → returns pixel frames → hits the 330-345 branch unchanged. No `sample` modification.
- **Existing `_sample_staged` (samplers.py:538) coexists:** it is denoise-only (load_dit→denoise→unload_dit), returns latent, decode done by a separate `FusionVAEDecoderNode`. The P3 all-stages-inside path is a DIFFERENT path (full text-encode→denoise→decode inside one call). Both stay; decision #5.
- **StageContext (P4) is greenfield** — only a no-op `PipelineStageContext` stub (lifecycle.py:68) + ad-hoc embed dicts. Out of P3 scope; P3 keeps the ad-hoc embed-dict convention.

## Decisions (locked, user-approved)

1. **Transparent staged default** — rewire native KSampler internally, not promote explicit nodes.
2. **Auto-detect fallback from inputs** — check typed latent/conditioning keys; image/vace/camera → monolith.
3. **All stages inside KSampler** — single node does full staged pipeline; memory sawtooth inside one node call.
4. **Strict sequential offload** — unload + gc + `FusionMemoryGuardian.maybe_purge()` between every stage.
5. **Keep explicit Fusion nodes** — `FusionKSamplerNode`/`TextEncoder`/`VAEDecoder` stay for power-user explicit-stage graphs (cacheable text-encode). Coexist.

## Auto-detect routing matrix (exact — no input sniffing)

| Case | Detection signal (latent/cond key) | Route |
|---|---|---|
| Video T2V (pure text) | none of `_i2v_*` / `_vace_*` | **staged** |
| Video I2V | `latent_image["_i2v_image_path"]` present | monolith |
| Video VACE | any of `_vace_control_video` / `_vace_control_mask` / `_vace_reference_images` | monolith |
| Image cascade stage_b | `positive`/`negative` has `stable_cascade_prior` | pass-through (no engine — unchanged) |
| Image img2img | `latent_image["_image_init_path"]` + `denoise < 1.0` | monolith (edit_image) |
| Image txt2img (FLUX.2) | none of above, `model_type == "image"` | **staged** |

Staged applies to exactly two cases: **video T2V** and **image txt2img**. All other paths keep their current monolith/pass-through behavior byte-for-byte.

## Architecture

### Files touched

- **Modify:** `fusion_comfyui_plugin/nodes/samplers.py` — `_generate_monolithic` gains a staged branch (or a new `_generate_staged` sibling called from `sample` when auto-detect says staged). The monolith path stays intact for fallback.
- **Modify:** `fusion_comfyui/core/engine_wrapper.py` — confirm/strengthen the 10 stage methods already present; add a single orchestration helper `_run_staged_pipeline(...)` that does load→op→unload+purge per stage (DRY for the 3 stages), returning normalized numpy float32 [0,1] output.
- **No fusion-mlx changes** (ComfyUI-only phase). I2V/VACE staged gap → filed as fusion-mlx issue.
- **Tests:** new `tests/test_staged_routing.py` — auto-detect matrix (parametrized: each row routes to staged/monolith/pass-through); staged pipeline happy-path (mock engine, assert stage call order + unload+purge between each); fallback regression (I2V/VACE/img2img still hit monolith); memory-guardian-purge-between-stages assertion.
- **Docs:** README P3 entry; CONSTRUCTION_PLAN checkmark.

### Staged pipeline shape (inside KSampler)

```
auto-detect → staged?
  YES (video T2V / image txt2img):
    await engine.load_text_encoder()
    pos_cond = await engine.encode_text(prompt)                      # {"embed": mx.array}
    neg_cond = await engine.encode_text(neg_prompt) if cfg > 1.0 else None
    await engine.unload_text_encoder(); FusionMemoryGuardian.purge_memory()
    await engine.load_dit()
    latent = await engine.denoise(init_latent, pos_cond, neg_cond, steps, cfg, seed,
                                  num_frames=num_frames)              # video path passes num_frames kwarg
    await engine.unload_dit(); FusionMemoryGuardian.purge_memory()
    await engine.load_vae()
    pixels = await engine.decode(latent)                              # returns mx.array float
    await engine.unload_vae(); FusionMemoryGuardian.purge_memory()
    return _staged_pixels_to_numpy(pixels, model_type)                # float32 [0,1], see Result normalization
  NO → existing _generate_monolithic (unchanged)
```

**Negative-embed contract (verified, risk #3 RESOLVED):** `encode_text(prompt)` (engine_wrapper.py:204 → Wan2Backend.encode_text) encodes ONE prompt, returns `{"embed": mx.array}`. The `negative_prompt` arg the wrapper accepts is **stored in the dict but never encoded** — it is metadata only. So staged MUST call `encode_text` twice when `cfg > 1.0` (positive then negative), both under the single `load_text_encoder`→`unload_text_encoder` span (encoder loaded once, encode twice, then unload — strict offload still holds: one load, one unload, two encodes between). When `cfg <= 1.0`, skip the negative encode entirely; `denoise` sets `cfg_disabled = neg_embed is None or cfg <= 1.0` (wan2.py denoise) and uses `context_null=None`. Pass `neg_cond=None` (not an empty dict) so the wrapper's `neg_embed = None` branch (engine_wrapper.py:233-234) triggers correctly.

**Wrapper `denoise` conditioning-dict contract:** `FusionEngineWrapper.denoise(latent, positive, negative, ...)` extracts `pos_embed = positive["embed"]`, `neg_embed = negative["embed"] if negative else None` (engine_wrapper.py:231-234). Pass the `pos_cond`/`neg_cond` dicts straight through — the wrapper does the extraction. Pass `neg_cond=None` (not `{}`) for cfg-disabled so the wrapper's `if negative:` is False.

### Init latent for T2V staged

T2V starts from pure noise; `denoise` generates its own seeded noise from `target_shape` (verified in wan2.py denoise source). The passed `latent` is the empty zeros latent from `create_empty_latent` — denoise ignores it as init, uses shape + seed. Staged path passes the same zeros latent (shape carries `num_frames`/spatial dims). No change to latent construction.

### Result normalization (exact — match monolith contracts)

`_staged_pixels_to_numpy(pixels_mx, model_type)` — single helper:

- **Video:** `decode` returns mx.array. `to_numpy` → float32. Expected shape `[T,H,W,3]` (monolith raw-path line 120-121) or `[1,T,H,W,3]` (mp4-decode stack line 154). If `[T,H,W,3]`, return as-is (sample wraps to `[1,T,H,W,3]` at samplers.py:336). If `[1,T,H,W,3]`, squeeze leading 1. Values already [0,1] float (VAE decode output) — NO `/255.0` (unlike monolith uint8 path). Clamp to [0,1] defensively.
- **Image:** `decode` returns mx.array `[batch,c,h,w]`. `to_numpy` → float32 → if `c==4` slice `[:3]` (line 238-239) → transpose `[0,2,3,1]` (NCHW→NHWC) → squeeze batch if 1 → `[H,W,3]` (monolith image contract line 237/243). Values [0,1] float, NO `/255.0`. Clamp defensively.
- **monolith path:** unchanged (uint8→/255.0). Staged path produces float [0,1] directly from VAE — the only difference is the absent `/255.0` divide, which is correct because VAE decode yields float not uint8.
- **Fallback guard:** if `decode` returns non-mx (e.g. numpy uint8 — should not happen but defensive), route through the monolith-style `/255.0` to match. Log the type.

## Global Constraints

- Zero `import torch` in non-test code (P2 invariant — do not regress).
- 4-space indent, no docstrings, logging in every function.
- Surgical: monolith path must stay byte-for-byte unchanged (fallback regression test guards this).
- Stage calls go through `FusionEngineWrapper` methods (engine_wrapper.py), NOT directly to `engine._engine` — keep the wrapper as the single engine boundary (matches existing staged-node pattern).
- `FusionMemoryGuardian.maybe_purge()` between every stage (strict offload). Log each stage transition.
- No autograd wrappers, no model freeze (ComfyUI AGENTS.md rules).

## Verification / Exit Criteria

1. New routing tests pass: every matrix row routes correctly (parametrized).
2. Staged happy-path: stage call order asserted (load_text_encoder→encode→unload→load_dit→denoise→unload→load_vae→decode→unload), purge called between each.
3. Fallback regression: I2V/VACE/cascade/img2img still produce identical output path (mock asserts monolith `generate` called, not staged).
4. Full plugin suite green (449+ tests, no regressions).
5. ruff clean.
6. e2e (real model, one T2V): staged path produces valid video; memory sawtooth visible in logs (load/unload between stages). Marked inference-gated, auto-skip in CI.
7. Zero `import torch` non-test.
8. fusion-mlx issue filed for I2V/VACE staged gap; issue number recorded in memory + README.

## Out of Scope (deferred)

- **P4 StageContext** — greenfield context object threading stage data; P3 keeps ad-hoc embed dicts.
- **P5 explicit comfy/ fork** — separate phase.
- **P6 stub internalization** — separate phase.
- **I2V/VACE/camera staged path** — upstream fusion-mlx work (issue only this phase).
- **SkyReels staged** — API exists but no e2e test infra; routing supports it (T2V-like) but no real-model verification. Documented as supported-but-untested.

## Open Risks (all RESOLVED pre-plan)

- **SkyReels denoise signature** — RESOLVED: identical to Wan2. All 10 stage methods implemented with the same `(self, latent, pos_embed, neg_embed, steps, cfg, seed, num_frames)` signature (verified via `inspect.signature`). Staged routing works for Wan2 + SkyReels without backend detection.
- **decode_tiled vs decode** — CONVENTION DECIDED: P3 staged default uses `decode` only (matches `VAEDecode`/`_decode_via_engine` default, vae.py:109). `decode_tiled` is the explicit `VAEDecodeTiled` power-user node — out of P3 transparent-default scope. No internal tiling threshold to guess. If large-latent OOM appears in e2e, that's a fusion-mlx `decode` limitation, not a P3 routing concern.
- **CFG negative-encode (cfg<=1.0)** — RESOLVED: `encode_text(prompt)` encodes ONE prompt → `{"embed": mx.array}`; negative_prompt is stored metadata, never encoded upstream. Staged calls `encode_text` twice when `cfg > 1.0` (pos then neg, one encoder-load span), skips negative encode when `cfg <= 1.0` and passes `neg_cond=None`. Wan2 denoise `cfg_disabled = neg_embed is None or cfg <= 1.0` handles the None case.
