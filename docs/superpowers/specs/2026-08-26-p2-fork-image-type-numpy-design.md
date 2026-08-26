# P2: Fork ComfyUI IMAGE type to numpy (remove torch from I/O glue)

**Status:** DRAFT — awaiting user spec review
**Date:** 2026-08-26
**Phase:** P2 of P1-P6 fork roadmap (user approved "Full fork now, accept risk")
**Predecessor:** P1 core unification (commit `dc1a1db`)

## Goal

Remove every direct `import torch` from `fusion_comfyui/` and `fusion_comfyui_plugin/`.
Make the ComfyUI `IMAGE`/`MASK` types **numpy arrays** (NHWC float32, [0,1]) instead of
torch tensors, by overriding every native node that touches IMAGE/MASK with torch ops.

**Inference stays on MLX/Metal** (already true). This phase removes torch from the
**I/O glue** — the IMAGE wrapper, mask wrapping, checkpoint `.bin` loading — so the
plugin never imports real torch even when ComfyUI's `requirements.txt` installs it.

## Why (the conflict, surfaced honestly)

User goal "不使用pytorch" (no PyTorch) vs ComfyUI's contract: native image nodes
(`ImageBatch`, `ImageScale`, `RepeatImageBatch`, controlnets) call torch tensor methods
(`.movedim`, `torch.cat`, `.to(device)`, `torch.nn.functional.interpolate`). A numpy
IMAGE crashes these. The plugin does NOT currently override them.

Three options were offered; user chose **Full fork now** twice — overriding all native
IMAGE/MASK torch nodes + rewriting `comfy/utils.py` scaling kernels to numpy, accepting
scaling-quality-drift risk. This is the scope this spec covers.

## Blast radius (verified)

13 native node classes touch IMAGE/MASK with torch ops and are NOT currently overridden.
**Usage audit** (decisive): none of these appear in real fusion-mlx workflows — e2e
tests, drama scripts, and IPAdapter examples use only already-overridden nodes
(`LoadImage`, `SaveImage`, `Fusion*Gen`, `FusionIPAdapter*`, `FusionSaveVideo`).
Drama uses `FusionEngineWrapper` direct API (no IMAGE-typed nodes). So the 13 split:

**Dead paths** (no valid pure-MLX behavior to fork — stub with clear message):
- Model-layer routes (`comfy.controlnet`/patcher/ldm that fusion-mlx never loads —
  P5 territory): `VAEEncodeForInpaint`, `InpaintModelConditioning`, `ControlNetApply`,
  `ControlNetApplyAdvanced`, `PainterNode`, `QwenImageDiffsynthControlnet`.
- Format-incompat: `ConditioningSetMask`. Plugin `CLIPTextEncode` returns a flat dict
  `{"prompt": text, ...}`, but native `append` iterates conditioning as a list of
  `[embed, {...}]` pairs (`for t in conditioning`) → crashes/garbage; and `mask.unsqueeze`
  crashes on numpy. Engine never reads a mask from conditioning. Nothing to fork → stub.

**Pure image transforms** (numpy-safe, no model layer — fork to numpy):
`ImageScale`, `ImageScaleBy`, `ImageBatch`, `EmptyImage`, `ImagePadForOutpaint`,
`LoadImageMask`. These are the real IMAGE-contract surface a user *could* wire up.

Scaling kernels in `comfy/utils.py` (torch) — used by `LatentUpscale` override
(`samplers.py:630` calls `comfy.utils.common_upscale(arr,...)` with **mx/numpy** `arr`
→ `arr.movedim` already **AttributeError-broken** today; latent path is dead/buggy):
- `common_upscale` (line 1069) — `.movedim`, `.narrow`, `torch.nn.functional.interpolate`
- `bislerp` (line 983)
- `lanczos` (line 1059) — `torch.from_numpy`, `torch.stack`, `.cpu().numpy()`

3 direct `import torch` glue sites to remove:
- `core/bridge.py:74` `to_image_tensor` → return numpy instead of torch
- `nodes/image.py:66` `LoadImage` mask → route through bridge
- `nodes/ip_adapter.py:431` `_load_torch_ip_adapter` → `.bin` legacy loader

1 hidden torch dependency to fix: `samplers.py:630` `comfy.utils.common_upscale` on
numpy/mx latent → replace with numpy scaling helper (fixes latent-upscale dead path).

## Design

### IMAGE/MASK contract (the fork)

| Type | Representation | dtype | layout | range |
|------|---------------|-------|--------|-------|
| IMAGE | `np.ndarray` | float32 | NHWC `[B,H,W,C]` | [0,1] |
| MASK | `np.ndarray` | float32 | `[B,H,W]` | [0,1] |

Conversion boundary: `fusion_comfyui.core.bridge` is the single IMAGE/MASK conversion
seam. After P2 it touches NO torch — it interops via numpy/PIL/MLX only. Native nodes
still expecting torch IMAGE are either overridden (transforms) or stubbed (dead paths);
no plugin code converts numpy→torch to satisfy them.

### bridge.py changes

- `to_image_tensor(data)` → rename intent: returns **numpy NHWC float32 [0,1]**.
  Drop `import torch`. Name kept for call-site stability; add `to_image_numpy`
  alias. **Verified safe**: all callers (image.py, vae.py) only read `.shape`/`.dtype`
  (both work on numpy) and feed the result into IMAGE slot → SaveImage/FusionSaveVideo
  (already numpy-compatible). No caller invokes a torch-only method on the return.
- Add `to_mask_numpy(data)` → numpy float32 `[B,H,W]` [0,1].
- `to_image_array` already numpy — unchanged.
- Keep `to_numpy`, `to_mlx_array` (MLX interop, no torch).

### Interop with torch-expecting native nodes

Dead-path nodes have no valid pure-MLX behavior to fork. Two distinct reasons:

1. **Model-layer routes** — `ControlNetApply*`, `InpaintModelConditioning`,
   `VAEEncodeForInpaint`, `PainterNode`, `QwenImageDiffsynthControlnet` route into torch
   model layers (`comfy.controlnet`, patchers, ldm) that fusion-mlx never loads. Verified:
   no plugin node imports `comfy.controlnet`/`model_patcher`/`ldm` — the entire MLX path
   bypasses torch models.
2. **Format-incompat** — `ConditioningSetMask` (see Blast radius): plugin conditioning is
   a flat dict, native iterates a `[embed, {...}]` list + `mask.unsqueeze`; engine never
   reads the mask anyway.

Decision for all 7: **stub them** — override `FUNCTION` to raise `NotImplementedError`
with a node-specific message. Model-layer nodes: "X routes into a PyTorch model layer not
yet ported to MLX (P5); use the Fusion* equivalent or wait for the comfy/ core fork".
`ConditioningSetMask`: "regional-mask conditioning is not supported on the fusion-mlx
pipeline (engine has no mask hook); use Fusion* nodes or wait for P3 staged conditioning".
This satisfies the "no torch" goal (no torch import) without forking behavior that doesn't
exist on MLX. Register the stubs in NODE_CLASS_MAPPINGS so they shadow native (consistent
with existing override pattern).

### New node file: `fusion_comfyui_plugin/nodes/image_transform.py`

Pure-numpy overrides (no torch model layer — fully numpy-safe):
- `ImageScale`, `ImageScaleBy` — use `PIL.Image.resize` (lanczos/bilinear/bislerp)
- `ImageBatch` — `np.concatenate`, channel-pad with `np.pad`
- `EmptyImage` — `np.full`
- `ImagePadForOutpaint` — `np.ones`/`np.zeros`/`np.pad`
- `LoadImageMask` — PIL → numpy mask

(`ConditioningSetMask` NOT here — it's dead by format-incompat, see Dead paths above.)

### Scaling kernels: `fusion_comfyui_plugin/nodes/_scaling.py`

numpy/PIL reimplementation of `common_upscale` + `lanczos` + `bislerp`:
- `lanczos` → already PIL-based in torch version; strip torch wrapping → pure
  `np.array(Image.resize(...))`.
- `bislerp` → port the blend formula to numpy (blend coefficients + lerp).
- `common_upscale(samples, w, h, method, crop)` → numpy `reshape`/`transpose` (replace
  `.movedim`) + `slice` (replace `.narrow`) + the above. Handles 4D `[B,C,H,W]` and 5D.
- Parity: assert corr ≥ 0.999 vs torch reference on a fixed 64×64→128×128 image. PIL
  lanczos IS the torch version's own backend, so drift is sub-pixel at worst.

### Conditioning/controlnet/painter overrides — stubs (dead paths)

Per Interop decision: override to `NotImplementedError` stubs. New file
`fusion_comfyui_plugin/nodes/_deadpath_stubs.py` with one class per dead node, each
preserving native `INPUT_TYPES`/`RETURN_TYPES` so graph validation still passes, but
`FUNCTION` raises the clear P5 message. Register in NODE_CLASS_MAPPINGS + native-override
patch dict. This keeps the node graph loadable (no "unknown node") while honestly
signaling the MLX boundary.

### ip_adapter `.bin` loader

Verified: `~/.fusion-mlx/models/ipadapter-flux/` ships BOTH `ip_adapter_flux.safetensors`
AND `ip-adapter.bin`. The loader (`_load_ip_adapter_file`) prefers safetensors (dir scan
finds `*.safetensors` first; `.bin` is only the fallback). The e2e PASS path uses
safetensors. So `.bin` is legacy fallback only.

Decision: **drop `.bin`/`.pt`/`.ckpt` support** — delete `_load_torch_ip_adapter` and
its call sites (lines 404-405, 414-416). `.bin` files are pickle + torch-specific; reading
them without torch needs a pickle unpickler that reconstructs torch tensors — unsafe and
out of scope. If a user has only `.bin`, log a clear message: "IP-Adapter .bin needs
torch; download the .safetensors from HF mirror https://hf-mirror.com (same model repo)".
No upstream issue needed (the `.safetensors` already exists upstream); this is a local
loader-simplification.

## Testing

- Unit: each new numpy node — shape/dtype/range assertions vs known inputs.
- Scaling parity: `lanczos`/`bislerp` numpy output vs torch reference on a fixed
  image, assert correlation ≥ 0.999 (allow minor drift).
- Existing 485 tests must stay green (they mock torch; verify mocks still valid).
- e2e (RUN_E2E=1): LoadImage→ImageScale→VAEDecode→SaveImage workflow on real model.
- ComfyUI startup: 849 nodes, no import errors, IMAGE nodes resolve.

## Risks (accepted by user)

- Scaling-quality drift: numpy/PIL lanczos may differ sub-pixel from torch's. Mitigated
  by parity test + PIL is the torch version's own backend anyway.
- Conditioning/controlnet nodes may hit torch model layers not yet forked (P5). If
  reachable, P2 nodes convert at boundary; if dead, stub with clear message.
- ComfyUI caching (`caching.py:567`) special-cases `torch.Tensor` — numpy IMAGE won't
  match; verify caching still works (likely falls through to generic pickle — test).

## Out of scope (deferred to P5)

- Forking `comfy/` core model layers (ldm, controlnet, model_patcher) off torch.
- `comfy/utils.py` non-image torch helpers.

## Exit criteria

- [ ] `grep -rn "import torch" fusion_comfyui/ fusion_comfyui_plugin/` → 0 matches
- [ ] 485+ tests pass, ruff clean
- [ ] ComfyUI starts, IMAGE nodes resolve, e2e image workflow green
- [ ] Scaling parity test ≥ 0.999 correlation
- [ ] Memory updated, README updated, committed
