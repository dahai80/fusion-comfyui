---
doc_id: fusion-comfyui/changelog
product: fusion-comfyui
class: app
entry: service
doc_type: changelog
title: { zh: "fusion-comfyui 变更记录", en: "fusion-comfyui changelog" }
audience: engineer
status: draft
version: 0.2.20
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - CHANGELOG.md
  - pyproject.toml
---

# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow `pyproject.toml` (`0.2.20` as of this entry). Phase-level history and the falsification log (spec denoise, radix cache) live in `README.md`.

## Unreleased

## 0.2.20 — 2026-11-26

### Fixed

- **CI green again (first success since 2026-11-07)**: the `macos-14` hosted runner label was retired and every run queued then got cancelled; moved both jobs to `macos-15` (6462a6a). With jobs actually executing, three latent fresh-checkout failures surfaced and were fixed:
  - Module-level `import cv2` in `nodes/shortcuts.py` and `nodes/identity.py` broke plugin load on cv2-less environments; the sole use (RGB2BGR) is now a numpy channel flip (19fce63). Remaining cv2 call sites already import lazily.
  - `sr.py`/`voice.py` imported `fusion_core.config` outside their try block, raising `ModuleNotFoundError` instead of degrading to an anonymous key on fusion-core-less environments; import now inside try (82d804f).
  - `tests/test_process_title.py` asserted the gitignored `ComfyUI/` engine tree exists; both asserting tests now skip on checkouts without it (82d804f, 49ca5ce).

## 0.2.19 — 2026-11-26

### Fixed

- Cleared the lint stockpile (issues #78/#79): `ruff check .` 8 -> 0. All 8 were T201 (print) in `ci/validate_docs.py`; scoped `T201` off for `ci/*` via `[tool.ruff.lint.per-file-ignores]` — the validator is a stdlib-only argparse CLI whose prints are its intended reporting mechanism (PR #80).

## 0.2.18 — 2026-11-07

### Added

- Adopted the fusion docs-system v1.1 contract (app x service, PR #77): added `docs/product-page.md`, `docs/get-started.md`, `docs/api-reference.md`, `docs/config-reference.md`, `docs/troubleshooting.md`, this `CHANGELOG.md`, `ci/validate_docs.py`, and the `docs` GitHub Actions workflow; added 12-key frontmatter to all in-repo markdown docs.

### Fixed

- README / README_CN Quick Start no longer reference the removed `fusion-comfyui serve` console script and stale port 11443; they now document the real runtime path (`start.sh` / `launch_comfyui.py`, port 11445) that exists since the P1 unification (v0.2.13).
