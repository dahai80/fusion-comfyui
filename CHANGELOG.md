---
doc_id: fusion-comfyui/changelog
product: fusion-comfyui
class: app
entry: service
doc_type: changelog
title: { zh: "fusion-comfyui 变更记录", en: "fusion-comfyui changelog" }
audience: engineer
status: draft
version: 0.2.18
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - CHANGELOG.md
  - pyproject.toml
---

# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow `pyproject.toml` (`0.2.18` as of this entry). Phase-level history and the falsification log (spec denoise, radix cache) live in `README.md`.

## Unreleased

## 0.2.18 — 2026-11-07

### Added

- Adopted the fusion docs-system v1.1 contract (app x service, PR #77): added `docs/product-page.md`, `docs/get-started.md`, `docs/api-reference.md`, `docs/config-reference.md`, `docs/troubleshooting.md`, this `CHANGELOG.md`, `ci/validate_docs.py`, and the `docs` GitHub Actions workflow; added 12-key frontmatter to all in-repo markdown docs.

### Fixed

- README / README_CN Quick Start no longer reference the removed `fusion-comfyui serve` console script and stale port 11443; they now document the real runtime path (`start.sh` / `launch_comfyui.py`, port 11445) that exists since the P1 unification (v0.2.13).
