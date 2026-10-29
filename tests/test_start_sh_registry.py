import os
import re
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
START_SH = os.path.join(REPO_ROOT, "start.sh")


@pytest.fixture
def script_copy(tmp_path):
    dst = tmp_path / "start.sh"
    shutil.copy(START_SH, dst)
    return str(dst)


def _registry_with(tmp_path, port):
    path = tmp_path / "registry.yaml"
    path.write_text(f"services:\n  fusion-comfyui:\n    port: {port}\n", encoding="utf-8")
    return str(path)


def _run_start(script, registry, port="12399", extra_env=None):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "FUSION_VENV": os.path.join(os.path.dirname(script), "no-venv"),
        "FUSION_COMFYUI_PORT": port,
        "FUSION_REGISTRY": registry,
    }
    env.update(extra_env or {})
    return subprocess.run(["bash", script, "start"], env=env, capture_output=True, text=True, timeout=60)


class TestRegistrySelfCheck:
    def test_refuses_unregistered_port(self, script_copy, tmp_path):
        r = _run_start(script_copy, _registry_with(tmp_path, "11445"))
        assert r.returncode == 1
        assert "refusing to start" in r.stdout + r.stderr

    def test_passes_with_entry_present(self, script_copy, tmp_path):
        r = _run_start(script_copy, _registry_with(tmp_path, "12399"))
        assert r.returncode == 2
        assert "venv python not found" in r.stdout + r.stderr

    def test_bypass_env_skips_check(self, script_copy, tmp_path):
        r = _run_start(
            script_copy,
            _registry_with(tmp_path, "11445"),
            extra_env={"FUSION_SKIP_REGISTRY_CHECK": "1"},
        )
        assert r.returncode == 2
        assert "registry check bypassed" in r.stdout + r.stderr

    def test_missing_registry_file_warns_and_continues(self, script_copy, tmp_path):
        r = _run_start(script_copy, str(tmp_path / "nonexistent.yaml"))
        assert r.returncode == 2
        assert "registry self-check skipped" in r.stdout + r.stderr


class TestLegacyKeyCleanup:
    def test_no_legacy_key_literals_in_start_sh(self):
        with open(START_SH, encoding="utf-8") as f:
            content = f.read()
        assert not re.search(r"(?<![A-Z0-9_])FUSION_MLX_API_KEY", content)
        assert not re.search(r"(?<![A-Z0-9_])MLX_API_KEY", content)
        assert not re.search(r"(?<![A-Z0-9_])FUSION_API_KEY", content)

    def test_unset_glob_covers_canonical_and_legacy(self, tmp_path):
        snippet = tmp_path / "snippet.sh"
        snippet.write_text(
            "export FUSION_CLIENT_KEY=canon\n"
            "export FUSION_MLX_API_KEY=stale\n"
            "export FUSION_IMAGE_TIMEOUT=keep\n"
            "while IFS='=' read -r _env_k _; do\n"
            '    case "$_env_k" in\n'
            '        FUSION_*KEY|MLX_*KEY) unset "$_env_k" ;;\n'
            "    esac\n"
            "done < <(env)\n"
            'echo "client=${FUSION_CLIENT_KEY-unset} mlx=${FUSION_MLX_API_KEY-unset} timeout=${FUSION_IMAGE_TIMEOUT-keep}"\n',
            encoding="utf-8",
        )
        r = subprocess.run(["bash", str(snippet)], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        assert "client=unset" in r.stdout
        assert "mlx=unset" in r.stdout
        assert "timeout=keep" in r.stdout
