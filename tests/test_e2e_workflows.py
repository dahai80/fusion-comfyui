"""
End-to-end workflow tests for fusion-comfyui.
Submits real workflows through ComfyUI API and verifies output.

Uses Fusion-MLX shortcut nodes (FusionVideoGen, FusionImageGen, FusionImageToVideo)
which test the full generation pipeline in a single node.

Usage:
    pytest tests/test_e2e_workflows.py -v --timeout=600
"""

import json
import logging
import time
import urllib.request
import urllib.error

import pytest

logger = logging.getLogger("test_e2e_workflows")

COMFYUI_URL = "http://127.0.0.1:11443"
POLL_INTERVAL = 5
MAX_POLL_SECONDS = 3600


def _api_get(path):
    url = f"{COMFYUI_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _api_post(path, data):
    url = f"{COMFYUI_URL}{path}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _wait_for_prompt(prompt_id):
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        try:
            hist = _api_get(f"/history/{prompt_id}")
        except Exception:
            hist = {}
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            status = hist[prompt_id].get("status", {})
            if status.get("completed", False) or status.get("status_str") == "success":
                return outputs
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"Workflow failed: {msgs}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {MAX_POLL_SECONDS}s")


def _submit_workflow(workflow):
    data = {"prompt": workflow, "client_id": "e2e-test"}
    result = _api_post("/prompt", data)
    pid = result.get("prompt_id")
    logger.info("Submitted prompt_id=%s", pid)
    outputs = _wait_for_prompt(pid)
    logger.info("Completed prompt_id=%s outputs=%s", pid, list(outputs.keys()))
    return outputs


def _check_comfyui_alive():
    try:
        _api_get("/system_stats")
        return True
    except Exception:
        return False


def _available_node_types():
    try:
        data = _api_get("/object_info")
        return set(data.keys())
    except Exception:
        return set()


REQUIRED_SHORTCUT_NODES = {"FusionImageGen", "FusionVideoGen", "FusionSaveVideo"}


@pytest.fixture(scope="session", autouse=True)
def comfyui_running():
    if not _check_comfyui_alive():
        pytest.skip("ComfyUI not running on port 11443")
    available = _available_node_types()
    missing = REQUIRED_SHORTCUT_NODES - available
    if missing:
        pytest.skip(
            f"server on 11443 missing Phase-1 plugin shortcut nodes {sorted(missing)}; "
            f"run e2e against ComfyUI phase-1 server (loads fusion_comfyui_plugin nodes)"
        )


def _verify_output(outputs):
    has_output = False
    for node_id, node_out in outputs.items():
        if "videos" in node_out:
            has_output = True
            logger.info("  video output: %s", node_out["videos"])
            assert len(node_out["videos"]) > 0, "Expected at least one video"
        if "images" in node_out:
            has_output = True
            logger.info("  image output: %s", node_out["images"])
            assert len(node_out["images"]) > 0, "Expected at least one image"
    assert has_output, f"No video/image in outputs: {outputs}"


# ─── t2v: ModelLoader → VideoGen → SaveVideo ─────────────────────


def _build_t2v_shortcut(model_name, prompt, width, height, num_frames,
                         steps, cfg, seed, prefix):
    return {
        "1": {
            "class_type": "FusionModelLoader",
            "inputs": {
                "model_name": model_name,
                "offload_strategy": "sequential",
                "quant_bit": "fp8_e4m3",
            },
        },
        "2": {
            "class_type": "FusionVideoGen",
            "inputs": {
                "pipeline": ["1", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": 16,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
            },
        },
        "3": {
            "class_type": "FusionSaveVideo",
            "inputs": {
                "images": ["2", 0],
                "filename_prefix": prefix,
                "fps": 16,
                "codec": "libx264",
                "crf": 18,
            },
        },
    }


def _build_image_shortcut(model_name, prompt, width, height, steps, cfg, seed, prefix):
    return {
        "1": {
            "class_type": "FusionModelLoader",
            "inputs": {
                "model_name": model_name,
                "offload_strategy": "sequential",
                "quant_bit": "fp8_e4m3",
            },
        },
        "2": {
            "class_type": "FusionImageGen",
            "inputs": {
                "pipeline": ["1", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
            },
        },
        "3": {
            "class_type": "FusionSaveVideo",
            "inputs": {
                "images": ["2", 0],
                "filename_prefix": prefix,
                "fps": 24,
                "codec": "libx264",
                "crf": 18,
            },
        },
    }


# ─── i2v: LoadImage → ModelLoader → ImageToVideo → SaveVideo ─────


def _build_i2v_shortcut(model_name, source_image, prompt, width, height,
                         num_frames, steps, cfg, seed, prefix):
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": source_image,
            },
        },
        "2": {
            "class_type": "FusionModelLoader",
            "inputs": {
                "model_name": model_name,
                "offload_strategy": "sequential",
                "quant_bit": "fp8_e4m3",
            },
        },
        "3": {
            "class_type": "FusionImageToVideo",
            "inputs": {
                "pipeline": ["2", 0],
                "image": ["1", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": 16,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
            },
        },
        "4": {
            "class_type": "FusionSaveVideo",
            "inputs": {
                "images": ["3", 0],
                "filename_prefix": prefix,
                "fps": 16,
                "codec": "libx264",
                "crf": 18,
            },
        },
    }


# ─── ip-adapter: LoadImage → IPAdapterLoader → IPAdapterApply →
#       ModelLoader → IPAdapterInject → SaveVideo ────────────────────


def _build_ipadapter_flux(model_name, source_image, ipadapter_file, prompt,
                           width, height, steps, cfg, seed, prefix):
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": source_image,
            },
        },
        "2": {
            "class_type": "FusionIPAdapterLoader",
            "inputs": {
                "ipadapter": ipadapter_file,
                "siglip_model": "siglip-so400m-patch14-384",
                "num_tokens": 128,
                "dtype": "float16",
            },
        },
        "3": {
            "class_type": "FusionIPAdapterApply",
            "inputs": {
                "ip_adapter_model": ["2", 0],
                "image": ["1", 0],
                "weight": 1.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "4": {
            "class_type": "FusionModelLoader",
            "inputs": {
                "model_name": model_name,
                "offload_strategy": "sequential",
                "quant_bit": "fp8_e4m3",
            },
        },
        "5": {
            "class_type": "FusionIPAdapterInject",
            "inputs": {
                "pipeline": ["4", 0],
                "ip_adapter_embed": ["3", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
            },
        },
        "6": {
            "class_type": "FusionSaveVideo",
            "inputs": {
                "images": ["5", 0],
                "filename_prefix": prefix,
                "fps": 24,
                "codec": "libx264",
                "crf": 18,
            },
        },
    }


# ─── Test cases ───────────────────────────────────────────────────


class TestHunyuanVideo:
    def test_t2v(self):
        wf = _build_t2v_shortcut(
            model_name="hunyuan_video_t2v_720p_bf16.safetensors",
            prompt="A cat walking on a beach at sunset, cinematic lighting",
            width=848, height=480, num_frames=33,
            steps=6, cfg=6.0, seed=42, prefix="e2e_hunyuan_t2v",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)


class TestWan13B:
    def test_t2v(self):
        wf = _build_t2v_shortcut(
            model_name="wan2.1_t2v_1.3B_fp16.safetensors",
            prompt="A dog running in a park, 4k quality",
            width=832, height=480, num_frames=33,
            steps=10, cfg=5.0, seed=42, prefix="e2e_wan13b_t2v",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)


class TestWan22:
    def test_5b_t2v(self):
        wf = _build_t2v_shortcut(
            model_name="wan2.2-5b.safetensors",
            prompt="A bird flying over mountains, cinematic",
            width=832, height=480, num_frames=33,
            steps=10, cfg=5.0, seed=42, prefix="e2e_wan22_5b_t2v",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)


class TestCosmos:
    def test_t2v(self):
        wf = _build_t2v_shortcut(
            model_name="Cosmos-1_0-Diffusion-7B-Text2World.safetensors",
            prompt="A robot dancing in a neon-lit room",
            width=640, height=400, num_frames=25,
            steps=6, cfg=7.0, seed=42, prefix="e2e_cosmos_t2v",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)


class TestVACE:
    def test_vace_t2v(self):
        wf = _build_t2v_shortcut(
            model_name="wan2.1_vace_14B_fp16.safetensors",
            prompt="A person walking forward steadily",
            width=832, height=480, num_frames=33,
            steps=6, cfg=5.0, seed=42, prefix="e2e_vace",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)


class TestFluxImage:
    def test_image(self):
        wf = _build_image_shortcut(
            model_name="flux2-klein-4b.safetensors",
            prompt="A beautiful sunset over mountains, masterpiece",
            width=1024, height=1024,
            steps=10, cfg=3.5, seed=42, prefix="e2e_flux_image",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)


class TestWan22I2V:
    def test_14b_i2v(self):
        wf = _build_i2v_shortcut(
            model_name="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            source_image="sunset.png",
            prompt="The sun sets over the ocean, waves gently lapping, cinematic",
            width=768, height=480, num_frames=33,
            steps=10, cfg=5.0, seed=42, prefix="e2e_wan22_14b_i2v",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)


class TestIPAdapterFlux:
    def test_ipadapter_flux(self):
        wf = _build_ipadapter_flux(
            model_name="flux2-klein-4b.safetensors",
            source_image="sunset.png",
            ipadapter_file="ip_adapter_flux.safetensors",
            prompt="A beautiful landscape with sunset colors",
            width=1024, height=1024,
            steps=10, cfg=4.0, seed=42, prefix="e2e_ipadapter_flux",
        )
        outputs = _submit_workflow(wf)
        _verify_output(outputs)
