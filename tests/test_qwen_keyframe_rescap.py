

from fusion_comfyui_plugin.nodes.samplers import _apply_qwen_keyframe_res_cap


class TestQwenKeyframeResCap:
    # AICF generates qwen t2i keyframes at size "2560x1440" (size.ts ratioToSize).
    # H3 i2v DOWNSCALES the condition image to video res anyway (generate.py
    # _load_image_to_latent img.resize((target_w, target_h), BILINEAR)), so 2560
    # detail is wasted; worse, 2560x1440 takes ~9.6min (10x native 1024's 55s) and
    # is out-of-distribution for the qwen DiT (trained at 1024) -> mild quality loss.
    # FUSION_QWEN_KEYFRAME_768P=1 caps the qwen-image txt2img keyframe to 768p short
    # side (mult of 32, aspect preserved), matching the H3 768p video override so the
    # keyframe is generated AT the resolution H3 consumes. Only caps qwen-image
    # txt2img (not img2img init pass, not other image models). Only caps (never raises
    # a request already <= cap), so non-AICF/native callers are unaffected.

    def test_caps_2560x1440_to_768p_16x9(self, monkeypatch):
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        w, h = _apply_qwen_keyframe_res_cap(2560, 1440, "qwen-image-2512", is_img2img=False)
        # short side 1440 -> 768, 16:9 preserved, mult of 32. round(2560*768/1440/32)*32
        # = round(42.67)*32 = 43*32 = 1376. MUST match h3.py _apply_768p_override math
        # (same int(round(.../32))*32) so the keyframe lands at the exact res H3 consumes.
        assert (w, h) == (1376, 768)

    def test_off_keeps_2560x1440(self, monkeypatch):
        monkeypatch.delenv("FUSION_QWEN_KEYFRAME_768P", raising=False)
        w, h = _apply_qwen_keyframe_res_cap(2560, 1440, "qwen-image-2512", is_img2img=False)
        assert (w, h) == (2560, 1440)

    def test_caps_9x16_portrait_to_768p(self, monkeypatch):
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        w, h = _apply_qwen_keyframe_res_cap(1440, 2560, "qwen-image-2512", is_img2img=False)
        # portrait: short side 1440 -> 768, mult of 32 -> 768x1376 (mirror of 16:9)
        assert (w, h) == (768, 1376)

    def test_caps_1x1_square_to_768x768(self, monkeypatch):
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        w, h = _apply_qwen_keyframe_res_cap(2048, 2048, "qwen-image-2512", is_img2img=False)
        assert (w, h) == (768, 768)

    def test_already_at_cap_unchanged(self, monkeypatch):
        # 1376x768 short side 768 == cap -> no change (matches H3 768p video res)
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        w, h = _apply_qwen_keyframe_res_cap(1376, 768, "qwen-image-2512", is_img2img=False)
        assert (w, h) == (1376, 768)

    def test_native_1024_capped_to_768(self, monkeypatch):
        # 1024x1024 short side 1024 > 768 -> capped to 768x768 (matches video res)
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        w, h = _apply_qwen_keyframe_res_cap(1024, 1024, "qwen-image-2512", is_img2img=False)
        assert (w, h) == (768, 768)

    def test_img2img_init_pass_not_capped(self, monkeypatch):
        # img2img hires-fix 2nd pass must NOT be capped — it refines an existing
        # image at the caller's chosen resolution. Only the txt2img keyframe pass caps.
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        w, h = _apply_qwen_keyframe_res_cap(2560, 1440, "qwen-image-2512", is_img2img=True)
        assert (w, h) == (2560, 1440)

    def test_non_qwen_model_not_capped(self, monkeypatch):
        # Only qwen-image keyframes are capped; flux/sd3/etc unaffected.
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        w, h = _apply_qwen_keyframe_res_cap(2560, 1440, "flux2-klein-4b", is_img2img=False)
        assert (w, h) == (2560, 1440)

    def test_mult_of_32_preserved(self, monkeypatch):
        # H3 patchify /2 needs even latent dims -> width/height mult of 32.
        monkeypatch.setenv("FUSION_QWEN_KEYFRAME_768P", "1")
        for w_in, h_in in [(2560, 1440), (1440, 2560), (2048, 2048), (1920, 1080)]:
            w, h = _apply_qwen_keyframe_res_cap(w_in, h_in, "qwen-image", is_img2img=False)
            assert w % 32 == 0, f"{w_in}x{h_in} -> w={w} not mult of 32"
            assert h % 32 == 0, f"{w_in}x{h_in} -> h={h} not mult of 32"
