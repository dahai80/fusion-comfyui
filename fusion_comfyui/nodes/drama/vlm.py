import logging
import re

from fusion_comfyui.nodes.base import BaseNode
from fusion_comfyui.core.timer import NodeTimer
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian

logger = logging.getLogger("fusion_comfyui.nodes.drama.vlm")

SPLIT_PROMPT = """将以下小说/剧本章节拆分为3-5个场景。输出纯JSON数组，不要markdown包裹。

每个场景字段:
- scene_id: 整数
- description_en: 英文画面描述(1-2句话，简洁，用于AI生图)
- description_cn: 中文场景描述(1句话，用于字幕)
- characters: 角色列表(英文标识符，如主角A/protagonist等，留空则自动推断)
- dialogue: 对话列表[{{speaker,text}}]
- scene_type: battle/dialogue/travel/transformation
- duration_seconds: 3-8

要求: description_en简洁，不超过50词。JSON必须完整闭合。

章节内容:
{chapter_text}"""

DESCRIPTION_PROMPT = """为以下场景写一句英文画面描述(用于AI图像生成)。
要求: 简洁，不超过50词，描述画面内容，不要对话。

场景: {scene_cn}

输出格式: 直接输出英文描述，不要其他内容。"""


class DramaChapterParser(BaseNode):
    RETURN_TYPES = ("SCENE_SCRIPTS",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chapter_text": ("STRING", {"multiline": True, "default": ""}),
                "vlm_model": ("STRING", {"default": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"}),
                "max_tokens": ("INT", {"default": 4096, "min": 512, "max": 16384}),
            }
        }

    async def execute(self, chapter_text, vlm_model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit", max_tokens=4096):
        async with NodeTimer.timed("DramaChapterParser", "full", text_len=len(chapter_text)):
            scenes = self._split_text_to_scenes(chapter_text)

            from fusion_mlx.public_api import VLMBatchedEngine

            async with NodeTimer.timed("DramaChapterParser", "load_vlm"):
                engine = VLMBatchedEngine(vlm_model)
                await engine.start()
                logger.info("DramaChapterParser: VLM started %s", vlm_model)

            async with NodeTimer.timed("DramaChapterParser", "vlm_generate"):
                for scene in scenes:
                    if not scene.get("description_en"):
                        desc_cn = scene.get("description_cn", "")
                        if desc_cn:
                            prompt = DESCRIPTION_PROMPT.format(scene_cn=desc_cn[:200])
                            messages = [{"role": "user", "content": prompt}]
                            try:
                                result = await engine.chat(
                                    messages=messages,
                                    max_tokens=200,
                                    temperature=0.3,
                                    repetition_penalty=1.2,
                                )
                                en_desc = result.text if hasattr(result, "text") else str(result)
                                en_desc = en_desc.strip().strip('"').strip()
                                if en_desc:
                                    scene["description_en"] = en_desc
                            except Exception as e:
                                logger.warning("VLM en-desc failed: %s", e)

            async with NodeTimer.timed("DramaChapterParser", "unload_vlm"):
                await engine.stop()
                FusionMemoryGuardian.purge_memory()

            logger.info("DramaChapterParser: parsed %d scenes", len(scenes))
            return (scenes,)

    SCENE_TYPE_EN = {
        "battle": "an intense battle scene with weapons clashing and magical energies",
        "travel": "a journey through mountains, rivers, and mystical landscapes",
        "dialogue": "characters in conversation with expressive gestures",
        "transformation": "a magical transformation with swirling energy and light",
    }

    def split_only(self, chapter_text: str) -> list:
        scenes = self._split_text_to_scenes(chapter_text)
        char_names = {
            "sunwukong": "Sun Wukong the Monkey King",
            "tangseng": "Tang Seng the monk",
            "zhubajie": "Zhu Bajie the pig demon",
            "shaseng": "Sha Seng the monk",
            "bailongma": "Bai Long Ma the white dragon horse",
        }
        for scene in scenes:
            if not scene.get("description_en"):
                scene_type = scene.get("scene_type", "dialogue")
                chars = scene.get("characters", [])
                scene_en = self.SCENE_TYPE_EN.get(scene_type, self.SCENE_TYPE_EN["dialogue"])
                char_desc = ", ".join(char_names.get(c, c) for c in chars[:3])
                if char_desc:
                    scene["description_en"] = f"{scene_en}, featuring {char_desc}, traditional Chinese painting style, epic fantasy art"
                else:
                    scene["description_en"] = f"{scene_en}, traditional Chinese painting style, epic fantasy art"
                logger.info("split_only: scene %d -> %s", scene.get("scene_id", 0), scene["description_en"][:80])
        return scenes

    def _split_text_to_scenes(self, chapter_text: str) -> list:
        text = chapter_text.strip()
        if not text:
            return [{"scene_id": 1, "description_en": "Empty scene", "description_cn": "", "characters": [], "dialogue": [], "scene_type": "dialogue", "duration_seconds": 5.0}]

        segments = re.split(r'(?=却说|且说|话说|忽一日|次日|当下|此时|不多时|须臾|半晌)', text)
        segments = [s for s in segments if s.strip()]

        if len(segments) < 3:
            sentences = re.split(r'[。！？；]', text)
            chunk_size = max(1, len(sentences) // 4)
            segments = []
            for i in range(0, len(sentences), chunk_size):
                chunk = "。".join(sentences[i:i + chunk_size]).strip()
                if chunk:
                    segments.append(chunk)

        if len(segments) > 6:
            step = len(segments) / 5
            merged = []
            for i in range(5):
                start = int(i * step)
                end = int((i + 1) * step)
                merged.append("。".join(segments[start:end]))
            segments = merged

        scenes = []
        for idx, seg in enumerate(segments):
            scene = self._parse_segment(seg, idx + 1)
            scenes.append(scene)

        return scenes

    def _parse_segment(self, text: str, scene_id: int) -> dict:
        characters = []
        char_map = {
            "悟空": "sunwukong", "行者": "sunwukong", "大圣": "sunwukong",
            "猴王": "sunwukong", "孙悟空": "sunwukong", "美猴王": "sunwukong",
            "唐僧": "tangseng", "三藏": "tangseng", "玄奘": "tangseng",
            "师父": "tangseng", "御弟": "tangseng",
            "八戒": "zhubajie", "猪八戒": "zhubajie", "呆子": "zhubajie",
            "天蓬": "zhubajie",
            "沙僧": "shaseng", "沙悟净": "shaseng", "沙和尚": "shaseng",
            "白马": "bailongma", "龙马": "bailongma", "白龙马": "bailongma",
        }
        for cn, en in char_map.items():
            if cn in text and en not in characters:
                characters.append(en)

        dialogue = []
        for m in re.finditer(r'[""「」](.+?)[""「」]', text):
            quote = m.group(1).strip()
            if len(quote) > 2:
                speaker = "sunwukong"
                for cn, en in char_map.items():
                    if cn in text[:m.start()][-30:]:
                        speaker = en
                        break
                dialogue.append({"speaker": speaker, "text": quote})

        has_battle = any(w in text for w in ["战", "斗", "打", "杀", "攻", "阵", "兵器", "法宝"])
        has_travel = any(w in text for w in ["行", "走", "路", "山", "水", "林", "洞", "河"])
        scene_type = "battle" if has_battle else ("travel" if has_travel else "dialogue")

        desc_cn = text[:80].strip().replace("\n", " ")
        if len(desc_cn) < 10:
            desc_cn = text.strip()[:80]

        duration = 5.0
        if len(text) > 300:
            duration = 6.0
        if len(text) > 500:
            duration = 7.0
        if has_battle:
            duration = min(duration + 1, 8.0)

        return {
            "scene_id": scene_id,
            "description_en": "",
            "description_cn": desc_cn,
            "characters": characters if characters else ["sunwukong"],
            "dialogue": dialogue,
            "scene_type": scene_type,
            "duration_seconds": duration,
        }
