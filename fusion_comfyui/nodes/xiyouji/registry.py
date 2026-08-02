import logging

from fusion_comfyui.nodes.xiyouji.pulid import PuLIDIdentityExtract, PuLIDConditioningApply, PuLIDClearID
from fusion_comfyui.nodes.xiyouji.vlm import XiyoujiChapterParser
from fusion_comfyui.nodes.xiyouji.tts import FusionTTS
from fusion_comfyui.nodes.xiyouji.lipsync import FusionLipSync
from fusion_comfyui.nodes.xiyouji.assemble import SceneVideoAssembler, ChapterVideoConcat

logger = logging.getLogger("fusion_comfyui.nodes.xiyouji.registry")

NODE_CLASS_MAPPINGS = {
    "PuLIDIdentityExtract": PuLIDIdentityExtract,
    "PuLIDConditioningApply": PuLIDConditioningApply,
    "PuLIDClearID": PuLIDClearID,
    "XiyoujiChapterParser": XiyoujiChapterParser,
    "FusionTTS": FusionTTS,
    "FusionLipSync": FusionLipSync,
    "SceneVideoAssembler": SceneVideoAssembler,
    "ChapterVideoConcat": ChapterVideoConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PuLIDIdentityExtract": "🎭 PuLID Identity Extract",
    "PuLIDConditioningApply": "🎭 PuLID Conditioning Apply",
    "PuLIDClearID": "🎭 PuLID Clear ID",
    "XiyoujiChapterParser": "📖 Xiyouji Chapter Parser",
    "FusionTTS": "🔊 Fusion TTS",
    "FusionLipSync": "👄 Fusion Lip Sync",
    "SceneVideoAssembler": "🎬 Scene Video Assembler",
    "ChapterVideoConcat": "🎬 Chapter Video Concat",
}
