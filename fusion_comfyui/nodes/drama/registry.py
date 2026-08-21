import logging

from fusion_comfyui.nodes.drama.pulid import PuLIDIdentityExtract, PuLIDConditioningApply, PuLIDClearID
from fusion_comfyui.nodes.drama.vlm import DramaChapterParser
from fusion_comfyui.nodes.drama.tts import FusionTTS
from fusion_comfyui.nodes.drama.lipsync import FusionLipSync
from fusion_comfyui.nodes.drama.assemble import SceneVideoAssembler, ChapterVideoConcat

logger = logging.getLogger("fusion_comfyui.nodes.drama.registry")

NODE_CLASS_MAPPINGS = {
    "PuLIDIdentityExtract": PuLIDIdentityExtract,
    "PuLIDConditioningApply": PuLIDConditioningApply,
    "PuLIDClearID": PuLIDClearID,
    "DramaChapterParser": DramaChapterParser,
    "FusionTTS": FusionTTS,
    "FusionLipSync": FusionLipSync,
    "SceneVideoAssembler": SceneVideoAssembler,
    "ChapterVideoConcat": ChapterVideoConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PuLIDIdentityExtract": "🎭 PuLID Identity Extract",
    "PuLIDConditioningApply": "🎭 PuLID Conditioning Apply",
    "PuLIDClearID": "🎭 PuLID Clear ID",
    "DramaChapterParser": "📖 Drama Chapter Parser",
    "FusionTTS": "🔊 Fusion TTS",
    "FusionLipSync": "👄 Fusion Lip Sync",
    "SceneVideoAssembler": "🎬 Scene Video Assembler",
    "ChapterVideoConcat": "🎬 Chapter Video Concat",
}
