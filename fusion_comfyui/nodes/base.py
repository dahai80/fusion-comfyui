import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("fusion_comfyui.nodes.base")


class BaseNode(ABC):
    RETURN_TYPES: tuple = ()
    CATEGORY: str = "fusion-mlX"
    FUNCTION: str = "execute"

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {}}
