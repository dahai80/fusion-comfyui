import pytest

from fusion_comfyui.nodes.base import BaseNode


class TestBaseNode:
    def test_return_types_default(self):
        assert BaseNode.RETURN_TYPES == ()

    def test_category_default(self):
        assert BaseNode.CATEGORY == "fusion-mlX"

    def test_function_default(self):
        assert BaseNode.FUNCTION == "execute"

    def test_input_types_default(self):
        assert BaseNode.INPUT_TYPES() == {"required": {}}

    @pytest.mark.asyncio
    async def test_abstract_execute(self):
        with pytest.raises(TypeError):
            BaseNode()

    @pytest.mark.asyncio
    async def test_concrete_subclass(self):
        class ConcreteNode(BaseNode):
            RETURN_TYPES = ("STRING",)

            async def execute(self, **kwargs):
                return ("result",)

        node = ConcreteNode()
        result = await node.execute()
        assert result == ("result",)

    @pytest.mark.asyncio
    async def test_subclass_input_types(self):
        class NodeWithInputs(BaseNode):
            RETURN_TYPES = ("INT",)

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"value": ("INT", {"default": 0})}}

            async def execute(self, **kwargs):
                return (kwargs.get("value", 0),)

        inputs = NodeWithInputs.INPUT_TYPES()
        assert "required" in inputs
        assert "value" in inputs["required"]
