import pytest

from fusion_comfyui.dag.executor import parse_workflow, DAGExecutor
from fusion_comfyui.dag.types import Workflow, NodeDef, LinkDef
from fusion_comfyui.nodes.base import BaseNode


class _DummyNode(BaseNode):
    RETURN_TYPES = ("STRING",)

    async def execute(self, **kwargs):
        return ("hello",)


class _AddNode(BaseNode):
    RETURN_TYPES = ("INT",)

    async def execute(self, a=0, b=0, **kwargs):
        return (a + b,)


class TestParseWorkflow:
    def test_basic_prompt_format(self):
        data = {
            "1": {"class_type": "FusionModelLoader", "inputs": {"model_name": "flux2-dev"}},
            "2": {"class_type": "FusionKSampler", "inputs": {"model": ["1", 0], "steps": 20}},
        }
        wf = parse_workflow(data)
        assert "1" in wf.nodes
        assert wf.nodes["1"].type == "FusionModelLoader"
        assert wf.nodes["1"].inputs["model_name"] == "flux2-dev"
        assert "2" in wf.nodes

    def test_wrapped_prompt_format(self):
        data = {
            "prompt": {
                "1": {"class_type": "FusionModelLoader", "inputs": {}},
            }
        }
        wf = parse_workflow(data)
        assert "1" in wf.nodes

    def test_empty_input(self):
        wf = parse_workflow({})
        assert len(wf.nodes) == 0

    def test_non_dict_input(self):
        wf = parse_workflow("not a dict")
        assert len(wf.nodes) == 0

    def test_skips_non_dict_node_data(self):
        data = {"1": "bad", "2": {"class_type": "OK", "inputs": {}}}
        wf = parse_workflow(data)
        assert "1" not in wf.nodes
        assert "2" in wf.nodes


class TestParseFrontendWorkflow:
    def test_basic_frontend_format(self):
        data = {
            "nodes": [
                {"id": 1, "type": "FusionModelLoader", "inputs": [], "widgets_values": ["flux2-dev"]},
                {"id": 2, "type": "FusionKSampler", "inputs": [
                    {"name": "model", "type": "MODEL", "link": 1},
                ], "widgets_values": [20, 6.0, 0, 1024, 1024]},
            ],
            "links": [
                [1, 1, 0, 2, 0, "MODEL"],
            ],
        }
        wf = parse_workflow(data)
        assert "1" in wf.nodes
        assert wf.nodes["1"].type == "FusionModelLoader"
        assert "2" in wf.nodes
        assert len(wf.links) == 1
        assert wf.links[0].src_node == "1"
        assert wf.links[0].dst_node == "2"

    def test_frontend_with_no_links(self):
        data = {
            "nodes": [
                {"id": 1, "type": "FusionModelLoader", "inputs": [], "widgets_values": []},
            ],
            "links": [],
        }
        wf = parse_workflow(data)
        assert len(wf.nodes) == 1
        assert len(wf.links) == 0

    def test_frontend_widget_values_to_inputs(self):
        data = {
            "nodes": [
                {
                    "id": 1,
                    "type": "FusionKSampler",
                    "inputs": [
                        {"name": "steps", "type": "INT", "link": None, "widget": {"name": "steps"}},
                    ],
                    "widgets_values": [20],
                },
            ],
            "links": [],
        }
        wf = parse_workflow(data)
        assert wf.nodes["1"].inputs.get("steps") == 20

    def test_frontend_linked_input_resolved(self):
        data = {
            "nodes": [
                {"id": 1, "type": "Loader", "inputs": [], "widgets_values": []},
                {"id": 2, "type": "Sampler", "inputs": [
                    {"name": "model", "type": "MODEL", "link": 42},
                ], "widgets_values": []},
            ],
            "links": [
                [42, 1, 0, 2, 0, "MODEL"],
            ],
        }
        wf = parse_workflow(data)
        assert wf.nodes["2"].inputs["model"] == ["1", 0]


class TestDAGExecutor:
    @pytest.mark.asyncio
    async def test_single_node(self):
        registry = {"Dummy": _DummyNode}
        executor = DAGExecutor(registry)
        wf = Workflow(nodes={"1": NodeDef(id="1", type="Dummy", inputs={})}, links=[])
        result = await executor.execute(wf)
        assert result["status"] == "ok"
        assert "1" in result["outputs"]

    @pytest.mark.asyncio
    async def test_unknown_node_type(self):
        registry = {"Dummy": _DummyNode}
        executor = DAGExecutor(registry)
        wf = Workflow(nodes={"1": NodeDef(id="1", type="Missing")}, links=[])
        result = await executor.execute(wf)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_chained_nodes(self):
        registry = {"Add": _AddNode}
        executor = DAGExecutor(registry)
        wf = Workflow(
            nodes={
                "1": NodeDef(id="1", type="Add", inputs={"a": 1, "b": 2}),
                "2": NodeDef(id="2", type="Add", inputs={"a": ["1", 0], "b": 10}),
            },
            links=[
                LinkDef(id="L1", src_node="1", src_slot=0, src_type="INT",
                        dst_node="2", dst_slot=0, dst_type="INT"),
            ],
        )
        result = await executor.execute(wf)
        assert result["status"] == "ok"
        assert result["outputs"]["1"] == (3,)
        assert result["outputs"]["2"] == (13,)

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        registry = {"Dummy": _DummyNode}
        executor = DAGExecutor(registry)
        wf = Workflow(nodes={"1": NodeDef(id="1", type="Dummy", inputs={})}, links=[])
        progress = []

        async def cb(step, total, nid, ntype):
            progress.append((step, total, nid))

        await executor.execute(wf, progress_cb=cb)
        assert len(progress) == 1
        assert progress[0] == (1, 1, "1")

    @pytest.mark.asyncio
    async def test_node_event_cb_start_end(self):
        registry = {"Dummy": _DummyNode}
        executor = DAGExecutor(registry)
        wf = Workflow(nodes={"1": NodeDef(id="1", type="Dummy", inputs={})}, links=[])
        events = []

        async def evt(phase, nid, ntype, current, total):
            events.append((phase, nid, ntype, current, total))

        result = await executor.execute(wf, node_event_cb=evt)
        assert result["status"] == "ok"
        assert events == [("start", "1", "Dummy", 1, 1), ("end", "1", "Dummy", 1, 1)]

    @pytest.mark.asyncio
    async def test_cycle_detected_fails_visibly(self):
        registry = {"Dummy": _DummyNode}
        executor = DAGExecutor(registry)
        wf = Workflow(
            nodes={
                "1": NodeDef(id="1", type="Dummy", inputs={"x": ["2", 0]}),
                "2": NodeDef(id="2", type="Dummy", inputs={"x": ["1", 0]}),
            },
            links=[
                LinkDef(id="L1", src_node="1", src_slot=0, src_type="STRING",
                        dst_node="2", dst_slot=0, dst_type="STRING"),
                LinkDef(id="L2", src_node="2", src_slot=0, src_type="STRING",
                        dst_node="1", dst_slot=0, dst_type="STRING"),
            ],
        )
        result = await executor.execute(wf)
        assert result["status"] == "error"
        assert any("cycle" in str(e) for e in result["errors"])

    @pytest.mark.asyncio
    async def test_unresolved_link_fails_visibly(self):
        registry = {"Add": _AddNode}
        executor = DAGExecutor(registry)
        wf = Workflow(
            nodes={
                "1": NodeDef(id="1", type="Add", inputs={"a": ["99", 0], "b": 10}),
            },
            links=[],
        )
        result = await executor.execute(wf)
        assert result["status"] == "error"
        assert any("unresolved link" in str(e) for e in result["errors"])

