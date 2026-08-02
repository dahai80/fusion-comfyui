import pytest

from fusion_comfyui.dag.types import NodeDef, LinkDef, Workflow, KNOWN_TYPES, VALID_LINKS


class TestKnownTypes:
    def test_core_types_present(self):
        for t in ("MODEL", "CONDITIONING", "LATENT", "IMAGE", "VAE", "MASK", "CLIP"):
            assert t in KNOWN_TYPES

    def test_valid_links_symmetric(self):
        for src, dst in VALID_LINKS:
            assert src == dst, f"non-symmetric link: {src} -> {dst}"


class TestNodeDef:
    def test_defaults(self):
        n = NodeDef(id="1", type="FusionModelLoader")
        assert n.inputs == {}

    def test_with_inputs(self):
        n = NodeDef(id="2", type="FusionKSampler", inputs={"steps": 20})
        assert n.inputs["steps"] == 20


class TestWorkflow:
    def test_empty_validate(self):
        wf = Workflow()
        assert wf.validate() == []

    def test_topo_order_empty(self):
        wf = Workflow()
        assert wf.topo_order() == []

    def test_topo_order_linear(self):
        wf = Workflow()
        wf.nodes["1"] = NodeDef(id="1", type="FusionModelLoader")
        wf.nodes["2"] = NodeDef(id="2", type="FusionKSampler")
        wf.links.append(LinkDef(id="l1", src_node="1", src_slot=0, src_type="MODEL", dst_node="2", dst_slot=0, dst_type="MODEL"))
        order = wf.topo_order()
        assert order.index("1") < order.index("2")

    def test_validate_missing_node(self):
        wf = Workflow()
        wf.nodes["1"] = NodeDef(id="1", type="A")
        wf.links.append(LinkDef(id="l1", src_node="1", src_slot=0, src_type="MODEL", dst_node="99", dst_slot=0, dst_type="MODEL"))
        errors = wf.validate()
        assert any("99" in e for e in errors)

    def test_validate_type_mismatch(self):
        wf = Workflow()
        wf.nodes["1"] = NodeDef(id="1", type="A")
        wf.nodes["2"] = NodeDef(id="2", type="B")
        wf.links.append(LinkDef(id="l1", src_node="1", src_slot=0, src_type="MODEL", dst_node="2", dst_slot=0, dst_type="LATENT"))
        errors = wf.validate()
        assert any("mismatch" in e for e in errors)

    def test_topo_order_diamond(self):
        wf = Workflow()
        wf.nodes["1"] = NodeDef(id="1", type="A")
        wf.nodes["2"] = NodeDef(id="2", type="B")
        wf.nodes["3"] = NodeDef(id="3", type="C")
        wf.nodes["4"] = NodeDef(id="4", type="D")
        wf.links.append(LinkDef(id="l1", src_node="1", src_slot=0, src_type="MODEL", dst_node="2", dst_slot=0, dst_type="MODEL"))
        wf.links.append(LinkDef(id="l2", src_node="1", src_slot=0, src_type="MODEL", dst_node="3", dst_slot=0, dst_type="MODEL"))
        wf.links.append(LinkDef(id="l3", src_node="2", src_slot=0, src_type="MODEL", dst_node="4", dst_slot=0, dst_type="MODEL"))
        wf.links.append(LinkDef(id="l4", src_node="3", src_slot=0, src_type="MODEL", dst_node="4", dst_slot=0, dst_type="MODEL"))
        order = wf.topo_order()
        assert order.index("1") < order.index("2")
        assert order.index("1") < order.index("3")
        assert order.index("2") < order.index("4")
        assert order.index("3") < order.index("4")
