import logging
from typing import Any

from fusion_comfyui.dag.types import NodeDef, LinkDef, Workflow

logger = logging.getLogger("fusion_comfyui.dag.executor")


def parse_workflow(prompt_data: dict) -> Workflow:
    wf = Workflow()
    if not isinstance(prompt_data, dict):
        return wf
    # Detect frontend workflow format (top-level "nodes" list + "links" list)
    if "nodes" in prompt_data and isinstance(prompt_data["nodes"], list):
        return _parse_frontend_workflow(prompt_data)
    # API format: {node_id: {class_type, inputs}} or {"prompt": {node_id: ...}}
    raw_nodes = prompt_data.get("prompt", prompt_data)
    if not isinstance(raw_nodes, dict):
        return wf
    for nid, node_data in raw_nodes.items():
        if not isinstance(node_data, dict):
            continue
        nid = str(nid)
        wf.nodes[nid] = NodeDef(
            id=nid,
            type=node_data.get("class_type", "Unknown"),
            inputs=node_data.get("inputs", {}),
        )
    return wf


def _parse_frontend_workflow(data: dict) -> Workflow:
    wf = Workflow()
    nodes_list = data.get("nodes", [])
    links_list = data.get("links", [])
    # Build node index
    node_by_id = {}
    for n in nodes_list:
        nid = str(n.get("id", ""))
        if not nid:
            continue
        class_type = n.get("type", "Unknown")
        # If type is a UUID (ComfyUI v0.3+ definitions), look up from definitions
        definitions = data.get("definitions", {})
        subgraphs = definitions.get("subgraphs", []) if isinstance(definitions, dict) else []
        if _is_uuid(class_type) and subgraphs:
            for sub in subgraphs:
                if isinstance(sub, dict) and sub.get("id") == class_type:
                    class_type = sub.get("name", class_type)
                    break
        # Build inputs from widgets_values and linked inputs
        inputs = _extract_frontend_inputs(n, links_list)
        node_by_id[nid] = NodeDef(id=nid, type=class_type, inputs=inputs)
        wf.nodes[nid] = node_by_id[nid]
    # Build links
    # Frontend link format: [link_id, src_node_id, src_slot, dst_node_id, dst_slot, type_str]
    for link in links_list:
        if not isinstance(link, list) or len(link) < 5:
            continue
        link_id = str(link[0])
        src_node = str(link[1])
        src_slot = link[2]
        dst_node = str(link[3])
        dst_slot = link[4]
        type_str = link[5] if len(link) > 5 else "UNKNOWN"
        wf.links.append(LinkDef(
            id=link_id,
            src_node=src_node,
            src_slot=src_slot,
            src_type=type_str,
            dst_node=dst_node,
            dst_slot=dst_slot,
            dst_type=type_str,
        ))
    logger.info("parsed frontend workflow: %d nodes, %d links", len(wf.nodes), len(wf.links))
    return wf


def _extract_frontend_inputs(node_data: dict, links_list: list) -> dict:
    inputs = {}
    node_data.get("id")
    # Linked inputs from node's "inputs" array
    for inp in node_data.get("inputs", []):
        link_id = inp.get("link")
        name = inp.get("name", "")
        if link_id is not None:
            # Find the source node from links
            for link in links_list:
                if isinstance(link, list) and len(link) >= 5 and link[0] == link_id:
                    inputs[name] = [str(link[1]), link[2]]
                    break
    # Widget values
    widgets = node_data.get("widgets_values", [])
    widget_names = _infer_widget_names(node_data)
    for i, val in enumerate(widgets):
        if i < len(widget_names):
            key = widget_names[i]
            if key not in inputs:
                inputs[key] = val
    return inputs


def _infer_widget_names(node_data: dict) -> list[str]:
    # Use input names that don't have links (widget inputs)
    names = []
    for inp in node_data.get("inputs", []):
        if inp.get("link") is None:
            widget = inp.get("widget", {})
            if isinstance(widget, dict) and widget.get("name"):
                names.append(widget["name"])
    return names


def _is_uuid(s: str) -> bool:
    if not s or len(s) != 36:
        return False
    return s.count("-") == 4


class DAGExecutor:
    def __init__(self, node_registry: dict[str, type]):
        self.registry = node_registry
        self._outputs: dict[str, Any] = {}
        logger.info("DAGExecutor: registry has %d node types", len(self.registry))

    async def execute(self, workflow: Workflow, progress_cb=None) -> dict[str, Any]:
        errors = workflow.validate()
        if errors:
            logger.error("workflow validation failed: %s", errors)
            return {"status": "error", "errors": errors}
        order = workflow.topo_order()
        total = len(order)
        self._outputs = {}
        for idx, nid in enumerate(order):
            from fusion_comfyui.server.protocol import check_interrupt
            if check_interrupt():
                logger.warning("workflow interrupted at node %s", nid)
                return {"status": "interrupted", "errors": ["interrupted by user"]}
            ndef = workflow.nodes[nid]
            if ndef.type not in self.registry:
                logger.error("unknown node type: %s", ndef.type)
                return {"status": "error", "errors": [f"unknown node type: {ndef.type}"]}
            node_cls = self.registry[ndef.type]
            node = node_cls()
            resolved_inputs = self._resolve_inputs(ndef.inputs, workflow, nid)
            logger.info("executing [%d/%d] node=%s type=%s", idx + 1, total, nid, ndef.type)
            try:
                result = await node.execute(**resolved_inputs)
            except Exception as e:
                logger.exception("node %s (%s) failed", nid, ndef.type)
                return {"status": "error", "errors": [f"node {nid}: {e}"]}
            self._outputs[nid] = result
            if progress_cb:
                await progress_cb(idx + 1, total, nid, ndef.type)
        return {"status": "ok", "outputs": self._outputs}

    def _resolve_inputs(self, raw_inputs: dict, workflow: Workflow, node_id: str) -> dict:
        resolved = {}
        for key, val in raw_inputs.items():
            if isinstance(val, list) and len(val) == 2:
                src_nid, src_slot = str(val[0]), val[1]
                if src_nid in self._outputs:
                    out = self._outputs[src_nid]
                    if isinstance(out, (list, tuple)):
                        resolved[key] = out[src_slot] if src_slot < len(out) else out[-1]
                    else:
                        resolved[key] = out
                else:
                    logger.warning("unresolved link: %s <- %s[%s]", key, src_nid, src_slot)
                    resolved[key] = val
            else:
                resolved[key] = val
        return resolved
