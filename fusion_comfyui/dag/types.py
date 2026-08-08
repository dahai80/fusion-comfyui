import logging
from dataclasses import dataclass, field

logger = logging.getLogger("fusion_comfyui.dag.types")

KNOWN_TYPES = {
    "MODEL", "CONDITIONING", "LATENT", "IMAGE", "VAE",
    "MASK", "CLIP", "INT", "FLOAT", "STRING", "COMBO",
}

VALID_LINKS = {
    ("MODEL", "MODEL"),
    ("CONDITIONING", "CONDITIONING"),
    ("LATENT", "LATENT"),
    ("IMAGE", "IMAGE"),
    ("VAE", "VAE"),
    ("MASK", "MASK"),
    ("CLIP", "CLIP"),
    ("INT", "INT"),
    ("FLOAT", "FLOAT"),
    ("STRING", "STRING"),
}


@dataclass
class NodeDef:
    id: str
    type: str
    inputs: dict = field(default_factory=dict)


@dataclass
class LinkDef:
    id: str
    src_node: str
    src_slot: int
    src_type: str
    dst_node: str
    dst_slot: int
    dst_type: str


@dataclass
class Workflow:
    nodes: dict[str, NodeDef] = field(default_factory=dict)
    links: list[LinkDef] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors = []
        for link in self.links:
            if link.dst_node not in self.nodes:
                errors.append(f"link {link.id}: dst node {link.dst_node} not found")
            if link.src_node not in self.nodes:
                errors.append(f"link {link.id}: src node {link.src_node} not found")
            if (link.src_type, link.dst_type) not in VALID_LINKS:
                if link.src_type != link.dst_type:
                    errors.append(
                        f"link {link.id}: type mismatch {link.src_type} -> {link.dst_type}"
                    )
        return errors

    def topo_order(self) -> list[str]:
        deps: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        for link in self.links:
            if link.dst_node in deps:
                deps[link.dst_node].add(link.src_node)
        order = []
        visited = set()
        temp = set()

        def visit(nid):
            if nid in visited:
                return
            if nid in temp:
                logger.error("cycle detected at node %s", nid)
                raise ValueError(f"cycle detected at node {nid}")
            temp.add(nid)
            for dep in deps.get(nid, set()):
                visit(dep)
            temp.discard(nid)
            visited.add(nid)
            order.append(nid)

        for nid in self.nodes:
            visit(nid)
        return order
