from dataclasses import dataclass, field


@dataclass
class StageContext:
    model_wrapper: object
    latent: object = field(default=None)
    pos_cond: object = field(default=None)
    neg_cond: object = field(default=None)
    pixels: object = field(default=None)
    model_type: str = "video"
