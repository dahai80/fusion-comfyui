import logging

logger = logging.getLogger("fusion_comfyui.nodes.deadpath_stubs")


def _stub_factory(native_cls_name, native_module, message):
    import importlib
    try:
        mod = importlib.import_module(native_module)
        native = getattr(mod, native_cls_name)
    except Exception as e:
        logger.warning("deadpath stub: native %s not found (%s)", native_cls_name, e)
        native = object

    class _Stub(native):
        pass

    _Stub.FUNCTION = "stub_run"

    def stub_run(self, *args, **kwargs):
        raise NotImplementedError(message)

    _Stub.stub_run = stub_run
    _Stub.__name__ = native_cls_name + "Stub"
    _Stub.__qualname__ = _Stub.__name__
    return _Stub


ConditioningSetMaskStub = _stub_factory(
    "ConditioningSetMask", "nodes",
    "ConditioningSetMask: regional-mask conditioning is not supported on the fusion-mlx "
    "pipeline (engine has no mask hook); use Fusion* nodes or wait for P3 staged conditioning.",
)
VAEEncodeForInpaintStub = _stub_factory(
    "VAEEncodeForInpaint", "nodes",
    "VAEEncodeForInpaint: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
InpaintModelConditioningStub = _stub_factory(
    "InpaintModelConditioning", "nodes",
    "InpaintModelConditioning: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
ControlNetApplyStub = _stub_factory(
    "ControlNetApply", "nodes",
    "ControlNetApply: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
ControlNetApplyAdvancedStub = _stub_factory(
    "ControlNetApplyAdvanced", "nodes",
    "ControlNetApplyAdvanced: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
PainterNodeStub = _stub_factory(
    "PainterNode", "comfy_extras.nodes_painter",
    "PainterNode: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
QwenImageDiffsynthControlnetStub = _stub_factory(
    "QwenImageDiffsynthControlnet", "comfy_extras.nodes_model_patch",
    "QwenImageDiffsynthControlnet: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
