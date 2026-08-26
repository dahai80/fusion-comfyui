import logging

logger = logging.getLogger("fusion_comfyui.nodes.deadpath_stubs")


def _stub_factory(native_cls_name, native_module, message):
    import importlib
    try:
        mod = importlib.import_module(native_module)
        native = getattr(mod, native_cls_name)
    except (ImportError, AttributeError) as e:
        logger.warning("deadpath stub: native %s not found (%s)", native_cls_name, e)
        native = object

    class _Stub(native):
        pass

    _Stub.FUNCTION = "stub_run"

    def stub_run(self, *args, **kwargs):
        logger.warning("deadpath stub %s invoked (upstream issue #653); raising", native_cls_name)
        raise NotImplementedError(message)

    _Stub.stub_run = stub_run
    _Stub.__name__ = native_cls_name + "Stub"
    _Stub.__qualname__ = _Stub.__name__
    return _Stub


ConditioningSetMaskStub = _stub_factory(
    "ConditioningSetMask", "nodes",
    "ConditioningSetMask: regional-mask conditioning is not supported on the fusion-mlx "
    "pipeline (engine has no mask hook, upstream issue #653); use Fusion* nodes or wait "
    "for the upstream mask-conditioning surface.",
)
VAEEncodeForInpaintStub = _stub_factory(
    "VAEEncodeForInpaint", "nodes",
    "VAEEncodeForInpaint: VAE-encode + mask surfaces not exposed on the fusion-mlx engine "
    "(upstream issue #653); use the Fusion* equivalent or wait for the upstream surface.",
)
InpaintModelConditioningStub = _stub_factory(
    "InpaintModelConditioning", "nodes",
    "InpaintModelConditioning: inpaint/mask conditioning surface not exposed on the fusion-mlx "
    "engine (upstream issue #653); use the Fusion* equivalent or wait for the upstream surface.",
)
ControlNetApplyStub = _stub_factory(
    "ControlNetApply", "nodes",
    "ControlNetApply: controlnet conditioning surface not exposed on the fusion-mlx engine "
    "(upstream issue #653); use the Fusion* equivalent or wait for the upstream surface.",
)
ControlNetApplyAdvancedStub = _stub_factory(
    "ControlNetApplyAdvanced", "nodes",
    "ControlNetApplyAdvanced: controlnet conditioning surface not exposed on the fusion-mlx "
    "engine (upstream issue #653); use the Fusion* equivalent or wait for the upstream surface.",
)
QwenImageDiffsynthControlnetStub = _stub_factory(
    "QwenImageDiffsynthControlnet", "comfy_extras.nodes_model_patch",
    "QwenImageDiffsynthControlnet: controlnet conditioning surface not exposed on the fusion-mlx "
    "engine (upstream issue #653); use the Fusion* equivalent or wait for the upstream surface.",
)
