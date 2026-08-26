import pytest


STUBS = [
    ("ConditioningSetMaskStub", "regional-mask conditioning is not supported"),
    ("VAEEncodeForInpaintStub", "PyTorch model layer"),
    ("InpaintModelConditioningStub", "PyTorch model layer"),
    ("ControlNetApplyStub", "PyTorch model layer"),
    ("ControlNetApplyAdvancedStub", "PyTorch model layer"),
    ("PainterNodeStub", "PyTorch model layer"),
    ("QwenImageDiffsynthControlnetStub", "PyTorch model layer"),
]


@pytest.mark.parametrize("stub_name,frag", STUBS)
def test_stub_raises(stub_name, frag):
    from nodes import _deadpath_stubs
    cls = getattr(_deadpath_stubs, stub_name)
    inst = cls()
    fn = getattr(inst, cls.FUNCTION)
    with pytest.raises(NotImplementedError) as exc:
        fn()
    assert frag in str(exc.value)
