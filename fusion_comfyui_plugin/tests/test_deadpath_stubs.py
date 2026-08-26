import pytest


STUBS = [
    ("ConditioningSetMaskStub", "upstream issue #653"),
    ("VAEEncodeForInpaintStub", "upstream issue #653"),
    ("InpaintModelConditioningStub", "upstream issue #653"),
    ("ControlNetApplyStub", "upstream issue #653"),
    ("ControlNetApplyAdvancedStub", "upstream issue #653"),
    ("QwenImageDiffsynthControlnetStub", "upstream issue #653"),
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
