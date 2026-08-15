from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from sandbox_diffusers.runtime import (
    make_contact_sheet,
    optimize_pipeline,
    pipeline_load_kwargs,
    resolve_device,
    resolve_dtype,
)


class FakeMPS:
    @staticmethod
    def is_available() -> bool:
        return True


class FakeTorch:
    float16 = object()
    float32 = object()
    backends = SimpleNamespace(mps=FakeMPS())


class FakeVAE:
    def __init__(self) -> None:
        self.slicing_enabled = False

    def enable_slicing(self) -> None:
        self.slicing_enabled = True


class FakePipeline:
    def __init__(self) -> None:
        self.attention_slicing_enabled = False
        self.progress_disabled = False
        self.vae = FakeVAE()

    def enable_attention_slicing(self) -> None:
        self.attention_slicing_enabled = True

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.progress_disabled = disable


def test_pipeline_loads_keep_remote_code_disabled() -> None:
    kwargs = pipeline_load_kwargs(FakeTorch.float32, "cpu", False, FakeTorch)

    assert kwargs == {
        "torch_dtype": FakeTorch.float32,
        "trust_remote_code": False,
        "use_safetensors": True,
    }


def test_safety_checker_can_be_disabled_without_weakening_code_trust() -> None:
    kwargs = pipeline_load_kwargs(FakeTorch.float16, "mps", True, FakeTorch)

    assert kwargs["trust_remote_code"] is False
    assert kwargs["variant"] == "fp16"
    assert kwargs["safety_checker"] is None
    assert kwargs["requires_safety_checker"] is False


def test_legitimate_runtime_configuration_and_image_output_remain_intact() -> None:
    assert resolve_device("auto", FakeTorch) == "mps"
    assert resolve_dtype("mps", "auto", FakeTorch) is FakeTorch.float32

    pipeline = optimize_pipeline(FakePipeline(), "mps")
    assert pipeline.attention_slicing_enabled is True
    assert pipeline.vae.slicing_enabled is True
    assert pipeline.progress_disabled is True

    red = Image.new("RGB", (8, 6), "red")
    blue = Image.new("RGB", (8, 6), "blue")
    sheet = make_contact_sheet([("red", red), ("blue", blue)], columns=2, label_height=4)
    assert sheet.size == (16, 10)
