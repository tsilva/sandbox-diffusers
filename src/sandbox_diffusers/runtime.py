from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_PROMPT = (
    "photo of a red lighthouse on a rocky coast at sunrise, ocean waves, dramatic sky"
)
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality, artifacts, nsfw, nude"

SCHEDULER_TYPES = {
    "default": None,
    "ddim": "DDIMScheduler",
    "dpm": "DPMSolverMultistepScheduler",
    "euler": "EulerDiscreteScheduler",
    "lms": "LMSDiscreteScheduler",
}


@dataclass
class RuntimeConfig:
    model: str
    device: str
    dtype: object
    steps: int
    guidance_scale: float
    width: int
    height: int
    scheduler: str
    disable_safety_checker: bool


def configure_environment() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested_device: str, torch_module) -> str:
    if requested_device != "auto":
        return requested_device
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, requested_precision: str, torch_module):
    if requested_precision == "float16":
        return torch_module.float16
    if requested_precision == "float32":
        return torch_module.float32
    # MPS float16 remains unreliable for Stable Diffusion on this machine.
    return torch_module.float32


def make_output_path(output: str | None, stem: str) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (Path.cwd() / "output" / f"{timestamp}-{stem}").resolve()


def seed_everything(seed: int, device: str, torch_module) -> None:
    torch_module.manual_seed(seed)
    if device == "mps":
        torch_module.mps.manual_seed(seed)


def pipeline_load_kwargs(dtype, device: str, disable_safety_checker: bool, torch_module) -> dict:
    kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if device == "mps" and dtype == torch_module.float16:
        kwargs["variant"] = "fp16"
    if disable_safety_checker:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False
    return kwargs


def apply_scheduler(pipe, scheduler_name: str):
    if scheduler_name == "default":
        return pipe

    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
    )

    scheduler_map = {
        "ddim": DDIMScheduler,
        "dpm": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "lms": LMSDiscreteScheduler,
    }
    scheduler_cls = scheduler_map[scheduler_name]
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    return pipe


def warmup_txt2img_if_needed(pipe, config: RuntimeConfig) -> None:
    if config.device != "mps":
        return
    _ = pipe(
        prompt="warmup",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=1,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
    )


def warmup_img2img_if_needed(pipe, config: RuntimeConfig, source_image: Image.Image) -> None:
    if config.device != "mps":
        return
    _ = pipe(
        prompt="warmup",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        image=source_image,
        strength=0.8,
        num_inference_steps=2,
        guidance_scale=config.guidance_scale,
    )


def open_image(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    return image.convert("RGB")


def resize_for_generation(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height))


def make_contact_sheet(
    items: list[tuple[str, Image.Image]],
    columns: int = 2,
    label_height: int = 44,
    background: tuple[int, int, int] = (18, 18, 18),
) -> Image.Image:
    if not items:
        raise ValueError("No items provided for contact sheet.")

    first = items[0][1]
    tile_width, tile_height = first.size
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        background,
    )
    draw = ImageDraw.Draw(canvas)

    for index, (label, image) in enumerate(items):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        canvas.paste(image, (x, y))
        draw.rectangle(
            [(x, y + tile_height), (x + tile_width, y + tile_height + label_height)],
            fill=(28, 28, 28),
        )
        draw.text((x + 10, y + tile_height + 12), label, fill=(235, 235, 235))

    return canvas
