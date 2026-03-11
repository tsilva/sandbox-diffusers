from __future__ import annotations

import argparse
import threading
from pathlib import Path

import gradio as gr

from sandbox_diffusers.runtime import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    RuntimeConfig,
    apply_scheduler,
    configure_environment,
    make_contact_sheet,
    make_output_path,
    open_image,
    pipeline_load_kwargs,
    resize_for_generation,
    resolve_device,
    resolve_dtype,
    seed_everything,
    warmup_img2img_if_needed,
    warmup_txt2img_if_needed,
)


PIPELINE_CACHE: dict[tuple[str, str, str, str, str, bool], object] = {}
PIPELINE_LOCK = threading.Lock()


APP_CSS = """
:root {
  --sand-bg: #f3efe4;
  --sand-panel: #f8f4ea;
  --sand-ink: #1f1b18;
  --sand-accent: #b54f28;
  --sand-accent-2: #235f72;
  --sand-line: #d5cab8;
}

.gradio-container {
  background:
    radial-gradient(circle at top right, rgba(181, 79, 40, 0.16), transparent 26%),
    radial-gradient(circle at bottom left, rgba(35, 95, 114, 0.14), transparent 30%),
    linear-gradient(180deg, #f7f1e4 0%, var(--sand-bg) 100%);
}

.app-shell {
  max-width: 1200px;
  margin: 0 auto;
}

.hero {
  background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,255,255,0.42));
  border: 1px solid rgba(31, 27, 24, 0.08);
  border-radius: 24px;
  padding: 24px 28px;
  box-shadow: 0 10px 30px rgba(31, 27, 24, 0.08);
  backdrop-filter: blur(6px);
}

.hero h1 {
  margin: 0;
  font-size: 2.2rem;
  line-height: 1;
}

.hero p {
  margin: 10px 0 0;
  max-width: 760px;
}

.panel-note {
  border-left: 4px solid var(--sand-accent);
  padding: 10px 14px;
  background: rgba(181, 79, 40, 0.08);
  border-radius: 10px;
}
"""


def _dtype_label(dtype) -> str:
    return str(dtype).split(".")[-1]


def _pipeline_key(
    mode: str,
    model: str,
    device: str,
    dtype,
    scheduler: str,
    disable_safety_checker: bool,
) -> tuple[str, str, str, str, str, bool]:
    return (mode, model, device, _dtype_label(dtype), scheduler, disable_safety_checker)


def _load_pipeline(
    mode: str,
    model: str,
    device: str,
    dtype,
    scheduler: str,
    disable_safety_checker: bool,
):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    key = _pipeline_key(mode, model, device, dtype, scheduler, disable_safety_checker)
    if key in PIPELINE_CACHE:
        return PIPELINE_CACHE[key]

    pipe_cls = StableDiffusionPipeline if mode == "txt2img" else StableDiffusionImg2ImgPipeline
    pipe = pipe_cls.from_pretrained(
        model,
        **pipeline_load_kwargs(dtype, device, disable_safety_checker, torch),
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe = apply_scheduler(pipe, scheduler)
    PIPELINE_CACHE[key] = pipe
    return pipe


def _txt2img(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    scheduler: str,
    precision: str,
    device_choice: str,
    disable_safety_checker: bool,
):
    configure_environment()
    import torch

    device = resolve_device(device_choice, torch)
    dtype = resolve_dtype(device, precision, torch)
    config = RuntimeConfig(
        model=DEFAULT_MODEL,
        device=device,
        dtype=dtype,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        scheduler=scheduler,
        disable_safety_checker=disable_safety_checker,
    )

    with PIPELINE_LOCK:
        pipe = _load_pipeline(
            "txt2img", DEFAULT_MODEL, device, dtype, scheduler, disable_safety_checker
        )
        seed_everything(seed, device, torch)
        warmup_txt2img_if_needed(pipe, config)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )

    image = result.images[0]
    output_path = make_output_path(None, "ui-txt2img.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    summary = (
        f"Saved to `{output_path}`\n\n"
        f"Model: `{DEFAULT_MODEL}`\n"
        f"Device: `{device}`\n"
        f"Precision: `{_dtype_label(dtype)}`\n"
        f"Scheduler: `{scheduler}`\n"
        f"Seed: `{seed}`"
    )
    return image, summary


def _img2img(
    source_image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    scheduler: str,
    precision: str,
    device_choice: str,
    disable_safety_checker: bool,
):
    if source_image is None:
        raise gr.Error("Upload a source image first.")

    configure_environment()
    import torch

    device = resolve_device(device_choice, torch)
    dtype = resolve_dtype(device, precision, torch)
    config = RuntimeConfig(
        model=DEFAULT_MODEL,
        device=device,
        dtype=dtype,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        scheduler=scheduler,
        disable_safety_checker=disable_safety_checker,
    )
    source = resize_for_generation(source_image.convert("RGB"), width, height)

    with PIPELINE_LOCK:
        pipe = _load_pipeline(
            "img2img", DEFAULT_MODEL, device, dtype, scheduler, disable_safety_checker
        )
        seed_everything(seed, device, torch)
        warmup_img2img_if_needed(pipe, config, source)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

    image = result.images[0]
    output_path = make_output_path(None, "ui-img2img.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    summary = (
        f"Saved to `{output_path}`\n\n"
        f"Strength: `{strength}`\n"
        f"Device: `{device}`\n"
        f"Precision: `{_dtype_label(dtype)}`\n"
        f"Scheduler: `{scheduler}`\n"
        f"Seed: `{seed}`"
    )
    return image, summary


def _compare(
    prompts_text: str,
    seeds_text: str,
    schedulers: list[str],
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    precision: str,
    device_choice: str,
    disable_safety_checker: bool,
):
    configure_environment()
    import torch

    prompts = [line.strip() for line in prompts_text.splitlines() if line.strip()]
    if not prompts:
        prompts = [DEFAULT_PROMPT]

    seeds = [int(item.strip()) for item in seeds_text.split(",") if item.strip()]
    if not seeds:
        seeds = [42, 123]

    scheduler_list = schedulers or ["default"]

    device = resolve_device(device_choice, torch)
    dtype = resolve_dtype(device, precision, torch)
    config = RuntimeConfig(
        model=DEFAULT_MODEL,
        device=device,
        dtype=dtype,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        scheduler="default",
        disable_safety_checker=disable_safety_checker,
    )

    if len(prompts) > 1:
        runs = [(f"prompt {idx + 1}", prompt, seeds[0], scheduler_list[0]) for idx, prompt in enumerate(prompts)]
    elif len(seeds) > 1:
        runs = [(f"seed {seed}", prompts[0], seed, scheduler_list[0]) for seed in seeds]
    else:
        runs = [(f"scheduler {scheduler}", prompts[0], seeds[0], scheduler) for scheduler in scheduler_list]

    rendered = []
    with PIPELINE_LOCK:
        base_pipe = _load_pipeline(
            "txt2img", DEFAULT_MODEL, device, dtype, "default", disable_safety_checker
        )
        warmup_txt2img_if_needed(base_pipe, config)
        for label, prompt, seed, scheduler in runs:
            pipe = _load_pipeline(
                "txt2img", DEFAULT_MODEL, device, dtype, scheduler, disable_safety_checker
            )
            seed_everything(seed, device, torch)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            rendered.append((label, result.images[0]))

    sheet = make_contact_sheet(rendered)
    output_path = make_output_path(None, "ui-compare.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    summary = (
        f"Saved to `{output_path}`\n\n"
        f"Rendered `{len(rendered)}` panels on `{device}` with `{_dtype_label(dtype)}`."
    )
    return sheet, summary


def build_demo() -> gr.Blocks:
    example_source = Path.cwd() / "output" / "lighthouse-cli-mps-fp32.png"
    example_source_value = str(example_source) if example_source.exists() else None

    with gr.Blocks(title="Sandbox Diffusers", theme=gr.themes.Soft(), css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                <div class="hero">
                  <h1>Sandbox Diffusers</h1>
                  <p>
                    Stable Diffusion v1.5 on Apple Silicon, tuned for this M1. Use the tabs below
                    to learn prompt changes, scheduler differences, seed drift, and image-to-image edits.
                  </p>
                </div>
                """
            )
            gr.Markdown(
                """
                <div class="panel-note">
                  Default path: <code>mps</code> + <code>float32</code> + attention slicing.
                  Keep the safety checker enabled for public apps; disable it here only for local experiments.
                </div>
                """
            )

            with gr.Tab("Text To Image"):
                with gr.Row():
                    with gr.Column(scale=4):
                        txt_prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            value=DEFAULT_PROMPT,
                        )
                        txt_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value=DEFAULT_NEGATIVE_PROMPT,
                        )
                        with gr.Row():
                            txt_steps = gr.Slider(1, 30, value=12, step=1, label="Steps")
                            txt_guidance = gr.Slider(
                                0, 15, value=7.5, step=0.5, label="Guidance Scale"
                            )
                            txt_seed = gr.Number(value=42, precision=0, label="Seed")
                        with gr.Row():
                            txt_width = gr.Dropdown(
                                choices=[256, 384, 512, 640],
                                value=512,
                                label="Width",
                            )
                            txt_height = gr.Dropdown(
                                choices=[256, 384, 512, 640],
                                value=512,
                                label="Height",
                            )
                            txt_scheduler = gr.Dropdown(
                                choices=["default", "ddim", "dpm", "euler", "lms"],
                                value="default",
                                label="Scheduler",
                            )
                        with gr.Row():
                            txt_precision = gr.Dropdown(
                                choices=["auto", "float32", "float16"],
                                value="auto",
                                label="Precision",
                            )
                            txt_device = gr.Dropdown(
                                choices=["auto", "mps", "cpu"],
                                value="auto",
                                label="Device",
                            )
                            txt_safety = gr.Checkbox(
                                value=True,
                                label="Disable Safety Checker",
                            )
                        txt_run = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=5):
                        txt_image = gr.Image(label="Output", type="pil")
                        txt_summary = gr.Markdown()

                txt_run.click(
                    fn=_txt2img,
                    inputs=[
                        txt_prompt,
                        txt_negative,
                        txt_steps,
                        txt_guidance,
                        txt_seed,
                        txt_width,
                        txt_height,
                        txt_scheduler,
                        txt_precision,
                        txt_device,
                        txt_safety,
                    ],
                    outputs=[txt_image, txt_summary],
                )

            with gr.Tab("Image To Image"):
                with gr.Row():
                    with gr.Column(scale=4):
                        img_source = gr.Image(
                            label="Source Image",
                            type="pil",
                            value=example_source_value,
                        )
                        img_prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            value="watercolor painting of the same lighthouse scene, soft brush strokes",
                        )
                        img_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value=DEFAULT_NEGATIVE_PROMPT,
                        )
                        with gr.Row():
                            img_strength = gr.Slider(
                                0.1, 0.9, value=0.45, step=0.05, label="Strength"
                            )
                            img_steps = gr.Slider(1, 30, value=18, step=1, label="Steps")
                            img_guidance = gr.Slider(
                                0, 15, value=7.5, step=0.5, label="Guidance Scale"
                            )
                        with gr.Row():
                            img_seed = gr.Number(value=42, precision=0, label="Seed")
                            img_width = gr.Dropdown(
                                choices=[256, 384, 512, 640],
                                value=512,
                                label="Width",
                            )
                            img_height = gr.Dropdown(
                                choices=[256, 384, 512, 640],
                                value=512,
                                label="Height",
                            )
                        with gr.Row():
                            img_scheduler = gr.Dropdown(
                                choices=["default", "ddim", "dpm", "euler", "lms"],
                                value="default",
                                label="Scheduler",
                            )
                            img_precision = gr.Dropdown(
                                choices=["auto", "float32", "float16"],
                                value="auto",
                                label="Precision",
                            )
                            img_device = gr.Dropdown(
                                choices=["auto", "mps", "cpu"],
                                value="auto",
                                label="Device",
                            )
                            img_safety = gr.Checkbox(
                                value=True,
                                label="Disable Safety Checker",
                            )
                        img_run = gr.Button("Transform", variant="primary")
                    with gr.Column(scale=5):
                        img_output = gr.Image(label="Output", type="pil")
                        img_summary = gr.Markdown()

                img_run.click(
                    fn=_img2img,
                    inputs=[
                        img_source,
                        img_prompt,
                        img_negative,
                        img_strength,
                        img_steps,
                        img_guidance,
                        img_seed,
                        img_width,
                        img_height,
                        img_scheduler,
                        img_precision,
                        img_device,
                        img_safety,
                    ],
                    outputs=[img_output, img_summary],
                )

            with gr.Tab("Compare"):
                with gr.Row():
                    with gr.Column(scale=4):
                        compare_prompts = gr.Textbox(
                            label="Prompts",
                            lines=5,
                            value=DEFAULT_PROMPT + "\nwatercolor painting of a red lighthouse on a rocky coast at sunrise",
                            info="Use multiple lines to compare prompts. If you enter one prompt, the app compares seeds or schedulers instead.",
                        )
                        compare_seeds = gr.Textbox(
                            label="Seeds",
                            value="42,123",
                            info="Comma-separated integers.",
                        )
                        compare_schedulers = gr.CheckboxGroup(
                            choices=["default", "ddim", "dpm", "euler", "lms"],
                            value=["default", "euler"],
                            label="Schedulers",
                        )
                        compare_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value=DEFAULT_NEGATIVE_PROMPT,
                        )
                        with gr.Row():
                            compare_steps = gr.Slider(1, 30, value=12, step=1, label="Steps")
                            compare_guidance = gr.Slider(
                                0, 15, value=7.5, step=0.5, label="Guidance Scale"
                            )
                        with gr.Row():
                            compare_width = gr.Dropdown(
                                choices=[256, 384, 512],
                                value=256,
                                label="Width",
                            )
                            compare_height = gr.Dropdown(
                                choices=[256, 384, 512],
                                value=256,
                                label="Height",
                            )
                            compare_precision = gr.Dropdown(
                                choices=["auto", "float32", "float16"],
                                value="auto",
                                label="Precision",
                            )
                        with gr.Row():
                            compare_device = gr.Dropdown(
                                choices=["auto", "mps", "cpu"],
                                value="auto",
                                label="Device",
                            )
                            compare_safety = gr.Checkbox(
                                value=True,
                                label="Disable Safety Checker",
                            )
                        compare_run = gr.Button("Build Contact Sheet", variant="primary")
                    with gr.Column(scale=5):
                        compare_image = gr.Image(label="Contact Sheet", type="pil")
                        compare_summary = gr.Markdown()

                compare_run.click(
                    fn=_compare,
                    inputs=[
                        compare_prompts,
                        compare_seeds,
                        compare_schedulers,
                        compare_negative,
                        compare_steps,
                        compare_guidance,
                        compare_width,
                        compare_height,
                        compare_precision,
                        compare_device,
                        compare_safety,
                    ],
                    outputs=[compare_image, compare_summary],
                )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Gradio UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
