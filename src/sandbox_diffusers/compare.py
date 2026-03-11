from __future__ import annotations

import argparse

from sandbox_diffusers.runtime import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    RuntimeConfig,
    apply_scheduler,
    configure_environment,
    make_contact_sheet,
    make_output_path,
    pipeline_load_kwargs,
    resolve_device,
    resolve_dtype,
    seed_everything,
    warmup_txt2img_if_needed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a comparison contact sheet across prompts, seeds, or schedulers."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to include. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Seed to include. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--scheduler",
        action="append",
        dest="schedulers",
        choices=("default", "ddim", "dpm", "euler", "lms"),
        help="Scheduler to include. Repeat for multiple schedulers.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt.",
    )
    parser.add_argument("--output", help="Destination PNG path.")
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu"), default="auto", help="Execution device."
    )
    parser.add_argument("--steps", type=int, default=12, help="Denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--width", type=int, default=512, help="Output width.")
    parser.add_argument("--height", type=int, default=512, help="Output height.")
    parser.add_argument(
        "--precision",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Tensor precision. 'auto' uses float32 on MPS for stability.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable the safety checker for local-only usage.",
    )
    return parser


def build_runs(args) -> list[tuple[str, str, int, str]]:
    prompts = args.prompts or [DEFAULT_PROMPT]
    seeds = args.seeds or [42, 123]
    schedulers = args.schedulers or ["default"]

    runs: list[tuple[str, str, int, str]] = []
    if len(prompts) > 1:
        for prompt in prompts:
            runs.append((f"prompt={prompt[:28]}", prompt, seeds[0], schedulers[0]))
    elif len(seeds) > 1:
        for seed in seeds:
            runs.append((f"seed={seed}", prompts[0], seed, schedulers[0]))
    else:
        for scheduler in schedulers:
            runs.append((f"scheduler={scheduler}", prompts[0], seeds[0], scheduler))
    return runs


def main() -> None:
    configure_environment()

    args = build_parser().parse_args()

    import torch
    from diffusers import StableDiffusionPipeline

    device = resolve_device(args.device, torch)
    dtype = resolve_dtype(device, args.precision, torch)
    config = RuntimeConfig(
        model=args.model,
        device=device,
        dtype=dtype,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        scheduler="default",
        disable_safety_checker=args.disable_safety_checker,
    )

    print(f"Loading model: {args.model}")
    print(f"Using device: {device}")
    print(f"Using precision: {dtype}")

    base_pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        **pipeline_load_kwargs(dtype, device, args.disable_safety_checker, torch),
    )
    base_pipe = base_pipe.to(device)
    base_pipe.enable_attention_slicing()

    seed_everything(42, device, torch)
    warmup_txt2img_if_needed(base_pipe, config)

    images = []
    for label, prompt, seed, scheduler in build_runs(args):
        pipe = apply_scheduler(base_pipe, scheduler)
        seed_everything(seed, device, torch)
        result = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
        )
        images.append((label, result.images[0]))
        print(f"Rendered {label}")

    sheet = make_contact_sheet(images)
    output_path = make_output_path(args.output, "compare.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Saved contact sheet to: {output_path}")


if __name__ == "__main__":
    main()
