from __future__ import annotations

import argparse

from sandbox_diffusers.runtime import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    RuntimeConfig,
    apply_scheduler,
    configure_environment,
    finalize_inference,
    make_output_path,
    optimize_pipeline,
    pipeline_load_kwargs,
    resolve_device,
    resolve_dtype,
    seed_everything,
    warmup_txt2img_if_needed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image with Diffusers on Apple Silicon or CPU."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt used for generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt passed to the pipeline.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id to load.",
    )
    parser.add_argument(
        "--output",
        help="Destination PNG path. Defaults to output/generated-<timestamp>.png.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Execution device. 'auto' prefers MPS and falls back to CPU.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=512, help="Output width in pixels.")
    parser.add_argument(
        "--height", type=int, default=512, help="Output height in pixels."
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Tensor precision. 'auto' uses float32 on MPS for stability.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("default", "ddim", "dpm", "euler", "lms"),
        default="default",
        help="Scheduler family to swap into the pipeline.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable the Stable Diffusion safety checker for local-only usage.",
    )
    return parser


def main() -> None:
    configure_environment()

    parser = build_parser()
    args = parser.parse_args()

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
        scheduler=args.scheduler,
        disable_safety_checker=args.disable_safety_checker,
    )

    print(f"Loading model: {args.model}")
    print(f"Using device: {device}")
    print(f"Using precision: {dtype}")
    print(f"Using scheduler: {args.scheduler}")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        **pipeline_load_kwargs(dtype, device, args.disable_safety_checker, torch),
    )
    pipe = pipe.to(device)
    pipe = optimize_pipeline(pipe, device)
    pipe = apply_scheduler(pipe, args.scheduler)
    seed_everything(args.seed, device, torch)
    warmup_txt2img_if_needed(pipe, config)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
        )
    image = result.images[0]
    finalize_inference(device, torch)

    output_path = make_output_path(args.output, "generated.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
