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
    open_image,
    optimize_pipeline,
    pipeline_load_kwargs,
    resize_for_generation,
    resolve_device,
    resolve_dtype,
    seed_everything,
    warmup_img2img_if_needed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion image-to-image on Apple Silicon or CPU."
    )
    parser.add_argument("input_image", help="Input image path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Target prompt.")
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    parser.add_argument("--output", help="Destination PNG path.")
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu"), default="auto", help="Execution device."
    )
    parser.add_argument("--steps", type=int, default=18, help="Denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=512, help="Output width.")
    parser.add_argument("--height", type=int, default=512, help="Output height.")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.45,
        help="How far to drift from the source image, between 0 and 1.",
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
        help="Disable the safety checker for local-only usage.",
    )
    return parser


def main() -> None:
    configure_environment()

    args = build_parser().parse_args()

    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

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

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        **pipeline_load_kwargs(dtype, device, args.disable_safety_checker, torch),
    )
    pipe = pipe.to(device)
    pipe = optimize_pipeline(pipe, device)
    pipe = apply_scheduler(pipe, args.scheduler)

    source = resize_for_generation(open_image(args.input_image), args.width, args.height)
    seed_everything(args.seed, device, torch)
    warmup_img2img_if_needed(pipe, config, source)
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=source,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
    image = result.images[0]
    finalize_inference(device, torch)

    output_path = make_output_path(args.output, "img2img.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
