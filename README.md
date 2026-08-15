# sandbox-diffusers

Minimal project scaffold for running the latest stable `diffusers` release on Apple Silicon.

As of August 14, 2026, this project pins `diffusers==0.38.0`. This release closes the
remote-code trust bypasses fixed in 0.38.0, and the application explicitly keeps
`trust_remote_code=False` for all model loads:

- PyPI: <https://pypi.org/project/diffusers/>
- Install docs: <https://huggingface.co/docs/diffusers/installation>
- Apple Silicon / MPS docs: <https://huggingface.co/docs/diffusers/en/optimization/mps>
- Diffusers pipeline loading / safety checker docs: <https://huggingface.co/docs/diffusers/v0.38.0/using-diffusers/loading>
- PyTorch MPS docs: <https://docs.pytorch.org/docs/stable/mps.html>
- Apple Metal + PyTorch page: <https://developer.apple.com/metal/pytorch/>

## Why the default changed to Stable Diffusion v1.5

After testing on this MacBook Pro M1, `sdxl-turbo` on `mps` produced black outputs. I checked current upstream guidance and changed the default example to `stable-diffusion-v1-5/stable-diffusion-v1-5` because:

- Hugging Face's current MPS page still demonstrates `stable-diffusion-v1-5/stable-diffusion-v1-5`
- the MPS docs explicitly recommend attention slicing and a `512x512` style workload
- PyTorch and Apple both still describe MPS as an evolving backend
- there are open/closed Diffusers issues documenting `fp16` instability on MPS, especially for SDXL-class pipelines

On this machine, the combination that actually worked was:

- `mps` on Apple Silicon when available
- `torch.float32` on MPS for stability
- 512x512 output
- attention slicing to reduce memory pressure
- a one-step warmup pass before the real generation call

The dependency set requires a patched Transformers 5.x release and verifies the
Diffusers pipeline imports against that runtime.

Custom pipelines and components from a model repository can execute Python code.
This sandbox rejects them by default. Do not enable `trust_remote_code` unless you
have pinned a revision and reviewed every executable file in that revision.

## Setup

Use an arm64 Python build on macOS. Apple Silicon acceleration will not work correctly from an x86_64 / Rosetta Python interpreter.

```bash
uv sync --config-file uv.toml --frozen --all-groups
source .venv/bin/activate
```

Optional but useful on MPS if an op falls back unexpectedly:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Learning workflow

If the goal is learning, use the repo in this order:

1. Start with text-to-image and change one variable at a time.
2. Compare seeds to understand composition drift.
3. Compare schedulers to understand the style and sharpness tradeoff.
4. Move to image-to-image once you want controlled edits instead of fresh generations.

The CLIs below are set up for exactly that.

## Local web UI

Launch the dependency-light Starlette UI:

```bash
source .venv/bin/activate
sandbox-diffusers-ui
```

Or choose a host and port explicitly:

```bash
sandbox-diffusers-ui --host 127.0.0.1 --port 7860
```

The server binds to `127.0.0.1` by default, sends restrictive browser security
headers, limits request/image sizes, and does not expose a public share-link mode.
The UI includes three tabs:

- text-to-image for prompt, seed, and scheduler experiments
- image-to-image for controlled edits from an uploaded source image
- compare for side-by-side prompt, seed, or scheduler contact sheets

## Text to image

```bash
sandbox-diffusers-generate \
  --prompt "photo of a red lighthouse on a rocky coast at sunrise, ocean waves, dramatic sky" \
  --output output/lighthouse.png \
  --disable-safety-checker
```

Equivalent module invocation:

```bash
python -m sandbox_diffusers.generate \
  --prompt "photo of a red lighthouse on a rocky coast at sunrise, ocean waves, dramatic sky" \
  --output output/lighthouse.png \
  --disable-safety-checker
```

The first run downloads model weights from Hugging Face, so expect a large download and a slower startup.

`--disable-safety-checker` is there because the built-in Stable Diffusion safety checker can false-positive and replace an otherwise valid image with a black frame. Hugging Face recommends keeping the safety checker enabled for public-facing apps; use this flag only for local experiments.

## Compare prompts, seeds, or schedulers

Compare two seeds with one prompt:

```bash
sandbox-diffusers-compare \
  --prompt "photo of a red lighthouse on a rocky coast at sunrise, ocean waves, dramatic sky" \
  --seed 42 \
  --seed 123 \
  --output output/seed-compare.png \
  --disable-safety-checker
```

Compare schedulers with one prompt and one seed:

```bash
sandbox-diffusers-compare \
  --prompt "photo of a red lighthouse on a rocky coast at sunrise, ocean waves, dramatic sky" \
  --seed 42 \
  --scheduler default \
  --scheduler euler \
  --scheduler dpm \
  --output output/scheduler-compare.png \
  --disable-safety-checker
```

Compare prompts with one seed:

```bash
sandbox-diffusers-compare \
  --prompt "photo of a red lighthouse on a rocky coast at sunrise" \
  --prompt "watercolor painting of a red lighthouse on a rocky coast at sunrise" \
  --output output/prompt-compare.png \
  --disable-safety-checker
```

The command writes a contact sheet so you can see the differences side by side.

## Image to image

Start from an existing image and push it toward a new style or scene:

```bash
sandbox-diffusers-img2img output/lighthouse.png \
  --prompt "watercolor painting of the same lighthouse scene, soft brush strokes" \
  --strength 0.45 \
  --output output/lighthouse-watercolor.png \
  --disable-safety-checker
```

Interpret `--strength` like this:

- `0.2` keeps the source composition close
- `0.4-0.6` makes a meaningful but still recognizable edit
- `0.7+` starts behaving more like a new generation

## Common options

```bash
python -m sandbox_diffusers.generate --help
```

Useful flags:

- `--device auto|mps|cpu`
- `--steps 12`
- `--precision auto|float32|float16`
- `--scheduler default|ddim|dpm|euler|lms`
- `--seed 1234`
- `--width 512 --height 512`
- `--model stable-diffusion-v1-5/stable-diffusion-v1-5`
- `--disable-safety-checker`

## Notes

- Hugging Face recommends PyTorch 2.0+ for MPS, and attention slicing when memory pressure is a concern on M1/M2 systems.
- The current MPS docs still use Stable Diffusion v1.5 as the reference example.
- On this M1, `float16` on MPS produced black images for both SDXL Turbo and Stable Diffusion v1.5, while `float32` on MPS produced a valid image.
- If you want the most conservative memory profile, keep resolution at `512x512`.
- The quickest way to learn is to hold everything constant except one parameter: prompt, seed, scheduler, or img2img strength.
