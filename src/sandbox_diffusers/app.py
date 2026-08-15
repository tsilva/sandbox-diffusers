from __future__ import annotations

import argparse
import base64
import binascii
import threading
from io import BytesIO

from PIL import Image
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from sandbox_diffusers.runtime import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    RuntimeConfig,
    apply_scheduler,
    configure_environment,
    finalize_inference,
    make_contact_sheet,
    make_output_path,
    optimize_pipeline,
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
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_777_216
SCHEDULERS = {"default", "ddim", "dpm", "euler", "lms"}
DEVICES = {"auto", "mps", "cpu"}
PRECISIONS = {"auto", "float32", "float16"}


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
    pipe = optimize_pipeline(pipe, device)
    pipe = apply_scheduler(pipe, scheduler)
    PIPELINE_CACHE[key] = pipe
    return pipe


def _runtime(payload: dict, *, default_steps: int = 12) -> tuple[object, RuntimeConfig]:
    configure_environment()
    import torch

    device = resolve_device(_choice(payload, "device", "auto", DEVICES), torch)
    precision = _choice(payload, "precision", "auto", PRECISIONS)
    dtype = resolve_dtype(device, precision, torch)
    config = RuntimeConfig(
        model=DEFAULT_MODEL,
        device=device,
        dtype=dtype,
        steps=_bounded_int(payload, "steps", default_steps, 1, 30),
        guidance_scale=_bounded_float(payload, "guidance_scale", 7.5, 0, 15),
        width=_bounded_int(payload, "width", 512, 256, 640),
        height=_bounded_int(payload, "height", 512, 256, 640),
        scheduler=_choice(payload, "scheduler", "default", SCHEDULERS),
        disable_safety_checker=bool(payload.get("disable_safety_checker", False)),
    )
    return torch, config


def _txt2img(payload: dict) -> tuple[Image.Image, str]:
    torch, config = _runtime(payload)
    prompt = _text(payload, "prompt", DEFAULT_PROMPT, 2_000)
    negative_prompt = _text(payload, "negative_prompt", DEFAULT_NEGATIVE_PROMPT, 2_000)
    seed = _bounded_int(payload, "seed", 42, 0, 2**32 - 1)

    with PIPELINE_LOCK:
        pipe = _load_pipeline(
            "txt2img",
            config.model,
            config.device,
            config.dtype,
            config.scheduler,
            config.disable_safety_checker,
        )
        seed_everything(seed, config.device, torch)
        warmup_txt2img_if_needed(pipe, config)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                width=config.width,
                height=config.height,
            )

    image = result.images[0]
    finalize_inference(config.device, torch)
    return _save_result(image, "ui-txt2img.png", config, f"Seed: {seed}")


def _img2img(payload: dict) -> tuple[Image.Image, str]:
    torch, config = _runtime(payload, default_steps=18)
    source = resize_for_generation(
        _decode_image(payload.get("source_image")), config.width, config.height
    )
    prompt = _text(
        payload,
        "prompt",
        "watercolor painting of the same lighthouse scene, soft brush strokes",
        2_000,
    )
    negative_prompt = _text(payload, "negative_prompt", DEFAULT_NEGATIVE_PROMPT, 2_000)
    seed = _bounded_int(payload, "seed", 42, 0, 2**32 - 1)
    strength = _bounded_float(payload, "strength", 0.45, 0.1, 0.9)

    with PIPELINE_LOCK:
        pipe = _load_pipeline(
            "img2img",
            config.model,
            config.device,
            config.dtype,
            config.scheduler,
            config.disable_safety_checker,
        )
        seed_everything(seed, config.device, torch)
        warmup_img2img_if_needed(pipe, config, source)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=source,
                strength=strength,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
            )

    image = result.images[0]
    finalize_inference(config.device, torch)
    return _save_result(image, "ui-img2img.png", config, f"Strength: {strength}")


def _compare(payload: dict) -> tuple[Image.Image, str]:
    torch, config = _runtime(payload)
    prompts = [line.strip() for line in str(payload.get("prompts", "")).splitlines() if line.strip()]
    prompts = prompts[:6] or [DEFAULT_PROMPT]
    seeds = _integer_list(payload.get("seeds", "42,123"), maximum=6)
    schedulers = [item for item in payload.get("schedulers", ["default"]) if item in SCHEDULERS]
    schedulers = schedulers[:6] or ["default"]
    negative_prompt = _text(payload, "negative_prompt", DEFAULT_NEGATIVE_PROMPT, 2_000)

    if len(prompts) > 1:
        runs = [(f"prompt {index + 1}", prompt, seeds[0], schedulers[0]) for index, prompt in enumerate(prompts)]
    elif len(seeds) > 1:
        runs = [(f"seed {seed}", prompts[0], seed, schedulers[0]) for seed in seeds]
    else:
        runs = [(f"scheduler {scheduler}", prompts[0], seeds[0], scheduler) for scheduler in schedulers]

    rendered: list[tuple[str, Image.Image]] = []
    with PIPELINE_LOCK:
        base_pipe = _load_pipeline(
            "txt2img", config.model, config.device, config.dtype, "default", config.disable_safety_checker
        )
        warmup_txt2img_if_needed(base_pipe, config)
        for label, prompt, seed, scheduler in runs:
            pipe = _load_pipeline(
                "txt2img", config.model, config.device, config.dtype, scheduler, config.disable_safety_checker
            )
            seed_everything(seed, config.device, torch)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=config.steps,
                    guidance_scale=config.guidance_scale,
                    width=config.width,
                    height=config.height,
                )
            rendered.append((label, result.images[0]))

    sheet = make_contact_sheet(rendered)
    finalize_inference(config.device, torch)
    return _save_result(sheet, "ui-compare.png", config, f"Panels: {len(rendered)}")


def _save_result(
    image: Image.Image, filename: str, config: RuntimeConfig, detail: str
) -> tuple[Image.Image, str]:
    output_path = make_output_path(None, filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    summary = (
        f"Saved to {output_path}. Device: {config.device}. "
        f"Precision: {_dtype_label(config.dtype)}. Scheduler: {config.scheduler}. {detail}."
    )
    return image, summary


def _bounded_int(payload: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = int(payload.get(key, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _bounded_float(
    payload: dict, key: str, default: float, minimum: float, maximum: float
) -> float:
    value = float(payload.get(key, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _choice(payload: dict, key: str, default: str, choices: set[str]) -> str:
    value = str(payload.get(key, default))
    if value not in choices:
        raise ValueError(f"invalid {key}")
    return value


def _text(payload: dict, key: str, default: str, maximum: int) -> str:
    value = str(payload.get(key, default)).strip()
    if not value or len(value) > maximum:
        raise ValueError(f"{key} must contain between 1 and {maximum} characters")
    return value


def _integer_list(value, *, maximum: int) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        return [42, 123]
    if len(values) > maximum or any(item < 0 or item > 2**32 - 1 for item in values):
        raise ValueError("invalid seed list")
    return values


def _decode_image(value) -> Image.Image:
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise ValueError("Upload a source image first")
    try:
        encoded = value.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
    except (IndexError, binascii.Error) as error:
        raise ValueError("invalid source image") from error
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("source image is too large")
    image = Image.open(BytesIO(raw))
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("source image has too many pixels")
    image.load()
    return image.convert("RGB")


def _image_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def homepage(request: Request) -> HTMLResponse:
    return HTMLResponse(APP_HTML)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "model": DEFAULT_MODEL, "remote_code": False})


async def generate(request: Request) -> JSONResponse:
    return await _run_request(request, _txt2img)


async def transform(request: Request) -> JSONResponse:
    return await _run_request(request, _img2img)


async def compare(request: Request) -> JSONResponse:
    return await _run_request(request, _compare)


async def _run_request(request: Request, operation) -> JSONResponse:
    length = int(request.headers.get("content-length", "0") or 0)
    if length > MAX_REQUEST_BYTES:
        return JSONResponse({"error": "request is too large"}, status_code=413)
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        image, summary = await run_in_threadpool(operation, payload)
        return JSONResponse({"image": _image_data_url(image), "summary": summary})
    except (ValueError, TypeError) as error:
        return JSONResponse({"error": str(error)}, status_code=400)


class SecurityHeadersMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"content-security-policy", b"default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def build_app() -> Starlette:
    app = Starlette(
        debug=False,
        routes=[
            Route("/", homepage),
            Route("/healthz", health),
            Route("/api/txt2img", generate, methods=["POST"]),
            Route("/api/img2img", transform, methods=["POST"]),
            Route("/api/compare", compare, methods=["POST"]),
        ],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    return app


APP_HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sandbox Diffusers</title><style>
:root{{--sand:#f3efe4;--panel:#fffaf0;--ink:#211b17;--accent:#b54f28;--blue:#235f72}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top right,#efd6c5,transparent 32%),var(--sand);color:var(--ink);font:16px system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:34px 22px}} .hero,.panel{{background:#fff9;border:1px solid #d5cab8;border-radius:22px;padding:24px;box-shadow:0 12px 32px #34200d12}}
h1{{margin:0;font-size:2.5rem}} .note{{border-left:4px solid var(--accent);padding:10px 14px;margin:18px 0;background:#fff8}}
.tabs{{display:flex;gap:8px;margin:22px 0 12px}} button{{border:0;border-radius:12px;padding:11px 16px;background:var(--blue);color:white;font-weight:700;cursor:pointer}}
.tabs button{{background:#d8d0c2;color:var(--ink)}} .tabs button.active{{background:var(--accent);color:white}}
.tab{{display:none}} .tab.active{{display:grid;grid-template-columns:1fr 1fr;gap:22px}} label{{display:block;font-weight:650;margin:12px 0 5px}}
textarea,input,select{{width:100%;padding:10px;border:1px solid #bcb09e;border-radius:10px;background:white}} textarea{{min-height:86px;resize:vertical}}
.row{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .output{{min-height:360px;border:1px dashed #bcb09e;border-radius:16px;display:grid;place-items:center;padding:14px;background:#fff}}
.output img{{max-width:100%;border-radius:10px}} .status{{white-space:pre-wrap;color:#544a42}} @media(max-width:760px){{.tab.active{{grid-template-columns:1fr}}.row{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><h1>Sandbox Diffusers</h1><p>Stable Diffusion v1.5 on Apple Silicon, with explicit remote-code rejection and a local-only web server.</p></section>
<p class="note">Custom Hub code is always disabled. The safety checker remains enabled unless you deliberately opt out for a local experiment.</p>
<nav class="tabs"><button class="active" data-tab="txt">Text to image</button><button data-tab="img">Image to image</button><button data-tab="cmp">Compare</button></nav>
<section class="panel tab active" id="txt"><div><label>Prompt</label><textarea name="prompt">{DEFAULT_PROMPT}</textarea><label>Negative prompt</label><textarea name="negative_prompt">{DEFAULT_NEGATIVE_PROMPT}</textarea><div class="row"><label>Steps<input name="steps" type="number" min="1" max="30" value="12"></label><label>Seed<input name="seed" type="number" min="0" value="42"></label><label>Scheduler<select name="scheduler"><option>default</option><option>ddim</option><option>dpm</option><option>euler</option><option>lms</option></select></label></div><label><input name="disable_safety_checker" type="checkbox"> Disable safety checker</label><button data-run="txt2img">Generate</button></div><div class="output"><p class="status">Ready.</p></div></section>
<section class="panel tab" id="img"><div><label>Source image</label><input name="source" type="file" accept="image/png,image/jpeg,image/webp"><label>Prompt</label><textarea name="prompt">watercolor painting of the same lighthouse scene, soft brush strokes</textarea><div class="row"><label>Strength<input name="strength" type="number" min="0.1" max="0.9" step="0.05" value="0.45"></label><label>Steps<input name="steps" type="number" min="1" max="30" value="18"></label><label>Seed<input name="seed" type="number" min="0" value="42"></label></div><button data-run="img2img">Transform</button></div><div class="output"><p class="status">Upload an image to begin.</p></div></section>
<section class="panel tab" id="cmp"><div><label>Prompts, one per line</label><textarea name="prompts">{DEFAULT_PROMPT}\nwatercolor painting of a red lighthouse on a rocky coast at sunrise</textarea><label>Seeds, comma separated</label><input name="seeds" value="42,123"><label>Schedulers, comma separated</label><input name="schedulers" value="default,euler"><button data-run="compare">Build contact sheet</button></div><div class="output"><p class="status">Ready.</p></div></section>
</main><script>
const tabs=document.querySelectorAll('[data-tab]');tabs.forEach(b=>b.onclick=()=>{{tabs.forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')}});
const value=(root,name)=>root.querySelector(`[name="${{name}}"]`)?.value;const number=(root,name)=>Number(value(root,name));
for(const button of document.querySelectorAll('[data-run]')) button.onclick=async()=>{{const root=button.closest('.tab'),status=root.querySelector('.status'),output=root.querySelector('.output');button.disabled=true;status.textContent='Working locally…';let payload={{prompt:value(root,'prompt'),steps:number(root,'steps')||12,seed:number(root,'seed')||42}};if(button.dataset.run==='txt2img'){{payload.negative_prompt=value(root,'negative_prompt');payload.scheduler=value(root,'scheduler');payload.disable_safety_checker=root.querySelector('[name="disable_safety_checker"]').checked}}if(button.dataset.run==='img2img'){{payload.strength=number(root,'strength');const file=root.querySelector('[name="source"]').files[0];if(!file){{status.textContent='Choose a source image first.';button.disabled=false;return}}payload.source_image=await new Promise((ok,no)=>{{const reader=new FileReader();reader.onload=()=>ok(reader.result);reader.onerror=no;reader.readAsDataURL(file)}})}}if(button.dataset.run==='compare'){{payload.prompts=value(root,'prompts');payload.seeds=value(root,'seeds');payload.schedulers=value(root,'schedulers').split(',').map(x=>x.trim())}}try{{const response=await fetch(`/api/${{button.dataset.run}}`,{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify(payload)}});const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');output.innerHTML='';const image=new Image();image.src=data.image;image.alt='Generated output';const summary=document.createElement('p');summary.className='status';summary.textContent=data.summary;output.append(image,summary)}}catch(error){{status.textContent=error.message}}finally{{button.disabled=false}}}};
</script></body></html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local Sandbox Diffusers UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    return parser


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
