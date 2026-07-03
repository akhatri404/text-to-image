"""
Lumora Image Studio — Streamlit front-end over free/paid, no-GPU image backends.

Backends:
  1. Hugging Face Inference Providers (official, needs a free HF_TOKEN in secrets).
  2. Pollinations.ai (official free public API, no token needed) — also used
     as automatic fallback when Hugging Face errors or runs out of credits.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import streamlit as st

# ----------------------------------------------------------------------------
# Config & constants
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Lumora Image Studio",
    page_icon="🎬",
    layout="wide",
)

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "None (raw prompt)": {
        "suffix": "",
        "negative": "",
    },
    "Realistic": {
        "suffix": (
            ", photorealistic, ultra detailed, natural lighting, "
            "shot on DSLR, 85mm lens, sharp focus, high dynamic range"
        ),
        "negative": "cartoon, painting, illustration, 3d render, anime, deformed",
    },
    "Cinematic": {
        "suffix": (
            ", cinematic still, dramatic lighting, anamorphic lens, "
            "film grain, shallow depth of field, teal and orange color grade, "
            "movie scene, 35mm film"
        ),
        "negative": "flat lighting, low contrast, amateur, deformed, watermark",
    },
    "Anime": {
        "suffix": ", anime style, vibrant colors, clean line art, studio quality, detailed background",
        "negative": "photorealistic, 3d, blurry, deformed hands",
    },
    "Digital Art": {
        "suffix": ", digital painting, concept art, trending on artstation, highly detailed, dramatic composition",
        "negative": "photo, watermark, signature, low quality",
    },
    "Fantasy": {
        "suffix": ", epic fantasy art, ethereal lighting, intricate detail, matte painting, majestic atmosphere",
        "negative": "modern, mundane, low detail, blurry",
    },
    "Cyberpunk": {
        "suffix": ", cyberpunk, neon lights, rain-soaked streets, futuristic city, blade runner aesthetic, volumetric fog",
        "negative": "daylight, rural, historical, low detail",
    },
    "Portrait Photography": {
        "suffix": (
            ", professional portrait photography, softbox lighting, bokeh background, "
            "skin texture detail, editorial quality"
        ),
        "negative": "cartoon, deformed face, extra fingers, plastic skin, oversaturated",
    },
}

SHAPES = {
    "Square (768×768)": (768, 768),
    "Square HD (1024×1024)": (1024, 1024),
    "Portrait (512×768)": (512, 768),
    "Landscape (768×512)": (768, 512),
    "Landscape HD (1920×1080)": (1920, 1080),
}

HF_MODEL_PRESETS: dict[str, str] = {
    "FLUX.1 schnell (fast, best free-tier quality)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion 3.5 medium": "stabilityai/stable-diffusion-3.5-medium",
    "Custom model ID…": "",
}

HF_API_URL = "https://router.huggingface.co/hf-inference/models/{model_id}"
HF_MAX_RETRIES = 3
HF_COLD_START_WAIT = 20  # seconds to wait when model is still loading

DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, jpeg artifacts"

MAX_IMAGES_PER_RUN = 4  # be a good citizen — don't hammer free backends


@dataclass
class GenResult:
    prompt: str
    style: str
    backend: str
    image_bytes: bytes | None = None
    error: str | None = None
    seed: int | None = None
    elapsed: float = 0.0
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

POLLINATIONS_MODELS: dict[str, str] = {
    "Flux (strong general-purpose, recommended)": "flux",
    "zimage (current default)": "zimage",
    "Nano Banana Pro (needs key, high quality)": "nanobanana-pro",
    "Seedream Pro (needs key, high quality)": "seedream-pro",
    "GPT Image Large (needs key, high quality)": "gptimage-large",
    "Ideogram v4 Quality (needs key, strong text rendering)": "ideogram-v4-quality",
    "Turbo (fast, lower quality)": "turbo",
}


def generate_via_pollinations(
    prompt: str, negative_prompt: str, width: int, height: int, seed: int,
    model: str = "flux", enhance: bool = False,
) -> GenResult:
    """Pollinations.ai. Uses the authenticated gen.pollinations.ai endpoint if
    POLLINATIONS_KEY is set in secrets (unlocks premium models); otherwise
    falls back to the legacy no-key image.pollinations.ai endpoint."""
    started = time.time()
    key = st.secrets.get("POLLINATIONS_KEY", "")

    params = {
        "width": width,
        "height": height,
        "nologo": "true",
        "model": model,
    }
    if seed and seed > 0:
        params["seed"] = seed
    if negative_prompt:
        params["negative_prompt"] = negative_prompt
    if enhance:
        params["enhance"] = "true"

    headers = {"User-Agent": "LumoraImageStudio/0.1"}
    if key:
        base = "https://gen.pollinations.ai/image/"
        headers["Authorization"] = f"Bearer {key}"
    else:
        base = "https://image.pollinations.ai/prompt/"

    url = base + urllib.parse.quote(prompt) + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            img = resp.read()
        backend_label = f"Pollinations.ai ({model}{'· keyed' if key else ''})"
        return GenResult(
            prompt=prompt, style="", backend=backend_label,
            image_bytes=img, seed=seed, elapsed=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return GenResult(
            prompt=prompt, style="", backend=f"Pollinations.ai ({model})",
            error=f"{type(exc).__name__}: {exc}", elapsed=time.time() - started,
        )


def generate_via_hf(
    prompt: str, negative_prompt: str, model_id: str, width: int, height: int,
    seed: int, guidance_scale: float = 7.0,
) -> GenResult:
    """Official Hugging Face Inference Providers API — no GPU needed locally."""
    started = time.time()

    token = st.secrets.get("HF_TOKEN", "")
    if not token:
        return GenResult(
            prompt=prompt, style="", backend=f"HF Inference ({model_id})",
            error=(
                "No HF_TOKEN found in Streamlit secrets. Add one under "
                "Settings → Secrets (or .streamlit/secrets.toml locally): "
                'HF_TOKEN = "hf_xxxxxxxx" (Read-role token from '
                "huggingface.co/settings/tokens)."
            ),
            elapsed=time.time() - started,
        )

    url = HF_API_URL.format(model_id=model_id)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt or None,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
        },
    }
    if seed and seed > 0:
        payload["parameters"]["seed"] = seed

    last_err = ""
    for attempt in range(1, HF_MAX_RETRIES + 1):
        try:
            import json as _json

            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={
                **headers, "Content-Type": "application/json",
            })
            with urllib.request.urlopen(req, timeout=120) as resp:
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read()

            if "application/json" in content_type:
                info = _json.loads(body)
                if isinstance(info, dict) and "estimated_time" in info:
                    wait = min(info.get("estimated_time", HF_COLD_START_WAIT), 60)
                    if attempt < HF_MAX_RETRIES:
                        time.sleep(wait)
                        continue
                    last_err = f"Model still loading after retries: {info}"
                    break
                last_err = f"Unexpected JSON response: {info}"
                break

            return GenResult(
                prompt=prompt, style="", backend=f"HF Inference ({model_id})",
                image_bytes=body, seed=seed, elapsed=time.time() - started,
            )

        except urllib.error.HTTPError as exc:
            body_txt = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 503:  # cold start
                if attempt < HF_MAX_RETRIES:
                    time.sleep(HF_COLD_START_WAIT)
                    continue
            last_err = f"HTTP {exc.code}: {body_txt[:300]}"
            break
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            break

    return GenResult(
        prompt=prompt, style="", backend=f"HF Inference ({model_id})",
        error=last_err, elapsed=time.time() - started,
    )


def generate_one(
    backend_choice: str,
    allow_fallback: bool,
    prompt: str,
    negative_prompt: str,
    shape_key: str,
    guidance_scale: float,
    seed: int,
    hf_model_id: str = "",
    pollinations_model: str = "flux",
    pollinations_enhance: bool = False,
) -> GenResult:
    width, height = SHAPES[shape_key]

    if backend_choice.startswith("Hugging Face"):
        result = generate_via_hf(
            prompt, negative_prompt, hf_model_id, width, height, seed, guidance_scale
        )
        if result.error and allow_fallback:
            fb = generate_via_pollinations(
                prompt, negative_prompt, width, height, seed,
                pollinations_model, pollinations_enhance,
            )
            fb.extras["fallback_from"] = result.error
            return fb
        return result

    return generate_via_pollinations(
        prompt, negative_prompt, width, height, seed,
        pollinations_model, pollinations_enhance,
    )


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list[GenResult]


# ----------------------------------------------------------------------------
# Sidebar — controls
# ----------------------------------------------------------------------------

with st.sidebar:
    st.title("🎬 Lumora Image Studio")
    st.caption("Free cloud generation — no local GPU required.")

    backend_choice = st.radio(
        "Backend",
        [
            "Hugging Face (official, free tier)",
            "Pollinations.ai (official free API)",
        ],
        key="backend_choice",
        help=(
            "Hugging Face: official API, real model choice, needs a free token. "
            "Pollinations: official, no token needed, no credit ceiling."
        ),
    )

    hf_model_id = ""
    if backend_choice.startswith("Hugging Face"):
        hf_preset = st.selectbox(
            "Model", list(HF_MODEL_PRESETS.keys()), key="hf_preset",
        )
        if hf_preset == "Custom model ID…":
            hf_model_id = st.text_input(
                "Hugging Face model ID", key="hf_custom_model",
                placeholder="e.g. black-forest-labs/FLUX.1-dev",
            )
        else:
            hf_model_id = HF_MODEL_PRESETS[hf_preset]

        if not st.secrets.get("HF_TOKEN", ""):
            st.warning(
                "No HF_TOKEN in secrets yet — add one under Settings → Secrets "
                "to use this backend.",
                icon="🔑",
            )

    pollinations_model = "flux"
    pollinations_enhance = False
    if backend_choice.startswith("Pollinations"):
        poll_preset = st.selectbox(
            "Model", list(POLLINATIONS_MODELS.keys()), key="poll_preset",
        )
        pollinations_model = POLLINATIONS_MODELS[poll_preset]
        pollinations_enhance = st.toggle(
            "Enhance prompt (AI-improved prompt before generating)",
            value=False, key="poll_enhance",
        )

        if st.secrets.get("POLLINATIONS_KEY", ""):
            st.caption("🔓 Pollinations key detected — premium models unlocked.")
        elif "needs key" in poll_preset:
            st.warning(
                "This model needs POLLINATIONS_KEY in secrets — get a free key "
                "at enter.pollinations.ai, or pick Flux/zimage/Turbo which work "
                "without one.",
                icon="🔑",
            )

    allow_fallback = st.toggle(
        "Auto-fallback to Pollinations on failure",
        value=True,
        key="allow_fallback",
        disabled=backend_choice.startswith("Pollinations"),
    )

    style_name = st.selectbox("Style preset", list(STYLE_PRESETS.keys()), index=2, key="style_name")
    shape_key = st.selectbox("Shape / resolution", list(SHAPES.keys()), key="shape_key")
    num_images = st.slider("Images per run", 1, MAX_IMAGES_PER_RUN, 1, key="num_images")
    guidance = st.slider(
        "Guidance scale (Hugging Face only)", 1.0, 20.0, 7.0, 0.5,
        key="guidance",
        help="Higher = follows the prompt more literally. Not used by Pollinations.",
    )
    seed = st.number_input(
        "Seed (-1 = random)", value=-1, step=1, key="seed",
        help="Fix a seed to make results reproducible.",
    )

    st.divider()
    st.caption(
        "Hugging Face free tier has a small monthly credit pool — expect it to "
        "run out with regular use. Pollinations has no such ceiling and is the "
        "reliable fallback."
    )


# ----------------------------------------------------------------------------
# Main — tabs
# ----------------------------------------------------------------------------

tab_img, tab_video, tab_history = st.tabs(["🖼️ Image", "🎥 Video (coming soon)", "🗂️ History"])

with tab_img:
    prompt = st.text_area(
        "Prompt",
        key="prompt",
        placeholder="A lone samurai walking through neon-lit rain at midnight…",
        height=100,
    )
    negative = st.text_input("Negative prompt", value=DEFAULT_NEGATIVE, key="negative")

    if st.button("✨ Generate", type="primary", use_container_width=True, disabled=not prompt.strip(), key="btn_generate"):
        if backend_choice.startswith("Hugging Face") and not hf_model_id.strip():
            st.error("Enter a model ID (or pick a preset) for the Hugging Face backend.")
            st.stop()

        preset = STYLE_PRESETS[style_name]
        final_prompt = prompt.strip() + preset["suffix"]
        final_negative = ", ".join(x for x in [negative.strip(), preset["negative"]] if x)

        results: list[GenResult] = []
        progress = st.progress(0.0, text="Generating…")
        for i in range(num_images):
            run_seed = int(seed) if int(seed) > 0 else -1
            r = generate_one(
                backend_choice, allow_fallback, final_prompt,
                final_negative, shape_key, guidance, run_seed,
                hf_model_id=hf_model_id,
                pollinations_model=pollinations_model,
                pollinations_enhance=pollinations_enhance,
            )
            r.style = style_name
            results.append(r)
            st.session_state.history.insert(0, r)
            progress.progress((i + 1) / num_images, text=f"Generated {i + 1}/{num_images}")
        progress.empty()

        st.session_state.last_batch = len(results)

    # Render the most recent batch from history (stable across reruns)
    batch = st.session_state.history[: st.session_state.get("last_batch", 0)]
    if batch:
        cols = st.columns(min(len(batch), 2))
        for i, r in enumerate(batch):
            with cols[i % len(cols)]:
                if r.image_bytes:
                    st.image(r.image_bytes, use_container_width=True)
                    note = f"{r.backend} · {r.elapsed:.1f}s"
                    if "fallback_from" in r.extras:
                        note += " · fell back after Hugging Face error"
                    st.caption(note)
                    st.download_button(
                        "⬇️ Download PNG",
                        data=r.image_bytes,
                        file_name=f"lumora_{i}.png",
                        mime="image/png",
                        key=f"dl_batch_{i}",
                        use_container_width=True,
                    )
                else:
                    st.error(f"{r.backend} failed: {r.error}")

with tab_video:
    st.info(
        "Video generation isn't wired up yet. When you're ready, this tab is "
        "designed to plug into a real text-to-video provider (Hugging Face "
        "Inference Providers, Replicate, or fal.ai — Wan2.1, Kling, "
        "HunyuanVideo, etc., billed per second of GPU time). The provider "
        "abstraction in this app makes that a drop-in addition."
    )
    st.text_area("Video prompt (disabled)", disabled=True, key="video_prompt",
                 placeholder="Available once a video provider is connected…")

with tab_history:
    if not st.session_state.history:
        st.caption("No generations yet — your session history will appear here.")
    else:
        if st.button("Clear history", key="btn_clear_history"):
            st.session_state.history = []
            st.rerun()
        for i, r in enumerate(st.session_state.history):
            with st.expander(
                f"{'✅' if r.image_bytes else '❌'} {r.style or 'raw'} · {r.backend} · {r.prompt[:70]}"
            ):
                st.write(f"**Prompt:** {r.prompt}")
                if r.image_bytes:
                    st.image(r.image_bytes, width=420)
                else:
                    st.error(r.error)
