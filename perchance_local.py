"""
Lumora Image Studio — Streamlit front-end over free, no-GPU image backends.

Backends:
  1. Perchance (unofficial, reverse-engineered via the `perchance` package;
     uses a headless Playwright browser to obtain a verification token).
  2. Pollinations.ai (official free public API) — used as automatic fallback
     when Perchance rate-limits or errors.

NOTE: Perchance has NO official API. This can break at any time and heavy
usage may get you rate-limited or blocked. Keep volume modest.
Video generation is NOT available on Perchance — the Video tab is a
placeholder wired for a future paid provider (Replicate / fal.ai).
"""

from __future__ import annotations

import asyncio
import io
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import streamlit as st

# ----------------------------------------------------------------------------
# Playwright bootstrap (needed on Streamlit Community Cloud, harmless locally)
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Installing headless browser (first run only)…")
def _ensure_playwright_browser() -> str:
    """Install Chromium for Playwright if missing. Cached for the app's lifetime."""
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            return f"playwright install failed: {proc.stderr[-500:]}"
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"playwright bootstrap error: {exc}"

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
    "Square (768×768)": ("square", 768, 768),
    "Portrait (512×768)": ("portrait", 512, 768),
    "Landscape (768×512)": ("landscape", 768, 512),
}

DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, jpeg artifacts"

MAX_IMAGES_PER_RUN = 4          # be a good citizen — don't hammer free backends
PERCHANCE_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 4


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

def _run_async(coro):
    """Run an async coroutine safely from Streamlit's sync world.

    Executes in a dedicated thread with its own event loop so Playwright's
    async machinery never clashes with Streamlit's script-runner thread.
    """
    import threading

    result: dict = {}

    def _worker():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller
            result["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=300)
    if t.is_alive():
        raise TimeoutError("Generation timed out after 300s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _perchance_generate(
    prompt: str,
    negative_prompt: str,
    shape: str,
    guidance_scale: float,
    seed: int,
) -> bytes:
    """Generate one image via the unofficial perchance package."""
    import perchance  # imported lazily so the app still boots without it

    if not hasattr(perchance, "ImageGenerator"):
        raise ImportError(
            f"The imported 'perchance' module at {getattr(perchance, '__file__', '?')} "
            "has no ImageGenerator — you're likely shadowing the real package with a "
            "local file/folder named 'perchance', or the wrong package is installed. "
            "Expected: perchance 0.1.0 by EeeMoon (pip install -U perchance, Python 3.10+)."
        )

    async with perchance.ImageGenerator() as gen:
        result = await gen.image(
            prompt,
            negative_prompt=negative_prompt or None,
            shape=shape,  # 'portrait' | 'square' | 'landscape'
            guidance_scale=guidance_scale,
            seed=seed,
        )
        binary = await result.download()
        return binary.read() if hasattr(binary, "read") else bytes(binary)


def generate_via_perchance(
    prompt: str, negative_prompt: str, shape: str, guidance_scale: float, seed: int
) -> GenResult:
    started = time.time()
    boot = _ensure_playwright_browser()
    if boot != "ok":
        return GenResult(
            prompt=prompt, style="", backend="Perchance (unofficial)",
            error=boot, elapsed=time.time() - started,
        )
    last_err = ""
    for attempt in range(1, PERCHANCE_MAX_RETRIES + 1):
        try:
            img = _run_async(
                _perchance_generate(prompt, negative_prompt, shape, guidance_scale, seed)
            )
            return GenResult(
                prompt=prompt,
                style="",
                backend="Perchance (unofficial)",
                image_bytes=img,
                seed=seed,
                elapsed=time.time() - started,
            )
        except Exception as exc:  # noqa: BLE001 — surface everything to the UI
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < PERCHANCE_MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return GenResult(
        prompt=prompt,
        style="",
        backend="Perchance (unofficial)",
        error=last_err,
        elapsed=time.time() - started,
    )


def generate_via_pollinations(
    prompt: str, negative_prompt: str, width: int, height: int, seed: int
) -> GenResult:
    """Official free API — simple GET returns raw image bytes."""
    started = time.time()
    params = {
        "width": width,
        "height": height,
        "nologo": "true",
    }
    if seed and seed > 0:
        params["seed"] = seed
    if negative_prompt:
        params["negative_prompt"] = negative_prompt
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt)
        + "?"
        + urllib.parse.urlencode(params)
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LumoraImageStudio/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            img = resp.read()
        return GenResult(
            prompt=prompt,
            style="",
            backend="Pollinations.ai",
            image_bytes=img,
            seed=seed,
            elapsed=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return GenResult(
            prompt=prompt,
            style="",
            backend="Pollinations.ai",
            error=f"{type(exc).__name__}: {exc}",
            elapsed=time.time() - started,
        )


def generate_one(
    backend_choice: str,
    allow_fallback: bool,
    prompt: str,
    negative_prompt: str,
    shape_key: str,
    guidance_scale: float,
    seed: int,
) -> GenResult:
    shape, width, height = SHAPES[shape_key]

    if backend_choice.startswith("Perchance"):
        result = generate_via_perchance(prompt, negative_prompt, shape, guidance_scale, seed)
        if result.error and allow_fallback:
            fb = generate_via_pollinations(prompt, negative_prompt, width, height, seed)
            fb.extras["fallback_from"] = result.error
            return fb
        return result

    return generate_via_pollinations(prompt, negative_prompt, width, height, seed)


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
        ["Perchance (unofficial ⚠️)", "Pollinations.ai (official free API)"],
        key="backend_choice",
        help=(
            "Perchance has no official API — this uses a reverse-engineered "
            "wrapper that may break or rate-limit at any time."
        ),
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
        "Guidance scale (Perchance only)", 1.0, 20.0, 7.0, 0.5,
        key="guidance",
        help="Higher = follows the prompt more literally.",
    )
    seed = st.number_input(
        "Seed (-1 = random)", value=-1, step=1, key="seed",
        help="Fix a seed on Perchance to make results reproducible.",
    )

    st.divider()
    st.caption(
        "⚠️ Perchance route is a gray-area hack: keep volume modest, "
        "expect breakage, and don't build a business on it. "
        "Video is not available on Perchance."
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
                        note += " · fell back after Perchance error"
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
        "**Perchance has no video generation backend**, so there is nothing to "
        "reverse-engineer here. When you're ready, this tab is designed to plug "
        "into a real text-to-video provider (Replicate or fal.ai — WAN, Kling, "
        "Hunyuan Video, etc., billed per second of GPU time). "
        "The provider abstraction in this app makes that a drop-in addition."
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
