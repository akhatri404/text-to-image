"""
Lumora Image Studio — Streamlit front-end over Pollinations.ai (official free
public image-generation API, no token needed for the base models).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import streamlit as st


# ----------------------------------------------------------------------------
# Shared usage tracking (this app's API key is shared by ALL visitors —
# st.cache_resource persists across sessions/users on this running instance,
# unlike st.session_state which resets per browser tab).
# ----------------------------------------------------------------------------

@st.cache_resource
def _usage_counters() -> dict:
    return {"pollinations_calls": 0}


def _bump_usage(key: str) -> None:
    counters = _usage_counters()
    counters[key] = counters.get(key, 0) + 1


@st.cache_data(ttl=60)
def _fetch_pollinations_balance(key: str) -> dict:
    """Query Pollinations' real balance endpoint. Cached 60s and shared across
    all users, so we don't hammer it on every page load/rerun."""
    if not key:
        return {}
    try:
        req = urllib.request.Request(
            "https://gen.pollinations.ai/account/balance",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json as _json
            return _json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}

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

DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, jpeg artifacts"

MAX_IMAGES_PER_RUN = 4  # be a good citizen — don't hammer the free backend


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
# Backend
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
        _bump_usage("pollinations_calls")
        return GenResult(
            prompt=prompt, style="", backend=backend_label,
            image_bytes=img, seed=seed, elapsed=time.time() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return GenResult(
            prompt=prompt, style="", backend=f"Pollinations.ai ({model})",
            error=f"{type(exc).__name__}: {exc}", elapsed=time.time() - started,
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
    st.caption("Free cloud generation via Pollinations.ai — no local GPU required.")

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

    style_name = st.selectbox("Style preset", list(STYLE_PRESETS.keys()), index=2, key="style_name")
    shape_key = st.selectbox("Shape / resolution", list(SHAPES.keys()), key="shape_key")
    num_images = st.slider("Images per run", 1, MAX_IMAGES_PER_RUN, 1, key="num_images")
    seed = st.number_input(
        "Seed (-1 = random)", value=-1, step=1, key="seed",
        help="Fix a seed to make results reproducible.",
    )

    st.divider()

    with st.expander("📊 Usage & remaining credits", expanded=False):
        st.caption(
            "⚠️ This API key is shared by every visitor to this app — "
            "there's no per-user quota. One person's heavy use affects everyone."
        )

        counters = _usage_counters()
        st.write(f"**This session's app instance:** {counters['pollinations_calls']} Pollinations images")
        st.caption(
            "Counts reset if the app restarts/redeploys — not a lifetime total."
        )

        st.markdown("**Pollinations Pollen balance**")
        poll_key = st.secrets.get("POLLINATIONS_KEY", "")
        if poll_key:
            bal = _fetch_pollinations_balance(poll_key)
            if "_error" in bal:
                st.caption(f"Couldn't fetch balance: {bal['_error']}")
            elif bal:
                st.json(bal, expanded=False)
            else:
                st.caption("No balance data returned.")
        else:
            st.caption("Add POLLINATIONS_KEY to see live balance here.")


# ----------------------------------------------------------------------------
# Main — tabs
# ----------------------------------------------------------------------------

tab_img, tab_history = st.tabs(["🖼️ Image", "🗂️ History"])

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
        width, height = SHAPES[shape_key]

        results: list[GenResult] = []
        progress = st.progress(0.0, text="Generating…")
        for i in range(num_images):
            run_seed = int(seed) if int(seed) > 0 else -1
            r = generate_via_pollinations(
                final_prompt, final_negative, width, height, run_seed,
                pollinations_model, pollinations_enhance,
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
                    st.caption(f"{r.backend} · {r.elapsed:.1f}s")
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
