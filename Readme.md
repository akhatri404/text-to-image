# Lumora Image Studio (Streamlit)

Free/paid cloud image generation with **no local GPU** — Hugging Face Inference Providers with automatic Pollinations.ai fallback.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Backends

- **Hugging Face (official):** real model choice (FLUX.1-schnell, SDXL, SD3.5-medium, or any custom model ID). Needs a free `HF_TOKEN` — see below. Small monthly free-credit pool (roughly $0.10 → ~80 images depending on model), then `402` until next month or you add billing.
- **Pollinations.ai (official):** model picker (Flux, zimage, Turbo work with no key; Nano Banana Pro, Seedream Pro, GPT Image Large, Ideogram v4 Quality need a free `POLLINATIONS_KEY`), optional AI prompt-enhance toggle, resolutions up to 1920×1080. Also the automatic fallback if Hugging Face errors or runs dry.

Perchance was removed: it has no official API, relied on a reverse-engineered wrapper, and consistently broke (import shadowing, Playwright/browser bootstrap issues, and — the final blocker — Perchance rejecting Streamlit Cloud's shared datacenter IPs during token verification). Not worth maintaining.

## Token setup

**Hugging Face:**
1. Free account at huggingface.co.
2. Settings → Access Tokens → New token → **Read** role is sufficient.
3. `.streamlit/secrets.toml` locally, or Streamlit Cloud → Settings → Secrets:
   `HF_TOKEN = "hf_xxxxxxxx"`

**Pollinations (optional, unlocks premium models):**
1. Free account/sign-in at enter.pollinations.ai.
2. Copy the `sk_...` secret key.
3. Same secrets file: `POLLINATIONS_KEY = "sk_xxxxxxxx"`
4. Without this key, Flux/zimage/Turbo still work via the legacy no-key endpoint — nothing breaks, you just don't get the premium models.

## Style presets

Realistic, Cinematic, Anime, Digital Art, Fantasy, Cyberpunk, Portrait Photography — each appends curated prompt/negative-prompt modifiers. "None" sends your raw prompt.

## Video

Not wired up yet. The provider-abstraction pattern (`generate_one` → backend function → `GenResult`) makes it a drop-in addition once you pick a paid provider (HF Inference Providers text-to-video, Replicate, or fal.ai — Wan2.1 is available on HF's router and is the same model family used in past Kaggle experiments).
