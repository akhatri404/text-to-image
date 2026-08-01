# Lumora Image Studio (Streamlit)

Free cloud image generation with **no local GPU** — powered by Pollinations.ai.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Backend

- **Pollinations.ai (official):** model picker (Flux, zimage, Turbo work with no key; Nano Banana Pro, Seedream Pro, GPT Image Large, Ideogram v4 Quality need a free `POLLINATIONS_KEY`), optional AI prompt-enhance toggle, resolutions up to 1920×1080.

Perchance was removed: it has no official API, relied on a reverse-engineered wrapper, and consistently broke (import shadowing, Playwright/browser bootstrap issues, and — the final blocker — Perchance rejecting Streamlit Cloud's shared datacenter IPs during token verification). Not worth maintaining.

Hugging Face Inference Providers was removed: the free credit tier ended, so it stopped working without paid billing.

## Token setup

**Pollinations (optional, unlocks premium models):**
1. Free account/sign-in at enter.pollinations.ai.
2. Copy the `sk_...` secret key.
3. `.streamlit/secrets.toml` locally, or Streamlit Cloud → Settings → Secrets:
   `POLLINATIONS_KEY = "sk_xxxxxxxx"`
4. Without this key, Flux/zimage/Turbo still work via the legacy no-key endpoint — nothing breaks, you just don't get the premium models.

## Style presets

Realistic, Cinematic, Anime, Digital Art, Fantasy, Cyberpunk, Portrait Photography — each appends curated prompt/negative-prompt modifiers. "None" sends your raw prompt.
