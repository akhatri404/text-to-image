\# Lumora Image Studio (Streamlit)



Free cloud image generation with \*\*no local GPU\*\* — Perchance (unofficial) with automatic Pollinations.ai fallback.



\## Setup



```bash

pip install streamlit perchance

playwright install chromium   # required: the perchance wrapper drives a headless browser

streamlit run app.py

```



If `playwright` isn't pulled in automatically: `pip install playwright` first.



\## How it works



\- \*\*Perchance route (⚠️ unofficial):\*\* the `perchance` PyPI package launches headless Chromium, visits perchance.org to obtain a verification token, then calls the same internal image endpoint the website uses. There is \*\*no official API\*\* — this can break, rate-limit, or get blocked at any time. Keep usage modest; don't build production on it.

\- \*\*Pollinations.ai route (official \& free):\*\* a plain HTTP GET returns the image. Slower/queue-dependent at peak times but sanctioned and stable.

\- \*\*Fallback:\*\* if Perchance errors after retries, the app silently retries the same prompt on Pollinations (toggle in sidebar).



\## Style presets



Realistic, Cinematic, Anime, Digital Art, Fantasy, Cyberpunk, Portrait Photography — each appends curated prompt/negative-prompt modifiers. "None" sends your raw prompt.



\## Video



Perchance has \*\*no video backend\*\* — nothing exists to reverse-engineer. The Video tab is a wired placeholder for a paid provider (Replicate / fal.ai) later.



\## Honest limits vs Colab/Kaggle



This isn't "unlimited": Perchance rate-limits by IP/token, resolutions are capped (\~768px), and the backend is whatever SDXL variant the site currently runs — you can't pick models. What you gain over Colab/Kaggle is zero session timeouts, zero GPU quota management, and instant startup.

