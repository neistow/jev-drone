# OpenRouter Decision Models Analysis for `jev-drone` & `jev-rover`

> **Source Crawled:** [`https://openrouter.ai/models?output_modalities=decisions&order=latency-low-to-high`](https://openrouter.ai/models?output_modalities=decisions&order=latency-low-to-high) (and `/api/v1/models?output_modalities=decisions` + all 25 individual model pages)
> **Crawled Date:** 2026-10-16

---

## 1. Quick Copy-Paste Model Names for OpenRouter

### Tier 1: Best / Recommended Models (Fast p50 + Tight p90 Tail + Full System One Schema)

```text
microsoft/microsoft-decision-1
~typesafe/jev-latest
typesafe/jev-1.14
typesafe/jev-1.13
liquid/d1-ultra
jaredpalmer/kev-4b
openai/gpt-6-luna-decisions
nace-ai/drex-v2.0
```

### Tier 2: Secondary Suitable Models (Viable for ~2–3 Hz Drone/Rover, Higher Latency or Tail Variance)

```text
perplexity/pplx-decider-v1.1-27b
~perplexity/pplx-decider-latest
respan/span-02
nace-ai/drex-v1.5
inception/mercury-decide
cloudflare/clef-omni
cloudflare/clef-flash
upstage/solar-decide-flash
liquid/d1
cloudflare/clef
```

### Ready-to-Run `benchmark.py` Commands

**Tier 1 Benchmark Suite (Rover & Drone):**
```powershell
.venv\Scripts\python benchmark.py --scenario rover --runs 5 --models microsoft/microsoft-decision-1 ~typesafe/jev-latest liquid/d1-ultra jaredpalmer/kev-4b openai/gpt-6-luna-decisions nace-ai/drex-v2.0
.venv\Scripts\python benchmark.py --scenario drone --runs 5 --models microsoft/microsoft-decision-1 ~typesafe/jev-latest liquid/d1-ultra jaredpalmer/kev-4b openai/gpt-6-luna-decisions nace-ai/drex-v2.0
```

---

## 2. Control-Loop Requirements in `jev-drone` & `jev-rover`

To evaluate which of the 25 OpenRouter decision models are suitable for the experiments in this repository, we match each model against the three architectural constraints of [`tactics.py`](tactics.py), [`rover_tactics.py`](rover_tactics.py), [`tunnel_tactics.py`](tunnel_tactics.py), and [`api_adapter.py`](api_adapter.py):

1. **Multi-Question System One Schema Compatibility (`Choice` + `Score` + `Noul`)**:
   - **Drone (`tactics.py`)** sends a JSON `state` (~1.2k tokens) and asks **3 typed questions simultaneously** in one call: `maneuver` (`choice`, 6 options), `risk` (`score`, 3-level rubric), and `target_truly_lost` (`noul`, binary probability).
   - **Rover (`rover_tactics.py`)** sends a JSON `state` (~1.5k tokens) and asks **4 typed questions simultaneously**: `maneuver` (`choice`, 6 options), `risk` (`score`), `target_truly_lost` (`noul`), and `override_safety_reflex` (`noul`, required to smash through `cardboard` obstacles).
   - **Tunnel (`tunnel_tactics.py`)** asks **3 graded `score` questions** (`steer`, `height`, `risk`).
   - *Requirement:* The model must natively support the System One / OpenRouter Decisions API (`/api/alpha/decisions`) with all three primitives (`choice`, `score`, `noul`) over structured JSON state in a single forward pass.
2. **Real-Time Latency & Tail-Latency Constraints**:
   - Both `tactics.py` and `rover_tactics.py` run a single worker thread with a 1-slot queue (`queue.Queue(maxsize=1)`), polling up to **`call_hz = 3.0 Hz` (333 ms period)** and expiring judgments at **`stale_after_s = 1.5 s`**.
   - At top speed (`3.6 m/s`), every `100 ms` of latency equals `0.36 m` of blind travel. If **p90 latency exceeds `1,500 ms`**, the rover/drone drops into stale-judgment fallback (`brake` / `default`) mid-maneuver.
   - For **`tunnel.py`** (`8–9 m/s`), `stale_after_s` is **`300 ms`**, requiring **p50 < 220 ms** and **p90 < 350 ms**.
3. **Numeric & Spatial Reasoning over Telemetry JSON**:
   - Models must compare numeric ranges across 5 horizontal sectors (`sector_range_m`), vertical clearances (`sector_clearance_m > 0.4` for rover underpass vs `< 0.4` obstacle), vertical headroom (`free_ahead_above_m` vs `free_ahead_level_m` for drone `climb`), and material type (`nearest_obstacle_type == "cardboard"` for `override_safety_reflex`).

---

## 3. Complete Crawl of All 25 OpenRouter Decision Models (Sorted by Latency Low-to-High)

All 25 models currently listed under `output_modalities=decisions`, ordered by live endpoint **p50 latency**:

| Rank | Model Name | OpenRouter Model ID | p50 (ms) | p75 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Input $/1M | Context | Modality | Window Reqs | Suitability |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 1 | Together: Tev1 4B Experimental | `togethercomputer/tev1-4b-experimental` | 141 | 210 | 322 | 395 | 699 | $0.042 | 32k | `text->decisions` | 732 | **Excluded** (Chat-completions single-choice only) |
| 2 | LiquidAI: d1 | `liquid/d1` | 155 | 214 | 1,911 | 4,648 | 8,233 | $0.020 | 65k | `text+image->decisions` | 132,458 | **Tier 2** (Fast p50, but severe p90/p99 spikes > 1.5s) |
| 3 | Jared Palmer: Kev 4B | `jaredpalmer/kev-4b` | 171 | 203 | 325 | 665 | 3,606 | $0.042 | 8k | `text->decisions` | 18,561 | **Tier 1** (Fast open-weight 4B System One) |
| 4 | Microsoft: Microsoft-Decision-1 | `microsoft/microsoft-decision-1` | 174 | 221 | 302 | 392 | 758 | $0.042 | 32k | `text->decisions` | 53,460 | **Tier 1** (Best 9B latency + tight p90=302ms) |
| 5 | TypeSafe: Jev 1.14 | `typesafe/jev-1.14` | 182 | 230 | 346 | 488 | 838 | $0.042 | 262k | `text+image->decisions` | 629,615 | **Tier 1** (Latest flagship Jev reference) |
| 6 | TypeSafe: Jev Latest | `~typesafe/jev-latest` | 182 | 230 | 346 | 488 | 838 | $0.042 | 262k | `text+image->decisions` | 629,615 | **Tier 1** (OpenRouter alias for `typesafe/jev-1.14`) |
| 7 | TypeSafe: Jev 1.13 | `typesafe/jev-1.13` | 191 | 251 | 307 | 342 | 483 | $0.042 | 262k | `text+image->decisions` | 2,244,927 | **Tier 1** (Best p95/p99 tail latency; best for Tunnel) |
| 8 | Respan: Span-01 | `respan/span-01` | 194 | 342 | 386 | 425 | 559 | $0.020 | — | `text->decisions` | 356 | **Excluded** (Conversation behavior scorer only) |
| 9 | LiquidAI: d1-ultra | `liquid/d1-ultra` | 208 | 253 | 316 | 378 | 1,749 | $0.042 | 32k | `text+image->decisions` | 166,652 | **Tier 1** (High-capacity LiquidAI + tight p90=316ms) |
| 10 | Cloudflare: Clef | `cloudflare/clef` | 244 | 555 | 2,797 | 5,162 | 6,988 | $0.042 | 65k | `text+image->decisions` | 91,496 | **Tier 2** (27B model, good p50 but p90=2.8s > 1.5s) |
| 11 | OpenAI: GPT-6 Luna Decisions | `openai/gpt-6-luna-decisions` | 250 | 343 | 529 | 859 | 3,713 | $0.100 | 1.05M | `text+image->decisions` | 69,841 | **Tier 1** (Highest reasoning intelligence, p50=250ms) |
| 12 | Respan: Span-02 | `respan/span-02` | 293 | 310 | 1,173 | 2,101 | 6,762 | $0.010 | 131k | `text+image->decisions` | 6,919 | **Tier 2** (Cheapest full System One model at $0.01/M) |
| 13 | Perplexity: Decider Latest | `~perplexity/pplx-decider-latest` | 306 | 501 | 919 | 1,143 | 1,964 | $0.020 | 262k | `text+image->decisions` | 218,940 | **Tier 2** (Alias for `pplx-decider-v1.1-27b`) |
| 14 | Perplexity: Decider V1.1 27B | `perplexity/pplx-decider-v1.1-27b` | 306 | 501 | 919 | 1,143 | 1,964 | $0.020 | 262k | `text+image->decisions` | 218,940 | **Tier 2** (Strong 27B model at $0.02/M, p50=306ms) |
| 15 | Nace.AI: Drex v2.0 | `nace-ai/drex-v2.0` | 311 | 431 | 586 | 742 | 1,039 | $0.060 | 131k | `text->decisions` | 54,767 | **Tier 1** (35B MoE / 3B active, p99=1.04s < 1.5s) |
| 16 | Respan: Span-01 Lite | `respan/span-01-lite` | 342 | 802 | 1,185 | 1,256 | 1,383 | $0.000 | — | `text->decisions` | 52 | **Excluded** (Conversation behavior scorer only) |
| 17 | Nace.AI: Drex v1.5 | `nace-ai/drex-v1.5` | 358 | 658 | 985 | 1,181 | 1,856 | $0.040 | 131k | `text->decisions` | 103,674 | **Tier 2** (9B open-weight; slower than `drex-v2.0`) |
| 18 | ScaleDown: Classify-1 | `scaledown/classify-1` | 384 | 892 | 2,243 | 3,465 | 5,548 | $0.040 | 1.00M | `text+image+file->decisions` | 21,021 | **Excluded** (Document classifier; p90=2.24s > 1.5s) |
| 19 | Inception: Mercury Decide | `inception/mercury-decide` | 413 | 567 | 955 | 1,284 | 1,982 | $0.020 | 65k | `text+image->decisions` | 28,016 | **Tier 2** (Diffusion LLM System One; ~2.4 Hz rate) |
| 20 | Cloudflare: Clef Omni | `cloudflare/clef-omni` | 422 | 497 | 602 | 702 | 1,031 | $0.150 | 65k | `text+image+audio+video->decisions` | 4,096 | **Tier 2** (30B-A3B MoE; tight tail p99=1.03s, pricier) |
| 21 | Cloudflare: Clef Flash | `cloudflare/clef-flash` | 495 | 724 | 1,052 | 1,264 | 1,804 | $0.021 | 24k | `text+image->decisions` | 7,758 | **Tier 2** (9B Workers AI; ~2 Hz rate, 2K state limit) |
| 22 | Upstage: Solar Decide Flash | `upstage/solar-decide-flash` | 565 | 663 | 892 | 1,061 | 1,541 | $0.050 | 524k | `text->decisions` | 1,716 | **Tier 2** (Borderline p50=565ms; usable for Rover) |
| 23 | Respan: Span-01 Lite (free) | `respan/span-01-lite:free` | 569 | 819 | 879 | 908 | 956 | $0.000 | — | `text->decisions` | 91 | **Excluded** (Conversation behavior scorer only) |
| 24 | Inception: Mercury Decide (free) | `inception/mercury-decide:free` | 1,361 | 1,730 | 2,211 | 2,577 | 3,573 | $0.000 | 65k | `text+image->decisions` | 15,783 | **Excluded** (Too slow: p50=1.36s, p90=2.21s > 1.5s) |
| 25 | Upstage: Solar Decide | `upstage/solar-decide` | 20,492 | 22,268 | 24,504 | 25,819 | 29,312 | $0.050 | 524k | `text->decisions` | 7,076 | **Excluded** (Too slow: p50=20.5s breaks control loop) |

---

## 4. Detailed Model Analysis & Selection Rationale

### Tier 1: Primary Models for `jev-drone`, `jev-rover`, and `tunnel`

1. **`microsoft/microsoft-decision-1`**
   - **Why it fits:** Post-trained from Qwen3.5-9B specifically for single-pass calibrated decision probabilities. At **174 ms p50** and **302 ms p90**, over 90% of requests finish within the 333 ms (`3 Hz`) control cycle, and even its **p99 (758 ms)** is half of the `1.5 s` staleness limit.
   - **Best for:** Primary challenger to Jev across `drone`, `rover`, and `tunnel`.

2. **`~typesafe/jev-latest` / `typesafe/jev-1.14` & `typesafe/jev-1.13`**
   - **Why they fit:** The native System One models this codebase was built for.
     - **`typesafe/jev-1.14` (`~typesafe/jev-latest`)** offers the newest weights and fastest median latency (**182 ms p50**, **346 ms p90**).
     - **`typesafe/jev-1.13`** has the tightest tail latency of any model on OpenRouter (**307 ms p90**, **342 ms p95**, **483 ms p99** across 2.24M requests), making it uniquely reliable for the `tunnel.py` experiment (`stale_after_s = 0.3 s`).
   - **Zero-Code Routing:** Passing `jev-latest` or `jev-preview` uses native `TypeSafeClient` (`TYPESAFE_API_KEY`), while passing `~typesafe/jev-latest`, `typesafe/jev-1.14`, `typesafe/jev-1.13`, or setting `USE_OPENROUTER=1` automatically routes through `SystemOneAdapter` (`OPENROUTER_API_KEY`) with zero code changes.

3. **`liquid/d1-ultra`**
   - **Why it fits:** Liquid AI's largest System One decision model. While base `liquid/d1` suffers from severe p90/p99 tail spikes (1.91 s / 8.23 s), **`liquid/d1-ultra` fixes the tail completely** (**208 ms p50**, **316 ms p90**, **378 ms p95**) while providing stronger reasoning capacity for multi-variable spatial states.

4. **`jaredpalmer/kev-4b`**
   - **Why it fits:** Open-weight 4B model (LoRA adapter + pointer head on Qwen3.5-4B-Base) built explicitly as an Apache-2.0 drop-in replacement for TypeSafe's `/v1/systemone` contract (`choice`, `score`, `noul`). Very low latency (**171 ms p50**, **325 ms p90**). Its 8,192-token context easily fits the ~1.2k–1.5k token scene state.

5. **`openai/gpt-6-luna-decisions`**
   - **Why it fits:** Frontier GPT-6 Luna served via the Decisions API (`choice`, `score`, `noul`). Although slightly slower at the median (**250 ms p50**, **529 ms p90**) and pricier ($0.10/1M input), it still comfortably meets the 3 Hz / 1.5 s control budget and brings the highest reasoning capability for subtle multi-constraint rules (such as simultaneously evaluating `sector_clearance_m > 0.4`, `nearest_obstacle_type == "cardboard"`, and `override_safety_reflex` in `rover_tactics.py`).

6. **`nace-ai/drex-v2.0`**
   - **Why it fits:** 35B Mixture-of-Experts model with 3B active parameters, explicitly built for TypeSafe System One compatibility (`choice`, `score`, `noul`). Achieves **311 ms p50**, **586 ms p90**, and an impressively bounded **1,039 ms p99** (never breaching `stale_after_s = 1.5 s`), outperforming its predecessor `nace-ai/drex-v1.5` in both intelligence and speed.

---

### Tier 2: Secondary / Specialized Contenders

- **`perplexity/pplx-decider-v1.1-27b` (`~perplexity/pplx-decider-latest`)**: Full `choice` / `score` / `noul` support from a large 27B checkpoint at half the cost of Jev ($0.02/1M). With **306 ms p50** and **919 ms p90**, it is well suited for `jev-drone` and `jev-rover`, though too slow for `tunnel.py`.
- **`respan/span-02`**: Upgraded successor to `span-01` that adds general `choice`, `yes/no`, and `score` primitives. At **$0.01/1M input** and **293 ms p50** (310 ms p75), it is the best ultra-low-cost benchmark option, with moderate tail latency (p90 = 1.17 s).
- **`nace-ai/drex-v1.5`**: 9B open-weight System One model (**358 ms p50**, **985 ms p90**, $0.04/1M). Useful for comparing 9B dense (`drex-v1.5`) vs 35B-A3B MoE (`drex-v2.0`).
- **`inception/mercury-decide`**: Diffusion LLM System One endpoint (**413 ms p50**, **955 ms p90**, $0.02/1M). Runs at ~2.4 Hz sequentially, viable for `drone` and `rover`.
- **`cloudflare/clef-omni`**: 30B-A3B MoE on Workers AI (**422 ms p50**, **602 ms p90**, **1,031 ms p99**, $0.15/1M). Notably has the cleanest tail latency of Cloudflare's three Clef models.
- **`cloudflare/clef-flash` & `cloudflare/clef`**: 9B (`clef-flash`, p50 = 495 ms, p90 = 1.05 s) and 27B (`clef`, p50 = 244 ms, p90 = 2.80 s) Qwen fine-tunes on Workers AI. Note that Workers AI truncates text state at ~2,000 tokens (our states are ~1,200–1,500 tokens, so they fit, but `clef` suffers from 2.8 s p90 tail spikes).
- **`upstage/solar-decide-flash`**: System One compatible on Solar Mini 4 (**565 ms p50**, **892 ms p90**). Usable for slower `rover` experiments, but too high median latency for fast drone evasion.
- **`liquid/d1`**: Fast median (**155 ms p50**, $0.02/1M), but **1,911 ms p90** and **8,233 ms p99** cause frequent stale judgments; prefer `liquid/d1-ultra`.

---

### Excluded Models (Not Suitable for `jev-drone` / `jev-rover`)

1. **`togethercomputer/tev1-4b-experimental`**: Uses Qwen's standard next-token head over the regular Chat Completions API to return a single option letter (`A`–`X`) for a single multiple-choice question. Does not support `/api/alpha/decisions` or simultaneous `choice` + `score` + `noul` questions.
2. **`respan/span-01`, `respan/span-01-lite`, `respan/span-01-lite:free`**: Designed exclusively to score plain-language behavior presence over conversation spans (`context_length: 0`); superseded by `respan/span-02` for general decision tasks.
3. **`scaledown/classify-1`**: Built for 1M-context document classification; **p90 latency (2,243 ms)** exceeds the `1.5 s` control loop staleness threshold.
4. **`inception/mercury-decide:free`**: Free-tier queuing pushes **p50 to 1,361 ms** and **p90 to 2,211 ms**, violating real-time control loop timing.
5. **`upstage/solar-decide`**: Non-flash checkpoint has a **p50 latency of 20,492 ms (~20.5 s)**, unusable in a live simulation loop.

---

## 5. Multimodal (Image-Capable) Decision Models for Vision + Telemetry Experiments

15 of the 25 OpenRouter decision models advertise `image` in their catalog `input_modalities` (`text+image->decisions`), while **10 of those 15** are officially verified and documented in [OpenRouter's Multimodal Decisions Guide](https://openrouter.ai/docs/guides/community/multimodal-decisions) with explicit `image_url` parsers and limits. (By contrast, `microsoft/microsoft-decision-1`, `jaredpalmer/kev-4b`, `nace-ai/drex-v2.0`, `nace-ai/drex-v1.5`, and `upstage/solar-decide*` are **text-only**).

### Quick Copy-Paste: Verified Image-Capable Models for the Experiment

**Tier 1 (Officially Verified Multimodal + Low Latency for ~2–3 Hz Control Loop):**
```text
~typesafe/jev-latest
typesafe/jev-1.14
typesafe/jev-1.13
openai/gpt-6-luna-decisions
```
*(Native TypeSafe equivalents: `jev-preview` = `jev-1.14.0` with `Box` support; `jev-latest` = `jev-1.13.0`)*

**Tier 2 (Officially Verified Multimodal Contenders for ~1.5–2.5 Hz Control Loop):**
```text
perplexity/pplx-decider-v1.1-27b
~perplexity/pplx-decider-latest
inception/mercury-decide
cloudflare/clef-omni
cloudflare/clef-flash
cloudflare/clef
```

**Unverified Catalog-Listed `text+image` Models (Advertise `image` in `/api/v1/models`, not listed in OpenRouter Multimodal Guide):**
```text
liquid/d1-ultra
respan/span-02
liquid/d1
```

### Comparison of All 15 Image-Capable Decision Models

| Rank (by p50) | Model Name | OpenRouter Model ID | p50 (ms) | p90 (ms) | p99 (ms) | Input $/1M | Context | Official Multimodal Docs | Per-Model Image Limits & Experiment Compatibility Notes | Suitability |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| 1 |TypeSafe: Jev 1.14 | `typesafe/jev-1.14` | 182 | 346 | 838 | $0.042 | 262k | **Verified** | **JPEG/PNG only** (no WebP); **280–1,568 px/side** & **78,400–2,458,624 total px** ($\le$16 MiB/img). Supports **`Box`** grounding questions (`jev-preview` on native API) | **Tier 1** (Recommended) |
| 2 | TypeSafe: Jev Latest | `~typesafe/jev-latest` | 182 | 346 | 838 | $0.042 | 262k | **Verified** | Alias for `typesafe/jev-1.14` on OpenRouter (`jev-latest` on native API resolves to `jev-1.13.0`) | **Tier 1** (Recommended) |
| 3 | TypeSafe: Jev 1.13 | `typesafe/jev-1.13` | 191 | 307 | 483 | $0.042 | 262k | **Verified** | **JPEG/PNG only**; **280–1,568 px/side** & **78,400–2,458,624 total px**. Supports `Choice`/`Score`/`Noul` on images (no `Box`). Tightest p99 (483 ms) | **Tier 1** (Recommended) |
| 4 | OpenAI: GPT-6 Luna Decisions | `openai/gpt-6-luna-decisions` | 250 | 529 | 3,713 | $0.100 | 1.05M | **Verified** | JPEG/PNG/WebP; up to **128 images/req** & 200 questions/req. **Honors `"detail": "low"`** for fast low-token vision in 3 Hz loops | **Tier 1** (Recommended) |
| 5 | Perplexity: Decider Latest | `~perplexity/pplx-decider-latest` | 306 | 919 | 1,964 | $0.020 | 262k | **Verified** | Alias for `perplexity/pplx-decider-v1.1-27b`; top-level `state` array `image_url` | **Tier 2** |
| 6 | Perplexity: Decider V1.1 27B | `perplexity/pplx-decider-v1.1-27b` | 306 | 919 | 1,964 | $0.020 | 262k | **Verified** | 27B multimodal checkpoint; up to 128 questions/req; top-level `state` array `image_url` | **Tier 2** |
| 7 | Inception: Mercury Decide | `inception/mercury-decide` | 413 | 955 | 1,982 | $0.020 | 65k | **Verified** | Diffusion LLM; up to **16 images/req** (<7 MB JPEG each; $\ge$12 MB fails with `422`). Ignores `detail` | **Tier 2** |
| 8 | Cloudflare: Clef Omni | `cloudflare/clef-omni` | 422 | 602 | 1,031 | $0.150 | 65k | **Verified** | 30B-A3B MoE (`text+image+audio+video`); up to **4 images** & **2 videos/req** (2 fps); tight p90/p99 (602 ms / 1.03 s). Ignores `detail` | **Tier 2** (Also supports short video clips) |
| 9 | Cloudflare: Clef | `cloudflare/clef` | 244 | 2,797 | 6,988 | $0.042 | 65k | **Verified** | 27B Qwen3.8; up to **4 images/req**. Workers AI pre-estimates tokens from base64 length (~4 chars/tok): **keep image < 300 KB raw**. Text truncated at ~2K tokens | **Tier 2** (Watch 2.8s p90 tail) |
| 10 | Cloudflare: Clef Flash | `cloudflare/clef-flash` | 495 | 1,052 | 1,804 | $0.021 | 24k | **Verified** | 9B Qwen3.5; up to **4 images/req**. 24k context window with base64 token pre-estimation: **keep image < 70 KB raw**. Text truncated at ~2K tokens | **Tier 2** |
| 11 | LiquidAI: d1-ultra | `liquid/d1-ultra` | 208 | 316 | 1,749 | $0.042 | 32k | Catalog only | Model card describes *"state of text and images... for visual checks"*, but not listed in OpenRouter Multimodal Decisions guide | **Unverified** (Test when OR key active) |
| 12 | Respan: Span-02 | `respan/span-02` | 293 | 1,173 | 6,762 | $0.010 | 131k | Catalog only | Model card describes *"reads a state of text and images"*, but not listed in OpenRouter Multimodal Decisions guide | **Unverified** |
| 13 | LiquidAI: d1 | `liquid/d1` | 155 | 1,911 | 8,233 | $0.020 | 65k | Catalog only | Catalog lists `image`, but model card only mentions text state and p90 (1.91s) spikes | **Unverified** |
| 14 | ScaleDown: Classify-1 | `scaledown/classify-1` | 384 | 2,243 | 5,548 | $0.040 | 1.00M | Catalog only | Document/OCR classifier (`text`, `image`, `file`); p90 (2.24 s) exceeds 1.5 s staleness window | **Excluded** (Too slow) |
| 15 | Inception: Mercury Decide (free) | `inception/mercury-decide:free` | 1,361 | 2,211 | 3,573 | $0.000 | 65k | **Verified** | Free-tier queuing (p50 = 1.36 s, p90 = 2.21 s) exceeds 1.5 s control-loop staleness window | **Excluded** (Too slow) |

---

### Critical Image Requirements to Fit All Experiment Models

In [`flight.py`](flight.py#L161) and [`drive.py`](drive.py#L81), the onboard `Eye` perception camera uses a `4:3` aspect ratio at `W, H = 96, 72`. To ensure an onboard RGB frame works across **every** model above (Jev, GPT-6 Luna, Perplexity Decider, Cloudflare Clef, and Mercury Decide) without `400` or `413` errors:

1. **Resolution Must Be at Least `280 × 280` (`384 × 288` Recommended for 4:3 `Eye`):**
   - **Why:** TypeSafe Jev (`typesafe/jev-1.14`, `typesafe/jev-1.13`, `~typesafe/jev-latest`, and native `jev-latest` / `jev-preview`) enforces a **hard minimum of `280 × 280` pixels (`78,400` total pixels)** and a maximum of `1,568 × 1,568` pixels (`2,458,624` total pixels). Passing a `96 × 72` or `240 × 180` image directly to Jev returns **`400 Bad Request`**.
   - **Experiment Standard:** Render or resize the onboard RGB frame to **`384 × 288`** (`110,592` total pixels, preserving `Eye`'s exact `4:3` FOV aspect ratio).
2. **Format Must Be Baseline `image/jpeg` (< 70 KB Raw Size):**
   - **Why:** Jev rejects `image/webp` with `400` (accepting only `image/jpeg` and `image/png`), while Cloudflare Workers AI (`cloudflare/clef` and `cloudflare/clef-flash`) pre-estimates request tokens from the raw base64 string length (~4 base64 chars/token) *before* decoding the image and rejects with `413` if the estimate exceeds the context window (24k tokens / ~70 KB raw on `clef-flash`, 65k tokens / ~300 KB raw on `clef`).
   - **Experiment Standard:** A `384 × 288` baseline JPEG at `quality=80` is only **~10–15 KB** (~14k–20k base64 chars, ~620–735 input tokens on Jev), fitting **all** models with a 5×–20× safety margin.
3. **Top-Level `state` Array vs. Native TypeSafe `assets` (Handled Automatically by `api_adapter.py`):**
   - Non-Jev OpenRouter models (`openai/gpt-6-luna-decisions`, `perplexity/pplx-decider*`, `inception/mercury-decide`, `cloudflare/clef*`) only read `{"type": "image_url", ...}` parts placed directly in the **top-level `state` array** (nested image parts inside dicts are ignored by non-Jev models).
   - Native TypeSafe (`https://api.typesafe.ai/v1/systemone`) does **not** parse OpenAI-style `image_url` objects directly; it requires extracting images into a top-level `assets` map referenced by UUID strings plus the `"TypeSafe-Danger-Zone-Images": "true"` header.
   - **[`api_adapter.py`](api_adapter.py) bridges both automatically:**
     - [`encode_image_part(rgb_frame)`](api_adapter.py#L97-L138) converts any MuJoCo `(H, W, 3)` RGB array (even `96 × 72`) into a `384 × 288` baseline JPEG `image_url` part (`detail="low"`).
     - [`SystemOneAdapter.system_one()`](api_adapter.py#L399-L419) automatically lifts `image_url` parts to the top-level `state` array for OpenRouter `/api/alpha/decisions`, OR extracts them into `assets` with `"TypeSafe-Danger-Zone-Images": "true"` when running against native TypeSafe (`--provider typesafe`).

```python
from api_adapter import encode_image_part

# Works identically across Native TypeSafe (jev-latest / jev-preview) and all OpenRouter Multimodal Decision Models:
state = [
    {"mission": MISSION, "rover": ROVER, "observed": scene},
    encode_image_part(rgb_frame),  # Auto-resizes to 384x288 (>=280px Jev min) & encodes <15 KB JPEG (<70 KB Clef limit)
]
```


