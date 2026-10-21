import base64
import io
import json
import os
import urllib.error
import urllib.request
import uuid
from typesafe_sdk import TypeSafeClient

NATIVE_JEV_MODELS = {"jev-latest", "jev-preview", "jev-1.14", "jev-1.13"}
NATIVE_MODEL_ALIASES = {
    "jev-1.14": "jev-preview",
    "jev-1.13": "jev-latest",
}
CHAT_ONLY_DECISION_MODELS = {"togethercomputer/tev1-4b-experimental"}


class MockAnswer:
    def __init__(self, value):
        self.type = None
        self.choice = None
        self.score = 0.0
        self.noul = 0.0
        self.confidence = 1.0
        self.probabilities = {}
        self.legend = {}
        self.box = None

        if isinstance(value, dict):
            self.type = value.get("type")
            if "choice" in value and value["choice"] is not None:
                self.choice = str(value["choice"])
            if "score" in value and value["score"] is not None:
                try:
                    self.score = float(value["score"])
                except (ValueError, TypeError):
                    self.score = 0.0
            if "noul" in value and value["noul"] is not None:
                try:
                    self.noul = float(value["noul"])
                except (ValueError, TypeError):
                    self.noul = 0.0
                if "score" not in value:
                    self.score = self.noul
                if "choice" not in value:
                    self.choice = "true" if self.noul >= 0.5 else "false"
            if "box" in value and isinstance(value["box"], dict):
                self.box = value["box"]
            if "probabilities" in value and isinstance(value["probabilities"], dict):
                self.probabilities = {str(k): float(v) for k, v in value["probabilities"].items()}
            elif self.choice is not None:
                self.probabilities = {self.choice: 1.0}
            if "confidence" in value and value["confidence"] is not None:
                try:
                    self.confidence = float(value["confidence"])
                except (ValueError, TypeError):
                    self.confidence = 1.0
            elif "noul" in value and value["noul"] is not None:
                self.confidence = round(abs(self.noul - 0.5) * 2.0, 4)
            if "legend" in value and isinstance(value["legend"], dict):
                self.legend = value["legend"]
        else:
            self.choice = str(value)
            float_val = 0.0
            if isinstance(value, bool):
                float_val = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                float_val = float(value)
            elif isinstance(value, str):
                low = value.strip().lower()
                if low == "true":
                    float_val = 1.0
                elif low == "false":
                    float_val = 0.0
                else:
                    try:
                        float_val = float(value)
                    except ValueError:
                        pass
            self.score = float_val
            self.noul = float_val
            self.confidence = 1.0
            self.probabilities = {self.choice: 1.0}


class MockResponse:
    def __init__(self, answers_dict, usage_dict, model=""):
        self.model = model
        self.answers = {
            k: (v if isinstance(v, MockAnswer) else MockAnswer(v))
            for k, v in (answers_dict or {}).items()
        }
        in_tok = (usage_dict or {}).get("input_tokens")
        if in_tok is None:
            in_tok = (usage_dict or {}).get("prompt_tokens", 0)
        out_tok = (usage_dict or {}).get("output_tokens")
        if out_tok is None:
            out_tok = (usage_dict or {}).get("completion_tokens", 0)
        self.usage = type("Usage", (), {
            "input_tokens": int(in_tok or 0),
            "output_tokens": int(out_tok or 0),
        })()


def encode_image_part(image, width=384, height=288, quality=80, detail="low"):
    """
    Prepare an onboard camera frame (numpy RGB array, PIL.Image, or file path) as a
    cross-model compatible OpenRouter/TypeSafe `image_url` part.

    Enforces constraints required across all multimodal decision models:
    - Baseline RGB JPEG (`image/jpeg`)
    - Dimensions between 280x280 and 1568x1568 (default 384x288 = 110,592 px >= 78,400 px Jev minimum)
    - Compact file size (~10-20 KB, well under Cloudflare Clef's 300 KB / 70 KB base64 limits)
    """
    from PIL import Image as PILImage

    if isinstance(image, (str, os.PathLike)):
        pil_img = PILImage.open(image).convert("RGB")
    elif hasattr(image, "save") and hasattr(image, "convert"):
        pil_img = image.convert("RGB")
    else:
        import numpy as np
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(arr).convert("RGB")

    w, h = pil_img.size
    if w < 280 or h < 280 or (w * h) < 78400 or w > 1568 or h > 1568:
        target_w = max(280, min(1568, width))
        target_h = max(280, min(1568, height))
        if target_w * target_h < 78400:
            target_w, target_h = 384, 288
        pil_img = pil_img.resize((target_w, target_h), PILImage.Resampling.BILINEAR)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{b64}",
            "detail": detail,
        },
    }


def _is_image_url_part(obj):
    return (
        isinstance(obj, dict)
        and obj.get("type") == "image_url"
        and isinstance(obj.get("image_url"), dict)
        and isinstance(obj["image_url"].get("url"), str)
    )


def _extract_typesafe_assets(obj, assets):
    """
    Recursively replace `{"type": "image_url", ...}` parts with UUID asset references
    for native TypeSafe `/v1/systemone`.
    """
    if _is_image_url_part(obj):
        url = obj["image_url"]["url"]
        fmt = "jpg"
        raw_b64 = url
        if url.startswith("data:image/"):
            header, _, raw_b64 = url.partition(",")
            mime = header.split(";")[0].replace("data:image/", "").lower()
            if mime == "png":
                fmt = "png"
            else:
                fmt = "jpg"
        ref = f"img-{uuid.uuid4()}"
        assets[ref] = {"type": "image", "format": fmt, "data_base64": raw_b64}
        return ref
    if isinstance(obj, dict):
        return {k: _extract_typesafe_assets(v, assets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_extract_typesafe_assets(item, assets) for item in obj]
    return obj


def _normalize_openrouter_multimodal_state(state):
    """
    Ensure any `image_url` parts sit directly in the top-level `state` array,
    as required by OpenRouter's Multimodal Decisions API for non-Jev models.
    """
    images = []

    def _strip_images(obj):
        if _is_image_url_part(obj):
            images.append(obj)
            return "<camera_frame>"
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if _is_image_url_part(v):
                    images.append(v)
                else:
                    cleaned[k] = _strip_images(v)
            return cleaned
        if isinstance(obj, list):
            out = []
            for item in obj:
                if _is_image_url_part(item):
                    images.append(item)
                else:
                    out.append(_strip_images(item))
            return out
        return obj

    cleaned_state = _strip_images(state)
    if not images:
        return state

    if isinstance(cleaned_state, list):
        return cleaned_state + images
    return [cleaned_state] + images


def serialize_questions(questions):
    """Convert typesafe_sdk Question objects or raw dicts into JSON-serializable wire dicts."""
    serialized = {}
    for k, v in questions.items():
        if hasattr(v, "model_dump"):
            serialized[k] = v.model_dump(exclude_none=True)
        elif isinstance(v, dict):
            serialized[k] = dict(v)
        else:
            q_type = getattr(v, "type", None)
            criteria = getattr(v, "criteria", None)
            if hasattr(criteria, "model_dump"):
                criteria = criteria.model_dump(exclude_none=True)
            if not q_type:
                if isinstance(criteria, list):
                    q_type = "score"
                elif isinstance(criteria, dict) and set(criteria.keys()) <= {"true", "false"}:
                    q_type = "noul"
                elif isinstance(criteria, dict):
                    q_type = "choice"
                else:
                    q_type = "noul"
            q_dict = {"type": q_type, "instructions": getattr(v, "instructions", str(v))}
            if criteria is not None:
                q_dict["criteria"] = criteria
            serialized[k] = q_dict
    return serialized


def call_openrouter_chat(state, model, questions, api_key):
    """Fallback for models that only support standard Chat Completions."""
    sys_prompt = (
        "You are a tactical decision system. Analyze the state and answer every question. "
        "Return ONLY a valid JSON object mapping each question key to its answer."
    )
    state_str = state if isinstance(state, str) else json.dumps(state, indent=2)
    q_wire = serialize_questions(questions)

    q_lines = ["QUESTIONS:"]
    for k, q in q_wire.items():
        q_type = q.get("type", "choice")
        instr = q.get("instructions", "")
        crit = q.get("criteria")
        if q_type == "choice" and isinstance(crit, dict):
            opts = ", ".join(f'"{opt}"' for opt in crit.keys())
            q_lines.append(f"- {k} (choice: must be one of [{opts}]): {json.dumps(instr)}")
            q_lines.append(f"  Options: {json.dumps(crit)}")
        elif q_type == "score" and isinstance(crit, list):
            max_idx = max(0, len(crit) - 1)
            indexed = {i: desc for i, desc in enumerate(crit)}
            q_lines.append(f"- {k} (score: float from 0.0 to {max_idx}.0): {json.dumps(instr)}")
            q_lines.append(f"  Scale: {json.dumps(indexed)}")
        else:
            q_lines.append(f"- {k} (noul: float probability from 0.0 to 1.0): {json.dumps(instr)}")
            if crit:
                q_lines.append(f"  Criteria: {json.dumps(crit)}")

    prompt = (
        f"STATE:\n{state_str}\n\n"
        + "\n".join(q_lines)
        + f"\n\nReturn a JSON object with these exact keys: {list(q_wire.keys())}."
    )

    req_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(req_data).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "jev-drone/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter Chat API Error {e.code}: {err_body}") from e

    raw_content = res["choices"][0]["message"]["content"].strip()
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        raw_content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    content = json.loads(raw_content)
    return MockResponse(content, res.get("usage", {}), model=res.get("model", model))


def call_openrouter_decisions(state, model, questions, api_key):
    """Hits OpenRouter's /api/alpha/decisions endpoint directly."""
    norm_state = _normalize_openrouter_multimodal_state(state)
    payload = {
        "model": model,
        "state": norm_state,
        "questions": serialize_questions(questions),
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/alpha/decisions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "jev-drone/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter Decisions Error {e.code}: {err_body}") from e

    return MockResponse(res.get("answers", {}), res.get("usage", {}), model=res.get("model", model))


def call_typesafe_with_assets(state, model, questions, api_key, base_url=None):
    """Hits Native TypeSafe's /v1/systemone endpoint with image assets and danger-zone header."""
    assets = {}
    clean_state = _extract_typesafe_assets(state, assets)
    clean_questions = _extract_typesafe_assets(serialize_questions(questions), assets)
    native_model = NATIVE_MODEL_ALIASES.get(model, model)

    base = (base_url or os.environ.get("API_BASE_URL") or os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
    url = f"{base}/v1/systemone" if not base.endswith("/v1/systemone") else base

    payload = {
        "model": native_model,
        "state": clean_state,
        "assets": assets,
        "questions": clean_questions,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
            "User-Agent": "jev-drone/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TypeSafe Multimodal Error {e.code}: {err_body}") from e

    return MockResponse(res.get("answers", {}), res.get("usage", {}), model=res.get("model", native_model))


def resolve_model_and_backend(model: str | None):
    """
    Resolve a model identifier to (backend, resolved_model) without requiring code changes.
    backend is either 'typesafe' (native TypeSafeClient) or 'openrouter'.
    """
    raw = (model or os.environ.get("JEV_MODEL") or "jev-latest").strip()
    force_or = os.environ.get("USE_OPENROUTER")
    has_ts_key = bool(os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY"))
    has_or_key = bool(os.environ.get("OPENROUTER_API_KEY"))

    # Explicit prefix overrides
    if raw.startswith(("native/", "native:")):
        clean = raw.split("/", 1)[1] if "/" in raw else raw.split(":", 1)[1]
        clean = clean.removeprefix("~").removeprefix("typesafe/")
        return "typesafe", NATIVE_MODEL_ALIASES.get(clean, clean)

    if raw.startswith(("openrouter/", "openrouter:")):
        clean = raw.split("/", 1)[1] if "/" in raw else raw.split(":", 1)[1]
        if clean in ("jev-latest", "typesafe/jev-latest"):
            clean = "~typesafe/jev-latest"
        return "openrouter", clean

    # Explicit environment variable overrides
    if force_or == "1":
        if raw in ("jev-latest", "typesafe/jev-latest"):
            return "openrouter", "~typesafe/jev-latest"
        return "openrouter", raw
    if force_or == "0":
        clean = raw.removeprefix("~").removeprefix("typesafe/")
        return "typesafe", NATIVE_MODEL_ALIASES.get(clean, clean)

    # Automatic detection:
    # Native TypeSafe hosts 'jev-latest' (jev-1.13.0) and 'jev-preview' (jev-1.14.0).
    # OpenRouter models use '<provider>/<slug>' or '~<provider>/<slug>'.
    stripped_ts = raw.removeprefix("typesafe/")
    if not raw.startswith("~") and (raw in ("jev-latest", "jev-preview") or stripped_ts in ("jev-latest", "jev-preview")):
        if has_ts_key or not has_or_key:
            return "typesafe", stripped_ts
        return "openrouter", "~typesafe/jev-latest" if stripped_ts == "jev-latest" else "typesafe/jev-1.14"

    if "/" in raw:
        return "openrouter", raw

    # Bare model name: use TypeSafe if key present, else OpenRouter
    if has_ts_key:
        return "typesafe", NATIVE_MODEL_ALIASES.get(raw, raw)
    return "openrouter", raw


class SystemOneAdapter:
    """
    Unified client adapter that transparently handles:
    - Native TypeSafe Jev models (via TypeSafeClient for text, or /v1/systemone with assets for images)
    - OpenRouter native Decisions models (via /api/alpha/decisions, text + multimodal)
    - OpenRouter Chat Completions fallback models (via /api/v1/chat/completions)
    Switching between any model or backend requires only --model / --provider or JEV_MODEL.
    """
    def __init__(self, model=None):
        backend, resolved_model = resolve_model_and_backend(model)
        self.backend = backend
        self.model = resolved_model
        self.is_openrouter = (backend == "openrouter")
        self.requires_fallback = (
            os.environ.get("USE_FALLBACK_INTERFACE") == "1"
            or self.model in CHAT_ONLY_DECISION_MODELS
        )

        if self.is_openrouter:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise RuntimeError(
                    f"OPENROUTER_API_KEY not set for OpenRouter model '{self.model}' (see .env.example)."
                )
            self.client = None
        else:
            self.api_key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
            if not self.api_key:
                raise RuntimeError(
                    f"TYPESAFE_API_KEY not set for native TypeSafe model '{self.model}' (see .env.example)."
                )
            if not self.requires_fallback:
                env_url = os.environ.get("API_BASE_URL") or os.environ.get("TYPESAFE_BASE_URL")
                if env_url:
                    self.client = TypeSafeClient(api_key=self.api_key, base_url=env_url)
                else:
                    self.client = TypeSafeClient(api_key=self.api_key)
            else:
                self.client = None

    def system_one(self, state, model=None, questions=None):
        if model and model != self.model:
            _, actual_model = resolve_model_and_backend(model)
        else:
            actual_model = self.model

        if self.requires_fallback or actual_model in CHAT_ONLY_DECISION_MODELS:
            return call_openrouter_chat(state, actual_model, questions, self.api_key)
        if self.is_openrouter:
            return call_openrouter_decisions(state, actual_model, questions, self.api_key)

        # Check if state or questions contain image_url parts for native TypeSafe
        probe_assets = {}
        _extract_typesafe_assets(state, probe_assets)
        if probe_assets or any(
            isinstance(q, dict) and q.get("type") == "box" for q in (questions or {}).values()
        ):
            return call_typesafe_with_assets(state, actual_model, questions, self.api_key)

        return self.client.system_one(state=state, model=actual_model, questions=questions)

    def close(self):
        if self.client:
            self.client.close()
