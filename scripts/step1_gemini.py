#!/usr/bin/env python3
"""
Step 1: Gemini Scene Analysis

Extracts first and last frames from video, sends to Gemini Vision API
to identify all objects, determine which is manipulated, and estimate
material type (wood, plastic, metal, etc.) for physics simulation.

Environment: Any Python 3.10+ with opencv-python
Required: GEMINI_API_KEY environment variable

Input:
    - RGB video file (color_video.mp4)
    
Output:
    - gemini_scene.json with object prompts and material types
    - video_first.png, video_last.png (extracted frames)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2

# Ensure unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# Regex for stripping color words from prompts as a backstop, in case Gemini
# forgets to supply fallback_prompt. Matches leading "[modifier] color [and color]"
# patterns like "light blue", "dark red and white".
_COLOR_STRIP_RE = re.compile(
    r"^(light |dark |bright |pale |deep )?"
    r"(red|orange|yellow|green|blue|navy|teal|cyan|purple|violet|pink|magenta|"
    r"brown|tan|beige|black|white|gray|grey|gold|silver|clear|transparent)"
    r"(\s+and\s+(light |dark )?(red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|grey))?"
    r"\s+",
    re.IGNORECASE,
)


def _strip_color(prompt: str) -> str:
    """Remove a leading color word from a prompt. Returns original if no match."""
    stripped = _COLOR_STRIP_RE.sub("", prompt).strip()
    return stripped if stripped else prompt

_GEMINI_ENDPOINT_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)


def _normalize_model_name(model: str) -> str:
    """Normalize Gemini model name."""
    model = (model or "").strip()
    if model.startswith("models/"):
        return model[len("models/"):]
    return model


def _list_models(api_key: str, *, timeout_s: float = 30.0) -> list[dict]:
    """List available Gemini models."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("models") or [])


def _resolve_auto_pro_model(api_key: str, *, timeout_s: float = 30.0) -> str:
    """Pick the best available Gemini Pro model supporting generateContent."""
    print("  [DEBUG] Fetching available Gemini models...")
    models = _list_models(api_key, timeout_s=timeout_s)
    print(f"  [DEBUG] Found {len(models)} total models")
    candidates: list[str] = []
    for m in models:
        name = str(m.get("name") or "")
        methods = m.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        low = name.lower()
        if "gemini" in low and "pro" in low and "preview-tts" not in low:
            candidates.append(name)
            print(f"  [DEBUG] Candidate model: {name}")

    if not candidates:
        raise RuntimeError("No Gemini Pro models support generateContent for this API key")

    def score(n: str) -> tuple:
        n2 = n.lower()
        nums = re.findall(r"gemini-(\d+)(?:\.(\d+))?", n2)
        major, minor = 0, 0
        if nums:
            major = int(nums[0][0] or 0)
            minor = int(nums[0][1] or 0)
        # Prefer 2.5 over 3-preview, avoid preview/image models
        is_2_5 = 1 if "2.5" in n2 or "2-5" in n2 else 0
        img_penalty = 1 if "image-preview" in n2 else 0
        preview_penalty = 1 if "preview" in n2 and "2.5" not in n2 else 0
        return (is_2_5, major, minor, -preview_penalty, -img_penalty, n)

    best_full = sorted(candidates, key=score, reverse=True)[0]
    normalized = _normalize_model_name(best_full)
    print(f"  [DEBUG] Selected model: {normalized}")
    return normalized


def _extract_json(text: str) -> dict:
    """Extract JSON object from model response text."""
    if "{" in text:
        open_braces = text.count("{")
        close_braces = text.count("}")
        if close_braces == 0 or close_braces < open_braces:
            raise ValueError("Truncated JSON (unbalanced braces)")

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:2000]}")
    return json.loads(m.group(0))


def _encode_image_b64(png_path: str) -> str:
    """Encode image file as base64 string."""
    data = Path(png_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _normalize_task_type(task_type: str | None) -> str:
    """Normalize task type label from Gemini output."""
    t = (task_type or "").strip().lower()
    if t in {
        "object push",
        "object_push",
        "push",
        "push_object",
        "object pull",
        "object_pull",
        "pull",
        "pull_object",
        "drag",
        "drag_object",
        "put_object_to_object",
        "put_to_object",
        "put",
        "place",
        "place_on_object",
        "place_into_object",
    }:
        return "object push"
    if t in {
        "interactive object manip",
        "interactive_object_manip",
        "interactive",
        "interact",
        "hand interaction",
        "tool interaction",
    }:
        return "interactive object manip"
    if t in {
        "simple object manip",
        "simple_object_manip",
        "simple",
        "manipulate_object",
        "object manip",
        "object_manip",
    }:
        return "simple object manip"
    return "simple object manip"


def extract_first_middle_last_frames(video_path: str, out_dir: str) -> tuple[str, str, str]:
    """Extract first, middle, and last frames from video as PNG files."""
    print(f"  [DEBUG] Creating output directory: {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"  [DEBUG] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"  [DEBUG] Video frame count: {frame_count}")
    if frame_count <= 0:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read first frame from: {video_path}")
        first = frame
        middle = frame
        last = frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
        cap.release()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read first frame from: {video_path}")

        mid_idx = max((frame_count - 1) // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ok, middle = cap.read()
        if not ok:
            middle = first

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
        ok, last = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last = first
            while True:
                ok2, frame2 = cap.read()
                if not ok2:
                    break
                last = frame2
        cap.release()

    first_path = str(Path(out_dir) / "video_first.png")
    middle_path = str(Path(out_dir) / "video_middle.png")
    last_path = str(Path(out_dir) / "video_last.png")

    def _downscale(img, max_side: int = 512):
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= max_side:
            return img
        scale = max_side / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print(f"  [DEBUG] Downscaling frames to max 512px (to reduce API payload size)...")
    first = _downscale(first)
    middle = _downscale(middle)
    last = _downscale(last)
    print(f"  [DEBUG] First frame shape: {first.shape}")
    print(f"  [DEBUG] Middle frame shape: {middle.shape}")
    print(f"  [DEBUG] Last frame shape: {last.shape}")

    print(f"  [DEBUG] Writing {first_path}")
    cv2.imwrite(first_path, first)
    print(f"  [DEBUG] Writing {middle_path}")
    cv2.imwrite(middle_path, middle)
    print(f"  [DEBUG] Writing {last_path}")
    cv2.imwrite(last_path, last)

    return first_path, middle_path, last_path


def describe_scene_with_gemini(
    *,
    middle_frame_png: str,
    last_frame_png: str,
    scene_image_png: str | None = None,
    api_key: str | None = None,
    model: str = "gemini-1.5-pro",
    timeout_s: float = 60
) -> dict:
    """
    Call Gemini with middle/last frame (and optionally full scene image) and request scene description.
    
    Returns JSON with:
    - objects: list of {prompt, material_type}
    - manipulated_object: {prompt}
    """
    print("  [DEBUG] Validating API key...")
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key missing: set GEMINI_API_KEY or pass --gemini_api_key")
    print(f"  [DEBUG] API key found (length: {len(api_key)})")

    model = _normalize_model_name(model)
    print(f"  [DEBUG] Requested model: {model}")
    if model.lower() in {"auto-pro", "auto", "pro"}:
        print("  [DEBUG] Auto-selecting best Pro model...")
        model = _resolve_auto_pro_model(api_key, timeout_s=min(timeout_s, 30.0))

    endpoint = _GEMINI_ENDPOINT_TMPL.format(model=model, key=api_key)
    print(f"  [DEBUG] API endpoint: {endpoint.split('?')[0]}...")

    # Updated schema: material_type instead of mass/friction
    schema_hint = {
        "scene": {
            "objects": [
                # {
                #     "prompt": "string (short description and direction for segmentation, e.g., 'red cube on left')",
                #     "material_type": "string (one of: wood, plastic, metal, glass, rubber, ceramic, cardboard, foam, fabric, other)",
                # }
                {
                    "prompt": "string (dominant body color + generic noun + optional location, e.g., 'blue chips can on right')",
                    "fallback_prompt": "string (SAME object without color word, e.g., 'chips can on right')",
                    "material_type": "string (one of: wood, plastic, metal, glass, rubber, ceramic, cardboard, foam, fabric, other)",
                }
            ],
            "manipulated_object": {
                "prompt": "string (must match one of objects[].prompt exactly; choose the physically moved/grasped object, not its contents)",
            },
            "task_type": "string (one of: simple object manip, interactive object manip, object push; 'object push' covers push/pull/drag)",
            "target_object": {
                "prompt": "string or null (for object push, usually null because only the pushed object is listed)",
            },
            # --- NEW ---
            "hand_along_object_motion": "boolean or null (only for 'object push' task_type; true if the hand is on the side the object moves TOWARD, false if the hand is on the opposite side (push-like); null for other task types)",

        }
    }

    instruction = (
        "All objects are on the tabletop surface. "
        "You are annotating a tabletop manipulation video for robotics simulation. "
        "You are given a REFERENCE SCENE image (full view), a MIDDLE frame, and the LAST frame. "
        "Use the reference scene image to identify all objects clearly. "
        "Task: "
        "(1) List ALL distinct objects on the table as concise prompts for segmentation. "
        "(2) For each object, estimate its material type from: wood, plastic, metal, glass, rubber, ceramic, cardboard, foam, fabric, or other. "
        "(3) Decide which SINGLE object is being manipulated/moved between the middle and last frames. "
        # "IMPORTANT: manipulated_object is the object physically grasped/moved by the hand/tool. "
        # "For pouring tasks, the manipulated object is the container being held (e.g., cup/bottle), "
        # "NOT the liquid or small items inside that container. "
        "IMPORTANT: manipulated_object is the object physically grasped/moved by the hand/tool. "
        "CONTAINER RULE: When the manipulated object is a container holding or carrying other items "
        "(e.g., a tray with cups, a box with parts, a basket with fruit, a bowl with food, a cup with liquid, "
        "a pan with food on it, a plate with food on it, a pot with ingredients in it), "
        "the manipulated object is the CONTAINER itself, not its contents. "
        "In such cases, list ONLY the container in objects[]; do NOT list the items inside or on top of it "
        "as separate objects, because they move passively with the container and are not independently manipulated. "
        "This applies to pouring, pushing, pulling, dragging, lifting, and any similar container manipulation. "
        "(4) Classify task_type as exactly one of: 'simple object manip', 'interactive object manip', or 'object push'. "
        "Use 'simple object manip' for single-object manipulation with no meaningful object-to-object interaction. "
        "Use 'interactive object manip' for hand/tool interaction between two objects (e.g., placing, stacking, grasping another object). "
        # "Use 'object push' when the task is pushing an object; in this case, list ONLY the pushed object in objects[] and set target_object to null. "
        "Use 'object push' when the task is pushing, pulling, or dragging an object across the surface; "
        "set target_object.prompt to null for push tasks (there is no interaction target). "
        "Still list ALL tabletop objects in objects[] (the pushed object plus any surrounding objects); only target_object is null. "
        "If the pushed object is a container with items inside (see CONTAINER RULE above), list only the container, not its contents. "
        "(5) If task_type is 'object push', set target_object.prompt to null. Still list all tabletop objects in objects[] (do not drop surrounding objects). "
        "If task_type is 'interactive object manip', identify which other object (not the manipulated one) is the target/interacted-with object. Set target_object.prompt to that object. "
        "If task_type is 'simple object manip', set target_object.prompt to null. "
        "(6) If task_type is 'object push', also set 'hand_along_object_motion': "
        "compare the human hand position to the object's movement direction (from middle frame to last frame). "
        "Set hand_along_object_motion=true if the hand is on the side the object is moving TOWARD "
        "(the hand 'leads' the motion, hand in front of the object along its motion vector). "
        "Set hand_along_object_motion=false if the hand is on the OPPOSITE side, behind the object relative to its motion direction "
        "(the hand 'trails' the motion). "
        "For 'simple object manip' and 'interactive object manip', set hand_along_object_motion to null. "
        "Return ONLY valid JSON (no markdown, no extra text). "
        "The manipulated_object.prompt MUST exactly equal one of objects[].prompt. "
        "If target_object.prompt is set, it MUST also exactly equal one of objects[].prompt. "
        "PROMPT QUALITY RULES — these prompts are passed to SAM3 open-vocabulary segmentation, "
        "so they must describe the object the way SAM3 will see it. Follow these strictly: "
        "(a) DOMINANT BODY COLOR ONLY. Use the color of the largest visible surface of the object. "
        "IGNORE accent colors from labels, lids, caps, text, graphics, illustrations, or stickers. "
        "WORKED EXAMPLE: a chips can with a blue cylindrical body, a yellow lid, and a yellow chip illustration on the label is 'blue chips can' — NOT 'yellow chips can'. The body is blue; the yellow elements are accents. "
        "WORKED EXAMPLE 2: a spice jar or seasoning shaker with a dark plastic body mostly covered by a colored paper label is described by the label's dominant color, e.g., 'green spice jar' or 'red and white seasoning shaker'. Do NOT call it 'clear' or 'transparent' unless you can actually see through it. "
        "GENERAL RULE: 'clear' / 'transparent' is ONLY for objects where the contents inside are visible through the walls (e.g., a clear water bottle showing the water level, an empty glass). Containers with opaque labels are NOT clear. "
        "If the body is genuinely two roughly equal colors, combine them: 'blue and white box'. "
        "For transparent objects, use 'clear' (e.g., 'clear glass bottle'). "
        "(b) GENERIC NOUNS. Use common object nouns that SAM3 recognizes: can, bottle, cup, plate, bowl, box, jar, container, block, cube, tray, mug. Do NOT use brand names (say 'chips can', not 'Lay's Stax can'). "
        "(c) NO MATERIAL WORDS in prompt. Do not include 'metal', 'plastic', 'wooden', etc. — material belongs in the material_type field. "
        "(d) STRUCTURE: '[dominant body color] [generic object noun] [optional spatial cue]', e.g., 'blue chips can on right', 'light blue plate in center', 'white spray bottle on left'. "
        "(e) FALLBACK PROMPT. For every object, also provide 'fallback_prompt': the SAME prompt with the color word(s) removed (e.g., prompt='blue chips can on right' → fallback_prompt='chips can on right'). This is used as a retry if the colored prompt fails. "
        "Return ONLY valid JSON (no markdown, no extra text). "
        "The manipulated_object.prompt MUST exactly equal one of objects[].prompt. "
        "If target_object.prompt is set, it MUST also exactly equal one of objects[].prompt. "
        "If there are multiple similar objects, add distinguishing details (e.g., 'red cup near left edge', 'red cup near center')."
    )

    # Build parts list
    parts = [
        {"text": instruction},
        {"text": "Required JSON schema:"},
        {"text": json.dumps(schema_hint)},
    ]
    
    # Add scene image if provided
    if scene_image_png and Path(scene_image_png).exists():
        print(f"  [DEBUG] Adding reference scene image: {scene_image_png}")
        parts.append({"text": "Reference scene image (full view):"})  
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": _encode_image_b64(scene_image_png),
            }
        })
    
    # Add middle and last frames
    parts.append({"text": "Middle frame:"})
    parts.append({
        "inline_data": {
            "mime_type": "image/png",
            "data": _encode_image_b64(middle_frame_png),
        }
    })
    parts.append({"text": "Last frame:"})
    parts.append({
        "inline_data": {
            "mime_type": "image/png",
            "data": _encode_image_b64(last_frame_png),
        }
    })
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }

    def _get_candidate_text(resp_data: dict) -> str | None:
        try:
            c0 = (resp_data.get("candidates") or [])[0] or {}
        except Exception:
            return None
        content = c0.get("content") or {}
        parts = content.get("parts")
        if isinstance(parts, list) and parts:
            p0 = parts[0] or {}
            if isinstance(p0, dict) and "text" in p0 and p0.get("text") is not None:
                return str(p0.get("text"))
            return json.dumps(p0)
        return None

    last_err: Exception | None = None
    max_output_tokens = int(payload["generationConfig"].get("maxOutputTokens") or 4096)
    data: dict | None = None
    obj: dict | None = None
    
    print(f"  [DEBUG] Starting API call with up to 6 retries...")
    for attempt in range(6):
        print(f"  [DEBUG] Attempt {attempt + 1}/6 (maxTokens={max_output_tokens})")
        try:
            payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
            print(f"  [DEBUG] Encoding payload ({len(json.dumps(payload))} bytes)...")
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            print(f"  [DEBUG] Sending POST request (timeout={timeout_s}s)...")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                print(f"  [DEBUG] Received response, parsing JSON...")
                data = json.loads(resp.read().decode("utf-8"))

            # Debug: Print full response structure
            print(f"  [DEBUG] Response keys: {list(data.keys())}")
            
            cand0 = (data.get("candidates") or [{}])[0] or {}
            finish = str(cand0.get("finishReason") or "").upper()
            print(f"  [DEBUG] Response finishReason: {finish}")
            
            # Debug: Print safety ratings
            safety_ratings = cand0.get("safetyRatings") or []
            print(f"  [DEBUG] Safety ratings count: {len(safety_ratings)}")
            for sr in safety_ratings:
                cat = sr.get("category", "?")
                prob = sr.get("probability", "?")
                print(f"  [DEBUG]   - {cat}: {prob}")
            
            # Debug: Print content structure
            content = cand0.get("content") or {}
            parts = content.get("parts") or []
            print(f"  [DEBUG] Content parts count: {len(parts)}")
            for i, part in enumerate(parts[:3]):  # Only show first 3 parts
                part_keys = list(part.keys()) if isinstance(part, dict) else []
                print(f"  [DEBUG]   - Part {i}: keys={part_keys}")
                if "text" in part:
                    text_preview = str(part["text"])[:100]
                    print(f"  [DEBUG]   - Part {i} text preview: {text_preview}")
            
            # Debug: Print usage metadata
            usage = data.get("usageMetadata") or {}
            if usage:
                prompt_tokens = usage.get("promptTokenCount", 0)
                cand_tokens = usage.get("candidatesTokenCount", 0)
                total_tokens = usage.get("totalTokenCount", 0)
                print(f"  [DEBUG] Usage: prompt={prompt_tokens}, candidates={cand_tokens}, total={total_tokens}")
            
            text = _get_candidate_text(data) or ""
            print(f"  [DEBUG] Extracted text length: {len(text)}")

            # Check for empty response (possible rate limiting)
            if not text or text.strip() == "":
                import time
                wait_time = min(10 * (2 ** attempt), 120)  # Exponential backoff, cap at 120s
                last_err = RuntimeError(f"Empty response from Gemini (possible rate limit)")
                print(f"  [WARN] Empty response on attempt {attempt + 1}/6, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            try:
                print(f"  [DEBUG] Parsing JSON response...")
                obj = _extract_json(text)
                print(f"  [DEBUG] Successfully parsed JSON object")
                last_err = None
                break
            except Exception as e:
                print(f"  [ERROR] JSON parsing failed: {type(e).__name__}: {str(e)[:200]}")
                if finish == "MAX_TOKENS" and max_output_tokens < 8192:
                    max_output_tokens = min(max_output_tokens * 2, 8192)
                    last_err = RuntimeError(
                        f"Gemini returned truncated output; retrying with maxOutputTokens={max_output_tokens}"
                    )
                    print(f"  [WARN] Increasing max tokens to {max_output_tokens}")
                    continue
                # If JSON extraction fails and we have retries left, treat as empty response
                import time
                wait_time = min(10 * (2 ** attempt), 120)
                last_err = e
                print(f"  [WARN] JSON parse failed on attempt {attempt + 1}/6, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

        except TimeoutError as e:
            # Network timeout - retry with exponential backoff
            import time
            wait_time = min(5 * (2 ** attempt), 60)  # Cap at 60 seconds
            last_err = e
            print(f"  [ERROR] Timeout on attempt {attempt + 1}/6, retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            print(f"  [ERROR] HTTP {e.code} error")
            print(f"  [ERROR] Response body: {body[:500]}")

            if e.code == 400 and body:
                try:
                    parsed = json.loads(body)
                    msg = str(((parsed.get("error") or {}).get("message")) or "")
                except Exception:
                    msg = body
                if "thinking" in msg.lower() and "budget" in msg.lower():
                    tc = payload.setdefault("generationConfig", {}).setdefault("thinkingConfig", {})
                    cur = tc.get("thinkingBudget")
                    try:
                        cur_i = int(cur)
                    except Exception:
                        cur_i = 0
                    if cur_i <= 0:
                        tc["thinkingBudget"] = 1024
                    last_err = RuntimeError(f"Gemini HTTPError {e.code}: {body}")
                    continue

            if e.code == 429:
                retry_s = 2 ** attempt
                print(f"  [WARN] Rate limit (429) detected")
                try:
                    parsed = json.loads(body) if body else {}
                    details = (parsed.get("error") or {}).get("details") or []
                    for d in details:
                        if (d.get("@type") or "") == "type.googleapis.com/google.rpc.RetryInfo":
                            rd = str(d.get("retryDelay") or "")
                            m = re.match(r"^(\d+)s$", rd.strip())
                            if m:
                                retry_s = max(retry_s, int(m.group(1)))
                except Exception:
                    pass
                print(f"  [WARN] Waiting {retry_s}s before retry...")
                time.sleep(retry_s)
                last_err = RuntimeError(f"Gemini HTTPError {e.code}: {body}")
                continue

            last_err = RuntimeError(f"Gemini HTTPError {e.code}: {body}")
            break

    if last_err is not None:
        raise last_err

    if data is None or obj is None:
        raise RuntimeError(f"Unexpected Gemini response format: {data}")

    # Normalize and validate
    print("  [DEBUG] Validating Gemini response structure...")
    if isinstance(obj, list):
        obj = obj[0] if obj and isinstance(obj[0], dict) else {}

    scene = (obj or {}).get("scene")
    if not isinstance(scene, dict):
        raise ValueError(f"Gemini output missing scene: {obj}")
    print(f"  [DEBUG] Found scene object")

    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"Gemini output missing objects list: {obj}")
    print(f"  [DEBUG] Found {len(objects)} objects in scene")

    # Valid material types
    VALID_MATERIALS = {"wood", "plastic", "metal", "glass", "rubber", "ceramic", "cardboard", "foam", "fabric", "other"}
    
    norm_objects: list[dict] = []
    prompts: list[str] = []
    print("  [DEBUG] Normalizing objects and materials...")
    for item in objects:
        if not isinstance(item, dict):
            continue
        prompt = (item.get("prompt") or "").strip()
        if not prompt:
            continue
        
        # Normalize material type
        material = (item.get("material_type") or "other").strip().lower()
        if material not in VALID_MATERIALS:
            print(f"  [WARN] Invalid material '{material}' for '{prompt}', using 'other'")
            material = "other"
        
        # norm_objects.append({
        #     "prompt": prompt,
        #     "material_type": material,
        # })
        # prompts.append(prompt)
        # print(f"  [DEBUG]   - {prompt}: {material}")

        # Capture fallback prompt; if Gemini didn't supply one, derive it by
        # stripping the leading color word.
        fallback = (item.get("fallback_prompt") or "").strip()
        if not fallback or fallback == prompt:
            fallback = _strip_color(prompt)

        norm_objects.append({
            "prompt": prompt,
            "fallback_prompt": fallback,
            "material_type": material,
        })
        prompts.append(prompt)
        print(f"  [DEBUG]   - {prompt} (fallback: {fallback}): {material}")


    manipulated = scene.get("manipulated_object") or {}
    manipulated_prompt = (manipulated.get("prompt") or "").strip()
    print(f"  [DEBUG] Raw manipulated object: {manipulated_prompt}")

    if manipulated_prompt and manipulated_prompt not in prompts:
        print(f"  [WARN] Manipulated prompt not in objects list, fuzzy matching...")
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()

        target = norm(manipulated_prompt)
        best = (0.0, "")
        for p in prompts:
            score = SequenceMatcher(a=target, b=norm(p)).ratio()
            if score > best[0]:
                best = (score, p)

        manipulated_prompt = best[1] if best[0] >= 0.6 else ""
        print(f"  [DEBUG] Best fuzzy match: '{best[1]}' (score={best[0]:.2f})")

    task_type = _normalize_task_type(scene.get("task_type"))

    target_object = scene.get("target_object") or {}
    if isinstance(target_object, dict):
        target_prompt = (target_object.get("prompt") or "").strip()
    elif isinstance(target_object, str):
        target_prompt = target_object.strip()
    else:
        target_prompt = ""

    if target_prompt and target_prompt not in prompts:
        print(f"  [WARN] Target prompt not in objects list, fuzzy matching...")

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()

        target_norm = norm(target_prompt)
        best_target = (0.0, "")
        for p in prompts:
            score = SequenceMatcher(a=target_norm, b=norm(p)).ratio()
            if score > best_target[0]:
                best_target = (score, p)

        target_prompt = best_target[1] if best_target[0] >= 0.6 else ""
        print(f"  [DEBUG] Best target fuzzy match: '{best_target[1]}' (score={best_target[0]:.2f})")

    hand_raw = scene.get("hand_along_object_motion", None)
    hand_along_object_motion: bool | None = None
    if task_type == "object push":
        if isinstance(hand_raw, bool):
            hand_along_object_motion = hand_raw
        elif isinstance(hand_raw, str):
            s = hand_raw.strip().lower()
            if s in {"true", "yes", "pull", "along", "toward", "with"}:
                hand_along_object_motion = True
            elif s in {"false", "no", "push", "against", "opposite", "behind"}:
                hand_along_object_motion = False
            else:
                print(f"  [WARN] Unrecognized hand_along_object_motion value '{hand_raw}', leaving null")
                hand_along_object_motion = None
        elif hand_raw is None:
            print(f"  [WARN] hand_along_object_motion missing for object push task; leaving null")
        else:
            print(f"  [WARN] Unexpected hand_along_object_motion type {type(hand_raw).__name__}, leaving null")
        target_prompt = ""

    # if task_type == "object push":
    #     # For push tasks, target is always null
        # target_prompt = ""



    elif task_type == "interactive object manip":
        # For interactive tasks, keep the target if provided
        pass
    else:
        # For simple object manip, no target
        target_prompt = ""

    print(f"  [DEBUG] Final hand_along_object_motion: {hand_along_object_motion}")

    print(f"  [DEBUG] Final manipulated object: {manipulated_prompt}")
    print(f"  [DEBUG] Final task_type: {task_type}")
    print(f"  [DEBUG] Final target object: {target_prompt or None}")
    return {
        "objects": norm_objects,
        "manipulated_prompt": manipulated_prompt or None,
        "task_type": task_type,
        "target_prompt": target_prompt or None,
        "hand_along_object_motion": hand_along_object_motion,
        "raw": obj,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Gemini scene analysis - identify objects and materials"
    )
    parser.add_argument("--video", required=True, help="Path to input video (e.g., color_video.mp4)")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--model", default="auto-pro", help="Gemini model (default: auto-pro)")
    parser.add_argument("--gemini_api_key", default=None, help="Gemini API key (or set GEMINI_API_KEY env)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: Gemini Scene Analysis")
    print("=" * 80)
    print(f"Video: {args.video}")
    print(f"Output: {output_dir}")

    # Extract frames
    print("\nExtracting first, middle, and last frames...")
    first_png, middle_png, last_png = extract_first_middle_last_frames(args.video, str(output_dir))
    print(f"  Saved: {first_png}")
    print(f"  Saved: {middle_png}")
    print(f"  Saved: {last_png}")

    # Find scene image (0.png)
    scene_image = None
    scene_capture_dir = output_dir.parent / "scene_capture" / "image"
    if scene_capture_dir.exists():
        scene_img_path = scene_capture_dir / "0.png"
        if scene_img_path.exists():
            scene_image = str(scene_img_path)
            print(f"  Found scene image: {scene_image}")

    # Call Gemini
    print(f"\nCalling Gemini API (model: {args.model})...")
    scene = describe_scene_with_gemini(
        middle_frame_png=middle_png,
        last_frame_png=last_png,
        scene_image_png=scene_image,
        api_key=args.gemini_api_key,
        model=args.model,
    )

    # Save output
    output_file = output_dir / "gemini_scene.json"
    output_file.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_file}")

    # Print summary
    print(f"\nDetected {len(scene['objects'])} objects:")
    for i, obj in enumerate(scene['objects']):
        print(f"  {i}: {obj['prompt']} ({obj['material_type']})")
    print(f"\nManipulated object: {scene['manipulated_prompt']}")
    
    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
