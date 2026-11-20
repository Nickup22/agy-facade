#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local OpenAI-style HTTP adapter for the installed Antigravity CLI.

Unofficial and experimental. Uses stream-json for chat and starts a new CLI
process per turn from a warmed pool. See README.md for security boundaries.
"""
import argparse, base64, hashlib, json, mimetypes, os, queue, re, shutil, subprocess, sys, threading, time
import select, selectors, socket, urllib.parse, uuid
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(STATE_DIR, "agy_images")
AGY_HOME = os.path.expanduser("~/.gemini/antigravity-cli")
WORKDIR = os.path.join(STATE_DIR, "agy_workdir")

DEFAULT_CONFIG = {
    "listen_host": "127.0.0.1",
    "listen_port": 8321,
    "agy_bin": "",
    "default_model": "gemini-3.8-flash-medium",
    "extra_models": [],
    "pool_per_key": 1,                # Warm processes per (model, effort) pair.
    "pool_max_processes": 6,
    "warm_models": [],                # Defaults to warming only default_model.
    "spawn_timeout_sec": 60,          # Time allowed for the CLI init event.
    "request_timeout_sec": 300,
    "image_generation_timeout_sec": 120,
    "max_concurrent": 1,              # Legacy setting; turns always run one at a time.
    "max_queued_requests": 16,        # Waiting turns, not counting the active turn.
    "max_inflight_body_bytes": 96 * 1024**2,
    "queue_timeout_sec": 300,
    "http_io_timeout_sec": 15,        # Socket inactivity and stalled writes.
    "body_timeout_sec": 60,           # Wall-clock upload limit.
    "min_turn_gap_sec": 4,
    "turn_gap_jitter_sec": 1.5,
    "models_ttl_sec": 600,
    "cleanup_chats": False,           # Opt in to cleanup of matching CLI session records.
    "enable_image_generation": False,
    "max_request_bytes": 24 * 1024**2,
    "facade_key": "",
    "preamble": ("Reminder: You're in API mode. Reply with plain text only. "
                 "Please don't call tools, read or write files, or run commands "
                 "for this conversation."),
}
CFG = dict(DEFAULT_CONFIG)
_lock = threading.Lock()
STATS = {
    "requests": 0,
    "images": 0,
    "media_turns": 0,
    "by_model": {},
    "by_key": {},
    "tokens": {"prompt": 0, "completion": 0}
}

MIME_EXT_MAP = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
}


def log(msg):
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def find_agy():
    if CFG.get("agy_bin"):
        return CFG["agy_bin"]
    env = os.environ.get("AGY_BIN")
    if env:
        return env
    w = shutil.which("agy")
    if w:
        return w
    return "agy"


KEYS_PATH = os.path.join(STATE_DIR, "agy_keys.json")


def load_keys():
    try:
        with open(KEYS_PATH) as f:
            return json.load(f).get("keys", [])
    except Exception:
        return []


def save_keys(keys):
    with open(KEYS_PATH, "w") as f:
        json.dump({"keys": keys}, f, indent=1, ensure_ascii=False)


def active_keys():
    out = {}
    for k in load_keys():
        if k.get("enabled", True) and k.get("key"):
            out[k["key"]] = k.get("name", "?")
    if CFG.get("facade_key"):
        out[CFG["facade_key"]] = "config"
    return out


def key_management(argv):
    import secrets, string
    cmd = argv[0] if argv else "list"
    keys = load_keys()
    if cmd == "list":
        if not keys:
            print("No stored keys. An active key is required to start the facade.")
        for k in keys:
            masked = (k["key"][:10] + "…") if len(k.get("key", "")) > 10 else k.get("key", "")
            print(f"{k['name']:<16} {'on ' if k.get('enabled', True) else 'off'} {masked}  created {k.get('created', '?')}")
        return 0
    if cmd == "add" and len(argv) > 1:
        name = argv[1]
        if any(k["name"] == name for k in keys):
            print(f"A key named {name} already exists")
            return 1
        value = "agyf-" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        keys.append({"name": name, "key": value, "enabled": True,
                     "created": time.strftime("%Y-%m-%d %H:%M")})
        save_keys(keys)
        print(f"Key for {name} (shown once):\n{value}")
        return 0
    if cmd in ("remove", "enable", "disable") and len(argv) > 1:
        name = argv[1]
        if cmd == "remove":
            keys = [k for k in keys if k["name"] != name]
        else:
            for k in keys:
                if k["name"] == name:
                    k["enabled"] = (cmd == "enable")
        save_keys(keys)
        print(f"Done: {cmd} {name}")
        return 0
    print("Usage: keys list | add <name> | remove <name> | enable <name> | disable <name>")
    return 2


class ModelCatalog:
    def __init__(self):
        self.models = []
        self.loaded_at = 0.0
        self.fetching = False

    def get(self):
        with _lock:
            stale = (time.time() - self.loaded_at) >= CFG["models_ttl_sec"]
            if stale and not self.fetching:
                self.fetching = True
                threading.Thread(target=self._refresh, daemon=True).start()
            return list(self.models)

    def _refresh(self):
        try:
            out = subprocess.run([find_agy(), "models"], capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL, start_new_session=True,
                                 timeout=60, cwd=STATE_DIR).stdout
            ids = []
            for line in out.splitlines():
                parts = line.split("\t")
                mid = parts[0].strip()
                if mid and (mid.startswith(("gemini", "claude", "gpt"))):
                    ids.append(mid)
            if ids:
                with _lock:
                    self.models = ids
                    self.loaded_at = time.time()
        except Exception as e:
            log(f"Failed to refresh the agy model catalog: {e}")
        finally:
            with _lock:
                self.fetching = False
                if not self.models:
                    self.models = ["gemini-3.8-flash-high", "gemini-3.8-flash-medium",
                                   "gemini-3.8-flash-low", "claude-sonnet-4-6",
                                   "claude-opus-4-6-thinking", "gpt-oss-120b-medium"]
                    self.loaded_at = time.time()


CATALOG = ModelCatalog()


def sniff_mime(raw_bytes: bytes) -> str:
    """Identify the MIME type by inspecting file signatures."""
    if not raw_bytes or len(raw_bytes) < 2:
        return "application/octet-stream"
    if raw_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    if raw_bytes.startswith(b"BM"):
        return "image/bmp"
    if raw_bytes.startswith(b"II*\x00") or raw_bytes.startswith(b"MM\x00*"):
        return "image/tiff"
    if len(raw_bytes) >= 12 and raw_bytes[4:8] == b"ftyp":
        brand = raw_bytes[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"hevc", b"mif1", b"msf1"):
            return "image/heic"
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        return "audio/wav"
    if raw_bytes.startswith(b"ID3") or (len(raw_bytes) >= 2 and raw_bytes[0] == 0xff and (raw_bytes[1] & 0xe0) == 0xe0):
        return "audio/mpeg"
    if raw_bytes.startswith(b"OggS"):
        return "audio/ogg"
    if raw_bytes.startswith(b"fLaC"):
        return "audio/flac"
    return "application/octet-stream"


def _decode_b64(b64_str: str) -> bytes:
    """Decode base64 or base64url, restoring missing padding."""
    clean = re.sub(r"\s+", "", str(b64_str or "")).replace("-", "+").replace("_", "/")
    pad = -len(clean) % 4
    if pad:
        clean += "=" * pad
    return base64.b64decode(clean)


def parse_media_value(val):
    """Decode inline data URIs, raw base64, or dictionaries with inline media.
    Return (MIME type, bytes) or (None, None).
    """
    if not val:
        return None, None

    if isinstance(val, bytes):
        mime = sniff_mime(val)
        return mime, val

    if isinstance(val, dict):
        # Anthropic format: {"type": "base64", "media_type": "...", "data": "..."}
        src = val.get("source")
        if isinstance(src, dict):
            b64 = src.get("data")
            mime = src.get("media_type") or "image/png"
            if b64:
                try:
                    data = _decode_b64(b64)
                    return (mime or sniff_mime(data)), data
                except Exception:
                    pass

        if "image_url" in val:
            return parse_media_value(val["image_url"])
        if "input_image" in val:
            return parse_media_value(val["input_image"])
        if "input_audio" in val:
            return parse_media_value(val["input_audio"])

        for k in ("url", "b64", "base64", "data", "image", "file", "audio", "path"):
            if k in val and val[k]:
                mime_hint = val.get("mime_type") or val.get("mime") or val.get("media_type")
                m, d = parse_media_value(val[k])
                if d:
                    return (mime_hint or m), d
        return None, None

    if not isinstance(val, str):
        return None, None

    s = val.strip()

    if s.startswith("data:"):
        try:
            header, b64_part = s.split(",", 1)
            if "%" in b64_part:
                b64_part = urllib.parse.unquote(b64_part)
            data = _decode_b64(b64_part)
            mime = "application/octet-stream"
            parts = header[5:].split(";")
            if parts and parts[0] and "/" in parts[0]:
                mime = parts[0].strip().lower()
            sniffed = sniff_mime(data)
            if not mime or mime in ("application/octet-stream", "text/plain"):
                mime = sniffed
            return mime, data
        except Exception as e:
            log(f"debug: data: URI parse: {e}")
            return None, None

    # Inline base64 only; no fetching URLs or reading supplied paths.
    clean_s = re.sub(r"\s+", "", s)
    if len(clean_s) >= 32 and re.match(r"^[A-Za-z0-9+/=_\-]+$", clean_s):
        try:
            data = _decode_b64(clean_s)
            mime = sniff_mime(data)
            if mime.startswith(("image/", "audio/")):
                return mime, data
        except Exception:
            pass

    return None, None


# Keep the legacy helper name for callers.
def parse_data_uri(uri):
    return parse_media_value(uri)


def extract_data_uris_from_text(text):
    """Extract inline media from data URIs and Markdown, HTML, or custom image tags."""
    if not isinstance(text, str):
        return str(text or ""), []

    found = []

    def _repl_media(raw_val):
        mime, data = parse_media_value(raw_val)
        if data:
            found.append((mime, data))
            return "[Attached Image]" if mime.startswith("image/") else "[Attached Audio]"
        return None

    # Strip recognized media markup while collecting the attachments.
    def _sub_md_img(m):
        res = _repl_media(m.group(1).strip())
        return res if res else m.group(0)

    cleaned = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", _sub_md_img, text)

    def _sub_img_tag(m):
        res = _repl_media(m.group(1).strip())
        return res if res else m.group(0)

    cleaned = re.sub(r"""<img\s+[^>]*src=["']([^"']+)["'][^>]*>""", _sub_img_tag, cleaned, flags=re.IGNORECASE)

    def _sub_tagged(m):
        res = _repl_media(m.group(1).strip())
        return res if res else m.group(0)

    cleaned = re.sub(r"\[img\](.*?)\[/img\]", _sub_tagged, cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<image>(.*?)</image>", _sub_tagged, cleaned, flags=re.IGNORECASE | re.DOTALL)

    def _sub_data(m):
        res = _repl_media(m.group(0).strip())
        return res if res else m.group(0)

    pattern_data = r"data:(?:image|audio)/[a-zA-Z0-9\.\-\+]+(?:;[^\s,]+)*,[A-Za-z0-9+/=_\-\s\r\n]+"
    cleaned = re.sub(pattern_data, _sub_data, cleaned)

    return cleaned, found


def build_prompt_and_media(req_or_messages, maybe_messages=None):
    """Build a flattened prompt and (MIME type, bytes) attachments.

    Accept request-level attachments and message-level media, including inline
    markup and OpenAI-style or Anthropic-style content blocks. Deduplicate by
    SHA-256, attach request-level media to the last user turn, and prefer
    current attachments over older history.
    """
    if maybe_messages is not None:
        req = req_or_messages if isinstance(req_or_messages, dict) else {}
        messages = maybe_messages
    elif isinstance(req_or_messages, dict):
        req = req_or_messages
        messages = req.get("messages") or []
    else:
        req = {}
        messages = req_or_messages or []

    system, turns = [], []
    seen_hashes = set()

    def _hash_data(d):
        return hashlib.sha256(d).hexdigest()

    top_media = []
    for top_key in ("images", "image", "picture", "pictures", "media", "files", "file", "photos", "photo"):
        top_val = req.get(top_key)
        if top_val:
            val_list = top_val if isinstance(top_val, list) else [top_val]
            for item in val_list:
                m, d = parse_media_value(item)
                if d:
                    h = _hash_data(d)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        top_media.append((m, d))

    last_user_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            last_user_idx = idx
            break

    current_turn_media = list(top_media)
    history_media = []

    img_idx, aud_idx = 0, 0
    top_media_attached = False

    for i, m in enumerate(messages or []):
        role = m.get("role", "user")
        is_current_user_turn = (i == last_user_idx)
        turn_attached = []

        for msg_key in ("images", "image", "picture", "pictures", "media", "files", "file", "attachment", "attachments", "image_url"):
            val = m.get(msg_key)
            if val:
                val_list = val if isinstance(val, list) else [val]
                for item in val_list:
                    mime, data = parse_media_value(item)
                    if data:
                        h = _hash_data(data)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            if is_current_user_turn:
                                current_turn_media.append((mime, data))
                            else:
                                history_media.append((mime, data))
                        if mime.startswith("image/"):
                            img_idx += 1
                            turn_attached.append(f"[Attached Image #{img_idx}]")
                        else:
                            aud_idx += 1
                            turn_attached.append(f"[Attached Audio #{aud_idx}]")

        content = m.get("content", "")

        if isinstance(content, list):
            turn_texts = []
            for p in content:
                if isinstance(p, str):
                    cleaned_txt, inlines = extract_data_uris_from_text(p)
                    for item in inlines:
                        h = _hash_data(item[1])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            if is_current_user_turn:
                                current_turn_media.append(item)
                            else:
                                history_media.append(item)
                    turn_texts.append(cleaned_txt)
                    continue

                if not isinstance(p, dict):
                    continue

                ptype = p.get("type", "")

                if ptype == "text":
                    raw_txt = p.get("text", "")
                    cleaned_txt, inlines = extract_data_uris_from_text(raw_txt)
                    for item in inlines:
                        h = _hash_data(item[1])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            if is_current_user_turn:
                                current_turn_media.append(item)
                            else:
                                history_media.append(item)
                    turn_texts.append(cleaned_txt)

                elif ptype in ("image_url", "image", "input_image"):
                    m, d = parse_media_value(p.get("image_url") or p.get("image") or p.get("source") or p.get("url") or p.get("data") or p)
                    if d:
                        h = _hash_data(d)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            if is_current_user_turn:
                                current_turn_media.append((m, d))
                            else:
                                history_media.append((m, d))
                        img_idx += 1
                        turn_texts.append(f"[Attached Image #{img_idx}]")

                elif ptype in ("input_audio", "audio_url", "audio"):
                    m, d = parse_media_value(p.get("input_audio") or p.get("audio_url") or p.get("audio") or p.get("source") or p.get("url") or p.get("data") or p)
                    if d:
                        h = _hash_data(d)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            if is_current_user_turn:
                                current_turn_media.append((m, d))
                            else:
                                history_media.append((m, d))
                        aud_idx += 1
                        turn_texts.append(f"[Attached Audio #{aud_idx}]")

                else:
                    m, d = parse_media_value(p)
                    if d:
                        h = _hash_data(d)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            if is_current_user_turn:
                                current_turn_media.append((m, d))
                            else:
                                history_media.append((m, d))
                        if m.startswith("image/"):
                            img_idx += 1
                            turn_texts.append(f"[Attached Image #{img_idx}]")
                        else:
                            aud_idx += 1
                            turn_texts.append(f"[Attached Audio #{aud_idx}]")
                    elif p.get("text"):
                        cleaned_txt, inlines = extract_data_uris_from_text(p["text"])
                        for item in inlines:
                            h = _hash_data(item[1])
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                if is_current_user_turn:
                                    current_turn_media.append(item)
                                else:
                                    history_media.append(item)
                        turn_texts.append(cleaned_txt)

            content = "\n".join(turn_texts)
        else:
            raw_txt = str(content or "")
            cleaned_txt, inlines = extract_data_uris_from_text(raw_txt)
            for item in inlines:
                h = _hash_data(item[1])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    if is_current_user_turn:
                        current_turn_media.append(item)
                    else:
                        history_media.append(item)
            content = cleaned_txt

        if turn_attached:
            content = (content + "\n" + "\n".join(turn_attached)).strip()

        # Request-level attachments belong to the latest user turn.
        if is_current_user_turn and not top_media_attached and top_media:
            top_markers = []
            for top_m, _ in top_media:
                if top_m.startswith("image/"):
                    img_idx += 1
                    top_markers.append(f"[Attached Image #{img_idx}]")
                elif top_m.startswith("audio/"):
                    aud_idx += 1
                    top_markers.append(f"[Attached Audio #{aud_idx}]")
            if top_markers:
                content = (content + "\n" + "\n".join(top_markers)).strip()
                top_media_attached = True

        if role in ("system", "developer"):
            system.append(content)
        else:
            turns.append(f"[{role}]\n{content}")

    # Keep all current attachments; use the remaining slots for recent history.
    MAX_ATTACHED_MEDIA = 8
    active_media = list(current_turn_media)
    if history_media:
        needed = MAX_ATTACHED_MEDIA - len(active_media)
        if needed > 0:
            active_media = history_media[-needed:] + active_media

    parts = []
    if CFG.get("preamble"):
        parts.append(CFG["preamble"])
    if system:
        parts.append("<system_instructions>\n" + "\n\n".join(system) + "\n</system_instructions>")
    parts.append("<conversation>\n" + "\n\n".join(turns) + "\n</conversation>")
    if active_media:
        parts.append(
            "Write the next assistant message in plain text. "
            "You may use the read_file tool to inspect the attached media file(s). "
            "Do not invoke terminal commands, write_file, edit_file, or subagents."
        )
    else:
        parts.append(
            "Write the next assistant message only in plain text. Do not invoke tools, commands, or subagents."
        )
    return "\n\n".join(parts), active_media


def generate_image_via_agy(prompt, size=None, timeout=120):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    brain_dir = os.path.join(AGY_HOME, "brain")
    existing_brain_images = set()
    if os.path.exists(brain_dir):
        for root, _, files in os.walk(brain_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    existing_brain_images.add(os.path.join(root, f))

    instruction = (
        f"Generate an image: {prompt}. "
        f"{f'Preferred size/aspect: {size}. ' if size else ''}"
        "Use the image-generator subagent and save the result as a session artifact."
    )
    cmd = [find_agy(), "-p", instruction]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=STATE_DIR)
    stdout = proc.stdout or ""

    found_path = None
    if os.path.isdir(brain_dir):
        root_real = os.path.realpath(brain_dir)
        new_files = []
        for root, _, files in os.walk(brain_dir):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                fp = os.path.join(root, f)
                fp_real = os.path.realpath(fp)
                if (fp not in existing_brain_images and not os.path.islink(fp)
                        and os.path.commonpath([root_real, fp_real]) == root_real
                        and os.path.isfile(fp_real)):
                    try:
                        new_files.append((os.path.getmtime(fp_real), fp_real))
                    except OSError:
                        pass
        if new_files:
            found_path = max(new_files)[1]

    if not found_path or not os.path.isfile(found_path):
        err_msg = stdout.strip() or proc.stderr.strip() or "image generation failed to produce an image file"
        raise AgyError(f"Image generation failed: {err_msg[:300]}", 502)

    ext = os.path.splitext(found_path)[1].lower() or ".jpg"
    img_name = f"img_{uuid.uuid4().hex[:16]}{ext}"
    dest_path = os.path.join(IMAGE_DIR, img_name)
    shutil.copy2(found_path, dest_path)
    return dest_path, img_name


class AgyError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.message = message
        self.status = status


def cleanup_conversation(conv_id):
    """Try to remove local CLI records for one facade conversation."""
    if not conv_id or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", conv_id) or not CFG.get("cleanup_chats"):
        return
    for base in ("conversations", "brain", "annotations"):
        d = os.path.join(AGY_HOME, base)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name == conv_id or name == conv_id + ".json":
                path = os.path.join(d, name)
                try:
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path)
                    else:
                        os.unlink(path)
                except OSError:
                    pass


class Worker:
    """Wait for CLI initialization in advance; use each process for one request."""

    def __init__(self, model, effort):
        self.model = model
        self.effort = effort
        self.events = queue.Queue()
        self.conversation_id = None
        self.spawn_error = None
        self.ready = threading.Event()
        self.dead = False
        os.makedirs(WORKDIR, exist_ok=True)
        cmd = [find_agy(), "--input-format", "stream-json", "--output-format", "stream-json",
               "-p=", "--disable-slash-commands", "--model", model]
        if effort:
            cmd += ["--effort", effort]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=WORKDIR,
                                     start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                line_lower = line.lower()
                if any(w in line_lower for w in ("safety", "blocked", "filter", "refusal", "harm", "prohibit")):
                    clean_msg = line.strip()
                    if len(clean_msg) > 300:
                        clean_msg = clean_msg[:300] + "..."
                    log(f"⚠️ [SAFETY LINE] {clean_msg}")

                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("event") == "init":
                    self.conversation_id = (ev.get("init") or {}).get("conversation_id") \
                        or ev.get("conversation_id")
                    self.ready.set()
                elif ev.get("event") == "result" and not self.ready.is_set():
                    self.spawn_error = (ev.get("result") or {}).get("error") or "agy failed to start"
                self.events.put(ev)
        except Exception:
            pass
        self.dead = True
        self.events.put(None)
        self.proc.stdout.close()

    def wait_ready(self, timeout):
        return self.ready.wait(timeout) and not self.dead

    def run_turn(self, prompt, media_items=None, on_delta=None, timeout=300, abort_event=None, stop=None):
        """Stage media in the session directory and send one user event.

        Supports cancellation and stop sequences.
        """
        media_payload = []
        if media_items:
            conv_id = self.conversation_id
            if not isinstance(conv_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", conv_id):
                raise AgyError("agy did not return a usable media session ID")
            upload_dir = os.path.join(AGY_HOME, "brain", conv_id, ".user_uploaded")
            os.makedirs(upload_dir, exist_ok=True)
            images_paths, audios_paths = [], []
            ws_dir = os.path.join(WORKDIR, "uploads")
            os.makedirs(ws_dir, exist_ok=True)
            for i, (mime, data) in enumerate(media_items):
                ext = MIME_EXT_MAP.get(mime, ".jpg" if mime.startswith("image/") else ".wav")
                fname = f"media_{i}_{uuid.uuid4().hex[:8]}{ext}"
                dest_path = os.path.join(upload_dir, fname)
                dest_ws = os.path.join(ws_dir, fname)
                try:
                    with open(dest_path, "wb") as f:
                        f.write(data)
                    with open(dest_ws, "wb") as f:
                        f.write(data)
                    dest_fwd = dest_path.replace("\\", "/")
                    media_payload.append({"mime_type": mime, "uri": dest_fwd})
                    if mime.startswith("image/"):
                        images_paths.append(dest_fwd)
                    elif mime.startswith("audio/"):
                        audios_paths.append(dest_fwd)
                except OSError as e:
                    log(f"warning: save upload media: {e}")

            log(f"media ready: {len(media_payload)} file(s) saved to {upload_dir} and {ws_dir}")

            meta_parts = []
            if images_paths:
                meta_parts.append(f"The user has attached {len(images_paths)} image file(s):")
                for idx, img in enumerate(images_paths, start=1):
                    meta_parts.append(f"- Image #{idx}: {img}")
                meta_parts.append(
                    "You MUST use the read_file tool to inspect the image file(s) listed above before answering. "
                    "Make sure to inspect every image file so you can distinguish between different photos or compare them. "
                    "Directly call the read_file tool without generating thought commentary. "
                    "After reading the image(s), write your final assistant response in plain text. "
                    "Do not run terminal commands, edit files, or invoke any other tools."
                )
            if audios_paths:
                meta_parts.append(f"The user has attached {len(audios_paths)} audio file(s):")
                for idx, aud in enumerate(audios_paths, start=1):
                    meta_parts.append(f"- Audio #{idx}: {aud}")
                meta_parts.append(
                    "You MUST use the read_file tool to inspect the audio file(s) listed above before answering. "
                    "Directly call the read_file tool without generating thought commentary. "
                    "After reading, write your final assistant response in plain text. "
                    "Do not run terminal commands, edit files, or invoke any other tools."
                )

            if meta_parts:
                prompt += "\n\n<ADDITIONAL_METADATA>\n" + "\n".join(meta_parts) + "\n</ADDITIONAL_METADATA>"

        user_msg = {"role": "user", "content": prompt}
        if media_payload:
            user_msg["media"] = media_payload

        line = json.dumps({"event": "user", "message": user_msg}, ensure_ascii=False)
        deadline = time.monotonic() + timeout
        try:
            fd = self.proc.stdin.fileno()
            os.set_blocking(fd, False)
            remaining = memoryview((line + "\n").encode(self.proc.stdin.encoding or "utf-8"))
            with selectors.DefaultSelector() as selector:
                selector.register(fd, selectors.EVENT_WRITE)
                while remaining:
                    if abort_event and abort_event.is_set():
                        raise AgyError("client aborted turn", 499)
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise AgyError("agy timed out writing the request", 504)
                    if selector.select(min(left, 0.25)):
                        try:
                            remaining = remaining[os.write(fd, remaining):]
                        except BlockingIOError:
                            continue
        except (OSError, ValueError) as e:
            raise AgyError(f"agy stdin: {e}") from e

        stop_list = []
        if isinstance(stop, str) and stop:
            stop_list = [stop]
        elif isinstance(stop, list):
            stop_list = [s for s in stop if isinstance(s, str) and s]
        for def_stop in ("\n[user]", "\n[User]", "\n<conversation>", "\n</conversation>"):
            if def_stop not in stop_list:
                stop_list.append(def_stop)

        text_parts, usage = [], None
        last_step_idx = None
        has_streamed_in_step = False
        stopped = False
        safety_detected_in_turn = False
        while True:
            if abort_event and abort_event.is_set():
                raise AgyError("client aborted turn", 499)

            left = deadline - time.monotonic()
            if left <= 0:
                raise AgyError("agy timed out", 504)
            try:
                ev = self.events.get(timeout=min(left, 0.25))
            except queue.Empty:
                if self.dead:
                    raise AgyError("agy process died mid-turn")
                continue
            if ev is None:
                raise AgyError("agy closed the stream mid-turn")
            kind = ev.get("event")
            if kind == "step_update":
                su = ev.get("step_update") or {}
                stype = su.get("step_type")
                step_idx = su.get("step_index")

                su_str = json.dumps(su, ensure_ascii=False)
                if any(w in su_str.lower() for w in ("safety", "blocked", "refusal", "filter", "harm", "prohibit")):
                    log(f"⚠️ [SAFETY IN STEP] step={step_idx} {su_str[:250]}")
                    safety_detected_in_turn = True

                if stype == "agent_response":
                    is_tool_call_step = bool(su.get("tool_calls") or su.get("tool_call") or su.get("call"))
                    if last_step_idx is not None and step_idx != last_step_idx:
                        log(f"⚠️ [MULTI-STEP] agent_response step changed: {last_step_idx} -> {step_idx}")
                        # Stop streaming if the CLI retries after a blocked step.
                        if safety_detected_in_turn and has_streamed_in_step:
                            log(f"⚠️ [SAFETY RETRY DETECTED] The previous step was blocked; stopping the stream.")
                            stopped = True
                        else:
                            text_parts = []
                    last_step_idx = step_idx

                    delta = su.get("text_delta") or ""
                    if is_tool_call_step and media_items:
                        delta = ""
                    if delta and not stopped:
                        text_parts.append(delta)
                        has_streamed_in_step = True

                        if any(kw in delta.lower() for kw in ("blocked by content safety filters", "safety guidelines", "cannot fulfill this request")):
                            log(f"⚠️ [REFUSAL IN TEXT] {delta.strip()[:150]}")
                            safety_detected_in_turn = True

                        full_so_far = "".join(text_parts)
                        hit_stop = None
                        for s in stop_list:
                            if s in full_so_far:
                                hit_stop = s
                                break
                        if hit_stop:
                            stopped = True
                            cutoff_idx = full_so_far.find(hit_stop)
                            emitted_len = len("".join(text_parts[:-1]))
                            valid_delta = full_so_far[emitted_len:cutoff_idx]
                            if valid_delta and on_delta:
                                on_delta(valid_delta)
                        else:
                            if on_delta:
                                on_delta(delta)
                if su.get("usage"):
                    usage = su["usage"]
            elif kind == "result":
                r = ev.get("result") or {}
                if r.get("usage"):
                    usage = r["usage"]
                r_err = str(r.get("error") or "")
                r_resp = str(r.get("response") or "")
                if any(w in r_err.lower() for w in ("safety", "blocked", "refusal", "filter")):
                    log(f"⚠️ [SAFETY IN RESULT] {r_err[:300]}")
                    safety_detected_in_turn = True

                is_safety = safety_detected_in_turn or any(
                    w in r_err.lower() or w in r_resp.lower()
                    for w in ("safety", "blocked", "refusal", "filter", "policy", "prohibit")
                )

                if r.get("status") == "SUCCESS":
                    if not on_delta and r_resp.strip():
                        text = r_resp
                    else:
                        text = "".join(text_parts)
                        if not text.strip() and r_resp.strip():
                            text = r_resp
                            if on_delta and not has_streamed_in_step:
                                on_delta(text)
                    for s in stop_list:
                        if s in text:
                            text = text.split(s, 1)[0]
                    return text, usage or {}

                # The CLI may include a partial reply with an error status.
                rescued_text = "".join(text_parts).strip() or r_resp.strip()
                if rescued_text and is_safety:
                    log(f"⚠️ [SAFETY RESCUE] Returning partial text ({len(rescued_text)} characters) despite the ERROR status")
                    for s in stop_list:
                        if s in rescued_text:
                            rescued_text = rescued_text.split(s, 1)[0]
                    if on_delta and not has_streamed_in_step:
                        on_delta(rescued_text)
                    return rescued_text, usage or {}

                # Return a readable refusal when the CLI supplies no partial reply.
                if is_safety:
                    refusal_msg = "I am unable to fulfill this request or process the attached image due to content safety policies."
                    log(f"⚠️ [SAFETY REFUSAL] Returning a refusal for the blocked response: {r_err[:150]}")
                    if on_delta and not has_streamed_in_step:
                        on_delta(refusal_msg)
                    return refusal_msg, usage or {}

                raise AgyError(str(r.get("error") or "agy turn failed"),
                              429 if "quota" in r_err.lower() or "rate" in r_err.lower() else 502)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
        finally:
            self.proc.stdin.close()
            cleanup_conversation(self.conversation_id)


class Pool:
    def __init__(self):
        self.idle = {}            # (model, effort) -> [Worker]
        self.total = 0
        self.starting = {}
        self.cond = threading.Condition()

    def _spawn_async(self, key):
        # Called with cond held; one pending spawn counts against capacity.
        self.total += 1
        self.starting[key] = self.starting.get(key, 0) + 1

        def finished(worker):
            self.starting[key] -= 1
            if not self.starting[key]:
                del self.starting[key]
            if worker is None:
                self.total -= 1
            else:
                self.idle.setdefault(key, []).append(worker)
            self.cond.notify_all()

        def job():
            w = None
            ready = False
            try:
                w = Worker(*key)
                ready = w.wait_ready(CFG["spawn_timeout_sec"])
                if not ready:
                    log(f"spawn {key}: worker failed to start: {w.spawn_error or 'no init event'}")
            except Exception as e:
                log(f"spawn {key}: {e}")
            finally:
                try:
                    if w is not None and not ready:
                        w.close()
                finally:
                    with self.cond:
                        finished(w if ready else None)
        try:
            threading.Thread(target=job, daemon=True).start()
        except BaseException:
            finished(None)
            raise

    def warm(self, model, effort=None):
        key = (model, effort)
        with self.cond:
            have = len(self.idle.get(key, [])) + self.starting.get(key, 0)
            for _ in range(max(0, CFG["pool_per_key"] - have)):
                if self.total >= CFG["pool_max_processes"]:
                    break
                self._spawn_async(key)

    def acquire(self, model, effort, abort_event=None):
        key = (model, effort)
        deadline = time.monotonic() + CFG["spawn_timeout_sec"] + 10
        requested_spawn = False
        while True:
            with self.cond:
                if abort_event and abort_event.is_set():
                    raise AgyError("client aborted turn", 499)
                left = deadline - time.monotonic()
                if left <= 0:
                    raise AgyError("no agy worker became ready in time", 503)
                workers = self.idle.get(key)
                if workers:
                    w = workers.pop(0)
                    if not workers:
                        self.idle.pop(key)
                    if not w.dead:
                        if self.total < CFG["pool_max_processes"] and not workers and not self.starting.get(key):
                            self._spawn_async(key)
                        return w
                elif self.starting.get(key):
                    self.cond.wait(timeout=min(left, 0.25))
                    continue
                elif requested_spawn:
                    raise AgyError("agy worker failed to start", 503)
                elif self.total < CFG["pool_max_processes"]:
                    self._spawn_async(key)
                    requested_spawn = True
                    continue
                elif self.idle:
                    other_key = next(iter(self.idle))
                    w = self.idle[other_key].pop(0)
                    if not self.idle[other_key]:
                        del self.idle[other_key]
                else:
                    self.cond.wait(timeout=min(left, 0.25))
                    continue
            self.release_dead(w)

    def release_dead(self, w):
        try:
            w.close()
        finally:
            with self.cond:
                self.total -= 1
                self.cond.notify_all()


POOL = Pool()


class Pacer:
    """Serialize CLI turns and enforce a gap between them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_end = 0.0
        self.turns = 0
        self.waited_sec = 0.0

    def run(self, fn, abort_event=None):
        with self.lock:
            gap = float(CFG.get("min_turn_gap_sec", 0) or 0)
            jitter = float(CFG.get("turn_gap_jitter_sec", 0) or 0)
            if jitter > 0:
                import random
                gap += random.uniform(0, jitter)
            wait = self.last_end + gap - time.monotonic()
            if wait > 0:
                started_wait = time.monotonic()
                if abort_event:
                    abort_event.wait(wait)
                else:
                    time.sleep(wait)
                self.waited_sec += time.monotonic() - started_wait
            if abort_event and abort_event.is_set():
                raise AgyError("client aborted turn", 499)
            try:
                return fn()
            finally:
                self.last_end = time.monotonic()
                self.turns += 1


PACER = Pacer()


EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium",
              "high": "high", "xhigh": "xhigh", "max": "max"}
IMAGE_MODELS = ("image-generator", "imagen-3", "dall-e-3")


def openai_error(status, message, etype="upstream_error"):
    return {"error": {"message": message, "type": etype, "code": status}}


def map_usage(u):
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {"prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens",
                                  u.get("input_tokens", 0) + u.get("output_tokens", 0))}


class QueueError(Exception):
    def __init__(self, status, message, code):
        super().__init__(message)
        self.status = status
        self.code = code


class RequestQueue:
    """Bound uploads by bytes and run fully uploaded requests in FIFO order."""

    def __init__(self):
        self.cond = threading.Condition()
        self.waiting = deque()
        self.active = False
        self.body_bytes = 0

    @contextmanager
    def reserve_body(self, size):
        with self.cond:
            if self.body_bytes + size > CFG["max_inflight_body_bytes"]:
                raise QueueError(429, "request body budget is full; retry later", "body_budget_full")
            self.body_bytes += size
        try:
            yield
        finally:
            with self.cond:
                self.body_bytes -= size

    @contextmanager
    def turn(self, disconnected):
        ticket = object()
        deadline = time.monotonic() + CFG["queue_timeout_sec"]
        with self.cond:
            if self.active or self.waiting:
                if len(self.waiting) >= CFG["max_queued_requests"]:
                    raise QueueError(429, "request queue is full; retry later", "queue_full")
                self.waiting.append(ticket)
                try:
                    while True:
                        if disconnected():
                            raise QueueError(499, "client disconnected", "client_disconnected")
                        left = deadline - time.monotonic()
                        if left <= 0:
                            raise QueueError(503, "request expired waiting in queue; retry later", "queue_timeout")
                        if not self.active and self.waiting[0] is ticket:
                            break
                        self.cond.wait(min(left, 0.25))
                finally:
                    self.waiting.remove(ticket)
                    self.cond.notify_all()
            elif disconnected():
                raise QueueError(499, "client disconnected", "client_disconnected")
            self.active = True
        try:
            yield
        finally:
            with self.cond:
                self.active = False
                self.cond.notify_all()

    def status(self):
        with self.cond:
            return {"active": int(self.active), "waiting": len(self.waiting),
                    "max_waiting": CFG["max_queued_requests"], "body_bytes": self.body_bytes,
                    "max_body_bytes": CFG["max_inflight_body_bytes"]}


class SafeThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64
    max_connections = 48

    def __init__(self, *args, **kwargs):
        self.turn_queue = RequestQueue()
        self.connections = threading.BoundedSemaphore(self.max_connections)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def handle_error(self, request, client_address):
        err = sys.exc_info()[1]
        if isinstance(err, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "agy-facade/1.0"

    def setup(self):
        self.request.settimeout(CFG["http_io_timeout_sec"])
        super().setup()

    def _client_disconnected(self):
        try:
            # POST connections are not reused. EOF means we can discard the turn.
            poller = select.poll()
            poller.register(self.connection, select.POLLIN)
            return bool(poller.poll(0)) and not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, fmt, *args):
        pass

    def _json(self, status, obj, extra_headers=None):
        try:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if self.close_connection:
                self.send_header("Connection", "close")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"warning: _json error: {e}")

    def _authorized(self):
        import hmac
        act = active_keys()
        if not act:
            return None
        provided = []
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            provided.append(auth[7:])
        if self.headers.get("x-api-key"):
            provided.append(self.headers.get("x-api-key"))
        for p in provided:
            for value, name in act.items():
                if isinstance(value, str) and hmac.compare_digest(p.encode(), value.encode()):
                    return name
        return None

    def do_GET(self):
        path = self.path.rstrip("/")

        if path.startswith("/v1/images/") or path.startswith("/images/"):
            if not self._authorized():
                return self._json(401, openai_error(401, "missing or invalid api key", "auth_error"))
            fname = os.path.basename(path)
            fpath = os.path.join(IMAGE_DIR, fname)
            if not re.fullmatch(r"img_[0-9a-f]{16}\.(jpg|jpeg|png|webp)", fname) or os.path.islink(fpath) or not os.path.isfile(fpath):
                return self._json(404, openai_error(404, "image not found", "invalid_request_error"))
            ext = os.path.splitext(fname)[1].lower()
            ctype = "image/jpeg"
            if ext == ".png":
                ctype = "image/png"
            elif ext == ".webp":
                ctype = "image/webp"
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if path in ("/v1/models", "/models"):
            if not self._authorized():
                return self._json(401, openai_error(401, "missing or invalid api key", "auth_error"))
            ids = CATALOG.get()
            for extra in CFG.get("extra_models", []):
                if extra not in ids:
                    ids.append(extra)
            for img_mod in (("image-generator", "imagen-3") if CFG.get("enable_image_generation") else ()):
                if img_mod not in ids:
                    ids.append(img_mod)
            data = [{"id": m, "object": "model", "created": 0, "owned_by": "antigravity"}
                    for m in sorted(ids)]
            return self._json(200, {"object": "list", "data": data})

        if self.path == "/":
            if not self._authorized():
                return self._json(401, openai_error(401, "missing or invalid api key", "auth_error"))
            with POOL.cond:
                pool_info = {"processes": POOL.total,
                             "idle": {f"{k[0]}|{k[1] or '-'}": len(v) for k, v in POOL.idle.items()}}
            return self._json(200, {"service": "agy-facade", "backend": "official agy CLI (stream-json)",
                                    "multimodal": {"images": True, "audio": True, "image_generation": bool(CFG.get("enable_image_generation"))},
                                    "pool": pool_info, "stats": STATS,
                                    "queue": self.server.turn_queue.status(),
                                    "pacing": {"turn_gap_sec": CFG.get("min_turn_gap_sec"),
                                               "jitter_sec": CFG.get("turn_gap_jitter_sec"),
                                               "turns_done": PACER.turns,
                                               "total_wait_sec": round(PACER.waited_sec, 1)}})
        return self._json(404, openai_error(404, "not found", "invalid_request_error"))

    def _body_length(self):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get_all("Transfer-Encoding") or len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
            raise QueueError(400, "expected one valid Content-Length and no Transfer-Encoding", "invalid_request_error")
        digits = lengths[0].lstrip("0") or "0"
        max_bytes = CFG["max_request_bytes"]
        if len(digits) > len(str(max_bytes)) or int(digits) > max_bytes:
            raise QueueError(413, "request body too large", "invalid_request_error")
        return int(digits)

    def _read_body(self, length):
        body = bytearray()
        deadline = time.monotonic() + CFG["body_timeout_sec"]
        io_timeout = CFG["http_io_timeout_sec"]
        try:
            while len(body) < length:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError
                self.connection.settimeout(min(left, io_timeout))
                chunk = self.rfile.read1(min(64 * 1024, length - len(body)))
                if not chunk:
                    raise QueueError(400, "incomplete request body", "invalid_request_error")
                body.extend(chunk)
        except TimeoutError:
            raise QueueError(408, "timed out reading request body", "request_body_timeout") from None
        finally:
            self.connection.settimeout(io_timeout)
        return body

    def do_POST(self):
        self.close_connection = True
        path = self.path.rstrip("/")
        images = path in ("/v1/images/generations", "/images/generations")
        if not images and path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, openai_error(404, "not found", "invalid_request_error"))
        key_name = self._authorized()
        if not key_name:
            return self._json(401, openai_error(401, "missing or invalid api key", "auth_error"))
        if images and not CFG.get("enable_image_generation"):
            return self._json(404, openai_error(404, "image generation is disabled", "invalid_request_error"))

        turn_queue = self.server.turn_queue
        try:
            length = self._body_length()
            with turn_queue.reserve_body(length):
                body = self._read_body(length)
                with turn_queue.turn(self._client_disconnected):
                    try:
                        req = json.loads(body)
                        if not isinstance(req, dict):
                            raise ValueError("expected a JSON object")
                    except (ValueError, RecursionError):
                        raise QueueError(400, "bad JSON request", "invalid_request_error") from None
                    finally:
                        del body
                    if images:
                        return self._handle_images_generations(key_name, req)
                    return self._handle_chat(key_name, req)
        except QueueError as e:
            if e.status == 499:
                return
            headers = {"Retry-After": "5"} if e.status in (429, 503) else None
            error = openai_error(e.status, str(e), e.code)
            return self._json(e.status, error, headers)

    def _handle_chat(self, key_name, req):
        model = req.get("model") or CFG["default_model"]
        messages = req.get("messages") or []
        if not isinstance(model, str) or not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
            return self._json(400, openai_error(400, "messages required", "invalid_request_error"))

        t0 = time.time()

        if model in IMAGE_MODELS:
            if not CFG.get("enable_image_generation"):
                return self._json(404, openai_error(404, "image generation is disabled", "invalid_request_error"))
            prompt = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        prompt = c
                    elif isinstance(c, list):
                        prompt = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
                    if prompt:
                        break
            if not prompt:
                return self._json(400, openai_error(400, "user prompt required for image generation", "invalid_request_error"))
            return self._serve_image_generation_chat(prompt, model, key_name, req.get("stream", False), t0)

        known = CATALOG.get()
        if known and model not in known and model not in CFG.get("extra_models", []) and model not in IMAGE_MODELS:
            return self._json(400, openai_error(
                400, f"model {model} is not selectable by agy; see GET /v1/models",
                "invalid_request_error"))

        effort = EFFORT_MAP.get(str(req.get("reasoning_effort") or "").lower())
        prompt, media_items = build_prompt_and_media(req, messages)

        log(f"--> POST /v1/chat/completions: model={model}, msgs={len(messages)}, media={len(media_items)} {[m[0] for m in media_items]}")

        has_audio = any(m[0].startswith("audio/") for m in media_items)
        if has_audio and (model.startswith("claude") or model.startswith("gpt")):
            return self._json(400, openai_error(
                400,
                f"Model '{model}' does not support audio input. Please select a Gemini model (such as gemini-3.8-flash-high).",
                "invalid_request_error"))

        with _lock:
            STATS["requests"] += 1
            if media_items:
                STATS["media_turns"] = STATS.get("media_turns", 0) + 1
            STATS["by_model"][model] = STATS["by_model"].get(model, 0) + 1
            STATS["by_key"][key_name] = STATS["by_key"].get(key_name, 0) + 1

        stop = req.get("stop")

        if req.get("stream"):
            return self._serve(prompt, model, effort, key_name, stream=True, t0=t0, media_items=media_items, stop=stop)
        return self._serve(prompt, model, effort, key_name, stream=False, t0=t0, media_items=media_items, stop=stop)

    def _serve_image_generation_chat(self, prompt, model, key_name, stream, t0):
        try:
            dest_path, img_name = PACER.run(lambda: generate_image_via_agy(
                prompt, timeout=CFG.get("image_generation_timeout_sec", 120)))
        except AgyError as e:
            return self._json(e.status, openai_error(e.status, e.message))
        except Exception as e:
            return self._json(500, openai_error(500, str(e)))

        host = f"{CFG['listen_host']}:{CFG['listen_port']}"
        img_url = f"http://{host}/v1/images/{img_name}"
        reply_text = f"![{prompt}]({img_url})\n\nGenerated image: {img_url}"

        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": 25, "total_tokens": len(prompt.split()) + 25}

        with _lock:
            STATS["requests"] += 1
            STATS["images"] = STATS.get("images", 0) + 1
            STATS["by_model"][model] = STATS["by_model"].get(model, 0) + 1
            STATS["by_key"][key_name] = STATS["by_key"].get(key_name, 0) + 1

        log(f"{model}: image ready in {time.time()-t0:.1f}s -> {img_name}")

        if not stream:
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}],
                "usage": usage
            })

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            d = {"role": "assistant", "content": reply_text}
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": model, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
            fin = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": usage}
            self.wfile.write(b"data: " + json.dumps(fin, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log(f"{model}: client closed the image stream")

    def _handle_images_generations(self, key_name, req):
        prompt = req.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return self._json(400, openai_error(400, "prompt is required", "invalid_request_error"))

        size = req.get("size")
        response_format = req.get("response_format", "url")
        model = req.get("model") or "image-generator"
        t0 = time.time()

        try:
            dest_path, img_name = PACER.run(lambda: generate_image_via_agy(
                prompt, size, timeout=CFG.get("image_generation_timeout_sec", 120)))
        except AgyError as e:
            log(f"image-generator: {e.message}")
            return self._json(e.status, openai_error(e.status, e.message))
        except Exception as e:
            log(f"image-generator: {e}")
            return self._json(500, openai_error(500, str(e)))

        with _lock:
            STATS["requests"] += 1
            STATS["images"] = STATS.get("images", 0) + 1
            STATS["by_model"][model] = STATS["by_model"].get(model, 0) + 1
            STATS["by_key"][key_name] = STATS["by_key"].get(key_name, 0) + 1

        log(f"image-generator: ready in {time.time()-t0:.1f}s -> {img_name}")
        created = int(time.time())
        if response_format == "b64_json":
            with open(dest_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("ascii")
            return self._json(200, {
                "created": created,
                "data": [{"b64_json": b64_data, "revised_prompt": prompt}]
            })
        else:
            host = f"{CFG['listen_host']}:{CFG['listen_port']}"
            img_url = f"http://{host}/v1/images/{img_name}"
            return self._json(200, {
                "created": created,
                "data": [{"url": img_url, "revised_prompt": prompt}]
            })

    def _serve(self, prompt, model, effort, key_name, stream, t0, media_items=None, stop=None):
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        chunks = queue.Queue(maxsize=64)
        result = {}
        abort_event = threading.Event()
        finished = threading.Event()

        def on_delta(text):
            while not abort_event.is_set():
                try:
                    chunks.put(text, timeout=0.25)
                    return
                except queue.Full:
                    pass

        def next_chunk():
            while True:
                try:
                    return chunks.get_nowait()
                except queue.Empty:
                    if finished.is_set():
                        # A last delta may have been queued before the event was set.
                        try:
                            return chunks.get_nowait()
                        except queue.Empty:
                            return None
                    if self._client_disconnected():
                        raise ConnectionResetError
                    if not stream:
                        finished.wait(0.25)
                    else:
                        try:
                            return chunks.get(timeout=0.25)
                        except queue.Empty:
                            pass

        def job():
            def run_turn():
                w = POOL.acquire(model, effort, abort_event=abort_event)
                try:
                    return w.run_turn(prompt, media_items, on_delta if stream else None,
                                      CFG["request_timeout_sec"], abort_event=abort_event, stop=stop)
                finally:
                    POOL.release_dead(w)

            try:
                text, usage = PACER.run(run_turn, abort_event)
                result.update(text=text, usage=usage)
            except AgyError as e:
                result.update(error=e)
            except Exception as e:
                result.update(error=AgyError(str(e)))
            finally:
                finished.set()

        thread = threading.Thread(target=job, daemon=True)
        thread.start()
        try:
            return self._deliver_chat(cid, created, model, stream, t0, result, next_chunk)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            log(f"{model}: client disconnected or stopped reading")
        finally:
            # Do not hand off the FIFO slot until CLI I/O and cleanup have stopped.
            abort_event.set()
            thread.join()

    def _deliver_chat(self, cid, created, model, stream, t0, result, next_chunk):
        if not stream:
            next_chunk()
            if "error" in result:
                e = result["error"]
                log(f"{model}: agy {e.status} in {time.time()-t0:.1f}s")
                return self._json(e.status, openai_error(e.status, e.message))
            usage = map_usage(result.get("usage"))
            with _lock:
                STATS["tokens"]["prompt"] += usage["prompt_tokens"]
                STATS["tokens"]["completion"] += usage["completion_tokens"]
            log(f"{model}: completed in {time.time()-t0:.1f}s, tokens: {usage['total_tokens']}")
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": result["text"]},
                             "finish_reason": "stop"}],
                "usage": usage})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        sent_role = False
        try:
            while True:
                delta = next_chunk()
                if delta is None:
                    break
                d = {"content": delta}
                if not sent_role:
                    d = dict(d, role="assistant")
                    sent_role = True
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                         "model": model,
                         "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            if "error" in result:
                e = result["error"]
                err_chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": model, "choices": [],
                             "error": {"message": e.message, "type": "upstream_error", "code": e.status}}
                self.wfile.write(b"data: " + json.dumps(err_chunk, ensure_ascii=False).encode() + b"\n\n")
            else:
                usage = map_usage(result.get("usage"))
                with _lock:
                    STATS["tokens"]["prompt"] += usage["prompt_tokens"]
                    STATS["tokens"]["completion"] += usage["completion_tokens"]
                fin = {"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                       "usage": usage}
                self.wfile.write(b"data: " + json.dumps(fin, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            log(f"{model}: stream ended in {time.time()-t0:.1f}s")


def load_config(path):
    if not os.path.exists(path):
        try:
            with open(path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=1, ensure_ascii=False)
            log(f"Created default configuration: {path}")
        except OSError:
            pass
    if os.path.exists(path):
        with open(path) as f:
            CFG.update(json.load(f))


def detach_stdin():
    """Prevent the facade and its children from reading from the terminal."""
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd, 0)
    except OSError:
        pass


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "keys":
        sys.exit(key_management(sys.argv[2:]))
    ap = argparse.ArgumentParser(description="OpenAI-style HTTP adapter for the Antigravity CLI")
    ap.add_argument("--config", default=os.path.join(STATE_DIR, "agy_config.json"))
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()
    detach_stdin()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    load_config(args.config)
    if args.host:
        CFG["listen_host"] = args.host
    if args.port:
        CFG["listen_port"] = args.port
    if CFG["listen_host"] != "127.0.0.1":
        ap.error("only 127.0.0.1 is supported by this local-only snapshot")
    for key, minimum in (("max_queued_requests", 0), ("max_request_bytes", 1),
                         ("max_inflight_body_bytes", CFG["max_request_bytes"])):
        value = CFG[key]
        if type(value) is not int or value < minimum:
            ap.error(f"{key} must be an integer >= {minimum}")
    for key in ("queue_timeout_sec", "http_io_timeout_sec", "body_timeout_sec"):
        value = CFG[key]
        if type(value) not in (int, float) or not 0 < value <= 86_400:
            ap.error(f"{key} must be a positive number <= 86400")
    if not active_keys():
        ap.error("no active key: run agy_facade.py keys add <name> first")
    agy_path = find_agy()
    if not shutil.which(agy_path) and not (os.path.isfile(agy_path) and os.access(agy_path, os.X_OK)):
        log(f"ERROR: agy executable not found ({agy_path}). Set agy_bin in "
            f"agy_config.json or set AGY_BIN. The facade did not start.")
        sys.exit(1)
    log(f"agy: {agy_path}")
    for m in (CFG.get("warm_models") or [CFG["default_model"]]):
        POOL.warm(m)
    CATALOG.get()
    srv = SafeThreadingHTTPServer((CFG["listen_host"], CFG["listen_port"]), Handler)
    log(f"agy-facade is listening at http://{CFG['listen_host']}:{CFG['listen_port']}/v1")

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
