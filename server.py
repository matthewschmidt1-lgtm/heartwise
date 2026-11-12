"""Heartwise: one quiet conversation. Serves the page and relays it to Claude.

Run with ./run.sh. The API key is read from a .env file beside this script
(ANTHROPIC_API_KEY=...) or from the environment. `--rehearsal` streams a canned
reply so the page can be checked without a key.
"""
import hmac
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
MODEL = "claude-opus-5-5"
REHEARSAL = "--rehearsal" in sys.argv
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
MAX_BODY = 2_000_000


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def has_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def gate():
    """'open' on your own machine with no code set, 'locked' when ACCESS_CODE is set,
    'unset' when hosted without one (a public address never runs unprotected)."""
    if os.environ.get("ACCESS_CODE", "").strip():
        return "locked"
    return "unset" if os.environ.get("RAILWAY_ENVIRONMENT") else "open"


def admitted(code):
    state = gate()
    if state == "open":
        return True
    if state == "unset" or not isinstance(code, str):
        return False
    return hmac.compare_digest(code.strip().encode(), os.environ["ACCESS_CODE"].strip().encode())


def clean(messages):
    """Keep only well-formed user/assistant text turns, starting with the user."""
    out = []
    for m in messages if isinstance(messages, list) else []:
        if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str) and m["content"].strip()):
            out.append({"role": m["role"], "content": m["content"]})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def open_stream(client, messages, with_fallback):
    params = dict(
        model=MODEL,
        max_tokens=16000,
        # teacher.md is read on every turn, so edits to it apply to the next reply.
        system=(ROOT / "teacher.md").read_text(),
        messages=messages,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        cache_control={"type": "ephemeral"},
    )
    if with_fallback:
        # If a safety classifier declines a turn, the API retries it on another model.
        return client.beta.messages.stream(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
    return client.messages.stream(**params)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # conversations are private; log nothing about requests

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/ready":
            return self.send_json(200, {"ready": REHEARSAL or has_key(), "gate": gate()})
        target = (PUBLIC / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        if PUBLIC not in target.parents or not target.is_file():
            return self.send_json(404, {"error": "not found"})
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def event(self, payload):
        self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
        self.wfile.flush()

    def do_POST(self):
        if self.path not in ("/api/talk", "/api/enter"):
            return self.send_json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self.send_json(413, {"error": "too large"})
        try:
            body = json.loads(self.rfile.read(length))
            body = body if isinstance(body, dict) else {}
        except ValueError:
            body = {}
        if not admitted(body.get("code")):
            if gate() == "unset":
                return self.send_json(403, {"error": "no_code"})
            time.sleep(1)  # slows down guessing
            return self.send_json(401, {"error": "locked"})
        if self.path == "/api/enter":
            return self.send_json(200, {"ok": True})
        messages = clean(body.get("messages"))
        if not messages or messages[-1]["role"] != "user":
            return self.send_json(400, {"error": "nothing to answer"})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.reply(messages)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the person closed the page mid-reply

    def reply(self, messages):
        if REHEARSAL:
            for word in ("That sounds like a hard evening. I am listening.\n\n"
                         "Tell me as much or as little as you like.").split(" "):
                self.event({"t": word + " "})
                time.sleep(0.04)
            return self.event({"done": True})
        if not has_key():
            return self.event({"error": "no_key"})
        client = anthropic.Anthropic()
        try:
            try:
                self.relay(open_stream(client, messages, with_fallback=True))
            except anthropic.BadRequestError:
                # Nothing has been sent yet when a request is rejected, so it is
                # safe to try once more without the optional fallback feature.
                self.relay(open_stream(client, messages, with_fallback=False))
        except anthropic.AuthenticationError:
            self.event({"error": "bad_key"})
        except anthropic.RateLimitError:
            self.event({"error": "busy"})
        except anthropic.APIStatusError as e:
            print("API error", e.status_code, e.message, file=sys.stderr)
            self.event({"error": "busy" if e.status_code >= 500 else "failed"})
        except anthropic.APIConnectionError:
            self.event({"error": "offline"})

    def relay(self, manager):
        with manager as stream:
            for text in stream.text_stream:
                self.event({"t": text})
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            self.event({"error": "declined"})
        elif final.stop_reason == "max_tokens":
            self.event({"error": "cut_short"})
        else:
            self.event({"done": True})


if __name__ == "__main__":
    load_env()
    port = int(os.environ.get("PORT", "4400"))
    print("Heartwise is at http://localhost:%d%s" % (port, "  (rehearsal: canned replies)" if REHEARSAL else ""), flush=True)
    if not (REHEARSAL or has_key()):
        print("No API key yet. Put ANTHROPIC_API_KEY=... in %s" % (ROOT / ".env"))
    # On a host such as Railway the server must accept outside connections; on your own
    # machine it stays private to it.
    hosted = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("HOST"))
    host = os.environ.get("HOST") or ("0.0.0.0" if hosted else "127.0.0.1")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
