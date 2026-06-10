#!/usr/bin/env python3
"""Proxy for DashScope chat/completions that injects cache_control: ephemeral.

Why: DashScope's OpenAI-compatible endpoint only returns `cached_tokens`
when the request contains an Anthropic-style `cache_control: {"type":
"ephemeral"}` marker on the last user/assistant text block. GenericAgent's
NativeOAISession doesn't inject this automatically, so we sit a thin proxy
in front of DashScope to do it transparently. Same trick the ZeroClaw
adapter uses (see claw_configs/zeroclaw/tool_filter_proxy.py).

Usage:
    python3 dashscope_cache_proxy.py 18093 /path/to/logs/proxy_usage.jsonl

Then point GA's mykey.py apibase to http://127.0.0.1:18093/v1 (or
http://host.docker.internal:18093/v1 when called from inside a container
that has --add-host host.docker.internal:host-gateway).
"""

import http.server, json, urllib.request, urllib.error, ssl, sys, socketserver, threading
from datetime import datetime, timezone

REAL_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18093
USAGE_LOG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dashscope_proxy_usage.jsonl"

_usage_lock = threading.Lock()


def normalize_empty_assistant_content(messages):
    """DashScope cache bug workaround: assistant messages with content=""
    (which Hermes emits when the assistant only does tool_calls without text)
    cause the cache to fail entirely. Replace empty content with a single
    space — this fixes cache hit rate from 0% to 100% on multi-turn dialogs.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if content == "" or content is None:
            msg["content"] = " "


def add_cache_control(messages):
    """Add `cache_control: {"type": "ephemeral"}` markers to maximize cache hits.

    Strategy: mark the LAST user/assistant text message. DashScope/Anthropic
    cache_control acts as a "breakpoint" — everything BEFORE the marker is
    eligible for caching. The longer the prefix (more turns), the more we hit.

    Each turn, the previous turns' content is already in cache from a prior
    request, so a new request marking its current last message extends the
    cache forward. This is exactly how OpenClaw/pi-ai does it (see
    maybeAddOpenRouterAnthropicCacheControl in pi-ai).

    Prerequisite: call normalize_empty_assistant_content() first to fix
    a DashScope cache bug where empty assistant content kills cache hits.
    """
    normalize_empty_assistant_content(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
            return
        if not isinstance(content, list):
            continue
        for j in range(len(content) - 1, -1, -1):
            part = content[j]
            if isinstance(part, dict) and part.get("type") == "text":
                part["cache_control"] = {"type": "ephemeral"}
                return


class Proxy(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Inject cache_control on the request. Pass streaming flag through
        # untouched — GA's parser is chosen at request time based on `stream`,
        # so forcing stream=false would break GA's SSE-mode parser.
        is_stream = False
        try:
            data = json.loads(body)
            if "messages" in data:
                add_cache_control(data["messages"])
            is_stream = bool(data.get("stream"))
            body = json.dumps(data, ensure_ascii=False).encode()
        except (json.JSONDecodeError, KeyError):
            pass

        # GA sends to /v1/chat/completions (apibase already has /v1). REAL_BASE
        # also ends in /v1. Strip the leading "/v1" from self.path to avoid
        # double /v1 in the target URL.
        path = self.path
        if path.startswith("/v1/"):
            path = path[3:]
        target = f"{REAL_BASE}{path}"
        h = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length",
                                 "transfer-encoding", "accept-encoding")
        }
        h["Host"] = "dashscope.aliyuncs.com"
        h["Content-Length"] = str(len(body))
        h["Accept-Encoding"] = "identity"

        req = urllib.request.Request(target, data=body, headers=h, method="POST")
        try:
            r = urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=300)
            rb = r.read()

            # Log usage. For stream=false the body is a single JSON. For
            # stream=true (default for GA) it's SSE — scan chunks for usage.
            try:
                if is_stream:
                    usage = None; model_name = ""
                    for line in rb.split(b"\n"):
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            continue
                        try:
                            evt = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("usage"):
                            usage = evt["usage"]
                            model_name = evt.get("model", model_name)
                        elif not model_name and evt.get("model"):
                            model_name = evt["model"]
                    if usage:
                        entry = {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "model": model_name,
                            "usage": usage,
                        }
                        with _usage_lock:
                            with open(USAGE_LOG, "a") as f:
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                else:
                    d = json.loads(rb)
                    u = d.get("usage", {})
                    if u:
                        entry = {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "model": d.get("model", ""),
                            "usage": u,
                        }
                        with _usage_lock:
                            with open(USAGE_LOG, "a") as f:
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except (json.JSONDecodeError, KeyError):
                pass

            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(rb)))
            self.end_headers()
            self.wfile.write(rb)
        except urllib.error.HTTPError as e:
            rb = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(rb)))
            self.end_headers()
            self.wfile.write(rb)
        except Exception as e:
            err = str(e).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def log_message(self, *a):
        pass


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    print(f"[proxy] listening on 0.0.0.0:{PORT}  forwarding to {REAL_BASE}  log={USAGE_LOG}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Proxy).serve_forever()
