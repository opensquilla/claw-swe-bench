#!/usr/bin/env python3
"""Proxy that filters ZeroClaw's 42 tools down to essential ones before forwarding to LLM API.

Features:
- Filters 42 tools → 6 essential ones (file_read, file_write, file_edit, glob_search, content_search, git_operations)
- Converts Responses API format (input) to Chat Completions format (messages)
- Forces non-streaming mode
- Cleans OpenRouter-specific response fields
- Patches DeepSeek reasoning_content for multi-turn conversations
- Handles gzip by requesting identity encoding
"""
import http.server, json, urllib.request, ssl, sys, socketserver, threading
from datetime import datetime, timezone
from pathlib import Path

REAL_BASE = sys.argv[1] if len(sys.argv) > 1 else "https://openrouter.ai/api/v1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18090
USAGE_LOG = sys.argv[3] if len(sys.argv) > 3 else "/tmp/proxy_usage.jsonl"

# Thread-safe usage log writer
_usage_lock = threading.Lock()

ALLOWED_TOOLS = {
    "shell",
    "file_read", "file_write", "file_edit",
    "glob_search", "content_search", "git_operations",
}

# Check target API
IS_DEEPSEEK = "deepseek.com" in REAL_BASE
IS_OPENROUTER = "openrouter.ai" in REAL_BASE
IS_DASHSCOPE = "dashscope.aliyuncs.com" in REAL_BASE


def normalize_empty_assistant_content(messages):
    """DashScope cache bug workaround: assistant messages with content=""
    (emitted when assistant does pure tool_calls without text) cause cache to
    fail entirely. Replace empty content with a single space — fixes cache
    hit rate from 0% to 90%+ on multi-turn dialogs.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if content == "" or content is None:
            msg["content"] = " "


def add_dashscope_cache_control(messages):
    """Add Anthropic-style cache_control to last user/assistant text message.
    DashScope (Qwen) uses this to enable caching and return cached_tokens.
    Mimics pi-ai's maybeAddOpenRouterAnthropicCacheControl logic."""
    normalize_empty_assistant_content(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content,
                 "cache_control": {"type": "ephemeral"}},
            ]
            return
        if not isinstance(content, list):
            continue
        for j in range(len(content) - 1, -1, -1):
            part = content[j]
            if isinstance(part, dict) and part.get("type") == "text":
                part["cache_control"] = {"type": "ephemeral"}
                return


class FilterProxy(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
            # Filter tools
            if "tools" in data:
                data["tools"] = [t for t in data["tools"]
                                 if t.get("function", {}).get("name", "") in ALLOWED_TOOLS]
                if not data["tools"]:
                    del data["tools"]
                    if "tool_choice" in data:
                        del data["tool_choice"]
            # Convert Responses API format (input) to Chat Completions format (messages)
            if "input" in data and "messages" not in data:
                input_val = data.pop("input")
                messages = []
                if isinstance(input_val, str):
                    messages = [{"role": "user", "content": input_val}]
                elif isinstance(input_val, list):
                    for item in input_val:
                        if isinstance(item, str):
                            messages.append({"role": "user", "content": item})
                        elif isinstance(item, dict):
                            messages.append(item)
                data["messages"] = messages
                # Remove Responses API specific fields
                for key in ["instructions", "previous_response_id", "truncation"]:
                    data.pop(key, None)

            # DeepSeek: ensure reasoning_content exists on all assistant messages
            # DeepSeek requires reasoning_content to be present in multi-turn requests
            # when thinking mode is enabled. ZeroClaw may strip it.
            if IS_DEEPSEEK and "messages" in data:
                for msg in data["messages"]:
                    if msg.get("role") == "assistant":
                        if "reasoning_content" not in msg:
                            msg["reasoning_content"] = ""
                # Inject reasoning_effort xhigh for DeepSeek
                if "reasoning_effort" not in data:
                    data["reasoning_effort"] = "xhigh"

            # OpenRouter: inject reasoning_effort=high
            if IS_OPENROUTER and "reasoning_effort" not in data:
                data["reasoning_effort"] = "high"

            # DashScope (Qwen): add cache_control to enable cached_tokens reporting
            # + enable_thinking=true to ensure thinking mode is on
            if IS_DASHSCOPE and "messages" in data:
                add_dashscope_cache_control(data["messages"])
                if "enable_thinking" not in data:
                    data["enable_thinking"] = True

            # Force non-streaming (easier to proxy)
            data["stream"] = False
            body = json.dumps(data, ensure_ascii=False).encode()
        except (json.JSONDecodeError, KeyError):
            pass

        # Always forward to /chat/completions
        target_url = f"{REAL_BASE}/chat/completions"
        headers = {}
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "transfer-encoding", "accept-encoding"):
                headers[k] = v
        headers["Host"] = REAL_BASE.split("//")[1].split("/")[0]
        headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"  # no gzip

        req = urllib.request.Request(target_url, data=body, headers=headers, method="POST")
        ctx = ssl.create_default_context()
        try:
            resp = urllib.request.urlopen(req, context=ctx, timeout=300)
            resp_body = resp.read()

            # Clean response and log full usage
            try:
                resp_data = json.loads(resp_body)

                # Log full API usage (including cache_read, reasoning_tokens)
                api_usage = resp_data.get("usage", {})
                if api_usage:
                    log_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "client_ip": self.client_address[0],
                        "model": resp_data.get("model", ""),
                        "usage": api_usage,
                    }
                    with _usage_lock:
                        with open(USAGE_LOG, "a") as f:
                            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                for choice in resp_data.get("choices", []):
                    msg = choice.get("message", {})
                    # Remove OpenRouter-specific fields
                    if "reasoning" in msg:
                        msg.pop("reasoning", None)
                    if "reasoning_details" in msg:
                        msg.pop("reasoning_details", None)
                    choice.pop("native_finish_reason", None)
                resp_body = json.dumps(resp_data, ensure_ascii=False).encode()
            except (json.JSONDecodeError, KeyError):
                pass

            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception as e:
            err_msg = str(e).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)

    def log_message(self, *a): pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


print(f"Tool filter proxy on :{PORT} → {REAL_BASE} (allowing: {ALLOWED_TOOLS})"
      f"{' [DeepSeek mode]' if IS_DEEPSEEK else ''}", file=sys.stderr, flush=True)
ThreadedHTTPServer(("0.0.0.0", PORT), FilterProxy).serve_forever()
