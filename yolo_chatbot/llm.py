"""Optional server-side LLM. No credentials or endpoints are entered in the UI."""
import json
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "none"
    model: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)

    @property
    def enabled(self):
        return self.provider != "none"


def load_config(values=None):
    values = values or {}
    def setting(key, default=""):
        return str(values.get(key, os.environ.get("LLM_" + key.upper(), default))).strip()
    provider = setting("provider", "none").lower()
    if provider == "none":
        return LLMConfig()
    if provider not in {"openai", "ollama"}:
        raise ValueError("Unsupported LLM provider")
    model = setting("model")
    if not model:
        raise ValueError("LLM model must be configured")
    base = setting("base_url", "https://api.openai.com/v1" if provider == "openai" else "http://127.0.0.1:11434").rstrip("/")
    parsed = urlsplit(base)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Invalid server endpoint")
    if parsed.scheme != "https" and not (provider == "ollama" and parsed.scheme == "http" and local):
        raise ValueError("A hosted LLM must use HTTPS")
    api_key = setting("api_key")
    if provider == "openai" and not api_key:
        raise ValueError("LLM API key must be configured")
    return LLMConfig(provider, model, base, api_key)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward authorization credentials to a redirect destination.
        return None


def complete_json(messages, config):
    if not config.enabled:
        raise ValueError("LLM is disabled")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = "Bearer " + config.api_key
    if config.provider == "openai":
        endpoint = config.base_url + "/chat/completions"
        payload = {"model": config.model, "messages": messages,
                   "response_format": {"type": "json_object"},
                   "max_completion_tokens": 900, "store": False}
    else:
        endpoint = config.base_url + "/api/chat"
        payload = {"model": config.model, "messages": messages, "stream": False,
                   "format": "json", "options": {"temperature": 0, "num_predict": 700}}
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with build_opener(NoRedirect()).open(request, timeout=25) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("LLM response too large")
    output = json.loads(raw)
    content = (output["choices"][0]["message"]["content"] if config.provider == "openai"
               else output["message"]["content"])
    return json.loads(content)
