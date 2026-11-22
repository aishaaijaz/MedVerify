"""Tiny Ollama client shared by the apps and the batch runner."""
import requests

OLLAMA_URL = "http://localhost:11434"


def installed_models(url=OLLAMA_URL):
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", []) if m.get("name")]
    except requests.RequestException:
        return []


def chat(model, prompt, temperature=0.2, max_tokens=1024, url=OLLAMA_URL, timeout=300):
    """Send one prompt to a local Ollama model and return the text (raises requests exceptions on failure)."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
               "options": {"temperature": temperature, "num_predict": max_tokens}}
    r = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()
