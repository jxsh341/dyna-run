import json
import requests


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def list_models(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.ConnectionError:
            pass
        return []

    def generate(self, model, prompt, stream=False, **kwargs):
        payload = {"model": model, "prompt": prompt, "stream": stream, **kwargs}
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error: {resp.status_code} {resp.text}")
        if not stream:
            return resp.json()
        return [json.loads(line) for line in resp.iter_lines() if line]

    def is_available(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
