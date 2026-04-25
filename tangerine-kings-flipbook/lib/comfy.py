"""ComfyUI HTTP API client + multi-instance dispatcher."""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


class ComfyClient:
    """Minimal client for the ComfyUI HTTP API."""

    def __init__(self, host: str = "localhost", port: int = 8188, timeout: float = 600.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def queue_prompt(
        self,
        workflow: dict,
        prompt_text: str,
        negative_text: str,
        seed: int,
    ) -> bytes:
        """Submit a workflow with prompt + seed substituted, poll, fetch image bytes."""
        wf = self._substitute(workflow, prompt_text, negative_text, seed)

        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base}/prompt", json={"prompt": wf})
            r.raise_for_status()
            prompt_id = r.json()["prompt_id"]

            outputs = self._poll_history(client, prompt_id)
            save_node = next(v for v in outputs.values() if "images" in v)
            img = save_node["images"][0]
            r = client.get(
                f"{self.base}/view",
                params={
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                },
            )
            r.raise_for_status()
            return r.content

    def _substitute(
        self, workflow: dict, prompt_text: str, negative_text: str, seed: int
    ) -> dict:
        wf = copy.deepcopy(workflow)
        for node in wf.values():
            if node.get("class_type") == "CLIPTextEncode":
                if node["inputs"].get("text") == "PROMPT_PLACEHOLDER":
                    node["inputs"]["text"] = prompt_text
                elif "blurry" in node["inputs"].get("text", "").lower():
                    node["inputs"]["text"] = negative_text
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = seed
        return wf

    def _poll_history(
        self, client: httpx.Client, prompt_id: str, interval: float = 0.5
    ) -> dict:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = client.get(f"{self.base}/history/{prompt_id}")
            r.raise_for_status()
            data = r.json()
            if prompt_id in data and data[prompt_id].get("outputs"):
                return data[prompt_id]["outputs"]
            time.sleep(interval)
        raise TimeoutError(
            f"ComfyUI did not produce output for {prompt_id} in {self.timeout}s"
        )


class Dispatcher:
    """Round-robins render work across N ComfyClient instances in parallel."""

    def __init__(self, clients: list[ComfyClient]):
        if not clients:
            raise ValueError("at least one ComfyClient required")
        self.clients = clients

    def render(self, workflow: dict, work: list[dict]) -> dict[str, bytes]:
        """Submit `work` round-robin across clients in parallel.

        Each work item: {"prompt_text", "negative_text", "seed", "tag"}.
        Returns: {tag: image_bytes}.
        """
        assignments = [
            (self.clients[i % len(self.clients)], item)
            for i, item in enumerate(work)
        ]

        def run(client_item):
            client, item = client_item
            img = client.queue_prompt(
                workflow=workflow,
                prompt_text=item["prompt_text"],
                negative_text=item["negative_text"],
                seed=item["seed"],
            )
            return item["tag"], img

        results: dict[str, bytes] = {}
        with ThreadPoolExecutor(max_workers=len(self.clients)) as exe:
            for tag, img in exe.map(run, assignments):
                results[tag] = img
        return results
