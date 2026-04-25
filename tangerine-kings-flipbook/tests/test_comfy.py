import json
from unittest.mock import MagicMock

import pytest
import respx
from httpx import Response

from lib.comfy import ComfyClient, Dispatcher


@respx.mock
def test_queue_prompt_submits_workflow_and_polls_history():
    workflow = {"3": {"inputs": {"seed": 42}, "class_type": "KSampler"}}

    respx.post("http://localhost:8188/prompt").mock(
        return_value=Response(200, json={"prompt_id": "abc-123"})
    )
    history_resp = {
        "abc-123": {
            "outputs": {
                "9": {"images": [{"filename": "keyframe_00001_.png", "subfolder": "", "type": "output"}]}
            }
        }
    }
    respx.get("http://localhost:8188/history/abc-123").mock(
        return_value=Response(200, json=history_resp)
    )
    respx.get("http://localhost:8188/view").mock(
        return_value=Response(200, content=b"\x89PNG\r\n\x1a\nFAKEDATA")
    )

    client = ComfyClient(host="localhost", port=8188)
    image_bytes = client.queue_prompt(
        workflow=workflow,
        prompt_text="TANGSTICK style stick figure singer",
        negative_text="blurry",
        seed=42,
    )
    assert image_bytes.startswith(b"\x89PNG")


@respx.mock
def test_queue_prompt_substitutes_prompt_into_text_node():
    """The workflow has a 'PROMPT_PLACEHOLDER' that must be replaced."""
    workflow = {
        "6": {
            "inputs": {"text": "PROMPT_PLACEHOLDER", "clip": ["10", 1]},
            "class_type": "CLIPTextEncode",
        }
    }
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return Response(200, json={"prompt_id": "x"})

    respx.post("http://localhost:8188/prompt").mock(side_effect=capture)
    respx.get("http://localhost:8188/history/x").mock(
        return_value=Response(200, json={
            "x": {"outputs": {"9": {"images": [{"filename": "f.png", "subfolder": "", "type": "output"}]}}}
        })
    )
    respx.get("http://localhost:8188/view").mock(
        return_value=Response(200, content=b"\x89PNG_DATA")
    )

    client = ComfyClient(host="localhost", port=8188)
    client.queue_prompt(
        workflow=workflow, prompt_text="my-prompt", negative_text="neg", seed=1
    )

    assert captured["body"]["prompt"]["6"]["inputs"]["text"] == "my-prompt"


def test_dispatcher_round_robins_across_clients():
    c1 = MagicMock()
    c1.queue_prompt.return_value = b"img1"
    c2 = MagicMock()
    c2.queue_prompt.return_value = b"img2"

    dispatcher = Dispatcher([c1, c2])
    work = [
        {"prompt_text": "a", "negative_text": "n", "seed": 1, "tag": "frame_0"},
        {"prompt_text": "b", "negative_text": "n", "seed": 2, "tag": "frame_1"},
        {"prompt_text": "c", "negative_text": "n", "seed": 3, "tag": "frame_2"},
        {"prompt_text": "d", "negative_text": "n", "seed": 4, "tag": "frame_3"},
    ]
    workflow: dict = {}

    results = dispatcher.render(workflow, work)

    assert c1.queue_prompt.call_count == 2
    assert c2.queue_prompt.call_count == 2
    assert results == {"frame_0": b"img1", "frame_1": b"img2", "frame_2": b"img1", "frame_3": b"img2"}


def test_dispatcher_empty_clients_raises():
    with pytest.raises(ValueError):
        Dispatcher([])
