import json

import src.modules.wavjourney.planner as planner_module
from src.modules.wavjourney.planner import AudioCreativePlanner


def test_generate_script_parses_llm_json(monkeypatch):
    script_payload = {
        "description": "demo",
        "duration": 5,
        "elements": [
            {"type": "music", "content": "calm", "start_time": 0, "duration": 5, "volume": 0.5}
        ],
    }

    def fake_call_llm(self, prompt):
        return json.dumps(script_payload)

    monkeypatch.setattr(AudioCreativePlanner, "_call_llm", fake_call_llm, raising=False)

    planner = AudioCreativePlanner(llm_provider="openai", llm_model="stub", api_key="test")
    script = planner._generate_script("demo instruction")

    assert script == script_payload


def test_generate_script_falls_back_on_invalid_json(monkeypatch):
    def fake_call_llm(self, prompt):
        return "not json at all"

    monkeypatch.setattr(AudioCreativePlanner, "_call_llm", fake_call_llm, raising=False)

    planner = AudioCreativePlanner(llm_provider="openai", llm_model="stub", api_key="test")
    script = planner._generate_script("fallback instruction")

    assert script["description"] == "fallback instruction"
    assert script["elements"]
