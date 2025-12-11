import numpy as np
import pathlib

import pytest

import src.modules.tts.tts_engine as tts_engine
from src.modules.tts.tts_engine import TTSEngine


def test_auto_engine_falls_back_to_pyttsx3(monkeypatch):
    load_order = []

    monkeypatch.setattr(tts_engine.torch.cuda, "is_available", lambda: False)

    def fake_load_coqui(self, model_name=None):
        load_order.append("coqui")
        raise RuntimeError("missing model")

    def fake_load_pyttsx3(self):
        load_order.append("pyttsx3")
        self.model = object()

    monkeypatch.setattr(TTSEngine, "_load_coqui", fake_load_coqui, raising=False)
    monkeypatch.setattr(TTSEngine, "_load_pyttsx3", fake_load_pyttsx3, raising=False)

    engine = TTSEngine(engine_type="auto", device="cuda")

    assert engine.engine_type == "pyttsx3"
    assert load_order == ["coqui", "pyttsx3"]


def test_coqui_synthesize_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_engine.torch.cuda, "is_available", lambda: False)

    class DummyCoqui:
        def __init__(self):
            self.speakers = ["demo"]
            self.languages = ["en"]

        def tts_to_file(self, text, file_path, **kwargs):
            path = pathlib.Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"RIFF....WAVE")

        def tts(self, text, **kwargs):
            return np.ones(8, dtype=np.float32)

    def fake_load_coqui(self, model_name=None):
        self.model = DummyCoqui()
        self.speakers = self.model.speakers
        self.languages = self.model.languages

    monkeypatch.setattr(TTSEngine, "_load_coqui", fake_load_coqui, raising=False)

    engine = TTSEngine(engine_type="coqui", device="cpu")

    out_file = tmp_path / "coqui.wav"
    engine.synthesize("hello world", output_path=out_file)

    assert out_file.exists()

    audio = engine.synthesize("hello again")
    assert isinstance(audio, np.ndarray)
    assert audio.shape[0] == 8
