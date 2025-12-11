import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.utils import av_alignment


def test_score_prefers_closer_audio(monkeypatch, tmp_path):
    def fake_load_models(device, use_fp16):
        class Dummy:
            def __init__(self, device, use_fp16):
                self.device = device
                self.use_fp16 = use_fp16

        return Dummy(device, use_fp16)

    def fake_video_emb(video_path, models):
        return torch.tensor([[1.0, 0.0]])

    def fake_audio_emb(audio_path, models):
        if "good" in str(audio_path):
            return torch.tensor([[1.0, 0.0]])
        return torch.tensor([[0.0, 1.0]])

    monkeypatch.setattr(av_alignment, "_load_models", fake_load_models)
    monkeypatch.setattr(av_alignment, "_extract_video_embedding", fake_video_emb)
    monkeypatch.setattr(av_alignment, "_extract_audio_embedding", fake_audio_emb)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"stub")
    audio_good = tmp_path / "good.wav"
    audio_good.write_bytes(b"stub")
    audio_bad = tmp_path / "bad.wav"
    audio_bad.write_bytes(b"stub")

    good_score = av_alignment.score(str(video), str(audio_good), device="cpu")
    bad_score = av_alignment.score(str(video), str(audio_bad), device="cpu")

    assert isinstance(good_score, float)
    assert good_score > bad_score
