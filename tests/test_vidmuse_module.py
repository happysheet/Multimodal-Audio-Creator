import numpy as np
import torch

import src.modules.vidmuse.inference as vidmuse_inference


def test_vidmuse_initializes_on_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(vidmuse_inference.torch.cuda, "is_available", lambda: False)

    generator = vidmuse_inference.VidMuseGenerator(device="cuda")

    assert generator.device == "cpu"
    assert generator.model is None
    assert generator.cache_dir == generator.config["cache_dir"]


def test_generate_requires_merge_path(monkeypatch, tmp_path):
    generator = vidmuse_inference.VidMuseGenerator(device="cpu")

    # Stub out heavy components
    class DummyModel:
        sample_rate = 32000

        def __init__(self):
            self.params = None

        def set_generation_params(self, **kwargs):
            self.params = kwargs

        def generate(self, tensors, progress=False, return_tokens=False):
            assert len(tensors) == 2
            assert tensors[0].shape[0] == tensors[1].shape[0]
            return torch.zeros(1, 1, self.sample_rate, dtype=torch.float32)

    dummy_model = DummyModel()
    generator.model = dummy_model
    generator._load_model = lambda: None  # noqa: E731

    def fake_process(*_, **__):
        local = torch.zeros(3, 4, 224, 224)
        global_ = torch.zeros(3, 32, 224, 224)
        return local, global_, 1.0

    monkeypatch.setattr(generator.processor, "process", fake_process)

    # merge_video=True without path should raise
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    try:
        generator.generate(video_path, merge_video=True)
        raised = False
    except ValueError:
        raised = True

    assert raised

    audio = generator.generate(video_path, merge_video=False)
    assert isinstance(audio, np.ndarray)
    assert audio.shape[-1] == dummy_model.sample_rate
