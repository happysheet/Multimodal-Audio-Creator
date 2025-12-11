"""
Lightweight video captioner used by the autopipeline mode.
Default backend: OpenAI vision (via chat.completions with image inputs).
Falls back to a stub caption when API/key is missing or any error occurs.
"""

import base64
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptionerConfig:
    backend: str = "openai"  # "openai" or "stub"
    max_len: int = 40
    num_frames: int = 4
    device: str = "auto"
    openai_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None


class Captioner:
    """
    Generate a short narration-ready caption from a video.
    """

    def __init__(self, backend: str = "openai", max_len: int = 40, num_frames: int = 4, device: str = "auto"):
        self.cfg = CaptionerConfig(backend=backend, max_len=max_len, num_frames=max(1, num_frames), device=device)

    def generate(self, video_path: str) -> str:
        """
        Generate caption text for a video.
        """
        video_path = str(video_path)
        if self.cfg.backend == "openai":
            try:
                return self._caption_with_openai(video_path)
            except Exception as exc:  # pragma: no cover - best-effort path
                logger.warning(f"OpenAI captioner failed ({exc}); falling back to stub.")

        return self._fallback_caption(video_path)

    # --------------------------------------------------------------------- #
    # OpenAI backend
    # --------------------------------------------------------------------- #
    def _caption_with_openai(self, video_path: str) -> str:
        if not self.cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        frames = self._sample_frames(video_path, self.cfg.num_frames)
        if not frames:
            raise RuntimeError("Failed to sample frames for captioning")

        from openai import OpenAI

        client = OpenAI(
            api_key=self.cfg.openai_api_key,
            base_url=self.cfg.openai_base_url,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a concise video captioner. Summarize the scene for narration.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe the video in one sentence, <= {self.cfg.max_len} words, suitable as a narration script.",
                    },
                    *frames,
                ],
            },
        ]

        resp = client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=messages,
            max_tokens=120,
            temperature=0.4,
        )

        text = resp.choices[0].message.content.strip()
        return text

    def _sample_frames(self, video_path: str, num_frames: int) -> List[dict]:
        """
        Sample frames uniformly and return OpenAI-compatible image payloads.
        """
        try:
            from decord import VideoReader, cpu
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(f"Missing dependencies for frame sampling: {exc}")

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        idx = np.linspace(0, total - 1, num_frames, dtype=int)
        images: List[dict] = []
        for i in idx:
            frame = vr[i].asnumpy()
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        return images

    # --------------------------------------------------------------------- #
    # Fallback
    # --------------------------------------------------------------------- #
    def _fallback_caption(self, video_path: str) -> str:
        name = Path(video_path).stem.replace("_", " ")
        return f"A short narration describing the visuals in {name}."
