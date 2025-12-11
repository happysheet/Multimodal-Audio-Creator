"""
VidMuse inference wrapper using the official long-short-term pipeline.
"""

from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.config import MODELS_DIR, OUTPUT_DIR, VIDMUSE_CONFIG
from .vidmuse_model import VidMuse
from .video_processor import VideoProcessor, merge_video_audio

logger = logging.getLogger(__name__)


class VidMuseGenerator:
    """High-fidelity VidMuse generator (official pipeline)."""

    def __init__(
        self,
        device: str = "cuda",
        use_float16: bool = True,
        lora_path: Optional[str] = None,
    ):
        self.config = VIDMUSE_CONFIG
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.use_float16 = use_float16 and self.device == "cuda"
        self.cache_dir = self.config.get("cache_dir", MODELS_DIR / "vidmuse")
        self.lora_path = lora_path

        # Ensure AudioCraft caches large weights under src/models/vidmuse
        os.environ.setdefault("AUDIOCRAFT_CACHE_DIR", str(self.cache_dir))

        self.processor = VideoProcessor(frame_size=self.config["video_frame_size"])
        self.model: Optional[VidMuse] = None

    def _load_model(self):
        if self.model is not None:
            return

        logger.info("Loading VidMuse model from %s", self.config["pretrained_id"])
        self.model = VidMuse.get_pretrained(
            self.config["pretrained_id"], device=self.device, lora_path=self.lora_path
        )
        self._apply_generation_params(self.config["max_duration"])
        logger.info("VidMuse model loaded successfully")

    def _apply_generation_params(self, duration: float):
        assert self.model is not None
        params = self.config
        self.model.set_generation_params(
            use_sampling=True,
            top_k=params["top_k"],
            top_p=params["top_p"],
            temperature=params["temperature"],
            duration=min(duration, params["max_duration"]),
            cfg_coef=params["cfg_coef"],
            two_step_cfg=False,
            extend_stride=params["extend_stride"],
        )

    def generate(
        self,
        video_path: Union[str, Path],
        duration: Optional[float] = None,
        merge_video: bool = False,
        merged_video_path: Optional[Union[str, Path]] = None,
        progress: bool = True,
        rerank: int = 1,
    ) -> np.ndarray:
        """
        Generate music aligned with the given video.

        Args:
            video_path: Source video.
            duration: Optional override duration (seconds). Defaults to video length.
            merge_video: If True, muxes generated audio back into the input video.
            merged_video_path: Optional path for merged mp4 (required if merge_video is True).
            progress: Show token generation progress.
            rerank: Number of candidate audios to generate for AV reranking (>=1) using CLAP cosine.
        """
        video_path = Path(video_path)
        self._load_model()
        assert self.model is not None

        logger.info("Processing video frames via VidMuse video processor...")
        local_tensor, global_tensor, video_duration = self.processor.process(
            video_path=video_path,
            target_fps=self.config["short_term_fps"],
            global_mode=self.config["global_mode"],
            global_num_frames=self.config["long_term_frames"],
        )

        target_duration = duration or video_duration
        self._apply_generation_params(target_duration)

        def _generate_once() -> np.ndarray:
            tensors = [
                local_tensor.to(self.device, dtype=torch.float32),
                global_tensor.to(self.device, dtype=torch.float32),
            ]
            outputs = self.model.generate(tensors, progress=progress, return_tokens=False)
            return outputs.detach().cpu().float().numpy()

        logger.info("Generating music (duration=%.2fs)...", target_duration)
        if rerank <= 1:
            audio = _generate_once()
        else:
            audio = self._generate_with_rerank(
                video_path=video_path,
                generate_fn=_generate_once,
                sample_rate=self.model.sample_rate,
                rerank=rerank,
            )

        if merge_video:
            if merged_video_path is None:
                raise ValueError("merged_video_path is required when merge_video=True")
            tmp_audio = OUTPUT_DIR / "vidmuse_temp.wav"
            tmp_audio.parent.mkdir(parents=True, exist_ok=True)
            self.save(audio, tmp_audio, sample_rate=self.model.sample_rate)
            merge_video_audio(video_path, tmp_audio, Path(merged_video_path))

        return audio

    def save(self, audio: np.ndarray, output_path: Union[str, Path], sample_rate: Optional[int] = None):
        """Save generated waveform."""
        import soundfile as sf

        if sample_rate is None:
            if self.model is None:
                raise RuntimeError("Model must be loaded to infer sample rate.")
            sample_rate = self.model.sample_rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        waveform = audio[0, 0] if audio.ndim == 3 else audio
        sf.write(output_path, waveform, sample_rate)
        logger.info("Audio saved to %s", output_path)

    def clear_memory(self):
        """Release GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VidMuse memory cleared")

    def _generate_with_rerank(
        self,
        video_path: Path,
        generate_fn,
        sample_rate: int,
        rerank: int,
    ) -> np.ndarray:
        """
        Generate multiple candidates and pick the best via CLAP cosine (AV alignment).
        """
        from src.utils.av_alignment import score as av_score

        rerank = max(1, int(rerank))
        tempdir = Path(tempfile.mkdtemp(prefix="vidmuse_rerank_", dir=str(OUTPUT_DIR)))

        candidates = []
        logger.info("Rerank enabled: generating %d candidates...", rerank)
        for idx in range(rerank):
            audio = generate_fn()
            cand_path = tempdir / f"candidate_{idx}.wav"
            self.save(audio, cand_path, sample_rate=sample_rate)
            score_val = av_score(str(video_path), str(cand_path), device=self.device)
            logger.info("Candidate %d score: %.4f", idx, score_val)
            candidates.append((score_val, cand_path, audio))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path, best_audio = candidates[0]
        logger.info("Best candidate: %s (score=%.4f)", best_path.name, best_score)

        # Cleanup all temp files/dir
        for _, cand_path, _ in candidates:
            try:
                cand_path.unlink()
            except OSError:
                pass
        try:
            tempdir.rmdir()
        except OSError:
            pass

        return best_audio
