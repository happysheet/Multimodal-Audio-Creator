"""
Audio-Video alignment scoring using CLAP (audio) and CLIP (video).

Expose a lightweight `score(video_path, audio_path, device="auto", model="clap")`
utility that returns a cosine similarity float in the range [-1, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _select_device(device: str) -> Tuple[str, bool]:
    """
    Resolve device and whether to run in float16.
    Returns (device_str, use_fp16).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device.startswith("cuda")
    return device, use_fp16


@dataclass
class _Models:
    clap: object  # laion-clap CLAP_Module
    clip_model: torch.nn.Module
    clip_preprocess: object
    use_fp16: bool
    device: str


@lru_cache(maxsize=2)
def _load_models(device: str, use_fp16: bool) -> _Models:
    """
    Lazy-load CLAP and CLIP models once per device/precision combo.
    """
    try:
        from laion_clap import CLAP_Module
    except ImportError as exc:  # pragma: no cover - dependency issues
        raise RuntimeError("laion-clap is required for AV alignment") from exc

    try:
        import clip  # OpenAI CLIP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai-CLIP is required for AV alignment") from exc

    logger.info("Loading CLAP and CLIP models (device=%s, fp16=%s)...", device, use_fp16)

    clap = CLAP_Module(enable_fusion=False, device=device)
    # load default public checkpoint (downloaded/cached by the package)
    clap.load_ckpt()
    # Force float32 for stability
    clap.model.float()

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    # Force float32 for stability with downstream layer norms
    clip_model = clip_model.float()

    clip_model.eval()
    return _Models(
        clap=clap,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        use_fp16=use_fp16,
        device=device,
    )


def _extract_video_embedding(
    video_path: Path, models: _Models, num_frames: int = 8
) -> torch.Tensor:
    """
    Sample frames uniformly and encode with CLIP. Returns a normalized 1xD tensor.
    """
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("decord is required for video frame decoding") from exc
    from PIL import Image

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise ValueError(f"No frames decoded from video: {video_path}")

    # Uniformly sample up to num_frames frames
    step = max(total // num_frames, 1)
    indices = list(range(0, total, step))[:num_frames]
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)

    # Preprocess for CLIP
    imgs = [Image.fromarray(f) for f in frames]
    tensors = [models.clip_preprocess(img) for img in imgs]
    video_tensor = torch.stack(tensors).to(models.device)
    if models.use_fp16 and models.device.startswith("cuda"):
        video_tensor = video_tensor.half()

    with torch.no_grad():
        feats = models.clip_model.encode_image(video_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        video_emb = feats.mean(dim=0, keepdim=True)
    return video_emb


def _extract_audio_embedding(audio_path: Path, models: _Models) -> torch.Tensor:
    """
    Encode audio file with CLAP. Returns a normalized 1xD tensor.
    """
    with torch.no_grad():
        audio_emb = models.clap.get_audio_embedding_from_filelist(
            x=[str(audio_path)],
            use_tensor=True,
        )
        # Ensure tensor on target device
        audio_emb = audio_emb.to(models.device)
        if models.use_fp16 and models.device.startswith("cuda"):
            audio_emb = audio_emb.half()
        audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
    return audio_emb


def score(
    video_path: str,
    audio_path: str,
    device: str = "auto",
    model: str = "clap",
) -> float:
    """
    Compute AV alignment score using cosine similarity.

    Args:
        video_path: Path to the video file.
        audio_path: Path to the audio file.
        device: 'auto', 'cuda', or 'cpu'.
        model: Placeholder for future model selection (currently only 'clap').

    Returns:
        Cosine similarity in [-1, 1] as a Python float.
    """
    if model != "clap":
        raise ValueError(f"Unsupported model '{model}', only 'clap' is available.")

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    resolved_device, use_fp16 = _select_device(device)
    models = _load_models(resolved_device, use_fp16)

    video_emb = _extract_video_embedding(video_path, models)
    audio_emb = _extract_audio_embedding(audio_path, models)

    similarity = F.cosine_similarity(video_emb, audio_emb).mean()
    return float(similarity.item())


__all__ = ["score"]
