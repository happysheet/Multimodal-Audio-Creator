"""
Video preprocessing utilities for VidMuse.
Ported from the official VidMuse repository (https://huggingface.co/Zeyue7/VidMuse).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
from decord import VideoReader, cpu
import einops
import torchvision.transforms as transforms
try:  # moviepy 2.x
    from moviepy import AudioFileClip, VideoFileClip
except ImportError:  # moviepy 1.x fallback
    from moviepy.editor import AudioFileClip, VideoFileClip


class VideoProcessor:
    """Extract long/short-term frame tensors expected by VidMuse."""

    def __init__(self, frame_size: Tuple[int, int] = (224, 224)):
        self.resize_transform = transforms.Resize(frame_size)

    def get_video_duration(self, video_path: Path) -> float:
        """Return video duration (seconds)."""
        clip = VideoFileClip(str(video_path))
        try:
            return float(clip.duration)
        finally:
            clip.close()

    @staticmethod
    def _adjust_duration(video_tensor: torch.Tensor, duration: float, target_fps: int) -> torch.Tensor:
        """Pad/trim tensor so its length matches target duration."""
        current = video_tensor.shape[1]
        target = int(duration * target_fps)
        if current > target:
            return video_tensor[:, :target]
        if current < target:
            last_frame = video_tensor[:, -1:]
            repeat = target - current
            pad = last_frame.repeat(1, repeat, 1, 1)
            return torch.cat((video_tensor, pad), dim=1)
        return video_tensor

    def _read_frames(
        self,
        filepath: Path,
        duration: float,
        target_fps: int,
        global_mode: str,
        global_num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vr = VideoReader(str(filepath), ctx=cpu(0))
        fps = vr.get_avg_fps()
        frame_count = len(vr)

        if duration > 0:
            total = target_fps * duration
            frame_interval = int(math.ceil(fps / target_fps))
            start_frame = 0
            end_frame = int(frame_interval * total)
            frame_ids = list(range(start_frame, min(end_frame, frame_count), frame_interval))
        else:
            frame_ids = list(range(0, frame_count, int(math.ceil(fps / target_fps))))

        local_frames = vr.get_batch(frame_ids)
        local_frames = torch.from_numpy(local_frames.asnumpy()).permute(0, 3, 1, 2)
        local_frames = [self.resize_transform(frame) for frame in local_frames]
        local_video = torch.stack(local_frames)
        local_video = einops.rearrange(local_video, "t c h w -> c t h w")
        local_video = self._adjust_duration(local_video, duration, target_fps)

        if global_mode != "average":
            raise ValueError(f"Unsupported global_mode: {global_mode}")

        global_frame_ids = torch.linspace(0, frame_count - 1, global_num_frames).long()
        global_frames = vr.get_batch(global_frame_ids)
        global_frames = torch.from_numpy(global_frames.asnumpy()).permute(0, 3, 1, 2)
        global_frames = [self.resize_transform(frame) for frame in global_frames]
        global_video = torch.stack(global_frames)
        global_video = einops.rearrange(global_video, "t c h w -> c t h w")

        if global_video.shape[1] != global_num_frames:
            raise RuntimeError(f"Global frames mismatch: expected {global_num_frames}, got {global_video.shape}")
        return local_video, global_video

    def process(
        self,
        video_path: Path,
        target_fps: int = 2,
        global_mode: str = "average",
        global_num_frames: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return (local_tensor, global_tensor, duration_seconds)."""
        duration = self.get_video_duration(video_path)
        local_video_tensor, global_video_tensor = self._read_frames(
            video_path,
            duration=duration,
            target_fps=target_fps,
            global_mode=global_mode,
            global_num_frames=global_num_frames,
        )
        return local_video_tensor, global_video_tensor, duration


def merge_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Merge generated audio back into the source video."""
    video = VideoFileClip(str(video_path)).without_audio()
    audio = AudioFileClip(str(audio_path))
    final_video = video.set_audio(audio)
    final_video.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
