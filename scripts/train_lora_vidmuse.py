"""
LoRA fine-tuning for VidMuse (MusicGen LM) with video conditioning.

This is a light-weight training script intended for small datasets and 8GB GPUs.
It:
 - Loads paired (video, audio) samples.
 - Extracts video features via VidMuse VideoProcessor.
 - Encodes audio to EnCodec tokens.
 - Trains LoRA adapters on the VidMuse LM to predict next audio tokens.

Examples
--------
conda activate audio-creator
python scripts/train_lora_vidmuse.py \
  --data-root data/train \
  --epochs 3 --rank 4 --alpha 16 \
  --lr 1e-4 --batch-size 1 --grad-accum 4 \
  --output src/models/lora/vidmuse_lora.pt

Manifest CSV/JSON (optional)
----------------------------
If provided via --manifest, must contain columns/fields:
  video_path,audio_path,start,end   # start/end optional (seconds)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import VIDMUSE_CONFIG
from src.modules.vidmuse.video_processor import VideoProcessor
from src.modules.vidmuse.vidmuse_model import VidMuse
from src.modules.vidmuse.lora import add_lora_adapters, extract_lora_state


def load_manifest(manifest: Path) -> List[Tuple[Path, Path, Optional[float], Optional[float]]]:
    samples: List[Tuple[Path, Path, Optional[float], Optional[float]]] = []
    if manifest.suffix.lower() == ".csv":
        import csv

        with manifest.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(
                    (
                        Path(row["video_path"]),
                        Path(row["audio_path"]),
                        float(row["start"]) if row.get("start") else None,
                        float(row["end"]) if row.get("end") else None,
                    )
                )
    else:
        data = json.loads(manifest.read_text())
        for item in data:
            samples.append(
                (
                    Path(item["video_path"]),
                    Path(item["audio_path"]),
                    float(item.get("start")) if item.get("start") is not None else None,
                    float(item.get("end")) if item.get("end") is not None else None,
                )
            )
    return samples


class VidMuseLoraDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path, Optional[float], Optional[float]]],
        video_processor: VideoProcessor,
        sample_rate: int,
    ):
        self.samples = samples
        self.video_processor = video_processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, audio_path, start, end = self.samples[idx]
        local_tensor, global_tensor, video_duration = self.video_processor.process(video_path)

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if start is not None or end is not None:
            start = start or 0.0
            end = end or video_duration
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            audio = audio[start_idx:end_idx]
        # resample if needed
        if sr != self.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        audio_tensor = torch.from_numpy(audio).float()[None, :]  # (1, T)
        return {
            "audio": audio_tensor,
            "local": local_tensor,
            "global": global_tensor,
        }


@dataclass
class TrainConfig:
    data_root: Path
    manifest: Optional[Path]
    output: Path
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    rank: int
    alpha: int
    device: str


def create_dataloader(cfg: TrainConfig, processor: VideoProcessor, sample_rate: int) -> DataLoader:
    if cfg.manifest:
        raw = load_manifest(cfg.manifest)
    else:
        raw = []
        for video_file in (cfg.data_root).glob("**/*.mp4"):
            audio_file = video_file.with_suffix(".wav")
            if audio_file.exists():
                raw.append((video_file, audio_file, None, None))

    samples = [
        (v, a, s, e)
        for (v, a, s, e) in raw
        if Path(v).exists() and Path(a).exists()
    ]
    if not samples:
        raise RuntimeError("No valid video/audio pairs found for LoRA training.")
    dataset = VidMuseLoraDataset(samples, processor, sample_rate)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)


def shift_tokens(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forcing shift: input=[:-1], target=[1:]."""
    return tokens[..., :-1], tokens[..., 1:]


def build_conditions(batch_size: int, sample_rate: int, device: str):
    from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition

    conds: List[ConditioningAttributes] = []
    for _ in range(batch_size):
        attr = ConditioningAttributes(text={"description": None})
        attr.wav["self_wav"] = WavCondition(
            torch.zeros((1, 1, 1), device=device),
            torch.tensor([0], device=device),
            sample_rate=[sample_rate],
            path=[None],
        )
        conds.append(attr)
    return conds


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    # Load base VidMuse
    vidmuse = VidMuse.get_pretrained(
        name=str(VIDMUSE_CONFIG.get("pretrained_id", "Zeyue7/VidMuse")),
        device=cfg.device,
    )
    compression = vidmuse.compression_model
    lm = vidmuse.lm.float()
    lm.train()

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    add_lora_adapters(lm, target_modules=target_modules, rank=cfg.rank, alpha=cfg.alpha, dropout=0.05)

    # Freeze non-LoRA params
    for name, p in lm.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    processor = VideoProcessor(frame_size=(224, 224))
    loader = create_dataloader(cfg, processor, compression.sample_rate)

    trainable = [p for p in lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            audio = batch["audio"].to(device)
            local = batch["local"].to(device)
            global_v = batch["global"].to(device)

            with torch.no_grad():
                codes, _ = compression.encode(audio.float())
            inputs, targets = shift_tokens(codes.long())

            conds = build_conditions(inputs.shape[0], compression.sample_rate, cfg.device)
            # Forward
            logits = lm(inputs, conds, [local.float(), global_v.float()])
            vocab = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab),
                targets.reshape(-1).long(),
                ignore_index=-100,
            )
            loss.backward()

            if (global_step + 1) % cfg.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                print(f"Epoch {epoch+1} step {global_step} loss {loss.item():.4f}")

    # Save LoRA adapter
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    lora_state = extract_lora_state(lm)
    payload = {
        "lora_state": lora_state,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "target_modules": target_modules,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cfg.output)
    print(f"[DONE] LoRA saved to {cfg.output}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/train"))
    p.add_argument("--manifest", type=Path, help="CSV or JSON manifest with video_path,audio_path[,start,end]")
    p.add_argument("--output", type=Path, default=Path("src/models/lora/vidmuse_lora.pt"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        manifest=args.manifest,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
