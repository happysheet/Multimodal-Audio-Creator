"""
Download a small subset of the VidMuse V2M dataset (YouTube IDs) for LoRA training.

What it does
------------
- Pulls a list of YouTube IDs from the HF repo HKUSTAudio/VidMuse-V2M-Dataset (default: V2M-20k.txt).
- Uses yt-dlp to download N videos with audio.
- Saves paired files as:
    data/v2m_subset/
      video_<idx>.mp4
      audio_<idx>.wav
      manifest.csv   # video_path,audio_path

Usage
-----
conda activate audio-creator
python scripts/download_v2m_subset.py \
  --num 50 \
  --output data/v2m_subset \
  --list-file V2M-20k.txt \
  --resolution 360

Notes
-----
- Requires ffmpeg in PATH (yt-dlp uses it to extract WAV).
- This script downloads from YouTube; ensure network/policy permits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from yt_dlp import YoutubeDL

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def load_ids_local(id_file: Path) -> List[str]:
    ids: List[str] = []
    with id_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line.split()[0])
    return ids


def load_ids_hf(repo: str, list_file: str) -> List[str]:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not installed; install or provide --id-file")
    local_path = hf_hub_download(repo_id=repo, filename=list_file)
    return load_ids_local(Path(local_path))


def download_one(
    yt_id: str,
    out_dir: Path,
    idx: int,
    resolution: int,
    cookies_from_browser: str | None,
    cookies_file: Path | None,
) -> tuple[Path, Path]:
    video_path = out_dir / f"video_{idx:05d}.mp4"
    audio_path = out_dir / f"audio_{idx:05d}.wav"

    ydl_opts = {
        "format": f"bestvideo[height<={resolution}]+bestaudio/best",
        "outtmpl": str(video_path.with_suffix(".%(ext)s")),
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file:
        ydl_opts["cookies"] = str(cookies_file)
    url = f"https://www.youtube.com/watch?v={yt_id}"
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # yt-dlp writes audio with same stem
    audio_candidate = video_path.with_suffix(".wav")
    if audio_candidate.exists():
        audio_candidate.rename(audio_path)
    return video_path, audio_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50, help="Number of samples to download")
    parser.add_argument("--output", type=Path, default=Path("data/yt_subset"))
    parser.add_argument("--id-file", type=Path, help="Local text file of YouTube IDs (one per line)")
    parser.add_argument("--hf-repo", type=str, help="Optional HF repo to fetch list file from")
    parser.add_argument("--list-file", type=str, default="youtube_ids.txt", help="List file name in HF repo")
    parser.add_argument("--resolution", type=int, default=360, help="Max video height to download")
    parser.add_argument("--cookies-from-browser", type=str, help="yt-dlp cookiesfrombrowser, e.g., chrome")
    parser.add_argument("--cookies-file", type=Path, help="yt-dlp --cookies <netscape.txt>")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.id_file and args.id_file.exists():
        ids = load_ids_local(args.id_file)
    elif args.hf_repo:
        ids = load_ids_hf(args.hf_repo, args.list_file)
    else:
        raise SystemExit("Please provide --id-file (local list of YouTube IDs) or --hf-repo + --list-file.")

    ids = ids[: args.num]

    rows = []
    for i, yt_id in enumerate(ids):
        try:
            v_path, a_path = download_one(
                yt_id,
                args.output,
                i,
                args.resolution,
                args.cookies_from_browser,
                args.cookies_file,
            )
            rows.append((v_path, a_path))
            print(f"[{i+1}/{len(ids)}] downloaded {yt_id}")
        except Exception as e:
            print(f"[{i+1}/{len(ids)}] failed {yt_id}: {e}")

    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "audio_path"])
        for v, a in rows:
            writer.writerow([v.as_posix(), a.as_posix()])

    print(f"\nDone. Saved {len(rows)} pairs to {args.output}, manifest: {manifest}")


if __name__ == "__main__":
    main()
