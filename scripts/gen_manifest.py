"""
Generic manifest generator for paired video/audio training data.

Scans video files, finds audio with matching stem, and writes a CSV manifest:
video_path,audio_path,start,end
`start` / `end` are left blank by default (full clip).

Example:
  conda activate audio-creator
  python scripts/gen_manifest.py \
    --video-dir data/trainingdata \
    --audio-dir data/trainingdata \
    --output data/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Set


def find_matches(
    video_dir: Path,
    audio_dir: Path,
    video_exts: Set[str],
    audio_exts: Set[str],
    recursive: bool,
) -> List[tuple[Path, Path]]:
    pattern = "**/*" if recursive else "*"
    matches: List[tuple[Path, Path]] = []
    for vpath in video_dir.glob(pattern):
        if not vpath.is_file() or vpath.suffix.lower() not in video_exts:
            continue
        stem = vpath.stem
        for aext in audio_exts:
            apath = audio_dir / vpath.relative_to(video_dir).with_suffix(aext)
            if apath.exists():
                matches.append((vpath.resolve(), apath.resolve()))
                break
        else:
            # fallback: search by stem anywhere under audio_dir
            for apath in audio_dir.glob(f"**/{stem}*"):
                if apath.is_file() and apath.suffix.lower() in audio_exts:
                    matches.append((vpath.resolve(), apath.resolve()))
                    break
    return matches


def write_csv(rows: Iterable[tuple[Path, Path]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "audio_path", "start", "end"])
        count = 0
        for v, a in rows:
            writer.writerow([str(v), str(a), "", ""])
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate paired video/audio manifest CSV.")
    parser.add_argument("--video-dir", type=Path, default=Path("data/trainingdata"), help="Directory with video files.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Directory with audio files (defaults to video-dir).")
    parser.add_argument("--video-exts", type=str, default=".mp4,.mov,.mkv", help="Comma-separated video extensions.")
    parser.add_argument("--audio-exts", type=str, default=".wav,.mp3,.flac", help="Comma-separated audio extensions.")
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"), help="Output CSV path.")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive search.")

    args = parser.parse_args()

    video_dir = args.video_dir
    audio_dir = args.audio_dir or video_dir
    recursive = not args.no_recursive
    video_exts = {e.strip().lower() for e in args.video_exts.split(",") if e.strip()}
    audio_exts = {e.strip().lower() for e in args.audio_exts.split(",") if e.strip()}

    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")
    if not audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    pairs = find_matches(video_dir, audio_dir, video_exts, audio_exts, recursive)
    if not pairs:
        raise SystemExit("No video/audio pairs found.")

    count = write_csv(pairs, args.output)
    print(f"Wrote {count} pairs to {args.output}")


if __name__ == "__main__":
    main()
