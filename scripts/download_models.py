"""
Model Download Script
Downloads pre-trained assets for VidMuse
"""

import os
import sys
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Simple ASCII status tags
OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"
INFO = "[INFO]"

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"


def download_vidmuse():
    """Download VidMuse model"""
    print("\n" + "=" * 80)
    print("Downloading VidMuse Model")
    print("=" * 80)

    from src.config import MODELS_DIR, MODEL_URLS

    vidmuse_dir = MODELS_DIR / "vidmuse"
    vidmuse_dir.mkdir(parents=True, exist_ok=True)

    print("\n[WARN]  VidMuse uses AudioCraft/MusicGen as the base model")
    print("The core checkpoints are hosted on Hugging Face and cached locally.")
    print(f"Models will be cached in: {vidmuse_dir}")
    os.environ.setdefault("AUDIOCRAFT_CACHE_DIR", str(vidmuse_dir))

    # Check audiocraft availability without forcing installation
    try:
        import audiocraft

        print("[OK] AudioCraft already installed")
    except Exception as exc:
        print(
            "[WARN]  AudioCraft is not available in this environment. "
            "Skip automatic install so you can manage dependencies manually."
        )
        print(f"[WARN]  Import error: {exc}")
        print("[INFO]  Install audiocraft (and its pinned Torch stack) in a separate env if needed.")

    # Download CLIP model
    try:
        import clip

        print("[OK] CLIP already installed")

        # Trigger CLIP model download
        print("Downloading CLIP ViT-B/32 model...")
        clip.load("ViT-B/32", device="cpu", download_root=str(MODELS_DIR / "clip"))
        print("[OK] CLIP model downloaded")

    except ImportError:
        print("Installing CLIP...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"],
            stdout=subprocess.DEVNULL,
        )
        import clip

        clip.load("ViT-B/32", device="cpu", download_root=str(MODELS_DIR / "clip"))
        print("[OK] CLIP installed and model downloaded")

    repo_cfg = MODEL_URLS.get("vidmuse", {})
    repo_id = repo_cfg.get("repo_id")
    hf_files = repo_cfg.get("hf_files", [])
    if repo_id and hf_files:
        print("\n[INFO]  Downloading VidMuse weights from Hugging Face...")
        for filename in hf_files:
            target_path = vidmuse_dir / filename
            if target_path.exists():
                print(f"[OK] {filename} already present")
                continue
            print(f"[INFO]  Fetching {filename} ...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(vidmuse_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[OK]  Saved {filename} to {target_path}")
    else:
        print("\n[WARN]  VidMuse repo_id not configured; skipping HF weight download")

    print(f"\n[OK] VidMuse setup complete")


def check_dependencies():
    """Check required dependencies"""
    print("\n" + "=" * 80)
    print("Checking Dependencies")
    print("=" * 80)

    required = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "soundfile": "SoundFile",
        "librosa": "Librosa",
        "pydub": "Pydub",
        "cv2": "OpenCV (opencv-python)",
        "decord": "Decord",
        "moviepy": "MoviePy",
        "einops": "Einops",
    }

    missing = []

    for module, name in required.items():
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[FAIL] {name} - MISSING")
            missing.append(module)

    if missing:
        print(f"\n[WARN]  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n[OK] All required dependencies installed")
    return True


def main():
    """Main download script"""
    print("=" * 80)
    print("MULTIMODAL AUDIO CREATOR - MODEL DOWNLOAD")
    print("=" * 80)

    # Check dependencies
    if not check_dependencies():
        print("\n[WARN]  Please install dependencies first:")
        print("    pip install -r requirements.txt\n")
        return 1

    # Download models
    try:
        download_vidmuse()

        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)

        print("\n[OK] Model download completed successfully!")
        print("\nNext steps:")
        print("1. Test the system: python examples/minimal_example.py")
        print("2. Run your first generation:")
        print("   python main.py --mode tts --text 'Hello world' --output_path output.wav")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\n[WARN]  Download interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n[FAIL] Download failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
