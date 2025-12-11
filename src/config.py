"""
Global Configuration for Multimodal Audio Creator
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent

if load_dotenv:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, DATA_DIR, TEMP_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

# Device settings
USE_CUDA = True  # Set to False to force CPU mode
DEVICE = "cuda" if USE_CUDA else "cpu"

# Memory optimization flags
ENABLE_MEMORY_EFFICIENT_MODE = True
USE_SEQUENTIAL_PROCESSING = True  # Process modules one at a time
CLEAR_CACHE_BETWEEN_MODULES = True

# Precision settings (use float16 to save memory)
USE_FLOAT16 = True  # Mixed precision training/inference
TORCH_DTYPE = "float16" if USE_FLOAT16 else "float32"

# ============================================================================
# VIDMUSE CONFIGURATION (Video-to-Music)
# ============================================================================

VIDMUSE_CONFIG = {
    # Model settings
    "model_path": MODELS_DIR / "vidmuse" / "model.pt",
    "model_size": "small",  # Options: "small" (6GB), "medium" (10GB), "large" (16GB)
    "pretrained_id": "Zeyue7/VidMuse",
    "cache_dir": MODELS_DIR / "vidmuse",

    # Video processing
    "video_frame_size": (224, 224),
    "short_term_fps": 2,  # Frames per second for short-term analysis
    "long_term_frames": 32,  # Number of frames for long-term analysis
    "global_mode": "average",

    # Audio generation
    "sample_rate": 32000,
    "max_duration": 60,  # Maximum music duration in seconds
    "temperature": 0.8,  # Creativity control (0.0-1.0)
    "top_k": 250,
    "top_p": 0.95,
    "cfg_coef": 3.0,
    "extend_stride": 29.5,

    # Memory optimization
    "batch_size": 1,  # Reduce if OOM
    "max_batch_size": 1,
    "use_cache": True,
}

# ============================================================================
# TTS CONFIGURATION (Text-to-Speech) - Multiple Engine Support
# ============================================================================

TTS_CONFIG = {
    # Engine selection
    "default_engine": "auto",  # Options: "auto", "coqui", "pyttsx3"

    # Output settings
    "output_sample_rate": 24000,
    "audio_format": "wav",

    # Coqui TTS settings
    "coqui": {
        # Higher-quality single-speaker default (no speaker prompt required):
        "default_model": "tts_models/en/ljspeech/glow-tts",
        # Alternatives:
        # "tts_models/en/ljspeech/tacotron2-DDC"
        # "tts_models/multilingual/multi-dataset/xtts_v2"  # requires speaker reference, not default
        # "tts_models/en/vctk/vits"  # multi-speaker, requires speaker id
        "use_gpu": True,
        "speakers": [],  # Auto-detected from model
        "languages": [],  # Auto-detected from model
    },

    # pyttsx3 settings (offline, lightweight)
    "pyttsx3": {
        "default_voice": "male",  # Options: "male", "female"
        "default_rate": 150,  # Words per minute
        "default_volume": 1.0,  # 0.0 to 1.0
    },

    # Speech generation defaults
    "default_speed": 1.0,  # Speech speed multiplier (0.5-2.0)

    # Memory optimization
    "batch_size": 1,
    "use_cache": True,
}

# ============================================================================
# WAVJOURNEY CONFIGURATION (LLM-Driven Planning)
# ============================================================================

WAVJOURNEY_CONFIG = {
    # LLM settings
    "llm_provider": "openai",  # OpenAI only
    "llm_model": "gpt-4",
    "openai_base_url": os.getenv("OPENAI_BASE_URL") or None,  # Only from environment

    # API keys (set via environment variables)
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),

    # Audio composition
    "max_audio_clips": 10,
    "default_fade_duration": 0.5,  # Fade in/out duration in seconds

    # Script generation
    "max_script_lines": 999,
    "temperature": 0.7,
    "max_tokens": 2000,
}

# ============================================================================
# AUDIO MIXING CONFIGURATION
# ============================================================================

AUDIO_MIXER_CONFIG = {
    # Output settings
    "output_sample_rate": 44100,  # Standard CD quality
    "output_format": "wav",
    "output_bit_depth": 16,

    # Mixing parameters
    "music_volume": 0.6,  # Background music volume (0.0-1.0)
    "speech_volume": 1.0,  # Speech volume (0.0-1.0)
    "sfx_volume": 0.8,    # Sound effects volume (0.0-1.0)

    # Audio processing
    "normalize_audio": True,
    "target_loudness": -16.0,  # LUFS (Loudness Units Full Scale)
    "apply_compression": True,

    # Fade effects
    "fade_in_duration": 0.5,
    "fade_out_duration": 1.0,
    "crossfade_duration": 2.0,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": True,
    "log_file": PROJECT_ROOT / "logs" / "audio_creator.log",
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL DOWNLOAD URLS
# ============================================================================

MODEL_URLS = {
    "vidmuse": {
        "small": "https://huggingface.co/Zeyue7/VidMuse/resolve/main/vidmuse_small.pt",
        "medium": "https://huggingface.co/Zeyue7/VidMuse/resolve/main/vidmuse_medium.pt",
        "large": "https://huggingface.co/HKUSTAudio/VidMuse/resolve/main/vidmuse_large.pt",
        "repo_id": "Zeyue7/VidMuse",
        "hf_files": [
            "compression_state_dict.bin",
            "state_dict.bin",
        ],
    },
    "tts": {
        "coqui": "auto-download",  # Coqui TTS models are auto-downloaded
        "pyttsx3": "system",  # pyttsx3 uses system voices
    },
}

# ============================================================================
# RUNTIME FLAGS
# ============================================================================

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Performance profiling
ENABLE_PROFILING = False

# Cache management
CLEAR_TEMP_ON_EXIT = True  # Clear temporary files when done

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check CUDA availability
    if USE_CUDA:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA is enabled but not available. Set USE_CUDA=False to use CPU.")
        except ImportError:
            errors.append("PyTorch is not installed.")

    # Check API keys for WavJourney
    if WAVJOURNEY_CONFIG["llm_provider"] == "openai":
        if not WAVJOURNEY_CONFIG["openai_api_key"]:
            # Only warn, don't fail - user might not use WavJourney mode
            pass

    # Check model paths
    if not MODELS_DIR.exists():
        errors.append(f"Models directory not found: {MODELS_DIR}")

    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("Multimodal Audio Creator - Configuration")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Memory Efficient Mode: {ENABLE_MEMORY_EFFICIENT_MODE}")
    print(f"Float16 Precision: {USE_FLOAT16}")
    print(f"\nVidMuse Model Size: {VIDMUSE_CONFIG['model_size']}")
    print(f"TTS Engine: {TTS_CONFIG['default_engine']}")
    print(f"TTS Sample Rate: {TTS_CONFIG['output_sample_rate']} Hz")
    print(f"WavJourney LLM: {WAVJOURNEY_CONFIG['llm_provider']} ({WAVJOURNEY_CONFIG['llm_model']})")
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"Temp Directory: {TEMP_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    # Validate and print config when run directly
    if validate_config():
        print_config()
    else:
        print("\nWARNING  Configuration validation failed!")
