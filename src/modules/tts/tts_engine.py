"""
Universal TTS Engine
Supports multiple text-to-speech backends with unified interface
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Universal Text-to-Speech engine
    Supports multiple backends: coqui, pyttsx3
    """

    SUPPORTED_ENGINES = ["coqui", "pyttsx3", "auto"]

    def __init__(
        self,
        engine_type: Literal["coqui", "pyttsx3", "auto"] = "auto",
        device: str = "cuda",
        use_float16: bool = True,
        model_name: Optional[str] = None,
    ):
        """
        Initialize TTS engine

        Args:
            engine_type: TTS engine to use ("coqui", "pyttsx3", or "auto")
            device: Device to run on ('cuda' or 'cpu')
            use_float16: Use mixed precision (only for neural models)
            model_name: Specific model to use (engine-dependent)
        """
        from src.config import TTS_CONFIG

        self.config = TTS_CONFIG
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_float16 = use_float16 and self.device == "cuda"
        self.model = None
        self.engine_type = None

        logger.info(f"Initializing TTS Engine (requested: {engine_type}, device: {self.device})")

        # Auto-detect or load specified engine
        if engine_type == "auto":
            self._auto_select_engine()
        else:
            self._load_engine(engine_type, model_name)

    def _auto_select_engine(self):
        """Auto-select best available TTS engine"""
        logger.info("Auto-detecting available TTS engines...")

        # Try engines in order of preference
        for engine in ["coqui", "pyttsx3"]:
            try:
                self._load_engine(engine)
                logger.info(f"Auto-selected: {engine}")
                return
            except Exception as e:
                logger.debug(f"Engine {engine} not available: {e}")
                continue

        raise RuntimeError(
            "No TTS engine available. Please install one:\n"
            "  - Coqui TTS: pip install TTS\n"
            "  - pyttsx3: pip install pyttsx3"
        )

    def _load_engine(self, engine_type: str, model_name: Optional[str] = None):
        """Load specified TTS engine"""
        if engine_type == "coqui":
            self._load_coqui(model_name)
        elif engine_type == "pyttsx3":
            self._load_pyttsx3()
        else:
            raise ValueError(f"Unsupported engine: {engine_type}")

        self.engine_type = engine_type

    def _load_coqui(self, model_name: Optional[str] = None):
        """Load Coqui TTS"""
        from TTS.api import TTS

        if model_name is None:
            model_name = self.config["coqui"]["default_model"]

        logger.info(f"Loading Coqui TTS model: {model_name}")

        # Initialize Coqui TTS
        self.model = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=(self.device == "cuda")
        )

        # Get available speakers and languages
        self.speakers = self.model.speakers if hasattr(self.model, 'speakers') else []
        self.languages = self.model.languages if hasattr(self.model, 'languages') else []

        logger.info(f"Coqui TTS loaded: {len(self.speakers) if self.speakers else 0} speakers available")

    def _load_pyttsx3(self):
        """Load pyttsx3"""
        import pyttsx3

        logger.info("Loading pyttsx3...")

        self.model = pyttsx3.init()

        # Get available voices
        voices = self.model.getProperty('voices')
        logger.info(f"pyttsx3 loaded: {len(voices)} voices available")

    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech from text

        Args:
            text: Input text to synthesize
            output_path: Output file path (if None, returns audio array)
            speaker: Speaker/voice to use (engine-dependent)
            language: Language code (e.g., "en", "zh-cn")
            speed: Speech speed multiplier (0.5-2.0)
            **kwargs: Additional engine-specific parameters

        Returns:
            Audio array if output_path is None, otherwise None
        """
        if self.model is None:
            raise RuntimeError("TTS engine not initialized")

        logger.info(f"Synthesizing with {self.engine_type}: '{text[:50]}...'")

        if self.engine_type == "coqui":
            return self._synthesize_coqui(text, output_path, speaker, language, speed, **kwargs)
        elif self.engine_type == "pyttsx3":
            return self._synthesize_pyttsx3(text, output_path, speaker, speed, **kwargs)
        else:
            raise RuntimeError(f"Unknown engine type: {self.engine_type}")

    def _synthesize_coqui(
        self,
        text: str,
        output_path: Optional[Path],
        speaker: Optional[str],
        language: Optional[str],
        speed: float,
        **kwargs
    ) -> Optional[np.ndarray]:
        """Synthesize using Coqui TTS"""
        # Prepare parameters
        tts_kwargs = {}

        # Always pass speaker/language through; XTTS expects them even if lists are empty
        if speaker:
            sp_path = Path(str(speaker))
            if sp_path.exists():
                # If a file path is provided, treat it as speaker reference audio
                tts_kwargs["speaker_wav"] = str(sp_path)
            else:
                tts_kwargs["speaker"] = speaker
            if "speaker_wav" in kwargs:
                tts_kwargs["speaker_wav"] = kwargs["speaker_wav"]
        if language:
            tts_kwargs["language"] = language

        # Add speed control if supported
        if hasattr(self.model, 'speed'):
            tts_kwargs["speed"] = speed

        # Synthesize
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.tts_to_file(text=text, file_path=str(output_path), **tts_kwargs)
            logger.info(f"Audio saved to {output_path}")
            return None
        else:
            # Return audio array
            wav = self.model.tts(text=text, **tts_kwargs)
            return np.array(wav)

    def _synthesize_pyttsx3(
        self,
        text: str,
        output_path: Optional[Path],
        speaker: Optional[str],
        speed: float,
        **kwargs
    ) -> Optional[np.ndarray]:
        """Synthesize using pyttsx3"""
        import tempfile
        import soundfile as sf

        # Set properties
        voices = self.model.getProperty('voices')

        # Select voice
        if speaker:
            # Try to find matching voice
            for voice in voices:
                if speaker.lower() in voice.name.lower():
                    self.model.setProperty('voice', voice.id)
                    break
        else:
            # Use default voice from config
            default_voice = self.config["pyttsx3"]["default_voice"]
            if default_voice == "male" and len(voices) > 0:
                self.model.setProperty('voice', voices[0].id)
            elif default_voice == "female" and len(voices) > 1:
                self.model.setProperty('voice', voices[1].id)

        # Set speed (pyttsx3 uses words per minute)
        base_rate = self.config["pyttsx3"]["default_rate"]
        self.model.setProperty('rate', int(base_rate * speed))

        # Set volume
        self.model.setProperty('volume', 1.0)

        # Synthesize
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_to_file(text, str(output_path))
            self.model.runAndWait()
            logger.info(f"Audio saved to {output_path}")
            return None
        else:
            # Save to temp file and load
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            self.model.save_to_file(text, temp_path)
            self.model.runAndWait()

            # Load audio
            audio, sr = sf.read(temp_path)
            Path(temp_path).unlink()

            return audio if audio.ndim == 1 else audio.T

    def get_available_speakers(self) -> list:
        """Get list of available speakers/voices"""
        if self.engine_type == "coqui":
            return self.speakers if self.speakers else []
        elif self.engine_type == "pyttsx3":
            voices = self.model.getProperty('voices')
            return [v.name for v in voices]
        return []

    def get_available_languages(self) -> list:
        """Get list of available languages"""
        if self.engine_type == "coqui":
            return self.languages if self.languages else []
        elif self.engine_type == "pyttsx3":
            return ["en"]  # pyttsx3 uses system voices
        return []

    def save(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ):
        """
        Save audio array to file

        Args:
            audio: Audio array
            output_path: Output file path
            sample_rate: Sample rate (Hz), uses config default if None
        """
        import soundfile as sf

        if sample_rate is None:
            sample_rate = self.config["output_sample_rate"]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure correct shape
        if audio.ndim == 1:
            audio_to_save = audio
        elif audio.shape[0] < audio.shape[1]:
            audio_to_save = audio.T
        else:
            audio_to_save = audio

        sf.write(output_path, audio_to_save, sample_rate)
        logger.info(f"Audio saved to {output_path}")

    def clear_memory(self):
        """Clear GPU memory (for neural models)"""
        if self.engine_type == "coqui" and self.model is not None:
            # Coqui TTS cleanup
            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Coqui TTS memory cleared")
        elif self.engine_type == "pyttsx3":
            # pyttsx3 doesn't use GPU
            pass

    def __del__(self):
        """Cleanup on deletion"""
        if self.engine_type == "pyttsx3" and self.model is not None:
            try:
                self.model.stop()
            except:
                pass


# ============================================================================
# Standalone usage example
# ============================================================================

if __name__ == "__main__":
    # Example 1: Auto-detect engine
    print("Example 1: Auto-detect TTS engine")
    tts = TTSEngine(engine_type="auto")
    print(f"Selected engine: {tts.engine_type}")
    print(f"Available speakers: {tts.get_available_speakers()[:3]}")

    # Example 2: Generate speech
    print("\nExample 2: Generate speech")
    tts.synthesize(
        text="Hello, this is a test of the text to speech system.",
        output_path="output/test_tts.wav"
    )

    # Example 3: Get audio array
    print("\nExample 3: Get audio array")
    audio = tts.synthesize(
        text="This returns an audio array instead of saving to file.",
    )
    print(f"Audio shape: {audio.shape if audio is not None else 'None'}")

    # Cleanup
    tts.clear_memory()
    print("\nDone!")
