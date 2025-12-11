"""
Audio Mixing Utilities
Mix multiple audio tracks with volume control, fade effects, and normalization
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioMixer:
    """Audio mixing and processing utilities"""

    def __init__(self, target_sample_rate: int = 44100):
        """
        Initialize audio mixer

        Args:
            target_sample_rate: Target sample rate for all audio
        """
        self.target_sample_rate = target_sample_rate

    def load_audio(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file

        Args:
            path: Path to audio file

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        import soundfile as sf

        audio, sr = sf.read(path)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return audio, sr

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate

        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio

        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def normalize(
        self, audio: np.ndarray, target_loudness: float = -16.0
    ) -> np.ndarray:
        """
        Normalize audio to target loudness (LUFS)

        Args:
            audio: Input audio array
            target_loudness: Target loudness in LUFS

        Returns:
            Normalized audio
        """
        try:
            import pyloudnorm as pyln

            # Measure loudness
            meter = pyln.Meter(self.target_sample_rate)
            loudness = meter.integrated_loudness(audio)

            # Normalize
            normalized = pyln.normalize.loudness(audio, loudness, target_loudness)

            return normalized

        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}. Using peak normalization.")
            # Fallback to peak normalization
            peak = np.abs(audio).max()
            if peak > 0:
                return audio / peak * 0.9
            return audio

    def match_loudness(self, audio: np.ndarray, sample_rate: int, target_lufs: float = -16.0) -> np.ndarray:
        """
        Loudness match a waveform to target LUFS.
        """
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(audio)
            return pyln.normalize.loudness(audio, loudness, target_lufs)
        except Exception as e:  # pragma: no cover - fallback path
            logger.warning(f"match_loudness failed: {e}. Using raw audio.")
            return audio

    def duck_music(
        self,
        music: np.ndarray,
        speech: np.ndarray,
        sample_rate: int,
        ducking_db: float = 4.0,
        attack: float = 0.05,
        release: float = 0.2,
    ) -> np.ndarray:
        """
        Apply sidechain ducking to music based on speech envelope.
        """
        if len(music) == 0 or len(speech) == 0:
            return music

        speech_env = np.abs(speech)
        attack_samples = max(int(sample_rate * attack), 1)
        release_samples = max(int(sample_rate * release), 1)
        alpha_a = np.exp(-1.0 / attack_samples)
        alpha_r = np.exp(-1.0 / release_samples)

        env = np.zeros_like(speech_env)
        for i, x in enumerate(speech_env):
            if i == 0:
                env[i] = x
                continue
            prev = env[i - 1]
            if x > prev:
                env[i] = alpha_a * prev + (1 - alpha_a) * x
            else:
                env[i] = alpha_r * prev + (1 - alpha_r) * x

        # Normalize envelope to [0,1]
        env_max = env.max() + 1e-8
        env = env / env_max

        # Pad or trim envelope to music length
        if len(env) < len(music):
            env = np.pad(env, (0, len(music) - len(env)))
        elif len(env) > len(music):
            env = env[: len(music)]

        reduction = 10 ** (-ducking_db / 20.0)
        duck_gain = 1 - env * (1 - reduction)
        return music * duck_gain

    def apply_fade(
        self,
        audio: np.ndarray,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply fade in/out effects

        Args:
            audio: Input audio array
            fade_in: Fade in duration (seconds)
            fade_out: Fade out duration (seconds)
            sample_rate: Sample rate (uses target if None)

        Returns:
            Audio with fade effects
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        audio_faded = audio.copy()

        # Fade in
        if fade_in > 0:
            fade_in_samples = int(fade_in * sample_rate)
            fade_in_samples = min(fade_in_samples, len(audio))
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            audio_faded[:fade_in_samples] *= fade_in_curve

        # Fade out
        if fade_out > 0:
            fade_out_samples = int(fade_out * sample_rate)
            fade_out_samples = min(fade_out_samples, len(audio))
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            audio_faded[-fade_out_samples:] *= fade_out_curve

        return audio_faded

    def mix_tracks(
        self,
        tracks: List[Tuple[np.ndarray, float]],
        method: str = "overlay",
    ) -> np.ndarray:
        """
        Mix multiple audio tracks

        Args:
            tracks: List of (audio_array, volume) tuples
            method: Mixing method ('overlay' or 'concatenate')

        Returns:
            Mixed audio
        """
        if method == "overlay":
            return self._mix_overlay(tracks)
        elif method == "concatenate":
            return self._mix_concatenate(tracks)
        else:
            raise ValueError(f"Unknown mixing method: {method}")

    def _mix_overlay(self, tracks: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Mix tracks by overlaying (simultaneous playback)"""
        if not tracks:
            return np.array([])

        # Find maximum length
        max_length = max(len(audio) for audio, _ in tracks)

        # Mix tracks
        mixed = np.zeros(max_length)
        for audio, volume in tracks:
            # Pad if necessary
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)))

            # Add to mix with volume control
            mixed += audio * volume

        # Prevent clipping
        peak = np.abs(mixed).max()
        if peak > 1.0:
            mixed = mixed / peak * 0.95

        return mixed

    def _mix_concatenate(self, tracks: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Mix tracks by concatenating (sequential playback)"""
        mixed_parts = []
        for audio, volume in tracks:
            mixed_parts.append(audio * volume)

        return np.concatenate(mixed_parts)

    def crossfade(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        duration: float = 2.0,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Crossfade between two audio clips

        Args:
            audio1: First audio clip
            audio2: Second audio clip
            duration: Crossfade duration (seconds)
            sample_rate: Sample rate

        Returns:
            Crossfaded audio
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        fade_samples = int(duration * sample_rate)
        fade_samples = min(fade_samples, len(audio1), len(audio2))

        # Create crossfade curves
        fade_out_curve = np.linspace(1, 0, fade_samples)
        fade_in_curve = np.linspace(0, 1, fade_samples)

        # Apply crossfade
        crossfade_region = (
            audio1[-fade_samples:] * fade_out_curve
            + audio2[:fade_samples] * fade_in_curve
        )

        # Concatenate
        result = np.concatenate(
            [audio1[:-fade_samples], crossfade_region, audio2[fade_samples:]]
        )

        return result

    def save(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ):
        """
        Save audio to file

        Args:
            audio: Audio array
            output_path: Output file path
            sample_rate: Sample rate (uses target if None)
        """
        import soundfile as sf

        if sample_rate is None:
            sample_rate = self.target_sample_rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(output_path, audio, sample_rate)
        logger.info(f"Mixed audio saved to {output_path}")


def mix_audio(
    music_path: Union[str, Path],
    speech_path: Union[str, Path],
    output_path: Union[str, Path],
    music_volume: float = 0.6,
    speech_volume: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to mix music and speech

    Args:
        music_path: Path to music file
        speech_path: Path to speech file
        output_path: Output file path
        music_volume: Music volume (0.0-1.0)
        speech_volume: Speech volume (0.0-1.0)
        **kwargs: Additional arguments for AudioMixer

    Returns:
        Mixed audio array
    """
    mixer = AudioMixer()

    # Load audio files
    music, music_sr = mixer.load_audio(music_path)
    speech, speech_sr = mixer.load_audio(speech_path)

    # Resample to target rate
    music = mixer.resample(music, music_sr, mixer.target_sample_rate)
    speech = mixer.resample(speech, speech_sr, mixer.target_sample_rate)

    target_lufs = kwargs.get("target_lufs", -18.0)
    ducking_db = kwargs.get("ducking_db", 8.0)
    attack = kwargs.get("attack", 0.02)
    release = kwargs.get("release", 0.2)

    # Loudness match before ducking
    music = mixer.match_loudness(music, mixer.target_sample_rate, target_lufs=target_lufs)
    speech = mixer.match_loudness(speech, mixer.target_sample_rate, target_lufs=target_lufs)

    # Sidechain ducking
    music = mixer.duck_music(
        music,
        speech,
        mixer.target_sample_rate,
        ducking_db=ducking_db,
        attack=attack,
        release=release,
    )

    # Apply fade effects
    music = mixer.apply_fade(music, fade_in=0.5, fade_out=1.0)
    speech = mixer.apply_fade(speech, fade_in=0.2, fade_out=0.5)

    # Mix
    mixed = mixer.mix_tracks(
        [(music, music_volume), (speech, speech_volume)], method="overlay"
    )

    # Final normalization
    mixed = mixer.normalize(mixed, target_loudness=target_lufs)

    # Save
    mixer.save(mixed, output_path)

    return mixed


# ============================================================================
# Standalone usage example
# ============================================================================

if __name__ == "__main__":
    # Example: Mix music and speech
    mixed = mix_audio(
        music_path="output/generated_music.wav",
        speech_path="output/synthesized_speech.wav",
        output_path="output/final_mixed.wav",
        music_volume=0.5,
        speech_volume=1.0,
    )

    print(f"Mixed audio shape: {mixed.shape}")
    print(f"Duration: {len(mixed) / 44100:.2f} seconds")
