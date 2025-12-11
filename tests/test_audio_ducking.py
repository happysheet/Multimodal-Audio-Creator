import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.utils.audio_mixer import AudioMixer


def test_duck_music_reduces_level_during_speech():
    mixer = AudioMixer(target_sample_rate=100)
    sr = mixer.target_sample_rate

    # Speech present in first half, silence in second half
    speech = np.concatenate([np.ones(sr // 2) * 0.8, np.zeros(sr // 2)])
    music = np.ones(sr)

    ducked = mixer.duck_music(music, speech, sr, ducking_db=6.0, attack=0.01, release=0.01)

    with_speech = np.mean(ducked[: sr // 2])
    without_speech = np.mean(ducked[sr // 2 :])

    assert with_speech < without_speech
