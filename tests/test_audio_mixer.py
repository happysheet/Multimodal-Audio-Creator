import numpy as np

from src.utils.audio_mixer import AudioMixer


def test_overlay_mix_caps_peak():
    mixer = AudioMixer(target_sample_rate=10)
    track_a = (np.ones(4), 1.0)
    track_b = (np.ones(4) * 0.8, 1.0)

    mixed = mixer.mix_tracks([track_a, track_b], method="overlay")

    assert mixed.shape[0] == 4
    assert mixed.max() <= 0.95


def test_apply_fade_modifies_edges():
    mixer = AudioMixer(target_sample_rate=4)
    audio = np.ones(8)

    faded = mixer.apply_fade(audio, fade_in=0.25, fade_out=0.5, sample_rate=4)

    assert np.isclose(faded[0], 0.0)
    assert np.isclose(faded[-1], 0.0)
