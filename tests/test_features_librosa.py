import numpy as np
import soundfile as sf

from selector.audio.features_librosa import extract_features


def test_extract_features_on_synthetic_tone(tmp_path):
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype("float32")
    path = tmp_path / "tone.wav"
    sf.write(path, tone, sr)

    features = extract_features(str(path))

    assert features is not None
    assert "tempo" in features
    assert "spectral_centroid_mean" in features
    assert features["rms_mean"] > 0
    assert 0 <= features["estimated_key"] <= 11


def test_extract_features_returns_none_for_missing_file(tmp_path):
    assert extract_features(str(tmp_path / "does_not_exist.wav")) is None
