import json

from selector.audio.features_essentia import MODEL_POSITIVE_CLASS, _load_model_metadata


def test_load_model_metadata_reads_positive_class_index(tmp_path):
    (tmp_path / "danceability-musicnn-msd-2.pb").write_bytes(b"")
    (tmp_path / "danceability-musicnn-msd-2.json").write_text(
        json.dumps({"classes": ["not_danceable", "danceable"]})
    )

    metadata = _load_model_metadata(tmp_path)

    assert "danceability" in metadata
    assert metadata["danceability"]["positive_index"] == 1


def test_load_model_metadata_skips_missing_files(tmp_path):
    metadata = _load_model_metadata(tmp_path)
    assert metadata == {}


def test_all_configured_models_have_a_positive_class_label():
    assert len(MODEL_POSITIVE_CLASS) == 6
    assert all(isinstance(label, str) and label for label in MODEL_POSITIVE_CLASS.values())
