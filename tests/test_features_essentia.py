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


def test_extract_all_skips_done_clips_and_checkpoints_new_ones(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from selector.audio import features_essentia as fe

    monkeypatch.setattr(fe, "_load_model_metadata", lambda model_dir: {"m": ("m.pb", 0)})
    monkeypatch.setattr(fe, "_build_graphs", lambda models: {})
    monkeypatch.setattr(
        fe, "extract_features", lambda path, graphs: None if "bad" in path else {"mood_happy": 0.5}
    )
    matches = pd.DataFrame(
        {"track_id": ["done", "new", "bad", "nomatch"], "local_path": ["d.mp3", "n.mp3", "bad.mp3", None]}
    )
    ckpt = tmp_path / "ckpt.jsonl"

    out = fe.extract_all(matches, checkpoint_path=ckpt, skip=frozenset({"done"}))

    assert list(out["track_id"]) == ["new"]
    # Only successes are checkpointed, so a failed clip is retried next run.
    assert [json.loads(line)["track_id"] for line in ckpt.read_text().splitlines()] == ["new"]
