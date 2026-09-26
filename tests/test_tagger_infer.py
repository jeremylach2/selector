
import json

import httpx

from selector.tagger import infer
from selector.tagger.schema import PredictedLabels, TeacherInput


def _fallback() -> PredictedLabels:
    return PredictedLabels(valence=0.5, mood_tags=["chill"], era="2010s", lyrical_theme="x", intensity=0.4)


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    assert infer._load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_load_checkpoint_keys_by_track_id(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"track_id": "t1", "valence": 0.1}\n{"track_id": "t2", "valence": 0.9}\n', encoding="utf-8")
    cached = infer._load_checkpoint(path)
    assert set(cached) == {"t1", "t2"}
    assert cached["t2"]["valence"] == 0.9


def test_load_lyrics_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "LYRICS_DIR", tmp_path)
    assert infer._load_lyrics("missing_track") is None


def test_load_lyrics_empty_marker_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "LYRICS_DIR", tmp_path)
    (tmp_path / "t1.txt").write_text("", encoding="utf-8")
    assert infer._load_lyrics("t1") is None


def test_load_lyrics_returns_cached_text(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "LYRICS_DIR", tmp_path)
    (tmp_path / "t1.txt").write_text("some lyrics", encoding="utf-8")
    assert infer._load_lyrics("t1") == "some lyrics"


def test_infer_one_flags_finetuned_gpu_on_valid_parse(monkeypatch):
    valid_json = _fallback().model_dump_json()
    monkeypatch.setattr(infer, "generate_completion_gpu", lambda prompt, client, server_url: valid_json)

    item = TeacherInput(track_id="t1", track_name="Song", artist_name="Artist")
    row = infer._infer_one("t1", "A", item, _fallback(), client=object())

    assert row["track_id"] == "t1"
    assert row["arm"] == "A"
    assert row["label_source"] == "finetuned_gpu"
    assert row["valence"] == 0.5


def test_infer_one_falls_back_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(infer, "generate_completion_gpu", lambda prompt, client, server_url: "not json at all")

    fallback = _fallback()
    item = TeacherInput(track_id="t2", track_name="Song", artist_name="Artist")
    row = infer._infer_one("t2", "C", item, fallback, client=object())

    assert row["label_source"] == "parse_fallback"
    assert row["arm"] == "C"
    assert row["valence"] == fallback.valence


def test_infer_one_falls_back_on_request_error(monkeypatch):
    def raise_error(prompt, client, server_url):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(infer, "generate_completion_gpu", raise_error)

    fallback = _fallback()
    item = TeacherInput(track_id="t3", track_name="Song", artist_name="Artist")
    row = infer._infer_one("t3", "A", item, fallback, client=object())

    assert row["label_source"] == "parse_fallback"


def test_drop_from_checkpoint_removes_only_named_tracks(tmp_path):
    from selector.tagger.infer import _load_checkpoint, drop_from_checkpoint

    path = tmp_path / "ckpt.jsonl"
    path.write_text(
        "".join(json.dumps({"track_id": t, "valence": 0.5}) + "\n" for t in ["a", "b", "c"]),
        encoding="utf-8",
    )
    assert drop_from_checkpoint(path, {"b", "zzz"}) == 1
    assert list(_load_checkpoint(path)) == ["a", "c"]
