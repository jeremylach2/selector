from selector.tagger.label import _build_tool, _load_cached, _track_prompt
from selector.tagger.schema import LabelRecord, PredictedLabels, TeacherInput


def test_build_tool_schema_matches_predicted_labels_fields():
    tool = _build_tool()
    assert tool["name"] == "emit_labels"
    props = tool["input_schema"]["properties"]
    assert set(props) == {"valence", "mood_tags", "era", "lyrical_theme", "intensity"}


def test_track_prompt_includes_measured_features_when_present():
    item = TeacherInput(
        track_id="t1", track_name="Song", artist_name="Artist", measured={"tempo_scaled": 0.8}
    )
    prompt = _track_prompt(item, "default")
    assert "tempo_scaled" in prompt
    assert "none available" not in prompt


def test_track_prompt_notes_missing_measured_features():
    item = TeacherInput(track_id="t1", track_name="Song", artist_name="Artist")
    prompt = _track_prompt(item, "default")
    assert "none available for this track" in prompt


def test_track_prompt_alt_phrasing_differs_from_default():
    item = TeacherInput(track_id="t1", track_name="Song", artist_name="Artist")
    assert _track_prompt(item, "default") != _track_prompt(item, "alt_phrasing")


def test_load_cached_reads_existing_records(tmp_path):
    path = tmp_path / "labels.jsonl"
    item = TeacherInput(track_id="t1", track_name="Song", artist_name="Artist")
    labels = PredictedLabels(valence=0.5, mood_tags=["chill"], era="2020s", lyrical_theme="x", intensity=0.5)
    record = LabelRecord(track_id="t1", input=item, labels=labels, teacher_model="claude-sonnet-5")
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    cached = _load_cached(path)

    assert "t1" in cached
    assert cached["t1"].labels.valence == 0.5


def test_load_cached_empty_when_file_missing(tmp_path):
    assert _load_cached(tmp_path / "does_not_exist.jsonl") == {}
