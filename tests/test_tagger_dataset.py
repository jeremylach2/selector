from selector.tagger.dataset import build_prompt, split_by_artist, to_examples
from selector.tagger.schema import LabelRecord, PredictedLabels, TeacherInput


def _record(track_id: str, artist: str, lyrics: str | None = "some lyrics", measured: dict | None = None) -> LabelRecord:
    item = TeacherInput(
        track_id=track_id,
        track_name=f"Song {track_id}",
        artist_name=artist,
        lyrics=lyrics,
        measured=measured or {"tempo_scaled": 0.5},
    )
    labels = PredictedLabels(valence=0.5, mood_tags=["chill"], era="2020s", lyrical_theme="x", intensity=0.5)
    return LabelRecord(track_id=track_id, input=item, labels=labels, teacher_model="test")


def test_split_by_artist_keeps_each_artist_in_one_split():
    records = [_record(f"t{i}", f"artist{i % 6}") for i in range(20)]

    train, val, test = split_by_artist(records, seed=1)

    train_artists = {r.input.artist_name for r in train}
    val_artists = {r.input.artist_name for r in val}
    test_artists = {r.input.artist_name for r in test}
    assert not (train_artists & val_artists)
    assert not (train_artists & test_artists)
    assert not (val_artists & test_artists)
    assert len(train) + len(val) + len(test) == len(records)


def test_build_prompt_arm_a_excludes_measured_features():
    record = _record("t1", "artist", lyrics="la la", measured={"tempo_scaled": 0.9})
    prompt = build_prompt(record, "A")
    assert "la la" in prompt
    assert "tempo_scaled" not in prompt


def test_build_prompt_arm_b_excludes_lyrics():
    record = _record("t1", "artist", lyrics="la la", measured={"tempo_scaled": 0.9})
    prompt = build_prompt(record, "B")
    assert "tempo_scaled" in prompt
    assert "la la" not in prompt
    assert "Lyrics" not in prompt


def test_build_prompt_arm_c_includes_both():
    record = _record("t1", "artist", lyrics="la la", measured={"tempo_scaled": 0.9})
    prompt = build_prompt(record, "C")
    assert "la la" in prompt
    assert "tempo_scaled" in prompt


def test_to_examples_produces_valid_json_completions():
    records = [_record("t1", "artist")]
    examples = to_examples(records, "C")
    assert len(examples) == 1
    assert "valence" in examples[0]["completion"]
