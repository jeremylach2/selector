import pytest
from pydantic import ValidationError

from selector.tagger.schema import LabelRecord, PredictedLabels, TeacherInput


def test_predicted_labels_rejects_out_of_range_valence():
    with pytest.raises(ValidationError):
        PredictedLabels(valence=1.5, mood_tags=["chill"], era="2020s", lyrical_theme="test", intensity=0.5)


def test_predicted_labels_rejects_unknown_mood_tag():
    with pytest.raises(ValidationError):
        PredictedLabels(valence=0.5, mood_tags=["ecstatic"], era="2020s", lyrical_theme="test", intensity=0.5)


def test_teacher_input_allows_null_lyrics_and_optional_fields():
    item = TeacherInput(track_id="abc", track_name="Song", artist_name="Artist")
    assert item.lyrics is None
    assert item.album_name is None
    assert item.measured == {}


def test_label_record_round_trips_through_json():
    item = TeacherInput(track_id="abc", track_name="Song", artist_name="Artist")
    labels = PredictedLabels(
        valence=0.7, mood_tags=["euphoric", "triumphant"], era="2010s", lyrical_theme="victory", intensity=0.8
    )
    record = LabelRecord(track_id="abc", input=item, labels=labels, teacher_model="claude-sonnet-5")

    restored = LabelRecord.model_validate_json(record.model_dump_json())

    assert restored == record
