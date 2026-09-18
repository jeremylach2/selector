from selector.tagger.eval import evaluate_predictions, train_mean_baseline_predictions
from selector.tagger.schema import LabelRecord, PredictedLabels, TeacherInput


def _label(valence, intensity, era="2020s", mood_tags=("chill",), theme="x") -> PredictedLabels:
    return PredictedLabels(valence=valence, mood_tags=list(mood_tags), era=era, lyrical_theme=theme, intensity=intensity)


def _record(track_id: str, labels: PredictedLabels) -> LabelRecord:
    item = TeacherInput(track_id=track_id, track_name="s", artist_name="a")
    return LabelRecord(track_id=track_id, input=item, labels=labels, teacher_model="test")


def test_evaluate_predictions_perfect_match_gives_zero_error():
    true = [_label(0.5, 0.5), _label(0.8, 0.2)]
    metrics = evaluate_predictions(true, true)
    assert metrics["valence_mae"] == 0
    assert metrics["era_exact_match"] == 1.0
    assert metrics["mood_tags_exact_match"] == 1.0


def test_evaluate_predictions_computes_mae():
    true = [_label(0.5, 0.5), _label(0.9, 0.1)]
    pred = [_label(0.4, 0.6), _label(0.7, 0.3)]
    metrics = evaluate_predictions(true, pred)
    assert abs(metrics["valence_mae"] - 0.15) < 1e-9


def test_train_mean_baseline_uses_train_statistics_not_test_input():
    train = [
        _record("t1", _label(0.2, 0.2, era="1970s", mood_tags=("somber",))),
        _record("t2", _label(0.8, 0.8, era="1970s", mood_tags=("somber",))),
    ]
    preds = train_mean_baseline_predictions(train, n_test=3)
    assert len(preds) == 3
    assert all(p.valence == 0.5 for p in preds)
    assert all(p.era == "1970s" for p in preds)
