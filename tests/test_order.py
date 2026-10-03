"""`order_tracks`: sequencing a given list along a fitted energy arc."""

import json

import numpy as np
import pytest

from selector.dj import order
from selector.dj.arc import jarring_reason
from tests.test_dj import _crate


def _ids(n: int, start: int = 0) -> list[str]:
    return [f"t{i}" for i in range(start, start + n)]


def test_every_given_track_is_kept_once():
    crate = _crate()
    given = _ids(12)
    result = order.order_tracks(crate, given + ["t3"])
    assert sorted(result.track_ids) == sorted(given)
    assert result.duplicates == 1 and result.unplaced == []


def test_fitted_arc_spans_the_tracks_energy():
    energy = np.array([0.2, 0.35, 0.6])
    arc = order.fit_arc(energy, minutes=10)
    targets = [arc.target(t) for t in np.linspace(0, 1, 201)]
    assert min(targets) == pytest.approx(0.2) and max(targets) == pytest.approx(0.6)


def test_the_peak_is_louder_than_the_edges():
    crate = _crate()
    result = order.order_tracks(crate, _ids(16))
    picks = result.picks
    peak = np.mean([p.energy for p in picks if p.phase == "peak"])
    assert peak > picks[0].energy and peak > picks[-1].energy
    assert picks[0].energy == pytest.approx(min(p.energy for p in picks), abs=0.15)


def test_polish_never_makes_the_assignment_worse():
    crate = _crate()
    rows = crate.by_id().loc[_ids(16)]
    energy = rows["energy"].to_numpy(dtype=float)
    tempo = rows["tempo"].to_numpy(dtype=float)
    duration = rows["duration_ms"].to_numpy(dtype=float) / 1000
    arc = order.fit_arc(energy, duration.sum() / 60)
    cost = order._Cost(energy, tempo, duration, np.zeros((16, 16)), arc)
    start = order._assign(energy, arc)
    before = cost(start.copy())
    assert cost(order._polish(start.copy(), cost)) <= before


def test_polish_removes_an_avoidable_lurch():
    crate = _crate()
    tracks = crate.tracks
    # Equal energies, so only tempo decides; one 90 BPM track among 120s.
    tracks["energy"] = 0.5
    tracks["tempo"] = 120.0
    tracks.loc[tracks.track_id == "t2", "tempo"] = 90.0
    result = order.order_tracks(crate, _ids(6))
    jarring = [
        jarring_reason(a.tempo, a.energy, b.tempo, b.energy) for a, b in zip(result.picks, result.picks[1:])
    ]
    # The odd one out can't avoid one neighbour, so it should end up at an edge.
    assert sum(r is not None for r in jarring) == 1
    assert result.picks[0].track_id == "t2" or result.picks[-1].track_id == "t2"


def test_tracks_without_audio_are_appended_in_order():
    crate = _crate()
    result = order.order_tracks(crate, ["x9", *_ids(5), "x1"])
    assert result.unplaced == ["x9", "x1"]
    assert result.uris[-2:] == ["spotify:track:x9", "spotify:track:x1"]
    text = order.render(result, {"x9": ("Lost Song", "Someone")})
    assert "Appended at the end" in text and "Lost Song - Someone" in text and "(unknown)" in text
    assert json.loads(text.rsplit("\n", 1)[-1]) == result.uris


@pytest.mark.parametrize("ids", [["t1"], _ids(101), ["x1", "x2", "t1"]])
def test_lists_that_cant_be_ordered_are_refused(ids):
    with pytest.raises(ValueError):
        order.order_tracks(_crate(), ids)


def test_critique_reads_the_result_with_no_artist_cap():
    crate = _crate()
    crate.tracks["artist"] = "Same Artist"
    result = order.order_tracks(crate, _ids(10))
    assert all(i.kind != "artist_cap" for i in result.verdict.issues)
    text = order.render(result)
    assert "**Flow check.**" in text and "Reject" not in text
