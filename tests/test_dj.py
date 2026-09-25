import json
from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from selector.dj import agent
from selector.dj import commit as commit_mod
from selector.dj.arc import EnergyArc, jarring_reason, tempo_shift
from selector.dj.brief import THEMES, build_brief, resolve_theme
from selector.dj.critique import critique, revised_config
from selector.dj.pool import Crate, measured_energy
from selector.dj.select import Pick, SelectConfig, Selection, select


def _at(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


MOODS = ["chill", "melancholic", "euphoric", "aggressive", "nostalgic", "somber"]


def _crate(n: int = 120, n_kc: int = 60, seed: int = 0) -> Crate:
    rng = np.random.default_rng(seed)
    tracks = pd.DataFrame({
        "track_id": [f"t{i}" for i in range(n)],
        "name": [f"Song {i}" for i in range(n)],
        "artist": [f"Artist {i % 30}" for i in range(n)],
        "energy": rng.uniform(0, 1, n),
        "tempo": rng.normal(120, 6, n),
        "duration_ms": np.full(n, 200_000),
        "mood_tags": [[MOODS[i % len(MOODS)], MOODS[(i + 1) % len(MOODS)]] for i in range(n)],
        "familiar": np.arange(n) % 2 == 0,
        "fly_valence": rng.normal(30, 5, n),
        "tag_row": np.arange(n),
    })
    tracks["taste"] = tracks["fly_valence"].rank(pct=True)
    dense = np.zeros((n, n_kc))
    for i in range(n):
        dense[i, rng.choice(n_kc, 6, replace=False)] = 1
    return Crate(tracks=tracks, tags=sparse.csr_matrix(dense), as_of=_at(2026, 9, 15))


def _recent(crate: Crate, k: int = 10) -> pd.DataFrame:
    return crate.tracks.head(k).rename(columns={"artist": "artist_name"})[["track_id", "artist_name"]]


def _pick(i: int, energy: float, tempo: float | None, target: float, phase: str, artist: str = "") -> Pick:
    return Pick(
        track_id=f"p{i}", name=f"P{i}", artist=artist or f"A{i}", start_s=i * 200.0,
        duration_s=200.0, t_mid=0.0, phase=phase, target_energy=target, energy=energy,
        tempo=tempo, familiar=True, fly_valence=30.0, hamming_to_prev=None,
        scores={"taste": 0.8},
    )


# -- arc ----------------------------------------------------------------------


def test_arc_maps_shape_onto_theme_range():
    arc = EnergyArc(minutes=45, energy_floor=0.2, energy_ceiling=0.8)
    assert arc.phase(0.05) == "opener" and arc.phase(0.6) == "peak" and arc.phase(1.0) == "comedown"
    assert arc.target(0.0) == pytest.approx(0.2 + 0.2 * 0.6)
    assert arc.target(0.7) == pytest.approx(0.2 + 0.95 * 0.6)
    build = [arc.target(t) for t in np.linspace(0.16, 0.54, 10)]
    assert build == sorted(build)
    assert arc.target(0.99) < arc.target(0.7)


def test_tempo_shift_folds_octaves():
    assert tempo_shift(90, 180) == pytest.approx(0.0)
    assert tempo_shift(100, 110) == pytest.approx(0.1)
    assert tempo_shift(100, None) is None
    assert tempo_shift(100, float("nan")) is None


def test_jarring_reason():
    assert jarring_reason(120, 0.5, 121, 0.55) is None
    assert "energy jump" in jarring_reason(120, 0.2, 120, 0.6)
    assert "tempo lurch" in jarring_reason(100, 0.5, 130, 0.5)
    assert "move together" in jarring_reason(100, 0.5, 115, 0.65)


# -- brief --------------------------------------------------------------------


def test_resolve_theme_by_name_and_mood_words():
    assert resolve_theme("Night Drive").name == "night drive"
    custom = resolve_theme("sad rainy day")
    assert custom.mood_tags == ("melancholic",)
    assert 0 <= custom.energy_floor < custom.energy_ceiling <= 1
    with pytest.raises(ValueError, match="mood"):
        resolve_theme("xyzzy")


def test_brief_picks_theme_that_fits_the_hour():
    crate = _crate()
    b = build_brief(_recent(crate), crate, _at(2026, 9, 23, 2, 0))
    assert 2 in b.theme.hours
    assert b.seed_track_ids and all(s in set(crate.tracks.track_id) for s in b.seed_track_ids)
    assert b.theme.name in b.rationale


def test_friday_evening_counts_as_weekend():
    crate = _crate()
    assert build_brief(_recent(crate), crate, _at(2026, 9, 25, 22, 0)).is_weekend
    assert not build_brief(_recent(crate), crate, _at(2026, 9, 25, 10, 0)).is_weekend


# -- select -------------------------------------------------------------------


def _plan(crate, theme="golden hour", minutes=40, **config):
    b = build_brief(_recent(crate), crate, _at(2026, 9, 23, 17, 0), requested_theme=theme)
    a = EnergyArc.for_theme(b.theme, minutes)
    return b, a, select(crate, b, a, SelectConfig(**config))


def test_select_respects_artist_cap_band_and_length():
    crate = _crate()
    _, a, s = _plan(crate, max_per_artist=2)
    assert max(Counter(p.artist for p in s.picks).values()) <= 2
    assert all(abs(p.energy - p.target_energy) <= a.tolerance for p in s.picks)
    assert abs(s.seconds - a.seconds) <= 200
    assert len(s.steps) == len(s.picks) and s.steps[0]["shortlist"]


def test_select_honours_familiar_ratio():
    crate = _crate()
    for ratio in (0.0, 1.0):
        _, _, s = _plan(crate, familiar_ratio=ratio)
        assert all(p.familiar == bool(ratio) for p in s.picks)


def test_select_never_uses_excluded_tracks():
    crate = _crate()
    b, a, first = _plan(crate)
    banned = set(first.track_ids[:3])
    again = select(crate, b, a, SelectConfig(), exclude=banned)
    assert not banned & set(again.track_ids)


def test_hard_transitions_mode_avoids_jarring_moves():
    crate = _crate()
    crate.tracks["tempo"] = np.where(np.arange(len(crate.tracks)) % 2, 90.0, 131.0)
    _, _, s = _plan(crate, hard_transitions=True)
    for x, y in zip(s.picks, s.picks[1:]):
        assert jarring_reason(x.tempo, x.energy, y.tempo, y.energy) is None


def test_select_config_validates():
    with pytest.raises(ValueError):
        SelectConfig(familiar_ratio=1.5)


# -- critique -----------------------------------------------------------------


def _selection(picks):
    return Selection(picks=picks, config=SelectConfig())


def test_critique_passes_a_clean_arc():
    arc = EnergyArc(minutes=4 * 200 / 60, energy_floor=0.2, energy_ceiling=0.8)
    picks = [
        _pick(0, 0.35, 120, 0.35, "opener"),
        _pick(1, 0.55, 121, 0.55, "build"),
        _pick(2, 0.75, 122, 0.77, "peak"),
        _pick(3, 0.55, 121, 0.55, "comedown"),
    ]
    v = critique(_selection(picks), arc)
    assert v.passed, v.issues
    assert v.summary.startswith("Pass")


def test_critique_rejects_jarring_transition_and_names_incoming_track():
    arc = EnergyArc(minutes=4 * 200 / 60, energy_floor=0.2, energy_ceiling=0.8)
    picks = [
        _pick(0, 0.35, 120, 0.35, "opener"),
        _pick(1, 0.40, 120, 0.45, "build"),
        _pick(2, 0.80, 80, 0.77, "peak"),
        _pick(3, 0.55, 121, 0.55, "comedown"),
    ]
    v = critique(_selection(picks), arc)
    assert not v.passed
    kinds = {i.kind for i in v.issues}
    assert "jarring" in kinds
    assert "p2" in v.revision_exclude


def test_critique_rejects_flat_shape_and_off_arc():
    arc = EnergyArc(minutes=3 * 200 / 60, energy_floor=0.2, energy_ceiling=0.8)
    picks = [
        _pick(0, 0.6, 120, 0.35, "opener"),
        _pick(1, 0.6, 120, 0.77, "peak"),
        _pick(2, 0.6, 120, 0.4, "comedown"),
    ]
    v = critique(_selection(picks), arc)
    kinds = {i.kind for i in v.issues}
    assert {"flat_shape", "off_arc"} <= kinds


def test_revised_config_hardens_transitions():
    c = revised_config(SelectConfig())
    assert c.hard_transitions and c.w_transition == pytest.approx(0.3)


# -- commit -------------------------------------------------------------------


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.token_path = SimpleNamespace(exists=lambda: False)

    def create_playlist(self, name, description="", public=False, track_uris=None):
        self.calls.append((name, description, public, track_uris))
        return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}


def _passed_run(crate):
    b, a, s = _plan(crate)
    return b, s, critique(s, a)


def test_dry_run_never_touches_spotify():
    crate = _crate()
    b, s, v = _passed_run(crate)
    client = _FakeClient()
    r = commit_mod.commit(b, s, v, _at(2026, 9, 23), client, dry_run=True)
    assert r.dry_run and client.calls == []


def test_commit_refuses_failed_critique():
    crate = _crate()
    b, s, _ = _passed_run(crate)
    failed = critique(_selection([_pick(0, 0.9, 120, 0.2, "opener")]), EnergyArc(45, 0.1, 0.5))
    with pytest.raises(commit_mod.CommitRefused):
        commit_mod.commit(b, s, failed, _at(2026, 9, 23), _FakeClient(), dry_run=False)


def test_commit_writes_private_playlist_with_track_uris():
    crate = _crate()
    b, s, v = _passed_run(crate)
    v.passed = True
    client = _FakeClient()
    r = commit_mod.commit(b, s, v, _at(2026, 9, 23), client, dry_run=False)
    _name, description, public, uris = client.calls[0]
    assert not public and uris == [f"spotify:track:{t}" for t in s.track_ids]
    assert len(description) <= commit_mod.MAX_DESCRIPTION_CHARS
    assert r.playlist_url.endswith("pl1")


def test_liner_notes_cite_measured_tempo_and_energy():
    a = _pick(0, 0.40, 120.0, 0.4, "build")
    b = _pick(1, 0.55, 121.0, 0.55, "peak")
    note = commit_mod.transition_note(a, b)
    assert "121 BPM" in note and "0.40 -> 0.55" in note and "into the peak" in note


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (152.0, 78.0, "78 BPM runs at half time against 152"),
        (76.0, 152.0, "152 BPM runs at double time against 76"),
        (76.0, 144.0, "near double time"),
        (120.0, 121.0, "locks in at ~121 BPM"),
        (120.0, 130.0, "pushes 120 -> 130 BPM"),
    ],
)
def test_tempo_phrase_names_octave_relations(a, b, expected):
    note = commit_mod.transition_note(_pick(0, 0.5, a, 0.5, "build"), _pick(1, 0.5, b, 0.5, "build"))
    assert expected in note


# -- agent --------------------------------------------------------------------


def test_run_dj_logs_the_full_chain(tmp_path):
    crate = _crate()
    run = agent.run_dj(
        crate, theme="golden hour", minutes=30, recent=_recent(crate),
        now=_at(2026, 9, 23, 17, 0), log_dir=tmp_path,
    )
    assert 1 <= len(run.attempts) <= 1 + agent.MAX_REVISIONS
    log = json.loads(run.log_path.read_text(encoding="utf-8"))
    assert {"brief", "arc", "attempts", "commit", "liner_notes"} <= set(log)
    assert log["attempts"][0]["selection"]["steps"]
    assert run.log_path.with_suffix(".md").exists()
    assert "Dry run" in agent.render(run)


def test_measured_energy_is_rank_normalised():
    audio = pd.DataFrame({
        "rms_mean_scaled": [0.1, 0.5, 0.9],
        "onset_density_scaled": [0.2, 0.5, 0.8],
        "spectral_centroid_mean_scaled": [0.3, 0.5, 0.7],
    })
    e = measured_energy(audio)
    assert list(e.rank()) == [1, 2, 3] and e.max() == 1.0


def test_every_named_theme_resolves_to_itself():
    for theme in THEMES:
        assert resolve_theme(theme.name) is theme
