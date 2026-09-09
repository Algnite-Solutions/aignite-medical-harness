"""AMA Dataset v0 schema, loader, validator round-trip tests."""
import json

import pytest

from ama.data import Episode, load_dataset, validate_dataset


def make_dataset(tmp_path, episodes, dataset_extra=None, targets=None, policy=None):
    ds = {"schema": "ama-dataset-v0", "name": "t", "version": "0.1",
          "splits": {"all": [e["episode_id"] for e in episodes]}, "scorer": "unscored"}
    ds.update(dataset_extra or {})
    (tmp_path / "dataset.json").write_text(json.dumps(ds), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in episodes) + "\n", encoding="utf-8")
    if targets is not None:
        (tmp_path / "targets.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in targets) + "\n", encoding="utf-8")
    if policy is not None:
        (tmp_path / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return tmp_path


EP_1TURN = {"episode_id": "e1", "subject_id": "s1", "metadata": {},
            "turns": [{"turn_id": "t1", "time": None, "message": "hello",
                       "evidence": [{"evidence_id": "v1", "kind": "lab", "text": "x",
                                     "artifact": None, "source": "s", "metadata": {}}]}]}
EP_3TURN = {"episode_id": "e2", "subject_id": "s2", "metadata": {},
            "turns": [
                {"turn_id": "t1", "time": "2026-01-01T09:00:00+00:00", "message": "a", "evidence": []},
                {"turn_id": "t2", "time": "2026-01-03T09:00:00+00:00", "message": "b", "evidence": []},
                {"turn_id": "t3", "time": "2026-01-06T09:00:00+00:00", "message": "c", "evidence": []},
            ]}


def test_roundtrip_single_and_multi_turn(tmp_path):
    d = make_dataset(tmp_path, [EP_1TURN, EP_3TURN])
    assert validate_dataset(d) == []
    ds = load_dataset(d, with_targets=True)
    assert [e.episode_id for e in ds.episodes] == ["e1", "e2"]
    assert ds.episode("e1").turns[0].time is None  # single-turn may omit time
    assert len(ds.episode("e2").turns) == 3


def test_validator_catches_structural_errors(tmp_path):
    # non-monotonic multi-turn
    bad = json.loads(json.dumps(EP_3TURN))
    bad["turns"][1]["time"] = "2025-01-01T00:00:00+00:00"
    errs = validate_dataset(make_dataset(tmp_path, [bad]))
    assert any("non-monotonic" in e for e in errs)
    # null time in multi-turn
    (tmp_path2 := tmp_path / "x")
    tmp_path2.mkdir()
    bad2 = json.loads(json.dumps(EP_3TURN))
    bad2["turns"][1]["time"] = None
    assert any("null turn time" in e for e in validate_dataset(make_dataset(tmp_path2, [bad2])))


def test_validator_catches_ids_splits_targets_artifacts(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    dup = [json.loads(json.dumps(EP_1TURN)), json.loads(json.dumps(EP_1TURN))]
    errs = validate_dataset(make_dataset(d, dup))
    assert any("duplicate episode ids" in e for e in errs)

    d = tmp_path / "b"
    d.mkdir()
    ep = json.loads(json.dumps(EP_1TURN))
    ep["turns"][0]["evidence"][0]["artifact"] = "../escape.png"
    make_dataset(d, [ep])
    assert any("unsafe artifact" in e for e in validate_dataset(d))

    d2 = tmp_path / "c"
    d2.mkdir()
    make_dataset(d2, [EP_1TURN], dataset_extra={"splits": {"dev": ["ghost"]}})
    assert any("unknown episode 'ghost'" in e for e in validate_dataset(d2))

    d3 = tmp_path / "d"
    d3.mkdir()
    make_dataset(d3, [EP_1TURN], targets=[{"episode_id": "ghost", "turns": {}}])
    assert any("targets reference unknown episode" in e for e in validate_dataset(d3))

    d4 = tmp_path / "e"
    d4.mkdir()
    ep2 = json.loads(json.dumps(EP_1TURN))
    ep2["turns"][0]["evidence"][0]["text"] = ""
    make_dataset(d4, [ep2])
    assert any("neither text nor artifact" in e for e in validate_dataset(d4))


def test_with_targets_false_never_parses_targets(tmp_path):
    d = make_dataset(tmp_path, [EP_1TURN],
                     targets=[{"episode_id": "e1", "turns": {"t1": {"answers": ["SECRET"]}}}])
    ds = load_dataset(d, with_targets=False)
    assert ds.targets == {}
    assert validate_dataset(d) == []  # validator (eval-side) does see them
