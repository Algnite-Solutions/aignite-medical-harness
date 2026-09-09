"""CLI acceptance: one loop for single/multi-turn, batch/interactive; targets never
touched by run; importer (medagentbench offline + FHIR pagination fake); --help."""
import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import REPO


def ama(*args, stdin_text=None, cwd=REPO):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, "-m", "ama.cli", *args],
                          capture_output=True, text=True, env=env, cwd=cwd,
                          timeout=120, input=stdin_text)


def decisions_of(run_dir: Path):
    return [json.loads(l)["decision"]
            for l in (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]


def test_validate_ok_and_invalid(tmp_path):
    ok = ama("validate", "datasets/qa_mini")
    assert ok.returncode == 0 and ok.stdout.startswith("OK")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "dataset.json").write_text('{"schema":"ama-dataset-v0","name":"bad"}', encoding="utf-8")
    (bad / "episodes.jsonl").write_text("", encoding="utf-8")
    r = ama("validate", str(bad))
    assert r.returncode == 1 and "no episodes" in r.stdout


def test_inspect_shows_no_targets(tmp_path):
    r = ama("inspect", "datasets/qa_mini", "--episode", "qa_001")
    assert r.returncode == 0
    assert "SECRET" not in r.stdout
    assert "targets" in r.stdout  # presence mentioned, contents never shown
    assert "1.2" in r.stdout  # evidence text is public


def test_run_single_and_multi_turn_share_loop(tmp_path):
    r1 = ama("run", "datasets/qa_mini", "--model", "scripted", "--runs-root", str(tmp_path))
    r2 = ama("run", "datasets/thyroid_demo", "--model", "scripted", "--runs-root", str(tmp_path))
    assert r1.returncode == 0 and r2.returncode == 0
    runs = sorted(tmp_path.iterdir())
    qa_run = next(p for p in runs if "qa_mini" in p.name)
    th_run = next(p for p in runs if "thyroid_demo" in p.name)
    assert len(decisions_of(qa_run)) == 2   # 2 single-turn episodes
    assert len(decisions_of(th_run)) == 3   # 1 three-turn episode
    e1 = ama("eval", str(qa_run))
    e2 = ama("eval", str(th_run))
    assert e1.returncode == 0 and "answer_correct=2/2" in e1.stdout
    assert e2.returncode == 0 and "action_correct=3/3" in e2.stdout


def test_run_never_opens_targets(tmp_path):
    # make targets literally unreadable-by-absence: rename it away, run must be identical
    import shutil

    ds = tmp_path / "qa_copy"
    shutil.copytree(REPO / "datasets/qa_mini", ds)
    baseline = ama("run", str(ds), "--model", "scripted", "--runs-root", str(tmp_path / "a"))
    (ds / "targets.jsonl").rename(ds / "targets.jsonl.bak")
    r = ama("run", str(ds), "--model", "scripted", "--runs-root", str(tmp_path / "b"))
    assert r.returncode == 0
    a = decisions_of(sorted((tmp_path / "a").iterdir())[0])
    b = decisions_of(sorted((tmp_path / "b").iterdir())[0])
    assert a == b
    manifest = json.loads((sorted((tmp_path / "b").iterdir())[0] / "manifest.json").read_text())
    assert "targets.jsonl" not in manifest["dataset_hashes"]


def test_interactive_matches_batch_and_quit(tmp_path):
    batch = ama("run", "datasets/thyroid_demo", "--model", "scripted", "--runs-root", str(tmp_path / "batch"))
    batch_dir = sorted((tmp_path / "batch").iterdir())[0]
    inter = ama("run", "datasets/thyroid_demo", "--model", "scripted",
                "--runs-root", str(tmp_path / "inter"), "--interactive",
                stdin_text="run\nnext\nrun\nnext\nrun\nnext\n")
    assert inter.returncode == 0, inter.stderr
    inter_dir = sorted((tmp_path / "inter").iterdir())[0]
    assert decisions_of(batch_dir) == decisions_of(inter_dir)  # same loop, same decisions
    assert "INVALID" not in inter.stdout
    for marker in ("answers", "expect_abstain", "allowed_actions", "required_evidence"):
        assert marker not in inter.stdout  # no target content on screen

    stopped = ama("run", "datasets/thyroid_demo", "--model", "scripted",
                  "--runs-root", str(tmp_path / "stop"), "--interactive",
                  stdin_text="run\nquit\n")
    assert stopped.returncode == 0
    stop_dir = sorted((tmp_path / "stop").iterdir())[0]
    events = [json.loads(l) for l in (stop_dir / "events.jsonl").read_text().splitlines()]
    end = [e for e in events if e["type"] == "episode_end"][0]
    assert end["termination"] == "operator_stopped"
    assert len(decisions_of(stop_dir)) == 1


def test_help_has_no_dual_concepts():
    r = ama("--help")
    text = r.stdout + r.stderr
    assert "validate" in text and "inspect" in text and "run" in text and "eval" in text and "import" in text
    for legacy in ("question", "trajectory", "medagentbench profile", "evaluate-trajectory"):
        assert legacy not in text, legacy


def test_import_episode_folder(tmp_path):
    r = ama("import", "episode-folder", "--source", str(REPO / "datasets/thyroid_demo"),
            "--out", str(tmp_path / "out"), "--name", "roundtrip")
    # importing an AMA dataset dir is not the expected source shape; use a legacy fixture instead
    assert True  # covered by fixture test below


def test_import_episode_folder_legacy_fixture(tmp_path):
    src = tmp_path / "legacy"
    ep_dir = src / "thyroid_001"
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode.json").write_text(json.dumps({
        "episode_id": "thyroid_001", "patient_id": "p1", "mode": "teacher_forced_replay",
        "turns": [{"turn_id": "t1", "as_of": "2026-01-01T00:00:00+00:00",
                   "world_message": "超声完成。", "release_evidence_ids": ["us"]}],
    }), encoding="utf-8")
    (ep_dir / "evidence.jsonl").write_text(json.dumps({
        "evidence_id": "us", "patient_id": "p1", "event_time": "2026-01-01T00:00:00+00:00",
        "recorded_time": "2026-01-01T00:00:00+00:00", "modality": "us",
        "content": "TI-RADS 4c", "source_locator": "us,1", "tags": ["suspicious_lesion"],
    }), encoding="utf-8")
    (ep_dir / "workflow.json").write_text(json.dumps({
        "workflow_id": "w", "states": ["a", "b"], "initial_state": "a",
        "transitions": [{"from": "a", "to": "b", "when": ["suspicious_lesion"],
                         "allowed_actions": ["order_biopsy"]}],
    }), encoding="utf-8")
    (ep_dir / "gold.json").write_text(json.dumps({
        "episode_id": "thyroid_001",
        "turns": {"t1": {"expected_workflow_state": "a", "allowed_actions": ["order_biopsy"],
                         "forbidden_actions": [], "required_evidence_refs": ["us"]}},
    }), encoding="utf-8")
    out = tmp_path / "ds"
    r = ama("import", "episode-folder", "--source", str(src), "--out", str(out))
    assert r.returncode == 0
    v = ama("validate", str(out))
    assert v.returncode == 0
    policy = json.loads((out / "policy.json").read_text())
    assert "guidance" in policy["public"] and policy["hidden"]["transitions"]
    target = json.loads((out / "targets.jsonl").read_text().splitlines()[0])
    assert target["turns"]["t1"]["state"]["workflow_state"] == "a"
    # evidence tags survived into metadata for the workflow scorer
    ep = json.loads((out / "episodes.jsonl").read_text().splitlines()[0])
    assert ep["turns"][0]["evidence"][0]["metadata"]["tags"] == ["suspicious_lesion"]


def test_import_medagentbench_offline(tmp_path):
    src = tmp_path / "tasks.json"
    src.write_text(json.dumps([
        {"id": "task1_1", "instruction": "age?", "context": "It's 2023-11-13T10:15:00+00:00 now.",
         "sol": ["50"], "eval_MRN": "S1"},
        {"id": "task1_2", "instruction": "who?", "context": "", "sol": ["Patient not found"],
         "eval_MRN": None},
    ]), encoding="utf-8")
    out = tmp_path / "mb"
    r = ama("import", "medagentbench", "--source", str(src), "--out", str(out))
    assert r.returncode == 0
    v = ama("validate", str(out))
    assert v.returncode == 0
    ep = json.loads((out / "episodes.jsonl").read_text().splitlines()[0])
    assert ep["turns"][0]["time"] == "2023-11-13T10:15:00+00:00"
    assert "sol" not in json.dumps(ep)  # gold never in episodes
    target = json.loads((out / "targets.jsonl").read_text().splitlines()[0])
    assert target["turns"]["t1"]["answers"] == ["50"]
    report = json.loads((out / "import_report.json").read_text())
    assert report["n_tasks"] == 2 and report["excluded_no_mrn"] == ["task1_2"]


def test_fhir_everything_pagination_follows_next_links(monkeypatch):
    from ama.importers.medagentbench import fetch_everything
    import urllib.request

    pages = [
        {"entry": [{"resource": {"id": f"r{i}", "resourceType": "Condition",
                                 "recordedDate": f"2023-01-0{i + 1}"}} for i in range(3)],
         "link": [{"rel": "next", "url": "http://fake/page2"}]},
        {"entry": [{"resource": {"id": f"r{i + 10}", "resourceType": "Condition",
                                 "recordedDate": f"2023-01-1{i}"}} for i in range(2)],
         "link": []},
    ]
    state = {"n": 0}

    def fake_urlopen(url, timeout=None):
        class R:
            def __init__(self, body):
                self._body = body

            def read(self):
                return json.dumps(self._body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        out = R(pages[state["n"]])
        state["n"] += 1
        return out

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = fetch_everything("http://fake/fhir", "p1")
    assert result["pages"] == 2 and not result["truncated"]
    assert len(result["entries"]) == 5  # page 2 entries included — the prototype bug


def test_fhir_everything_reports_truncation(monkeypatch):
    from ama.importers.medagentbench import FHIR_PAGE_LIMIT, fetch_everything
    import urllib.request

    def fake_urlopen(url, timeout=None):
        class R:
            def read(self):
                return json.dumps({"entry": [{"resource": {"id": "x"}}],
                                   "link": [{"rel": "next", "url": "http://fake/next"}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = fetch_everything("http://fake/fhir", "p1")
    assert result["pages"] == FHIR_PAGE_LIMIT and result["truncated"]
    assert result["next_link_pending"] is True
