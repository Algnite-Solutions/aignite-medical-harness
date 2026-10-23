"""All source images/questions/answers in this module are synthetic fixtures."""
import hashlib
import io
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from ama.cli import _eval_run, main
from ama.data import load_dataset, validate_dataset
from ama.importers.vqa_rad import REPO, REVISION, import_vqa_rad
from ama.importers.vqa_rad_scoring import SCORER, answer_metrics, normalize_answer, score_vqa_rad
from ama.runner import execute
from ama.scorer import REGISTRY
from fakes import FakeModel

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
Image = pytest.importorskip("PIL.Image")


def source_fixture(root: Path, *, invalid=None):
    """Mimic the pinned snapshot's layout, never download or redistribute real QA."""
    rows = {}
    for split, answers in {"train": ["privateorchid", "yes", "two lesions"],
                           "test": ["no", "IVC", "yes", "positive"]}.items():
        rows[split] = []
        for i, answer in enumerate(answers):
            stream = io.BytesIO()
            Image.new("RGB", (6, 5), (i * 50, 100, 120)).save(stream, format="PNG")
            rows[split].append({"image": {"bytes": stream.getvalue(), "path": "unused.png"},
                                "question": f"Synthetic question {i}?", "answer": answer})
    if invalid:
        rows["test"][0].update(invalid)
    snapshot = root / "snapshot" / "data"
    snapshot.mkdir(parents=True)
    files = []
    for split, values in rows.items():
        path = snapshot / f"{split}-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(values), path)
        files.append({"path": f"data/{path.name}", "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    # A manifest is not proof of authenticity; these files are deliberately synthetic.
    (root / "source-manifest.json").write_text(json.dumps({
        "repo_id": REPO, "revision": REVISION, "files": files,
        "fixture": "synthetic test data, not the upstream snapshot",
    }))
    return root


def test_import_split_identity_and_answer_isolation(tmp_path):
    source = source_fixture(tmp_path / "source")
    out = tmp_path / "full"
    report = import_vqa_rad(source, out)
    assert validate_dataset(out) == []
    public, evaluator = load_dataset(out), load_dataset(out, with_targets=True)
    assert public.targets == {} and public.eval_config is None
    assert report["source_counts"] == report["split_counts"] == {"train": 3, "test": 4}
    assert set(public.info.splits["train"]).isdisjoint(public.info.splits["test"])
    assert public.info.splits["test"] == [f"vqarad-test-{i:06d}" for i in range(4)]
    assert all(len(ep.turns) == 1 and len(ep.turns[0].evidence) == 2 for ep in public.episodes)
    assert "privateorchid" not in (out / "episodes.jsonl").read_text()
    assert "question_type" not in (out / "episodes.jsonl").read_text()
    assert evaluator.targets["vqarad-train-000000"].turns["t1"]["answer"] == {"text": "privateorchid"}
    assert evaluator.targets["vqarad-test-000003"].turns["t1"]["question_type"] == "open"
    assert report["source_image_overlap_train_test"] == 3
    assert report["n_artifacts"] == 4
    checksums = json.loads((out / "checksums.json").read_text())
    assert set(checksums["files"]) == {str(p.relative_to(out)) for p in out.rglob("*")
                                      if p.is_file() and p.name != "checksums.json"}
    for name, sha in checksums["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    for name in ("DATASET_CARD.md", "DATASET_CARD.zh-CN.md", "provenance.jsonl", "instructions.txt"):
        assert (out / name).is_file()


def test_smoke_selection_stable_source_order_and_no_overwrite(tmp_path):
    source = source_fixture(tmp_path / "source")
    a, b = tmp_path / "a", tmp_path / "b"
    report = import_vqa_rad(source, a, subset="smoke", smoke_count=2, seed=42)
    import_vqa_rad(source, b, subset="smoke", smoke_count=2, seed=42)
    indices = sorted(random.Random(42).sample(range(4), 2))
    assert report["selection"]["selected_row_indices"] == {"train": [], "test": indices}
    assert load_dataset(a).info.splits == {"train": [], "test": [f"vqarad-test-{i:06d}" for i in indices]}
    for name in ("episodes.jsonl", "targets.jsonl", "provenance.jsonl", "import_report.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    before = (a / "checksums.json").read_bytes()
    with pytest.raises(FileExistsError):
        import_vqa_rad(source, a)
    assert (a / "checksums.json").read_bytes() == before


@pytest.mark.parametrize("invalid", [
    {"question": " "}, {"answer": ""}, {"image": {"bytes": b"corrupt", "path": ""}},
    {"image": {"bytes": None, "path": "/not/allowed.png"}},
])
def test_invalid_records_fail_without_partial_dataset(tmp_path, invalid):
    source = source_fixture(tmp_path / "source", invalid=invalid)
    out = tmp_path / "invalid"
    with pytest.raises((ValueError, OSError)):
        import_vqa_rad(source, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".ama-vqarad-*"))


@pytest.mark.parametrize("change", ["checksum", "revision", "missing_split", "unsafe", "extra_shard"])
def test_source_manifest_failures(tmp_path, change):
    source = source_fixture(tmp_path / "source")
    path = source / "source-manifest.json"
    manifest = json.loads(path.read_text())
    if change == "checksum":
        manifest["files"][0]["sha256"] = "0" * 64
    elif change == "revision":
        manifest["revision"] = "main"
    elif change == "missing_split":
        manifest["files"] = manifest["files"][:1]
    elif change == "unsafe":
        manifest["files"][0]["path"] = "../outside.parquet"
    else:
        (source / "snapshot/data/extra.parquet").write_bytes(b"unexpected")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        import_vqa_rad(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(("raw", "normalized"), [
    ("  YES!!! ", "yes"), ("positive", "yes"), ("absent", "no"),
    ("Two  lesions.", "2 lesions"), ("IVC", "inferior vena cava"),
    ("L MCA", "left middle cerebral artery"), ("left-sided", "leftsided"),
    ("the lung", "the lung"), ("ＮＯ", "ｎｏ"), ("yes, a lesion", "yes a lesion"),
])
def test_official_normalization_examples(raw, normalized):
    assert normalize_answer(raw) == normalized


def test_heuristic_f1_and_strict_yes_no():
    assert answer_metrics("yes, a lesion", "yes")["accuracy_heuristic"] == 0
    assert answer_metrics("TRUE!", "yes")["accuracy_heuristic"] == 1
    assert answer_metrics("lesion", "two lesions")["accuracy_heuristic"] == 0
    assert answer_metrics("lesions", "two lesions")["token_f1"] == pytest.approx(2 / 3)
    assert answer_metrics("lesions", "two lesions")["accuracy_heuristic"] == pytest.approx(1 / 3)
    assert answer_metrics("lung lung", "lung")["token_f1"] == pytest.approx(2 / 3)
    assert answer_metrics("", "yes")["accuracy_heuristic"] == 0


def make_row(answer, *, termination="completed", turn="t1"):
    return {"turn_id": turn, "decision": {"answer": answer}, "termination": termination,
            "model_calls": 1, "usage": None, "duration_ms": 0}


def test_scorer_denominators_and_malformed_decisions(tmp_path):
    out = tmp_path / "dataset"
    import_vqa_rad(source_fixture(tmp_path / "source"), out)
    ds = load_dataset(out, with_targets=True)
    ds.episodes = ds.select(split="test")
    ids = ds.info.splits["test"]
    decisions = {
        ids[0]: [make_row({"text": "negative"})],
        ids[1]: [make_row({"text": "inferior vena cava"})],
        ids[2]: [make_row({"text": "yes, it is present"})],
        # "positive" has staged type=open, but belongs to normalized yes/no metrics.
    }
    result = score_vqa_rad(ds, decisions)
    assert result["aggregate"]["accuracy_heuristic"] == {"num": 2.0, "den": 4, "value": 0.5}
    assert result["aggregate"]["yes_no_accuracy"]["den"] == 3
    assert result["by_question_type"]["open"]["exact_match"]["den"] == 2
    assert not result["per_episode"][ids[3]]["submitted"]
    for invalid in (None, 1, ["no"], {"text": 1}, {"text": ""}, {"text": "no", "extra": "yes"}):
        assert score_vqa_rad(ds, {ids[0]: [make_row(invalid)]})["aggregate"]["accuracy_heuristic"]["num"] == 0
    assert score_vqa_rad(ds, {ids[0]: [make_row({"text": "no"}, termination="api_error")]})["aggregate"]["accuracy_heuristic"]["num"] == 0
    for invalid in ({"unknown": []}, {ids[0]: [make_row({"text": "no"}, turn="t2")]},
                    {ids[0]: [make_row({"text": "no"})] * 2}):
        with pytest.raises(ValueError):
            score_vqa_rad(ds, invalid)
    ds.targets.pop(ids[0])
    with pytest.raises(ValueError, match="target"):
        score_vqa_rad(ds, {})


def test_cli_fake_run_eval_and_run_loader_never_reads_targets(tmp_path):
    out = tmp_path / "dataset"
    assert main(["import", "vqa-rad", "--source", str(source_fixture(tmp_path / "source")),
                 "--out", str(out)]) == 0
    ds = load_dataset(out, with_targets=True)
    model = FakeModel(*(json.dumps({"turn_id": "t1", "answer": target.turns["t1"]["answer"],
                                    "citations": ["image", "question"],
                                    "reasoning_summary": "Synthetic test response."})
                        for target in ds.targets.values()))
    run = execute(out, model, runs_root=tmp_path / "runs")
    assert len(model.requests) == 7
    assert "privateorchid" not in json.dumps(model.requests[0][0])
    assert _eval_run(run) == 0
    result = json.loads((run / "metrics.json").read_text())["scored"]
    assert result["aggregate"]["accuracy_heuristic"]["value"] == 1
    assert REGISTRY[SCORER] is score_vqa_rad
    # An invalid evaluator file must not affect execution or appear in prompts.
    for name in ("targets.jsonl", "eval.json", "provenance.jsonl", "import_report.json"):
        (out / name).write_text("INVALID evaluator-only secret")
    fake = FakeModel(json.dumps({"turn_id": "t1", "answer": {"text": "synthetic"},
                                 "citations": ["image", "question"], "reasoning_summary": "Test."}))
    execute(out, fake, episode_ids=[ds.episodes[0].id], runs_root=tmp_path / "isolated-runs")
    assert "evaluator-only secret" not in json.dumps(fake.requests)


def test_independent_verifier_checks_source_content_beyond_checksums(tmp_path):
    source = source_fixture(tmp_path / "source")
    out = tmp_path / "dataset"
    import_vqa_rad(source, out)
    script = Path(__file__).resolve().parents[1] / "scripts/verify_vqa_rad.py"
    command = [sys.executable, str(script), "--source", str(source), "--dataset", str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["offline_validation"] == "passed"
    lines = (out / "targets.jsonl").read_text().splitlines()
    changed = json.loads(lines[0])
    changed["turns"]["t1"]["answer"]["text"] = "wrong synthetic reference"
    lines[0] = json.dumps(changed)
    (out / "targets.jsonl").write_text("\n".join(lines) + "\n")
    # Even internally consistent generated hashes must not hide a changed target.
    checksums = json.loads((out / "checksums.json").read_text())
    checksums["files"]["targets.jsonl"] = hashlib.sha256((out / "targets.jsonl").read_bytes()).hexdigest()
    checksums["overall_sha256"] = hashlib.sha256(json.dumps(
        checksums["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (out / "checksums.json").write_text(json.dumps(checksums))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "reference answer changed" in result.stderr
