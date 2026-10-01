import json

import pytest

from ama.data import load_dataset, validate_dataset
from ama.cli import _eval_run
from ama.importers.mimic_cdm import import_mimic_cdm
from ama.importers.mimic_cdm_benchmark import (
    build_benchmark, score_mimic_cdm_open, _diagnosis_label, _select_ids,
)
from ama.importers.mimic_cdm_compare import compare_runs
from ama.runner import execute
from ama.scorer import REGISTRY
from fakes import FakeModel, call, calls
from test_mimic_cdm import _source


def test_paired_views_have_same_task_targets_and_hpi(tmp_path):
    processed = tmp_path / "processed"
    import_mimic_cdm(_source(tmp_path / "source"), processed)
    out = tmp_path / "benchmark"
    build_benchmark(processed / "mimic_cdm_full_info",
                    processed / "mimic_cdm_interactive", out, ids=["2", "1"])
    variants = [out / f"mimic_cdm_open_{name}" for name in ("hpi", "interactive", "full_info")]
    datasets = [load_dataset(folder, with_targets=True) for folder in variants]
    assert all(validate_dataset(folder) == [] for folder in variants)
    assert all(ds.info.splits["all"] == ["2", "1"] for ds in datasets)
    assert all(ds.targets == datasets[0].targets for ds in datasets)
    assert [ep.turns[0].evidence[0] for ep in datasets[0].episodes] == [
        ep.turns[0].evidence[0] for ep in datasets[1].episodes]
    assert len(datasets[0].episodes[0].turns[0].evidence) == 1
    assert len(datasets[2].episodes[0].turns[0].evidence) > 1
    assert all("appendicitis" not in (folder / "instructions.txt").read_text()
               for folder in variants)
    assert len({(folder / "instructions.txt").read_text().split("\n")[0]
                for folder in variants}) == 1
    assert REGISTRY["mimic_cdm_open"] is score_mimic_cdm_open


def test_open_score_is_conservative_and_exposes_unmapped_answers(tmp_path):
    processed = tmp_path / "processed"
    import_mimic_cdm(_source(tmp_path / "source"), processed)
    out = tmp_path / "benchmark"
    build_benchmark(processed / "mimic_cdm_full_info",
                    processed / "mimic_cdm_interactive", out, ids=["1", "2"])
    dataset = load_dataset(out / "mimic_cdm_open_full_info", with_targets=True)
    decisions = {
        "1": [{"turn_id": "t1", "decision": {"answer": {"diagnosis": "Acute appendicitis"}},
               "termination": "completed"}],
        "2": [{"turn_id": "t1", "decision": {"answer": {
            "diagnosis": "Could be cholecystitis or pancreatitis"}}, "termination": "completed"}],
    }
    aggregate = score_mimic_cdm_open(dataset, decisions)["aggregate"]
    assert aggregate["diagnosis_accuracy_auto"]["num"] == 1
    assert aggregate["diagnosis_mapped"]["num"] == 1
    assert aggregate["needs_review"] == 1
    assert aggregate["review_episode_ids"] == ["2"]
    assert _diagnosis_label("No evidence of appendicitis") == (None, "ambiguous")
    assert _diagnosis_label("Acute pancreatitis without necrosis") == ("pancreatitis", "mapped")


def test_mismatched_source_variant_fails_before_writing(tmp_path):
    processed = tmp_path / "processed"
    import_mimic_cdm(_source(tmp_path / "source"), processed)
    tool_dir = processed / "mimic_cdm_interactive"
    cases_path = tool_dir / "cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    cases[0]["hpi"] = "different admission"
    cases_path.write_text("\n".join(json.dumps(case) for case in cases) + "\n")
    out = tmp_path / "benchmark"
    with pytest.raises(ValueError, match="HPI mismatch"):
        build_benchmark(processed / "mimic_cdm_full_info", tool_dir, out, ids=["1"])
    assert not out.exists()


def test_balanced_holdout_selection_excludes_pilot():
    labels = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
    targets = [{"id": f"{label}-{index}", "turns": {"t1": {"answer": {"diagnosis": label}}}}
               for label in labels for index in range(3)]
    pilot = _select_ids(targets, None, 1, 42, set())
    holdout = _select_ids(targets, None, 1, 43, set(pilot))
    assert len(pilot) == len(holdout) == 4
    assert not set(pilot) & set(holdout)
    assert _select_ids(targets, None, 1, 42, set()) == pilot


def test_paired_comparison_separates_model_and_tool_effects(tmp_path):
    processed = tmp_path / "processed"
    import_mimic_cdm(_source(tmp_path / "source"), processed)
    out = tmp_path / "benchmark"
    build_benchmark(processed / "mimic_cdm_full_info",
                    processed / "mimic_cdm_interactive", out, ids=["1", "2"])
    matrix = {}
    for model_name in ("model-a", "model-b"):
        matrix[model_name] = {}
        for variant in ("hpi", "interactive", "full_info"):
            second = ("pancreatitis" if (model_name, variant) in {
                ("model-a", "hpi"), ("model-b", "interactive")} else "cholecystitis")
            def reply(diagnosis):
                return json.dumps({"turn_id": "t1", "answer": {"diagnosis": diagnosis},
                                   "citations": ["hpi"],
                                   "reasoning_summary": "The available findings support this diagnosis."})
            messages = [reply("acute appendicitis"), reply(second)]
            if model_name == "model-a" and variant == "interactive":
                messages.insert(0, calls(call("physical_examination", "{}")))
            model = FakeModel(*messages)
            model.name = model_name
            model.config = {"model": model_name}
            folder = out / f"mimic_cdm_open_{variant}"
            run = execute(folder, model, runs_root=tmp_path / "runs",
                          tools_path="src/ama/importers/mimic_cdm_tools.py"
                          if variant == "interactive" else None)
            assert _eval_run(run) == 0
            matrix[model_name][variant] = run
    report = compare_runs(matrix)
    assert report["paired"]["model-a"]["interactive_vs_hpi"]["left_only"] == 1
    assert report["paired"]["model-b"]["interactive_vs_hpi"]["right_only"] == 1
    assert report["paired"]["models_on_full_info"]["both_correct"] == 2
    assert report["results"]["model-a"]["interactive"]["tool_calls"] == 1
    history_file = matrix["model-a"]["hpi"] / "messages.json"
    histories = json.loads(history_file.read_text())
    histories["1"][0]["content"] = "A different diagnosis task"
    history_file.write_text(json.dumps(histories))
    with pytest.raises(ValueError, match="actual system prompt"):
        compare_runs(matrix)
