"""Native grading, transcript integrity, and optional upstream differential checks."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from ama.cli import _eval_run
from ama.data import load_dataset
from ama.importers import medagentbench as med
from ama.runner import execute
from ama.scorer import REGISTRY
from fakes import FakeModel, call, calls, decision
from test_medagentbench import resources, source, medication, observation, service, write_snapshot


UPSTREAM = Path(__file__).resolve().parents[2] / "MedAgentBench/src/server/tasks/medagentbench/refsol.py"


@pytest.fixture
def grading_resources(resources):
    for code, value, time in (("MG", 1.2, med._EVAL_TIME), ("K", 3.3, med._EVAL_TIME),
                              ("A1C", 7.0, "2023-11-01T00:00:00+00:00")):
        resources["Observation"].append({"resourceType": "Observation", "id": code,
            "subject": {"reference": "Patient/P1"}, "code": {"coding": [{"code": code}]},
            "effectiveDateTime": time, "valueQuantity": {"value": value, "unit": "fixture-unit"}})
    return resources


def build(source, raw, folder, tasks=range(1, 11)):
    path = source / "data/medagentbench/test_data_v2.json"
    rows = [{"id": f"task{task}_1", "instruction": "Fixture task.", "context": med._EVAL_TIME,
             "eval_MRN": "M1", **({"sol": ["M1"]} if task == 1 else {})} for task in tasks]
    med._json(path, rows)
    med.import_medagentbench(source, folder, snapshot=write_snapshot(folder.with_name(folder.name + "-snapshot"), raw))
    return load_dataset(folder, with_targets=True)


@pytest.fixture
def grading_dataset(source, grading_resources, tmp_path):
    return build(source, grading_resources, tmp_path / "grading")


def golden():
    answers = {1: {"mrn": "M1"}, 2: {"age": 43}, 3: {"completed": True}, 4: {"value": 1.2},
               5: {"completed": True}, 6: {"value": 4.25}, 7: {"value": 4.25},
               8: {"completed": True}, 9: {"completed": True},
               10: {"value": 7.0, "observed_at": "2023-11-01T00:00:00+00:00"}}
    actions = {task: [] for task in answers}
    actions[3] = [("record_observation", {"observation": observation()})]
    actions[5] = [("submit_order", {"order": medication(code={"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"},
                   dose={"value": 2, "unit": "g"}, rate={"value": 2, "unit": "h"}, route="IV")})]
    actions[8] = [("submit_order", {"order": service("referral", code={"system": "http://snomed.info/sct", "code": "306181000000106"}, note=med._SBAR)})]
    actions[9] = [("submit_order", {"order": medication()}),
                  ("submit_order", {"order": service(occurrence_at="2023-11-14T08:00:00+00:00")})]
    return answers, actions


def history(folder, episode, actions):
    tools = {tool.name: tool for tool in med.make_tools(folder, episode)}
    rows = []
    for index, (name, args) in enumerate(actions):
        call_id = f"call-{index}"
        rows.append(calls(call(name, json.dumps(args), id=call_id)))
        try:
            result = tools[name].invoke(json.dumps(args))
            content = result.text + "\nCiteable evidence IDs: " + json.dumps(result.evidence_ids)
        except ValueError as exc:
            content = json.dumps({"error": str(exc)})
        rows.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return rows


def inputs(dataset, answers=None, actions=None):
    golden_answers, golden_actions = golden()
    answers = golden_answers if answers is None else answers
    actions = golden_actions if actions is None else actions
    decisions, histories = {}, {}
    for ep in dataset.episodes:
        task = int(ep.id.split("_")[0][4:])
        decisions[ep.id] = [{"turn_id": "t1", "termination": "completed",
                             "decision": {"answer": answers[task]}}]
        histories[ep.id] = history(dataset.folder, ep.id, actions[task])
    return decisions, histories


def test_all_ten_tasks_score_native_answers_and_actual_actions(grading_dataset, monkeypatch):
    monkeypatch.setattr(med, "_get_json", lambda *a: pytest.fail("scoring must be offline"))
    decisions, histories = inputs(grading_dataset)
    assert REGISTRY["medagentbench"] is med.score_medagentbench
    result = med.score_medagentbench(grading_dataset, decisions, histories=histories)
    assert result["aggregate"]["task_success"] == {"num": 10, "den": 10, "value": 1}
    assert all(group["value"] == 1 for group in result["aggregate"]["per_task"].values())
    assert result["per_episode"]["task9_1"]["accepted_submissions"] == 2
    assert result["per_episode"]["task2_1"]["reference"]["age"] == 43
    # MRN M1 resolves to Patient.id P1. Neither patient identity nor the answer is autofilled.
    assert result["per_episode"]["task3_1"]["reference"]["patient_id"] == "P1"


@pytest.mark.parametrize("task,wrong", [(1, {"mrn": "wrong"}), (2, {"age": True}), (4, {"value": 9}),
    (5, {"value": 9}), (6, {"value": 4.4}), (7, {"value": 93}), (9, {"value": 9}),
    (10, {"value": 7, "observed_at": "wrong-time"})])
def test_wrong_answers_fail_independently_of_correct_actions(grading_dataset, task, wrong):
    answers, actions = golden()
    answers[task] = wrong
    decisions, histories = inputs(grading_dataset, answers, actions)
    row = med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"][f"task{task}_1"]
    assert not row["correct"] and not row["answer_correct"] and row["actions_correct"]


@pytest.mark.parametrize("task,field,value", [
    (3, "patient_id", "P2"), (3, "value", "120/80"), (3, "category", "wrong"),
    (3, "observed_at", "2023-11-13T10:16:00+00:00"),
    (5, "dose", {"value": 1, "unit": "g"}), (5, "rate", {"value": 2, "unit": "wrong"}),
    (5, "route", "oral"), (5, "authored_at", "2023-11-13T10:16:00+00:00"),
    (8, "note", "wrong SBAR"), (8, "priority", "routine"), (8, "patient_id", "P2"),
    (9, "dose", {"value": 40, "unit": "mEq"}), (9, "intent", "proposal"),
])
def test_wrong_structured_actions_are_not_rescued_by_final_answer(grading_dataset, task, field, value):
    answers, actions = golden()
    payload = actions[task][0][1]
    payload[next(iter(payload))][field] = value
    decisions, histories = inputs(grading_dataset, answers, actions)
    row = med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"][f"task{task}_1"]
    assert row["answer_correct"] and not row["actions_correct"] and not row["correct"]


def test_missing_duplicate_reversed_and_mistimed_orders_fail(grading_dataset):
    for task, transform in [(3, lambda a: []), (8, lambda a: a + a),
                            (9, lambda a: list(reversed(a))), (9, lambda a: a[:1])]:
        answers, actions = golden()
        actions[task] = transform(actions[task])
        decisions, histories = inputs(grading_dataset, answers, actions)
        assert not med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"][f"task{task}_1"]["correct"]
    answers, actions = golden()
    actions[9][1][1]["order"]["occurrence_at"] = "2023-11-15T08:00:00+00:00"
    decisions, histories = inputs(grading_dataset, answers, actions)
    assert not med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"]["task9_1"]["correct"]


def test_attempted_writes_fail_read_only_tasks_but_retry_can_succeed(grading_dataset):
    answers, actions = golden()
    failed = ("submit_order", {"order": medication(authored_at="bad-time")})
    actions[1] = [failed]
    actions[3].insert(0, failed)
    decisions, histories = inputs(grading_dataset, answers, actions)
    result = med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"]
    assert not result["task1_1"]["correct"]
    assert result["task1_1"]["submission_attempts"] == 1 and result["task1_1"]["accepted_submissions"] == 0
    assert result["task3_1"]["correct"] and result["task3_1"]["submission_attempts"] == 2


@pytest.mark.parametrize("damage", ["missing", "orphan", "arguments", "episode", "sequence", "order"])
def test_invalid_transcripts_cannot_supply_successful_operations(grading_dataset, damage):
    decisions, histories = inputs(grading_dataset)
    ep = "task9_1"
    if damage == "missing":
        del histories[ep]
    elif damage == "orphan":
        histories[ep].pop(0)
    elif damage == "arguments":
        histories[ep][0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"order": medication(patient_id="P2")})
    elif damage == "order":
        histories[ep][1]["tool_call_id"] = "unrelated-id"
    else:
        body = json.loads(histories[ep][1]["content"].split("\nCiteable")[0])
        body["episode_id" if damage == "episode" else "submission_id"] = "wrong"
        histories[ep][1]["content"] = json.dumps(body)
    row = med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"][ep]
    assert not row["correct"] and not row["history_valid"] and row["history_error"]


def test_missing_failed_and_abstained_answers_remain_in_denominator(grading_dataset):
    decisions, histories = inputs(grading_dataset)
    del decisions["task1_1"]
    decisions["task2_1"][0]["termination"] = "api_error"
    decisions["task3_1"][0]["decision"]["answer"] = None
    aggregate = med.score_medagentbench(grading_dataset, decisions, histories=histories)["aggregate"]
    assert aggregate["task_success"] == {"num": 7, "den": 10, "value": 0.7}
    assert med.score_medagentbench(grading_dataset, {}, histories=histories)["aggregate"]["task_success"]["num"] == 0
    with pytest.raises(ValueError, match="messages.json"):
        med.score_medagentbench(grading_dataset, decisions)


@pytest.mark.parametrize("born,age", [("1980-11-13", 43), ("1980-11-14", 42), ("1980-02-29", 43)])
def test_age_uses_fixed_benchmark_date(source, grading_resources, tmp_path, born, age):
    grading_resources["Patient"][0]["birthDate"] = born
    dataset = build(source, grading_resources, tmp_path / "age", tasks=[2])
    decisions, histories = inputs(dataset, {2: {"age": age}}, {2: []})
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 1


def test_not_found_missing_labs_and_reference_errors(grading_dataset):
    decisions, histories = inputs(grading_dataset)
    grading_dataset.targets["task1_1"].turns["t1"]["case_data"]["sol"] = ["Patient not found"]
    decisions["task1_1"][0]["decision"]["answer"] = {"mrn": "Patient not found"}
    assert med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"]["task1_1"]["correct"]
    del grading_dataset.targets["task1_1"]
    with pytest.raises(ValueError, match="target"):
        med.score_medagentbench(grading_dataset, decisions, histories=histories)


@pytest.mark.parametrize("k", [3.5, 3.6, None])
def test_potassium_without_indicated_replacement_requires_no_attempts(source, grading_resources, tmp_path, k):
    grading_resources["Observation"] = [row for row in grading_resources["Observation"] if row["id"] != "K" or k is not None]
    for row in grading_resources["Observation"]:
        if row["id"] == "K":
            row["valueQuantity"]["value"] = k
    dataset = build(source, grading_resources, tmp_path / "potassium", tasks=[9])
    decisions, histories = inputs(dataset, {9: {"value": k if k is not None else -1}}, {9: []})
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 1
    _, actions = golden()
    decisions, histories = inputs(dataset, {9: {}}, actions)
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 0


def test_cutoff_includes_boundary_and_future_and_ties_keep_source_order(source, grading_resources, tmp_path):
    raw = grading_resources
    mg = next(row for row in raw["Observation"] if row["id"] == "MG")
    raw["Observation"] = [copy.deepcopy(mg) for _ in range(4)]
    for index, (row, date, value) in enumerate(zip(raw["Observation"],
        ["2023-11-12T10:14:59Z", "2023-11-12T10:15:00Z", "2023-11-14T10:15:00Z", "2023-11-14T10:15:00Z"],
        [99, 1, 2, 3])):
        row.update(id=str(index), effectiveDateTime=date)
        row["code"]["coding"][0]["code"] = "GLU"
        row["valueQuantity"]["value"] = value
    dataset = build(source, raw, tmp_path / "times", tasks=[6, 7])
    decisions, histories = inputs(dataset, {6: {"value": 2}, 7: {"value": 2}}, {6: [], 7: []})
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 1
    # These unusual details intentionally match the original grader, which has
    # a lower cutoff only and resolves latest-time ties by first encounter.


def test_empty_lab_history_uses_minus_one_and_average_tolerance(source, grading_resources, tmp_path):
    grading_resources["Observation"] = []
    dataset = build(source, grading_resources, tmp_path / "missing-labs", tasks=[4, 6, 7])
    decisions, histories = inputs(dataset, {4: {"value": -1}, 6: {"value": -1}, 7: {"value": -1}}, {4: [], 6: [], 7: []})
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 1
    decisions["task6_1"][0]["decision"]["answer"]["value"] = -0.95
    assert med.score_medagentbench(dataset, decisions, histories=histories)["per_episode"]["task6_1"]["correct"]
    decisions["task6_1"][0]["decision"]["answer"]["value"] = -0.5
    assert not med.score_medagentbench(dataset, decisions, histories=histories)["per_episode"]["task6_1"]["correct"]


def test_prose_claiming_submission_is_not_an_operation(grading_dataset):
    decisions, histories = inputs(grading_dataset)
    histories["task3_1"] = [{"role": "user", "content": "POST request accepted"},
                             {"role": "assistant", "content": "Blood pressure was recorded."}]
    row = med.score_medagentbench(grading_dataset, decisions, histories=histories)["per_episode"]["task3_1"]
    assert not row["correct"] and row["accepted_submissions"] == 0


@pytest.mark.parametrize("mg,dose", [(0.9, 4), (1.0, 2), (1.49, 2), (1.5, 1), (1.9, 1), (1.91, None), (None, None)])
def test_magnesium_thresholds_and_no_order_cases(source, grading_resources, tmp_path, mg, dose):
    raw = grading_resources
    raw["Observation"] = [r for r in raw["Observation"] if r["id"] != "MG" or mg is not None]
    for row in raw["Observation"]:
        if row["id"] == "MG":
            row["valueQuantity"]["value"] = mg
    dataset = build(source, raw, tmp_path / "mg", tasks=[5])
    actions = {5: [] if dose is None else [("submit_order", {"order": medication(
        code={"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"}, route="IV",
        dose={"value": dose, "unit": "g"}, rate={"value": dose, "unit": "h"})})]}
    decisions, histories = inputs(dataset, {5: {"completed": True}}, actions)
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 1
    if dose is None:
        actions[5] = [("submit_order", {"order": medication(authored_at="bad")})]
        decisions, histories = inputs(dataset, {5: {}}, actions)
        assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 0


@pytest.mark.parametrize("date,needs_order", [("2022-11-13T10:14:59+00:00", True),
    ("2022-11-13T10:15:00+00:00", False), ("2022-11-13T10:15:01+00:00", False), (None, True)])
def test_a1c_year_boundary_and_missing_measurements(source, grading_resources, tmp_path, date, needs_order):
    raw = grading_resources
    raw["Observation"] = [r for r in raw["Observation"] if r["id"] != "A1C" or date is not None]
    for row in raw["Observation"]:
        if row["id"] == "A1C":
            row["effectiveDateTime"] = date
    dataset = build(source, raw, tmp_path / "a1c", tasks=[10])
    action = ("submit_order", {"order": service(code={"system": "http://loinc.org", "code": "4548-4"})})
    answers = {10: {"value": 7.0, "observed_at": date} if date else {"value": -1}}
    actions = {10: [action] if needs_order else []}
    decisions, histories = inputs(dataset, answers, actions)
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 1
    decisions, histories = inputs(dataset, {10: {"completed": True}}, actions)
    assert med.score_medagentbench(dataset, decisions, histories=histories)["aggregate"]["task_success"]["value"] == 1


def test_cli_eval_reads_history_and_keeps_output_quality_separate(grading_dataset, tmp_path):
    model = FakeModel(calls(call("record_observation", json.dumps({"observation": observation()}))),
                      decision(answer={"completed": True}))
    run = execute(grading_dataset.folder, model, episode_ids=["task3_1"],
                  tools_path=Path("src/ama/importers/medagentbench.py"), runs_root=tmp_path / "runs")
    assert _eval_run(run) == 0
    scored = json.loads((run / "metrics.json").read_text())["scored"]
    assert scored["aggregate"]["task_success"]["value"] == 1
    assert scored["aggregate"]["output_quality"]["strict_output_compliance"]["value"] == 0
    (run / "messages.json").unlink()
    with pytest.raises(ValueError, match="messages.json"):
        _eval_run(run)


def original_grader(raw):
    """Run trusted upstream rules with an in-memory FHIR response, without requests."""
    tree = ast.parse(UPSTREAM.read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level and node.module == "utils")]
    def get(url):
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if parsed.path.endswith("Patient"):
            rows = [p for p in raw["Patient"] if any(i.get("value") == query["identifier"][0] for i in p.get("identifier", []))]
        else:
            patient = next(p for p in raw["Patient"] if any(i.get("value") == query["patient"][0] for i in p.get("identifier", [])))
            rows = [r for r in raw["Observation"]
                    if med._patient_reference(r["subject"]["reference"], {p["id"] for p in raw["Patient"]}) == patient["id"]
                    and query["code"][0] in [c.get("code") for c in r["code"].get("coding", [])]]
        return {"data": json.dumps({"entry": [{"resource": row} for row in rows]})}
    namespace = {"send_get_request": get}
    exec(compile(tree, str(UPSTREAM), "exec"), namespace)
    return namespace


def legacy_payload(name, args):
    field = "order" if name == "submit_order" else "observation"
    row = args[field]
    # The native interface uses Patient.id; original graders hardcode eval_MRN.
    patient = "M1" if row["patient_id"] == "P1" else row["patient_id"]
    if name == "record_observation":
        return "Observation", {"resourceType": "Observation", "subject": {"reference": f"Patient/{patient}"},
            "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category", "code": row["category"], "display": "Vital Signs"}]}],
            "code": {"text": row["code"]}, "effectiveDateTime": row["observed_at"], "status": row["status"],
            "valueString": f"{row['value']} {row['unit']}"}
    kind = "MedicationRequest" if row["order_type"] == "medication" else "ServiceRequest"
    payload = {"resourceType": kind, "subject": {"reference": f"Patient/{patient}"},
               "authoredOn": row["authored_at"], "status": row["status"], "intent": row["intent"]}
    if kind == "MedicationRequest":
        payload.update(medicationCodeableConcept={"coding": [row["code"]]}, dosageInstruction=[{
            "route": row["route"], "doseAndRate": [{"doseQuantity": row["dose"],
                **({"rateQuantity": row["rate"]} if row.get("rate") is not None else {})}]}])
    else:
        payload.update(code={"coding": [row["code"]]}, priority=row["priority"])
        if row.get("occurrence_at") is not None:
            payload["occurrenceDateTime"] = row["occurrence_at"]
        if row.get("note") is not None:
            payload["note"] = {"text": row["note"]}
    return kind, payload


@pytest.mark.skipif(not UPSTREAM.is_file(), reason="upstream MedAgentBench checkout is required for differential tests")
def test_native_rules_agree_with_original_grader(grading_dataset, grading_resources):
    upstream = original_grader(grading_resources)
    answers, actions = golden()
    variants = [(answers, actions)]
    for task in range(1, 11):
        bad_answers, bad_actions = golden()
        if task in (3, 8):
            bad_actions[task] = []
        else:
            bad_answers[task] = {"mrn": "wrong", "age": -99, "value": -99, "observed_at": "wrong"}
        variants.append((bad_answers, bad_actions))
    # Real accepted posts with wrong clinical contents, duplicate posts and reversed posts.
    for task in (3, 5, 8, 9):
        bad_answers, bad_actions = golden()
        payload = bad_actions[task][0][1]
        payload[next(iter(payload))]["patient_id"] = "P2"
        variants.append((bad_answers, bad_actions))
    for task in (3, 5, 8, 9):
        bad_answers, bad_actions = golden()
        bad_actions[task] += copy.deepcopy(bad_actions[task])
        variants.append((bad_answers, bad_actions))
    bad_answers, bad_actions = golden()
    bad_actions[9].reverse()
    variants.append((bad_answers, bad_actions))
    for answers, actions in variants:
        decisions, histories = inputs(grading_dataset, answers, actions)
        native = med.score_medagentbench(grading_dataset, decisions, histories=histories)
        for ep in grading_dataset.episodes:
            task = int(ep.id.split("_")[0][4:])
            answer = answers[task]
            finish = [answer["mrn"]] if task == 1 else [answer["age"]] if task == 2 else []
            if task in (4, 5, 6, 7, 9, 10) and "value" in answer:
                finish = [answer["value"]]
                if task == 10 and "observed_at" in answer:
                    finish.append(answer["observed_at"])
            posts = []
            for name, args in actions[task]:
                kind, payload = legacy_payload(name, args)
                posts += [SimpleNamespace(role="agent", content=f"POST http://fixture/{kind}\n" + json.dumps(payload)),
                          SimpleNamespace(role="user", content="POST request accepted")]
            result = SimpleNamespace(result=json.dumps(finish), history=posts)
            case = grading_dataset.targets[ep.id].turns["t1"]["case_data"]
            assert native["per_episode"][ep.id]["correct"] == upstream[f"task{task}"](case, result, "http://fixture/")
