import pytest

from ama.cxr_radgraph import combined_f1, prediction_report, score_mimic_cxr_radgraph
from ama.data import Dataset, DatasetInfo, Episode, TargetRecord


def graph(label="Observation::definitely present", *, relation=True):
    return {"entities": {
        "1": {"tokens": "opacity", "label": label,
              "relations": [["located_at", "2"]] if relation else []},
        "2": {"tokens": "lung", "label": "Anatomy::definitely present", "relations": []}}}


def test_negation_and_relation_change_graph_scores():
    ref = graph()
    assert combined_f1(ref, ref)["radgraph_combined"] == 1
    negated = graph("Observation::definitely absent")
    assert combined_f1(ref, negated) == {
        "entity_f1": 0.5, "relation_f1": 0.0, "radgraph_combined": 0.25}
    assert combined_f1(ref, graph(relation=False))["radgraph_combined"] == 0.5
    assert combined_f1({"entities": {}}, {"entities": {}})["radgraph_combined"] == 0


def dataset(tmp_path):
    episodes = [Episode(id=str(i), turns=[{"id": "t1", "evidence": [{"id": "img", "text": "synthetic"}]}])
                for i in range(2)]
    return Dataset(tmp_path, DatasetInfo(schema="ama-dataset", name="synthetic", splits={}), episodes,
                   {e.id: TargetRecord(id=e.id, turns={"t1": {"report": "SYNTHETIC REF"}})
                    for e in episodes}, None)


def test_missing_outputs_stay_in_denominator_and_reference_is_authoritative(tmp_path):
    calls = []

    class FakeExtractor:
        metadata = {"synthetic": True}

        def __call__(self, texts):
            calls.extend(texts)
            return [graph() for _ in texts]

    ds = dataset(tmp_path)
    decisions = {"0": [{"turn_id": "t1", "termination": "completed", "decision": {
        "answer": {"findings": "SYNTHETIC", "impression": ""}}}]}
    result = score_mimic_cxr_radgraph(ds, decisions, extractor=FakeExtractor())
    assert calls == ["SYNTHETIC REF", "Findings: SYNTHETIC Impression: "]
    assert result["aggregate"]["radgraph_combined_macro"] == 0.5
    assert result["aggregate"]["valid_report_radgraph_combined_macro"] == 1
    assert result["aggregate"]["failed_or_invalid_reports"] == 1


def test_bad_reference_and_extraction_fail_loudly(tmp_path):
    ds = dataset(tmp_path)
    ds.targets["0"].turns["t1"].pop("report")
    with pytest.raises(ValueError, match="authoritative"):
        score_mimic_cxr_radgraph(ds, {})
    ds = dataset(tmp_path)
    decisions = {"0": [{"turn_id": "t1", "termination": "completed", "decision": {
        "answer": {"findings": "x", "impression": ""}}}]}
    with pytest.raises(RuntimeError, match="count mismatch"):
        score_mimic_cxr_radgraph(ds, decisions, extractor=lambda _: [])


@pytest.mark.parametrize("value", [None, {}, {"findings": "x"}, {"findings": 3, "impression": "x"},
                                  {"findings": " ", "impression": ""}])
def test_invalid_report(value):
    assert prediction_report(value) is None
