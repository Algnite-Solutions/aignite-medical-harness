"""Regression cases from PR #16; no network or model-provider calls."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from ama import medical_tools as medical
from ama.agent import Agent
from ama.guideline_retrieval import classify
from fakes import FakeModel, call, calls
from test_medical_tools import ARTICLE, FULLTEXT, ncbi_responses


FIXTURE = Path(__file__).parent / "fixtures/guideline_retrieval/reference_records.json"
REFERENCES = {"appendicitis": "32295644", "cholecystitis": "33153472",
              "diverticulitis": "32381121", "pancreatitis": "31210778"}


def online(monkeypatch, tmp_path, records):
    index = tmp_path / "guidelines.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("AMA_GUIDELINES_PATH", str(index))
    urls = []
    monkeypatch.setattr(medical, "_get", lambda url: urls.append(url) or json.dumps({"resultList": {"result": records}}))
    return urls


def record(pmid, title, year="2024", **extra):
    return {"pmid": pmid, "title": title, "pubYear": year,
            "pubTypeList": {"pubType": ["Practice Guideline"]}, **extra}


@pytest.mark.parametrize("disease", REFERENCES)
def test_legacy_field_prefix_never_matches_ckd_index(monkeypatch, disease):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    results = medical.guideline_search(f"TITLE:{disease}", live=False)
    assert results and all("KDIGO" not in row["title"] for row in results)
    assert results[0]["pmid"] == REFERENCES[disease]


@pytest.mark.parametrize("query", ["CKD", "慢性肾脏病", "慢性肾病", "TITLE:CKD", "CKD SGLT2"])
def test_ckd_aliases_still_match(monkeypatch, query):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    rows = medical.guideline_search(query, live=False)
    assert len(rows) == 1 and rows[0]["org"] == "KDIGO"


@pytest.mark.parametrize("query,disease", [("阑尾炎", "appendicitis"), ("胆囊炎", "cholecystitis"),
                                         ("结肠憩室炎", "diverticulitis"), ("胰腺炎", "pancreatitis"),
                                         ("gallbladder inflammation", "cholecystitis")])
def test_chinese_and_synonym_topics(monkeypatch, query, disease):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    assert medical.guideline_search(query, live=False)[0]["pmid"] == REFERENCES[disease]


@pytest.mark.parametrize("query", ["org", "url", "year", "source", "key_points"])
def test_json_keys_and_urls_are_not_topic_content(monkeypatch, query):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    assert medical.guideline_search(query, live=False) == []


@pytest.mark.parametrize("disease", REFERENCES)
def test_fixed_four_disease_relevance_with_high_citation_distractors(monkeypatch, tmp_path, disease):
    references = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
    noise = [record("26462967", "Thyroid nodule guidelines", abstractText=f"Background mentions {disease}.", citedByCount=999999),
             record("28137831", f"STARD reporting guidelines for {disease}", citedByCount=999999),
             record("26799652", f"Laboratory protocol and autophagy assays for {disease}", citedByCount=999999)]
    urls = online(monkeypatch, tmp_path, noise + references)
    result = medical.guideline_search(disease)
    # Narrow fixture precision/coverage, not a claim about clinical diagnostic accuracy.
    assert [r["pmid"] for r in result] == [REFERENCES[disease]]
    assert result[0]["document_type"] == "clinical_practice_guideline"
    assert result[0]["classification_basis"] and result[0]["relevance_basis"] == "topic in title"
    params = parse_qs(urlsplit(urls[0]).query)
    assert f'TITLE_ABS:"{disease}"' in params["query"][0]
    assert "sort" not in params and "CITED" not in params["query"][0]
    assert result[0]["pmcid"] and result[0]["fulltext_url"]


@pytest.mark.parametrize("title,kind", [
    ("Pancreatitis clinical practice guideline", "clinical_practice_guideline"),
    ("Pancreatitis international consensus", "consensus"),
    ("STARD pancreatitis reporting guideline", "reporting_guideline"),
    ("Pancreatitis laboratory protocol", "laboratory_protocol"),
    ("Systematic review of pancreatitis guidelines", "other"),
])
def test_document_type_has_explicit_basis(title, kind):
    actual, basis = classify(title, ["Guideline"])
    assert actual == kind and basis


def test_live_fixture_excludes_studies_and_comments_about_guidelines():
    fixture = FIXTURE.with_name("live_candidates.json")
    records = {r.get("pmid") or r.get("pmcid"): r for r in json.loads(fixture.read_text(encoding="utf-8"))["records"]}
    excluded = {"42613617", "42169135", "41322888", "40825891", "42516005", "42112895", "41928009", "42065164",
                "42740752", "40679968", "37751156"}
    assert excluded <= records.keys()
    for pmid in excluded:
        row = records[pmid]
        kind, evidence = classify(row["title"], row["pubTypeList"]["pubType"])
        assert kind == "other", row["title"]
        assert evidence
    for pmid in ("41604201", "42390126", "42216617", "41493264"):
        row = records[pmid]
        assert classify(row["title"], row["pubTypeList"]["pubType"])[0] == "clinical_practice_guideline"


def test_explicit_query_subtype_is_not_discarded(monkeypatch, tmp_path):
    online(monkeypatch, tmp_path, [record("1", "Acute pancreatitis guidelines"),
                                   record("2", "Chronic pancreatitis guidelines")])
    assert [r["pmid"] for r in medical.guideline_search("chronic pancreatitis")] == ["2"]


def test_population_subtype_intent_and_year_filters(monkeypatch, tmp_path):
    rows = [record("1", "Adult acute pancreatitis diagnosis guidelines", "2018"),
            record("2", "Adult acute pancreatitis diagnosis guidelines", "2024"),
            record("3", "Pediatric acute pancreatitis diagnosis guidelines", "2025"),
            record("4", "Adult autoimmune pancreatitis diagnosis consensus", "2025"),
            record("5", "Adult acute pancreatitis surgery guidelines", "2023"),
            record("6", "Acute pancreatitis diagnosis guidelines", "2025")]
    urls = online(monkeypatch, tmp_path, rows)
    result = medical.guideline_search("pancreatitis", population="adult", subtype="acute", intent="diagnosis", year_to=2024)
    assert [r["pmid"] for r in result] == ["2", "1"]
    assert result[0]["version"] == {"publication_year": 2024, "latest_status": "not_verified", "year_to": 2024}
    assert result[0]["applicability"]["population"] == ["adult"]
    assert "FIRST_PDATE:[1000-01-01 TO 2024-12-31]" in parse_qs(urlsplit(urls[0]).query)["query"][0]
    autoimmune = medical.guideline_search("pancreatitis", subtype="autoimmune")
    assert [r["pmid"] for r in autoimmune] == ["4"]


def test_local_year_cutoff_and_diagnostic_reference_metadata(monkeypatch):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    assert medical.guideline_search("appendicitis", live=False, year_to=2019) == []
    for disease, pmid in REFERENCES.items():
        row = medical.guideline_search(disease, live=False, intent="diagnosis")[0]
        assert row["pmid"] == pmid and row["org"] == "World Society of Emergency Surgery"
        assert row["org_source"] and row["diagnostic_sections"]
        assert row["metadata_checked_at"] == "2026-10-21"
    pancreas = medical.guideline_search("pancreatitis", live=False)[0]
    assert pancreas["applicability"]["severity"] == ["severe"]


def test_abstract_truncation_and_org_not_invented(monkeypatch, tmp_path):
    online(monkeypatch, tmp_path, [record("1", "Pancreatitis guideline", pmcid="PMC123", doi="10.1/test",
           abstractText="x" * 900 + " Diagnostic criteria.", journalInfo={"journal": {"title": "Example Journal"}})])
    row = medical.guideline_search("pancreatitis", intent="diagnosis")[0]
    assert row["org"] == "" and row["journal"] == "Example Journal"
    assert len(row["abstract"]) == 800 and row["abstract_truncated"] is True
    assert row["pmid"] == "1" and row["pmcid"] == "PMC123" and row["doi"] == "10.1/test"


def test_nullable_identifiers_do_not_deduplicate_different_records(monkeypatch, tmp_path):
    online(monkeypatch, tmp_path, [
        record(None, "Pancreatitis guideline", source="PMC", id="PMC123", doi=None),
        record(None, "Pancreatitis consensus", source="PMC", id="PMC456", doi=None),
    ])
    rows = medical.guideline_search("pancreatitis")
    assert len(rows) == 2
    assert {r["pmcid"] for r in rows} == {"PMC123", "PMC456"}
    assert all(r["pmid"] == "" and r["doi"] == "" for r in rows)


def test_raw_expression_is_separate_from_local_topic(monkeypatch, tmp_path):
    urls = online(monkeypatch, tmp_path, [])
    medical.guideline_search("pancreatitis", search_expression='TITLE:"initial management"')
    assert 'AND (TITLE:"initial management")' in parse_qs(urlsplit(urls[0]).query)["query"][0]
    with pytest.raises(ValueError, match="search_expression"):
        medical.guideline_search("TITLE:pancreatitis AND FIRST_PDATE:[2010 TO 2020]")


@pytest.mark.parametrize("kwargs", [{"population": []}, {"subtype": "unknown"}, {"intent": "invalid"},
                                  {"year_to": True}, {"year_to": "2020"}, {"search_expression": []}])
def test_invalid_scope_cannot_contact_network(monkeypatch, kwargs):
    monkeypatch.setattr(medical, "_get", lambda url: pytest.fail("invalid request contacted server"))
    with pytest.raises(ValueError):
        medical.guideline_search("pancreatitis", **kwargs)


def test_pubmed_ids_preserved_from_xml_and_summary(monkeypatch):
    xml = ARTICLE.replace("</PubmedArticle>", '<PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC123</ArticleId>'
                          '<ArticleId IdType="doi">10.1/example</ArticleId></ArticleIdList></PubmedData></PubmedArticle>')
    ncbi_responses(monkeypatch, fetch=xml)
    row = medical.pubmed_search("kidney")[0]
    assert row["pmcid"] == "PMC123" and row["doi"] == "10.1/example"
    assert row["fulltext_url"].endswith("PMC123")

    def summary(endpoint, **kwargs):
        if endpoint == "esearch":
            return '{"esearchresult":{"idlist":["123"]}}'
        return json.dumps({"result": {"123": {"title": "Study", "articleids": [
            {"idtype": "pmc", "value": "PMC123"}, {"idtype": "doi", "value": "10.1/example"}]}}})
    monkeypatch.setattr(medical, "_ncbi", summary)
    assert medical.pubmed_search("kidney", abstract=False)[0]["pmcid"] == "PMC123"


@pytest.mark.parametrize("name", ["guideline_search", "pubmed_search"])
def test_search_to_fulltext_agent_chain(monkeypatch, name):
    xml = ARTICLE.replace("</PubmedArticle>", '<PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC123</ArticleId>'
                          '</ArticleIdList></PubmedData></PubmedArticle>')
    ncbi_responses(monkeypatch, fetch=xml)
    original_get = medical._get

    def get(url):
        return FULLTEXT if url.endswith("/fullTextXML") else original_get(url)
    monkeypatch.setattr(medical, "_get", get)
    if name == "guideline_search":
        monkeypatch.setattr(medical, "_local_guidelines", lambda query: [
            {"title": "Kidney guideline", "url": "https://example.org/guide", "pmcid": "PMC123"}])
    arguments = {"query": "kidney", **({"live": False} if name == "guideline_search" else {})}

    class ChainedModel(FakeModel):
        def complete(self, history, tools=None):
            if len(self.requests) == 1:
                result = json.loads(history[-1]["content"])
                self.pmcid = result[0]["pmcid"]
                assert self.pmcid == "PMC123"
                self.replies = iter([calls(call("paper_fulltext", json.dumps({"pmcid": self.pmcid, "section": "cohort"}), id="read")), "done"])
            return super().complete(history, tools)

    model = ChainedModel(calls(call(name, json.dumps(arguments), id="search")), "placeholder", "done")
    agent = Agent(model, tools=medical.TOOLS)
    assert agent.chat("Find and read the study") == "done"
    observations = [m for m in agent.history if m["role"] == "tool"]
    section = json.loads(observations[1]["content"])["sections"][0]
    assert section["id"] == "cohort" and section["text"] == "Participants\nSecond paragraph."
    assert section["source_url"] == "https://europepmc.org/articles/PMC123#cohort"
    assert [m["tool_call_id"] for m in observations] == ["search", "read"]


def test_pmid_resolution_and_missing_open_fulltext(monkeypatch):
    urls = []
    def get(url):
        urls.append(url)
        if url.endswith("/fullTextXML"):
            return FULLTEXT
        return json.dumps({"resultList": {"result": [{"pmid": "999", "pmcid": "PMC999"},
                                                     {"pmid": "123", "pmcid": "PMC123"}]}})
    monkeypatch.setattr(medical, "_get", get)
    result = medical.paper_fulltext(pmid="123", section="cohort")
    assert result["pmcid"] == "PMC123" and result["pmid"] == "123"
    assert "EXT_ID:123 AND SRC:MED" in parse_qs(urlsplit(urls[0]).query)["query"][0]
    assert len(urls) == 2 and result["sections"][0]["id"] == "cohort"
    monkeypatch.setattr(medical, "_get", lambda url: '{"resultList":{"result":[{"pmid":"123"}]}}')
    unavailable = medical.paper_fulltext(pmid="123")
    assert "error" in unavailable and unavailable["fulltext_status"] == "no_pmcid"
    assert unavailable["pmcid"] == ""
    monkeypatch.setattr(medical, "_get", lambda url: "<error>XML not available</error>")
    xml_unavailable = medical.paper_fulltext("PMC123")
    assert "error" in xml_unavailable and xml_unavailable["fulltext_status"] == "xml_unavailable"


def test_fixed_reference_set_metrics(monkeypatch, tmp_path):
    records = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
    online(monkeypatch, tmp_path, records)
    hits = sum(medical.guideline_search(disease, max_results=1)[0]["pmid"] == pmid
               for disease, pmid in REFERENCES.items())
    assert hits / len(REFERENCES) == 1.0  # P@1 on this four-reference fixture only.
    assert len({row["pmcid"] for row in records}) / len(REFERENCES) == 1.0


@pytest.mark.parametrize("kwargs", [{}, {"pmcid": "PMC123", "pmid": "123"}, {"pmid": "../123"}])
def test_fulltext_identifier_validation(monkeypatch, kwargs):
    monkeypatch.setattr(medical, "_get", lambda url: pytest.fail("invalid ID contacted server"))
    with pytest.raises(ValueError):
        medical.paper_fulltext(**kwargs)
