import json
from email.message import Message
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from ama import medical_tools as medical
from ama.agent import Agent
from ama.tools import load_tools
from fakes import FakeModel, call, calls


ARTICLE = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID Version="1">123</PMID>
<Article><Journal><Title>Test Journal</Title><JournalIssue><PubDate>
<MedlineDate>2024 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
<ArticleTitle>Kidney <i>disease</i> &amp; care</ArticleTitle><Abstract>
<AbstractText Label="METHODS">A <b>trial</b>.</AbstractText>
<AbstractText Label="RESULTS">Improved outcomes.</AbstractText>
</Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def ncbi_responses(monkeypatch, *, fetch=ARTICLE, ids=None):
    requests = []

    def get(url):
        requests.append(url)
        endpoint = urlsplit(url).path.rsplit("/", 1)[1]
        if endpoint == "esearch.fcgi":
            return json.dumps({"esearchresult": {"idlist": ["123"] if ids is None else ids}})
        if endpoint == "efetch.fcgi":
            if isinstance(fetch, Exception):
                raise fetch
            return fetch
        return json.dumps({"result": {"123": {"title": "Summary <i>title</i>",
                                               "fulljournalname": "Journal", "pubdate": "2024 Jan"}}})

    monkeypatch.setattr(medical, "_get", get)
    monkeypatch.setattr(medical.time, "sleep", lambda _: None)
    return requests


def test_pubmed_query_escaping_abstracts_and_caller_identity(monkeypatch):
    requests = ncbi_responses(monkeypatch)
    monkeypatch.setenv("NCBI_EMAIL", "test@example.org")
    monkeypatch.setenv("NCBI_API_KEY", "test-key")
    query = 'kidney AND ("2020"[dp] : "2026"[dp]) & SGLT2'
    result = medical.pubmed_search(query, retmax=2)
    assert result == [{"pmid": "123", "pmcid": "", "doi": "", "fulltext_url": "", "fulltext_links": [],
                       "fulltext_status": "no_pmcid", "title": "Kidney disease & care", "journal": "Test Journal",
                       "year": "2024", "abstract": "METHODS: A trial. RESULTS: Improved outcomes.",
                       "abstract_truncated": False,
                       "url": "https://pubmed.ncbi.nlm.nih.gov/123/"}]
    params = parse_qs(urlsplit(requests[0]).query)
    assert params["term"] == [query]
    assert params["email"] == ["test@example.org"] and params["api_key"] == ["test-key"]
    assert params["sort"] == ["relevance"] and params["retmax"] == ["2"]


@pytest.mark.parametrize("fetch", ["<broken", "<PubmedArticleSet/>", ValueError("offline")])
def test_pubmed_falls_back_to_summary(monkeypatch, fetch):
    requests = ncbi_responses(monkeypatch, fetch=fetch)
    result = medical.pubmed_search("kidney")
    assert result[0]["abstract"] == "" and result[0]["title"] == "Summary title"
    assert "esummary.fcgi" in requests[-1]


def test_pubmed_no_abstract_skips_fetch_and_empty_search_stops(monkeypatch):
    requests = ncbi_responses(monkeypatch)
    assert medical.pubmed_search("kidney", abstract=False)[0]["abstract"] == ""
    assert len(requests) == 2 and not any("efetch" in url for url in requests)
    requests = ncbi_responses(monkeypatch, ids=[])
    assert medical.pubmed_search("no matches") == [] and len(requests) == 1


def test_pubmed_unavailable_is_distinct_from_empty_results(monkeypatch):
    monkeypatch.setattr(medical, "_ncbi", lambda *args, **kwargs: '{"error":"rate limit"}')
    assert "error" in medical.pubmed_search("kidney")


def test_ncbi_rate_limit_applies_to_each_request(monkeypatch):
    sleeps = []
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.setattr(medical, "_NCBI_LAST", 10.0)
    monkeypatch.setattr(medical.time, "monotonic", lambda: 10.1)
    monkeypatch.setattr(medical.time, "sleep", sleeps.append)
    monkeypatch.setattr(medical, "_get", lambda url: "ok")
    medical._ncbi("esearch", term="test")
    assert sleeps[0] == pytest.approx(0.25)
    monkeypatch.setenv("NCBI_API_KEY", "key")
    medical._ncbi("efetch", id="123")
    assert sleeps[1] == pytest.approx(0.11)


def test_offline_guidelines_never_access_network(monkeypatch):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    monkeypatch.setattr(medical, "_get", lambda url: pytest.fail("offline search made a network request"))
    results = medical.guideline_search("CKD SGLT2", live=False)
    assert results and all(row["source"] == "local_index" for row in results)
    assert results[0]["org"] == "KDIGO" and results[0]["key_points"]
    assert medical.guideline_search("unrelated-condition-xyz", live=False) == []


def test_guideline_live_mapping_and_query(monkeypatch, tmp_path):
    index = tmp_path / "guidelines.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("AMA_GUIDELINES_PATH", str(index))
    urls = []

    def get(url):
        urls.append(url)
        return json.dumps({"resultList": {"result": [
            {"pmid": "123", "title": "Chronic kidney disease SGLT2 <i>guideline</i>", "pubYear": "2024",
             "journalInfo": {"journal": {"title": "Journal"}}, "pubTypeList": {"pubType": ["Guideline"]},
             "abstractText": "<p>Evidence &amp; care</p>"},
            {"source": "PMC", "id": "PMC999", "doi": "10.1/test", "title": "Another chronic kidney disease SGLT2 guideline"}]}})

    monkeypatch.setattr(medical, "_get", get)
    result = medical.guideline_search("chronic kidney disease SGLT2", max_results=3)
    params = parse_qs(urlsplit(urls[0]).query)
    assert 'TITLE_ABS:"chronic kidney disease"' in params["query"][0]
    assert 'TITLE_ABS:"sglt2"' in params["query"][0]
    assert params["resultType"] == ["core"] and "sort" not in params
    assert result[0]["abstract"] == "Evidence & care" and result[0]["org"] == ""
    assert result[0]["journal"] == "Journal"
    assert result[1]["url"] == "https://doi.org/10.1/test"


def test_guideline_partial_failure_preserves_local_results(monkeypatch):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    monkeypatch.setattr(medical, "_get", lambda url: "invalid JSON")
    result = medical.guideline_search("CKD")
    assert "Europe PMC" in result["error"] and result["results"][0]["source"] == "local_index"


def test_guideline_deduplication_preserves_local_metadata_and_order(monkeypatch):
    local = [{"title": "Curated kidney guideline", "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
              "source": "local_index", "key_points": ["curated detail"]}]
    monkeypatch.setattr(medical, "_local_guidelines", lambda query: local)
    monkeypatch.setattr(medical, "_get", lambda url: json.dumps({"resultList": {"result": [
        {"pmid": "999", "title": "Other kidney guideline"}, {"pmid": "123", "title": "Duplicate kidney guideline"}]}}))
    result = medical.guideline_search("kidney")
    assert [row["title"] for row in result] == ["Curated kidney guideline", "Other kidney guideline"]
    assert result[0]["key_points"] == ["curated detail"]


def test_bad_local_index_does_not_block_live_search(monkeypatch, tmp_path):
    monkeypatch.setenv("AMA_GUIDELINES_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(medical, "_get", lambda url: '{"resultList":{"result":[]}}')
    result = medical.guideline_search("kidney")
    assert "local guideline index" in result["error"] and result["results"] == []


def test_search_text_cleaning_removes_markup_and_noise():
    page = """<html><head><title>Guide &amp; care</title><style>hidden</style></head>
    <body><header>logo</header><nav>menu</nav><main><h1>Kidney care</h1>
    <p>Evidence &amp; <b>recommendations</b>.</p></main><script>bad()</script>
    <footer>copyright</footer></body></html>"""
    assert medical._clean(page) == "Kidney care Evidence & recommendations ."


def test_get_timeout_and_does_not_expose_request_secrets(monkeypatch):
    def fail(request, timeout):
        assert timeout == 20 and request.get_header("User-agent").startswith("ama/")
        raise URLError("https://example.org?api_key=SECRET")

    monkeypatch.setattr(medical, "urlopen", fail)
    with pytest.raises(ValueError, match="Network request failed") as error:
        medical._get("https://example.org?api_key=SECRET")
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("content_type,raw,message", [
    ("application/pdf", b"%PDF", "PDF extraction"),
    ("image/png", b"image", "Unsupported response"),
    ("text/html", b"123456", "download limit"),
])
def test_get_rejects_binary_or_oversized_responses(monkeypatch, content_type, raw, message):
    class Response:
        headers = Message()
        headers["Content-Type"] = content_type

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return raw[:size]

    monkeypatch.setattr(medical, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(medical, "_MAX_BYTES", 5)
    with pytest.raises(ValueError, match=message):
        medical._get("https://example.org/guide")


@pytest.mark.parametrize("kwargs", [{"retmax": 0}, {"retmax": True}, {"retmax": 51}, {"abstract": "false"}])
def test_pubmed_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        medical.pubmed_search("query", **kwargs)


def test_tools_load_and_execute_through_agent():
    tools = load_tools("src/ama/medical_tools.py")
    assert [tool.name for tool in tools] == [
        "pubmed_search", "guideline_search", "paper_fulltext", "clinical_trial_search", "analyze_images"]
    model = FakeModel(calls(call("guideline_search", '{"query":"CKD","live":false}')), "done")
    agent = Agent(model, tools=tools)
    assert agent.chat("Find a guideline offline") == "done"
    assert len(model.requests[0][1]) == 5
    assert json.loads(agent.history[2]["content"])[0]["source"] == "local_index"


FULLTEXT = """<article><front><article-meta><title-group>
<article-title>A <italic>clinical</italic> study</article-title></title-group>
<abstract><p>Short abstract.</p></abstract></article-meta></front><body>
<p>Opening paragraph.</p><sec id="methods"><title>Materials and Methods</title>
<p>First <italic>inline</italic> paragraph.</p><sec id="cohort"><title>Participants</title>
<p>Second paragraph.</p></sec></sec><sec id="results"><title>Results</title>
<p>Observed outcomes.</p></sec></body><back><ref-list><ref>Not body text.</ref></ref-list></back></article>"""


def test_fulltext_sections_preserve_nested_text_and_paragraphs(monkeypatch):
    urls = []
    monkeypatch.setattr(medical, "_get", lambda url: urls.append(url) or FULLTEXT)
    result = medical.paper_fulltext("pmc7092803")
    assert urls == [medical.EPMC_REST + "/PMC7092803/fullTextXML"]
    assert result["title"] == "A clinical study" and result["pmcid"] == "PMC7092803"
    assert [s["id"] for s in result["sections"]] == ["abstract", "body", "methods", "results"]
    assert result["sections"][2]["text"] == (
        "Materials and Methods\nFirst inline paragraph.\nParticipants\nSecond paragraph.")
    assert "Not body text" not in json.dumps(result)
    assert result["truncated"] is False


def test_fulltext_heading_selection_and_total_character_budget(monkeypatch):
    monkeypatch.setattr(medical, "_get", lambda url: FULLTEXT)
    selected = medical.paper_fulltext("PMC7092803", section="METHODS", max_chars=25)
    assert len(selected["sections"]) == 1
    assert selected["sections"][0]["id"] == "methods" and selected["truncated"]
    assert selected["sections"][0]["truncated"] and len(selected["sections"][0]["text"]) == 25
    partial = medical.paper_fulltext("PMC7092803", section="materials", max_chars=1000)
    assert partial["sections"][0]["id"] == "methods"
    limited = medical.paper_fulltext("PMC7092803", max_chars=30)
    assert sum(len(s["text"]) for s in limited["sections"]) == 30 and limited["truncated"]
    missing = medical.paper_fulltext("PMC7092803", section="unknown")
    assert "error" in missing and missing["available_sections"]


def test_fulltext_unstructured_body_and_missing_section_ids(monkeypatch):
    monkeypatch.setattr(medical, "_get", lambda url: "<article><body><p>Unstructured text.</p></body></article>")
    assert medical.paper_fulltext("PMC123")["sections"] == [
        {"id": "body", "title": "Body", "text": "Unstructured text.", "truncated": False,
         "source_url": "https://europepmc.org/articles/PMC123"}]
    monkeypatch.setattr(medical, "_get", lambda url: '<article><body><sec sec-type="methods"><p>Text</p></sec></body></article>')
    assert medical.paper_fulltext("PMC123")["available_sections"] == [{"id": "section-1", "title": "methods"}]


@pytest.mark.parametrize("response", ["<broken", "<response><error>Not available</error></response>",
                                      "<article><body/></article>"])
def test_fulltext_unavailable_or_malformed_returns_error(monkeypatch, response):
    monkeypatch.setattr(medical, "_get", lambda url: response)
    result = medical.paper_fulltext("PMC123")
    assert "error" in result and result["pmcid"] == "PMC123"


@pytest.mark.parametrize("arguments", [
    {"pmcid": "123"}, {"pmcid": "PMC123/../../private"}, {"pmcid": None},
    {"pmcid": "PMC123", "max_chars": 0}, {"pmcid": "PMC123", "section": []},
])
def test_fulltext_bad_arguments_do_not_fetch(monkeypatch, arguments):
    monkeypatch.setattr(medical, "_get", lambda url: pytest.fail("invalid arguments made a request"))
    with pytest.raises(ValueError):
        medical.paper_fulltext(**arguments)


def test_trials_query_filters_and_structured_fields(monkeypatch):
    urls = []
    study = {"protocolSection": {
        "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Clinical study"},
        "statusModule": {"overallStatus": "RECRUITING", "lastUpdatePostDateStruct": {"date": "2026-10-01"}},
        "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL"},
        "conditionsModule": {"conditions": ["Psoriasis"]},
        "armsInterventionsModule": {"interventions": [{"name": "Study drug", "type": "DRUG"}]},
        "descriptionModule": {"briefSummary": "s" * 1100},
        "eligibilityModule": {"eligibilityCriteria": "e" * 3100, "minimumAge": "18 Years", "sex": "ALL",
                              "healthyVolunteers": False},
        "contactsLocationsModule": {"locations": [{"facility": "Hospital", "city": "Boston", "country": "United States"}] * 6},
    }}
    monkeypatch.setattr(medical, "_get", lambda url: urls.append(url) or json.dumps({"studies": [study]}))
    result = medical.clinical_trial_search("psoriasis & arthritis", "drug A+B", "recruiting", max_results=2)
    params = parse_qs(urlsplit(urls[0]).query)
    assert params == {"query.cond": ["psoriasis & arthritis"], "query.intr": ["drug A+B"],
                      "filter.overallStatus": ["RECRUITING"], "pageSize": ["2"], "format": ["json"]}
    row = result[0]
    assert row["nct_id"] == "NCT12345678" and row["url"].endswith("/NCT12345678")
    assert row["phase"] == ["PHASE3"] and row["last_update"] == "2026-10-01"
    assert len(row["summary"]) == 1000 and len(row["eligibility"]["criteria"]) == 3000
    assert row["eligibility"]["healthy_volunteers"] is False and len(row["locations"]) == 5
    assert all(row["truncated"].values())


def test_trials_optional_fields_empty_results_and_errors(monkeypatch):
    monkeypatch.setattr(medical, "_get", lambda url: '{"studies":[]}')
    assert medical.clinical_trial_search("psoriasis") == []
    monkeypatch.setattr(medical, "_get", lambda url: json.dumps({"studies": [{
        "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}}}]}))
    row = medical.clinical_trial_search("psoriasis")[0]
    assert row["locations"] == [] and row["phase"] == [] and row["summary"] == ""
    monkeypatch.setattr(medical, "_get", lambda url: '<html>Access denied</html>')
    assert "error" in medical.clinical_trial_search("psoriasis")
    monkeypatch.setattr(medical, "_get", lambda url: '{"other":[]}')
    assert "error" in medical.clinical_trial_search("psoriasis")


@pytest.mark.parametrize("function,arguments", [
    (medical.paper_fulltext, {"pmcid": "PMC123"}),
    (medical.clinical_trial_search, {"condition": "psoriasis"}),
])
def test_online_tools_return_network_errors(monkeypatch, function, arguments):
    def fail(url):
        raise ValueError("Remote server returned HTTP 404")
    monkeypatch.setattr(medical, "_get", fail)
    assert "error" in function(**arguments)


@pytest.mark.parametrize("arguments", [
    {"condition": ""}, {"condition": "psoriasis", "max_results": True},
    {"condition": "psoriasis", "status": "bogus"}, {"condition": "psoriasis", "intervention": []},
])
def test_trials_validates_arguments(monkeypatch, arguments):
    monkeypatch.setattr(medical, "_get", lambda url: pytest.fail("invalid arguments made a request"))
    with pytest.raises(ValueError):
        medical.clinical_trial_search(**arguments)


@pytest.mark.parametrize("arguments", [
    {"query": ""}, {"query": "CKD", "max_results": 0},
    {"query": "CKD", "max_results": True}, {"query": "CKD", "live": "false"},
])
def test_guidelines_validate_before_network(monkeypatch, arguments):
    monkeypatch.setattr(medical, "_get", lambda url: pytest.fail("invalid arguments made a request"))
    with pytest.raises(ValueError):
        medical.guideline_search(**arguments)


@pytest.mark.parametrize("name,arguments,field", [
    ("pubmed_search", {"query": "kidney", "retmax": 1}, "pmid"),
    ("guideline_search", {"query": "kidney", "max_results": 1}, "title"),
    ("paper_fulltext", {"pmcid": "PMC123", "section": "methods"}, "sections"),
    ("clinical_trial_search", {"condition": "psoriasis", "max_results": 1}, "nct_id"),
])
def test_each_tool_round_trip_through_agent(monkeypatch, tmp_path, name, arguments, field):
    ncbi_responses(monkeypatch)
    index = tmp_path / "guidelines.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("AMA_GUIDELINES_PATH", str(index))
    if name == "guideline_search":
        monkeypatch.setattr(medical, "_get", lambda url: json.dumps({"resultList": {"result": [
            {"pmid": "123", "title": "Kidney clinical guideline"}]}}))
    elif name == "paper_fulltext":
        monkeypatch.setattr(medical, "_get", lambda url: FULLTEXT)
    elif name == "clinical_trial_search":
        monkeypatch.setattr(medical, "_get", lambda url: json.dumps({"studies": [{
            "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}}}]}))

    model = FakeModel(calls(call(name, json.dumps(arguments), id="retrieval-1")), "done")
    agent = Agent(model, tools=medical.make_tools())
    assert agent.chat("Retrieve evidence") == "done"
    observation = next(message for message in agent.history if message["role"] == "tool")
    assert observation["tool_call_id"] == "retrieval-1"
    result = json.loads(observation["content"])
    assert field in (result[0] if isinstance(result, list) else result)
    assert len(model.requests) == 2
    assert observation in model.requests[1][0]
    assert {tool["function"]["name"] for tool in model.requests[0][1]} == {
        "pubmed_search", "guideline_search", "paper_fulltext", "clinical_trial_search", "analyze_images"}
