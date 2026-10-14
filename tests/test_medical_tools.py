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
    assert result == [{"pmid": "123", "title": "Kidney disease & care", "journal": "Test Journal",
                       "year": "2024", "abstract": "METHODS: A trial. RESULTS: Improved outcomes.",
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
            {"pmid": "123", "title": "A <i>guideline</i>", "pubYear": "2024",
             "journalInfo": {"journal": {"title": "Journal"}}, "pubTypeList": {"pubType": ["Guideline"]},
             "abstractText": "<p>Evidence &amp; care</p>"},
            {"source": "PMC", "id": "PMC999", "doi": "10.1/test", "title": "Another guideline"}]}})

    monkeypatch.setattr(medical, "_get", get)
    result = medical.guideline_search("chronic kidney disease SGLT2", max_results=3)
    params = parse_qs(urlsplit(urls[0]).query)
    assert params["query"] == ['(chronic kidney disease SGLT2) AND (PUB_TYPE:"guideline")']
    assert params["resultType"] == ["core"] and params["sort"] == ["CITED desc"]
    assert result[0]["abstract"] == "Evidence & care" and result[0]["org"] == "Journal"
    assert result[1]["url"] == "https://doi.org/10.1/test"


def test_guideline_partial_failure_preserves_local_results(monkeypatch):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    monkeypatch.setattr(medical, "_get", lambda url: "invalid JSON")
    result = medical.guideline_search("CKD")
    assert "Europe PMC" in result["error"] and result["results"][0]["source"] == "local_index"


def test_guideline_deduplication_preserves_local_metadata_and_order(monkeypatch):
    local = [{"title": "Curated", "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
              "source": "local_index", "key_points": ["curated detail"]}]
    monkeypatch.setattr(medical, "_local_guidelines", lambda query: local)
    monkeypatch.setattr(medical, "_get", lambda url: json.dumps({"resultList": {"result": [
        {"pmid": "999", "title": "Other"}, {"pmid": "123", "title": "Duplicate"}]}}))
    result = medical.guideline_search("kidney")
    assert [row["title"] for row in result] == ["Curated", "Other"]
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
    assert [tool.name for tool in tools] == ["pubmed_search", "guideline_search"]
    model = FakeModel(calls(call("guideline_search", '{"query":"CKD","live":false}')), "done")
    agent = Agent(model, tools=tools)
    assert agent.chat("Find a guideline offline") == "done"
    assert len(model.requests[0][1]) == 2
    assert json.loads(agent.history[2]["content"])[0]["source"] == "local_index"
