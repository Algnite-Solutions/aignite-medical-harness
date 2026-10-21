"""Dummy HTTP service + FakeModel only; no DGX, real keys, images or patient data."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import URLError

import pytest

from ama import image_search_tools as search
from ama.agent import Agent
from ama.tools import load_tools
from fakes import FakeModel, call, calls, decision


# Synthetic format fixtures, not actual prepared-manifest paths or clinical data.
QUERY = "files/p10/p10000001/s50000001/00000000-0000-0000-0000-000000000001.jpg"
REFERENCE = "files/p11/p11000001/s050000002/00000000-0000-0000-0000-000000000002.jpg"
KEY = "dummy-test-key-not-a-real-credential"


def response():
    return {"results": [{
        "evidence_id": "mimic-train:050000002", "study_id": "050000002",
        "image_id": "00000000-0000-0000-0000-000000000002", "image_path": REFERENCE,
        "report": "SYNTHETIC REPORT\nNo actual clinical data.",
        "report_path": "reports/p11/p11000001/s050000002.txt", "split": "train", "score": 0.81,
    }], "similarity": "cosine", "elapsed_seconds": 0.25, "model": "google/medsiglip-448",
        "collection": "dummy-32-patient-gallery", "same_patient_excluded": True,
        "identical_image_excluded": True, "scope": "Dummy scope; not a stable enum."}


@pytest.fixture
def dummy(monkeypatch):
    state = {"status": 200, "payload": response(), "content_type": "application/json", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            state["requests"].append({"path": self.path, "body": body, "headers": dict(self.headers)})
            if self.headers.get("X-API-Key") != KEY:
                status, payload = 401, b'{"error":"unauthorized"}'
            else:
                status = state["status"]
                payload = state.get("raw", json.dumps(state["payload"]).encode("utf-8"))
            self.send_response(status)
            self.send_header("Content-Type", state["content_type"])
            self.send_header("Content-Length", str(len(payload)))
            if status == 302:
                self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    monkeypatch.setenv("MEDHEB_SEARCH_URL", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("MEDHEB_SEARCH_KEY", KEY)
    monkeypatch.delenv("MEDHEB_SEARCH_KEY_FILE", raising=False)
    # A proxy that cannot connect must never intercept this private API request.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_request_contract_and_complete_response_preserved(dummy):
    result = search.image_search(QUERY)
    assert json.loads(result.text) == dummy["payload"]
    assert result.images == [] and result.evidence_ids == ["mimic-train:050000002"]
    assert KEY not in result.text
    request, = dummy["requests"]
    assert request["path"] == "/search/image"
    assert json.loads(request["body"]) == {"image_path": QUERY, "top_k": 5}
    assert 1 <= len(request["body"]) <= 4096
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["X-Api-Key"] == KEY
    assert int(request["headers"]["Content-Length"]) == len(request["body"])
    assert json.loads(result.text)["results"][0]["study_id"] == "050000002"


def test_duplicate_study_results_preserved_but_citations_unique(dummy):
    second = copy.deepcopy(dummy["payload"]["results"][0])
    second["image_id"] = "00000000-0000-0000-0000-000000000003"
    second["image_path"] = REFERENCE.replace("000000000002.jpg", "000000000003.jpg")
    second["score"] = 0.70
    dummy["payload"]["results"].append(second)
    result = search.image_search(QUERY, top_k=2)
    assert len(json.loads(result.text)["results"]) == 2
    assert result.evidence_ids == ["mimic-train:050000002"]


def test_empty_and_short_results_are_not_padded(dummy):
    assert len(json.loads(search.image_search(QUERY, top_k=10).text)["results"]) == 1
    dummy["payload"]["results"] = []
    result = search.image_search(QUERY)
    assert json.loads(result.text)["results"] == [] and result.evidence_ids == []


def test_negative_cosine_and_runtime_collection_are_preserved(dummy):
    dummy["payload"]["results"][0]["score"] = -0.25
    dummy["payload"]["collection"] = "different-runtime-collection"
    dummy["payload"]["scope"] = "Different free-text deployment scope"
    dummy["payload"]["results"][0]["report"] = ""
    assert json.loads(search.image_search(QUERY).text) == dummy["payload"]


def test_server_image_paths_are_not_opened_locally(dummy, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Image search attempted local file access")
    monkeypatch.setattr(Path, "open", fail)
    assert "error" not in json.loads(search.image_search(QUERY).text)


@pytest.mark.parametrize("top_k", [0, 11, True, False, 1.0, "5", None])
def test_invalid_top_k_never_sends_request(dummy, top_k):
    with pytest.raises(ValueError, match="top_k"):
        search.image_search(QUERY, top_k)
    assert dummy["requests"] == []


@pytest.mark.parametrize("path", ["", None, "x" * 513, "/" + QUERY, "../" + QUERY,
    "http://example.test/image.jpg", "data:image/jpeg;base64,xxx", QUERY.replace("/", "\\"),
    QUERY.replace(".jpg", ".png"), QUERY.replace("s50000001/", "s50000001/../"),
    "files/pXX/pPATIENT_ID/sSTUDY_ID/DICOM_ID.jpg"])
def test_invalid_image_paths_never_sends_request(dummy, path):
    with pytest.raises(ValueError, match="image_path"):
        search.image_search(path)
    assert dummy["requests"] == []


@pytest.mark.parametrize("status,error", [(400, "invalid_request_or_image"), (401, "unauthorized"),
    (404, "not_found"), (503, "retrieval_failed"), (501, "http_error"), (302, "http_error")])
def test_status_checked_before_parsing_error_body_and_redirects_not_followed(dummy, status, error):
    dummy.update(status=status, content_type="text/html", raw=("<html>" + KEY + "</html>").encode())
    result = search.image_search(QUERY)
    assert json.loads(result.text) == {"error": error, "http_status": status}
    assert result.evidence_ids == [] and KEY not in result.text
    assert len(dummy["requests"]) == 1


def test_missing_key_and_key_file(dummy, monkeypatch, tmp_path):
    monkeypatch.delenv("MEDHEB_SEARCH_KEY")
    assert json.loads(search.image_search(QUERY).text)["error"] == "client_configuration"
    assert dummy["requests"] == []
    key_file = tmp_path / "protected-key.txt"
    key_file.write_text(KEY + "\n", encoding="utf-8")
    monkeypatch.setenv("MEDHEB_SEARCH_KEY_FILE", str(key_file))
    assert "error" not in json.loads(search.image_search(QUERY).text)
    monkeypatch.setenv("MEDHEB_SEARCH_KEY_FILE", str(tmp_path / "missing"))
    result = search.image_search(QUERY)
    assert json.loads(result.text)["error"] == "client_configuration" and KEY not in result.text


@pytest.mark.parametrize("url", ["ftp://example.test", "http://user:password@example.test", "http://example.test?key=value",
                                     "http://example.test#fragment", "http://127.0.0.1:0", "http://example.test:99999"])
def test_invalid_configuration_not_exposed(dummy, monkeypatch, url):
    monkeypatch.setenv("MEDHEB_SEARCH_URL", url)
    result = search.image_search(QUERY)
    assert json.loads(result.text)["error"] == "client_configuration"
    assert url not in result.text and KEY not in result.text and not dummy["requests"]


@pytest.mark.parametrize("exception", [URLError("private-url-" + KEY), TimeoutError(KEY), ConnectionResetError(KEY)])
def test_transport_failure_and_timeout_are_sanitized(dummy, monkeypatch, exception):
    class Opener:
        def open(self, request, timeout):
            assert timeout == 180
            raise exception
    monkeypatch.setattr(search, "build_opener", lambda *args: Opener())
    result = search.image_search(QUERY)
    assert json.loads(result.text)["error"] == "transport_error"
    assert KEY not in result.text and result.evidence_ids == []


@pytest.mark.parametrize("field,value", [("same_patient_excluded", False), ("identical_image_excluded", 1),
    ("similarity", "dot"), ("results", {}), ("elapsed_seconds", float("nan")), ("elapsed_seconds", True),
    ("collection", ""), ("model", None)])
def test_invalid_top_level_payload_does_not_release_evidence(dummy, field, value):
    dummy["payload"][field] = value
    result = search.image_search(QUERY)
    assert json.loads(result.text)["error"] == "invalid_response" and not result.evidence_ids


@pytest.mark.parametrize("field,value", [("split", "test"), ("score", True), ("score", float("inf")),
    ("score", 1.01), ("study_id", 50000002), ("evidence_id", "wrong"), ("report", None),
    ("report_path", "../secret.txt"), ("image_id", "wrong"), ("image_path", QUERY)])
def test_invalid_reference_does_not_release_evidence(dummy, field, value):
    dummy["payload"]["results"][0][field] = value
    result = search.image_search(QUERY)
    assert json.loads(result.text)["error"] == "invalid_response" and not result.evidence_ids


def test_same_patient_exclusion_independently_checked(dummy):
    dummy["payload"]["results"][0]["image_path"] = REFERENCE.replace("p11/p11000001", "p10/p10000001")
    assert json.loads(search.image_search(QUERY).text)["error"] == "invalid_response"


def test_too_many_or_unsorted_results_rejected(dummy):
    second = copy.deepcopy(dummy["payload"]["results"][0])
    second["score"] = 0.90
    dummy["payload"]["results"].append(second)
    assert json.loads(search.image_search(QUERY, top_k=1).text)["error"] == "invalid_response"
    assert json.loads(search.image_search(QUERY, top_k=2).text)["error"] == "invalid_response"


@pytest.mark.parametrize("raw,content_type", [(b"not json", "application/json"), (b"[]", "application/json"),
    (b"\xff", "application/json"), (b"{}", "text/html")])
def test_malformed_response(dummy, raw, content_type):
    dummy.update(raw=raw, content_type=content_type)
    assert json.loads(search.image_search(QUERY).text)["error"] == "invalid_response"


def test_response_size_limit(dummy, monkeypatch):
    monkeypatch.setattr(search, "MAX_RESPONSE_BYTES", 5)
    assert json.loads(search.image_search(QUERY).text)["error"] == "response_too_large"


def test_loaded_tool_agent_round_trip_and_citations(dummy):
    tools = load_tools("src/ama/image_search_tools.py")
    assert [tool.name for tool in tools] == ["image_search"]
    arguments = json.dumps({"image_path": QUERY, "top_k": 2})
    model = FakeModel(calls(call("image_search", arguments, id="retrieve")), "Evidence: mimic-train:050000002")
    agent = Agent(model, tools=tools)
    assert agent.chat("Find a similar training case") == "Evidence: mimic-train:050000002"
    observation = agent.history[2]
    assert observation["role"] == "tool" and observation["tool_call_id"] == "retrieve"
    assert "SYNTHETIC REPORT\nNo actual clinical data." in observation["content"].replace("\\n", "\n")
    assert agent.evidence_ids == {"mimic-train:050000002"}
    assert observation in model.requests[1][0]
    assert KEY not in json.dumps(agent.history) and "base64" not in json.dumps(agent.history)
    assert len(agent.calls) == 2


def test_agent_handles_unknown_arguments_and_server_failure(dummy):
    dummy["status"] = 503
    model = FakeModel(calls(call("image_search", json.dumps({"image_path": QUERY, "unknown": 1}))),
                      calls(call("image_search", json.dumps({"image_path": QUERY}))), "unavailable")
    agent = Agent(model, tools=search.TOOLS)
    assert agent.chat("Search") == "unavailable"
    assert len(dummy["requests"]) == 1 and agent.evidence_ids == set()
    assert '"http_status": 503' in agent.history[4]["content"]


def test_runner_saves_retrieved_report_and_accepts_source_citation(dummy, tmp_path):
    from ama.runner import execute
    from test_cli import dataset, messages, rows

    model = FakeModel(calls(call("image_search", json.dumps({"image_path": QUERY}), id="retrieve")),
                      decision(answer={"findings": "Synthetic prediction"}, citations=["mimic-train:050000002"]),
                      decision("t2"))
    run = execute(dataset(tmp_path), model, tools_path=Path("src/ama/image_search_tools.py"),
                  runs_root=tmp_path / "runs")
    assert rows(run)[0]["termination"] == "completed"
    assert rows(run)[0]["decision"]["citations"] == ["mimic-train:050000002"]
    history = messages(run)
    assert any(message.get("tool_call_id") == "retrieve" and "SYNTHETIC REPORT" in message["content"]
               for message in history)
    assert KEY not in json.dumps(history)
