"""MedHEB similar-image retrieval: --tools src/ama/image_search_tools.py.

Only the retrieval API is contacted. Paths refer to server-prepared JPEGs;
the client does not read/upload images or connect to Qdrant.
"""
from __future__ import annotations

from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ama.tools import Tool, ToolResult


TIMEOUT = 180
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_IMAGE_PATH = re.compile(r"files/p\d{2}/p(?P<patient>\d+)/s(?P<study>\d+)/(?P<image>[A-Za-z0-9-]+)\.jpg")
_HTTP_ERRORS = {400: "invalid_request_or_image", 401: "unauthorized", 404: "not_found", 503: "retrieval_failed"}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The service contract has no redirects; never forward its API key elsewhere.
        return None


def _image_path(value: str):
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise ValueError("image_path must be a server-relative JPEG path of 1-512 characters")
    matched = _IMAGE_PATH.fullmatch(value)
    if matched is None:
        raise ValueError("image_path must use files/pXX/pPATIENT_ID/sSTUDY_ID/IMAGE_ID.jpg")
    return matched


def _configuration() -> tuple[str, str]:
    base = os.environ.get("MEDHEB_SEARCH_URL", "http://127.0.0.1:8766").rstrip("/")
    try:
        parsed = urlsplit(base)
        port = parsed.port
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or any(char.isspace() for char in base) or port == 0):
            raise ValueError
    except ValueError:
        raise ValueError("MEDHEB_SEARCH_URL must be an HTTP(S) API base URL without credentials, query or fragment") from None
    key = os.environ.get("MEDHEB_SEARCH_KEY", "").strip()
    if not key and os.environ.get("MEDHEB_SEARCH_KEY_FILE"):
        try:
            with Path(os.environ["MEDHEB_SEARCH_KEY_FILE"]).expanduser().open(encoding="utf-8") as stream:
                key = stream.read(4097).strip()
        except (OSError, UnicodeError):
            raise ValueError("Could not read MEDHEB_SEARCH_KEY_FILE") from None
    if not key:
        raise ValueError("Set MEDHEB_SEARCH_KEY or MEDHEB_SEARCH_KEY_FILE")
    if len(key) > 4096 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError("Search API key must be a nonempty printable ASCII value without whitespace")
    return base, key


def _number(value, *, minimum: float, maximum: float = math.inf) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        return False


def _relative_path(value) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith("/")
            and not any(char in value for char in ("\\", ":", "\x00"))
            and all(part not in {"", ".", ".."} for part in value.split("/")))


def _validate_response(data, *, top_k: int, query_path: str) -> list[str]:
    if (not isinstance(data, dict) or data.get("similarity") != "cosine"
            or data.get("same_patient_excluded") is not True or data.get("identical_image_excluded") is not True
            or not _number(data.get("elapsed_seconds"), minimum=0)
            or any(not isinstance(data.get(field), str) or not data[field].strip()
                   for field in ("model", "collection", "scope"))
            or not isinstance(data.get("results"), list) or len(data["results"]) > top_k):
        raise ValueError("Invalid search response")
    query = _image_path(query_path)
    evidence_ids, previous = [], math.inf
    for row in data["results"]:
        if (not isinstance(row, dict) or row.get("split") != "train"
                or any(not isinstance(row.get(field), str) or not row[field].strip()
                       for field in ("evidence_id", "study_id", "image_id"))
                or not isinstance(row.get("report"), str)
                or not _relative_path(row.get("report_path"))
                or not _number(row.get("score"), minimum=-1, maximum=1)):
            raise ValueError("Invalid search result")
        reference = _image_path(row.get("image_path"))
        if (reference["patient"] == query["patient"] or row["image_path"] == query_path
                or reference["study"] != row["study_id"] or reference["image"] != row["image_id"]
                or row["evidence_id"] != f"mimic-train:{row['study_id']}" or row["score"] > previous):
            raise ValueError("Inconsistent search result")
        previous = row["score"]
        if row["evidence_id"] not in evidence_ids:
            evidence_ids.append(row["evidence_id"])
    return evidence_ids


def _error(code: str, *, http_status: int | None = None, message: str = "") -> ToolResult:
    result = {"error": code}
    if http_status is not None:
        result["http_status"] = http_status
    if message:
        result["message"] = message
    return ToolResult(json.dumps(result, ensure_ascii=False))


def image_search(image_path: str, top_k: int = 5) -> ToolResult:
    """Return paired train reports and provenance, preserving image-level results."""
    _image_path(image_path)
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer between 1 and 10 (not a boolean)")
    body = json.dumps({"image_path": image_path, "top_k": top_k}).encode("utf-8")
    if not 1 <= len(body) <= 4096:
        raise ValueError("Search request must contain 1-4096 bytes")
    try:
        base, key = _configuration()
    except ValueError as exc:
        return _error("client_configuration", message=str(exc))
    request = Request(base + "/search/image", data=body, method="POST", headers={
        "Content-Type": "application/json", "Content-Length": str(len(body)), "X-API-Key": key,
    })
    try:
        # Ignore proxy environment variables for the private API/SSH tunnel.
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=TIMEOUT) as response:
            if response.status != 200:
                return _error(_HTTP_ERRORS.get(response.status, "http_error"), http_status=response.status)
            if response.headers.get_content_type() != "application/json":
                return _error("invalid_response")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                return _error("response_too_large")
        data = json.loads(raw.decode("utf-8"))
        evidence_ids = _validate_response(data, top_k=top_k, query_path=image_path)
    except HTTPError as exc:
        # Check status first. Error bodies may be HTML and must not leak secrets.
        status = exc.code
        exc.close()
        return _error(_HTTP_ERRORS.get(status, "http_error"), http_status=status)
    except (URLError, OSError, HTTPException):
        return _error("transport_error", message="Search API connection failed, was interrupted or timed out")
    except (ValueError, UnicodeError, TypeError):
        return _error("invalid_response")
    return ToolResult(json.dumps(data, ensure_ascii=False), evidence_ids=evidence_ids)


TOOLS = [Tool(
    name="image_search",
    description=("Retrieve similar training MIMIC-CXR images and their paired original reports from the MedHEB API. "
                 "Supply an actual server-prepared JPEG relative path, never a local path, URL or image upload. "
                 "The service encodes images with MedSigLIP; it does not encode whole reports. Returns 0 to top_k "
                 "image-level matches with reports, provenance, collection and cosine scores, not diagnostic probabilities. "
                 "Train-only, same-patient and identical-image exclusions are required. Duplicate study evidence IDs "
                 "may occur; cite the returned unique IDs. Remote image/report paths are not downloadable URLs. "
                 "The configured collection may be a small demonstration gallery; inspect collection and scope."),
    parameters={"type": "object", "properties": {
        "image_path": {"type": "string", "minLength": 1, "maxLength": 512,
                       "description": "Actual prepared-manifest path: files/pXX/pPATIENT_ID/sSTUDY_ID/IMAGE_ID.jpg"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
    }, "required": ["image_path"], "additionalProperties": False},
    handler=image_search,
)]
