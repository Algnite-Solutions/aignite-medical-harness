"""Opt-in biomedical retrieval: --tools src/ama/medical_tools.py.

Only the standard library is used. NCBI_EMAIL and NCBI_API_KEY are optional;
AMA_GUIDELINES_PATH can point to a curated JSON list for offline searches.
"""
from __future__ import annotations

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from ama.tools import Tool

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_NCBI_LOCK = threading.Lock()
_NCBI_LAST = 0.0
_MAX_BYTES = 4 * 1024 * 1024


def _get(url: str) -> str:
    email = os.environ.get("NCBI_EMAIL", "")
    agent = "ama/0.1" + (f" (mailto:{email})" if email else "")
    try:
        with urlopen(Request(url, headers={"User-Agent": agent}), timeout=20) as response:
            media_type = response.headers.get_content_type()
            if media_type == "application/pdf":
                raise ValueError("PDF extraction is unsupported; use an HTML full-text URL")
            if not (media_type.startswith("text/") or media_type in {"application/json", "application/xml"}
                    or media_type.endswith(("+xml", "+json"))):
                raise ValueError("Unsupported response type; use a text or HTML URL")
            raw = response.read(_MAX_BYTES + 1)
            if len(raw) > _MAX_BYTES:
                raise ValueError("Response exceeds the 4 MiB download limit")
            return raw.decode(response.headers.get_content_charset() or "utf-8", errors="replace")
    except HTTPError as exc:
        raise ValueError(f"Remote server returned HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError):
        # Do not expose a request URL containing NCBI_API_KEY in model logs.
        raise ValueError("Network request failed or timed out") from None


def _ncbi(endpoint: str, **params) -> str:
    global _NCBI_LAST
    params.update(db="pubmed", tool="ama")
    for key, env in (("email", "NCBI_EMAIL"), ("api_key", "NCBI_API_KEY")):
        if os.environ.get(env):
            params[key] = os.environ[env]
    with _NCBI_LOCK:
        interval = 0.11 if params.get("api_key") else 0.35
        time.sleep(max(0, interval - (time.monotonic() - _NCBI_LAST)))
        _NCBI_LAST = time.monotonic()
        return _get(f"{EUTILS}/{endpoint}.fcgi?{urlencode(params)}")


def _count(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")


def _query(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query must be a non-empty string")
    return value.strip()


class _Text(HTMLParser):
    """Extract visible text, ignoring common navigation and executable content."""

    _skip = {"script", "style", "nav", "footer", "header", "noscript", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._skip:
            self.hidden.append(tag)

    def handle_endtag(self, tag):
        if tag in self.hidden:
            del self.hidden[self.hidden.index(tag):]

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _clean(markup: str) -> str:
    parser = _Text()
    parser.feed(markup)
    return " ".join(" ".join(parser.parts).split())


def _xml_text(node) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _parse_articles(xml: str) -> list[dict]:
    result = []
    for record in ET.fromstring(xml).findall(".//PubmedArticle"):
        pmid = _xml_text(record.find("./MedlineCitation/PMID"))
        article = record.find("./MedlineCitation/Article")
        if not pmid or article is None:
            continue
        year = _xml_text(article.find("./Journal/JournalIssue/PubDate/Year"))
        if not year:
            date = _xml_text(article.find("./Journal/JournalIssue/PubDate/MedlineDate"))
            match = re.search(r"\b\d{4}\b", date)
            year = match.group() if match else _xml_text(article.find("./ArticleDate/Year"))
        sections = []
        for section in article.findall("./Abstract/AbstractText"):
            label = section.get("Label")
            sections.append((f"{label}: " if label else "") + _xml_text(section))
        result.append({"pmid": pmid, "title": _xml_text(article.find("ArticleTitle")),
                       "journal": _xml_text(article.find("./Journal/Title")), "year": year,
                       "abstract": " ".join(sections), "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    return result


def pubmed_search(query: str, retmax: int = 5, abstract: bool = True):
    query = _query(query)
    _count(retmax, "retmax", 50)
    if type(abstract) is not bool:
        raise ValueError("abstract must be a boolean")
    try:
        search = json.loads(_ncbi("esearch", term=query, retmode="json", sort="relevance", retmax=retmax))
        if search.get("error") or "esearchresult" not in search:
            raise ValueError("Invalid PubMed search response")
        if search["esearchresult"].get("ERROR"):
            raise ValueError("PubMed rejected the search query")
        ids = search["esearchresult"]["idlist"]
        if not ids:
            return []
        if abstract:
            try:
                articles = _parse_articles(_ncbi("efetch", id=",".join(ids), retmode="xml"))
                if articles:
                    return articles
            except (ValueError, ET.ParseError):
                pass  # Fall back to metadata if abstract retrieval fails.
        summary = json.loads(_ncbi("esummary", id=",".join(ids), retmode="json"))["result"]
        return [{"pmid": pmid, "title": _clean(summary[pmid].get("title", "")),
                 "journal": summary[pmid].get("fulljournalname", summary[pmid].get("source", "")),
                 "year": summary[pmid].get("pubdate", "")[:4], "abstract": "",
                 "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"} for pmid in ids]
    except (ValueError, KeyError, TypeError):
        return {"error": "PubMed retrieval failed; no reliable results are available"}


def _local_guidelines(query: str) -> list[dict]:
    configured = os.environ.get("AMA_GUIDELINES_PATH")
    path = Path(configured) if configured else Path(__file__).with_name("guidelines.json")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(
            isinstance(entry, dict) and isinstance(entry.get("title"), str)
            and isinstance(entry.get("url"), str) for entry in entries):
        raise ValueError("Guideline index must be a JSON list with title and url fields")
    terms = set(re.findall(r"\w+", query.casefold())) - {"and", "or", "the", "guideline", "guidelines"}
    ranked = []
    for entry in entries:
        searchable = json.dumps(entry, ensure_ascii=False).casefold()
        score = sum(bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", searchable)) for term in terms)
        if score:
            ranked.append((score, dict(entry, source="local_index")))
    return [entry for _, entry in sorted(ranked, key=lambda row: row[0], reverse=True)]


def guideline_search(query: str, max_results: int = 5, live: bool = True):
    query = _query(query)
    _count(max_results, "max_results", 50)
    if type(live) is not bool:
        raise ValueError("live must be a boolean")
    results = []
    errors = []
    try:
        results = _local_guidelines(query)
    except (OSError, ValueError, TypeError):
        errors.append("Could not read the local guideline index")
    if live:
        try:
            params = urlencode({"query": f'({query}) AND (PUB_TYPE:"guideline")',
                                "format": "json", "pageSize": max_results,
                                "resultType": "core", "sort": "CITED desc"})
            data = json.loads(_get(f"{EPMC}?{params}"))
            articles = data["resultList"]["result"]
            for article in articles:
                journal = article.get("journalInfo", {}).get("journal", {}).get("title", "")
                pmid, doi = article.get("pmid"), article.get("doi")
                url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else
                       f"https://doi.org/{doi}" if doi else
                       f"https://europepmc.org/article/{article['source']}/{article['id']}")
                results.append({"source": "europepmc", "org": journal, "year": article.get("pubYear", ""),
                                "title": _clean(article.get("title", "")), "url": url,
                                "pub_type": article.get("pubTypeList", {}).get("pubType", []),
                                "abstract": _clean(article.get("abstractText", ""))[:800]})
        except (ValueError, KeyError, TypeError):
            errors.append("Europe PMC retrieval failed")
    unique = {}
    for row in results:
        unique.setdefault(row["url"].rstrip("/"), row)
    results = list(unique.values())
    return {"error": "; ".join(errors), "results": results} if errors else results


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOLS = [
    Tool("pubmed_search",
         "Search PubMed (NCBI E-utilities) for biomedical literature. Returns PMIDs, titles, journals, "
         "years, abstracts and URLs. Use for evidence-based medicine questions. Failures return an error object.",
         _schema({"query": {"type": "string", "description": "PubMed query; supports PubMed search syntax."},
                  "retmax": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                  "abstract": {"type": "boolean", "default": True}}, ["query"]), pubmed_search),
    Tool("guideline_search",
         "Search clinical practice guidelines using a curated local index and live Europe PMC results "
         "filtered to guideline publication type, ordered by citations (not a guarantee of quality or recency). "
         "Matches can include reporting guidelines; verify clinical scope and publication date. "
         "Returns titles, sources, years, URLs and abstracts. Set live=false for offline search. "
         "Partial failures return an error object with available results.",
         _schema({"query": {"type": "string", "description": "Guideline topic, e.g. chronic kidney disease SGLT2."},
                  "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5,
                                  "description": "Maximum live results; local matches are additional."},
                  "live": {"type": "boolean", "default": True}}, ["query"]), guideline_search),
]
