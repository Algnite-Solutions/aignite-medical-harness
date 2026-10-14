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
EPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
TRIALS = "https://clinicaltrials.gov/api/v2/studies"
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


def _article_text(node) -> str:
    blocks = {"sec", "title", "p", "list-item", "disp-quote", "table-wrap", "tr", "caption"}

    def render(element):
        pieces = [element.text or ""]
        for child in element:
            if child.tag in blocks:
                pieces.extend(["\n", render(child), "\n"])
            else:
                pieces.append(render(child))
            pieces.append(child.tail or "")
        return "".join(pieces)

    return "\n".join(line for raw in render(node).splitlines() if (line := " ".join(raw.split())))


def paper_fulltext(pmcid: str, section: str = "", max_chars: int = 20000):
    if not isinstance(pmcid, str) or not re.fullmatch(r"PMC[1-9]\d*", pmcid.strip(), re.I):
        raise ValueError("pmcid must be a PMC identifier, e.g. PMC7092803")
    if not isinstance(section, str):
        raise ValueError("section must be a string")
    _count(max_chars, "max_chars", 50000)
    pmcid = pmcid.strip().upper()
    url = f"https://europepmc.org/articles/{pmcid}"
    try:
        article = ET.fromstring(_get(f"{EPMC_REST}/{pmcid}/fullTextXML"))
        body = article.find("body")
        if article.tag != "article" or body is None:
            raise ValueError("Response does not contain a readable article body")
        sections = []
        abstract = article.find("./front/article-meta/abstract")
        if abstract is not None:
            sections.append({"id": "abstract", "title": "Abstract", "text": _article_text(abstract)})
        top_sections = body.findall("sec")
        if top_sections:
            preamble = ET.Element("body")
            preamble.text = body.text
            preamble.extend(child for child in body if child.tag != "sec")
            if _article_text(preamble):
                sections.append({"id": "body", "title": "Body preamble", "text": _article_text(preamble)})
            for index, node in enumerate(top_sections, 1):
                sections.append({"id": node.get("id") or f"section-{index}",
                                 "title": _xml_text(node.find("title")) or node.get("sec-type") or f"Section {index}",
                                 "text": _article_text(node)})
        else:
            sections.append({"id": "body", "title": "Body", "text": _article_text(body)})
        if not any(item["text"] for item in sections):
            raise ValueError("Article contains no readable section text")
        available = [{"id": item["id"], "title": item["title"]} for item in sections]
        if section.strip():
            requested = section.strip().casefold()
            exact = [item for item in sections if item["id"].casefold() == requested]
            sections = exact or [item for item in sections if requested in item["title"].casefold()]
            if not sections:
                return {"error": "No matching section; choose an available section ID or title",
                        "pmcid": pmcid, "url": url, "available_sections": available}
        returned = []
        remaining = max_chars
        total_chars = sum(len(item["text"]) for item in sections)
        for item in sections:
            if remaining <= 0:
                break
            text = item["text"][:remaining]
            returned.append(dict(item, text=text, truncated=len(text) < len(item["text"])))
            remaining -= len(text)
        return {"pmcid": pmcid, "title": _xml_text(article.find("./front/article-meta/title-group/article-title")),
                "url": url, "available_sections": available, "sections": returned,
                "truncated": total_chars > max_chars}
    except (ValueError, ET.ParseError) as exc:
        return {"error": f"Full text retrieval failed: {exc}. Full text may not be openly available.",
                "pmcid": pmcid, "url": url}


_TRIAL_STATUSES = ["NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING",
                   "SUSPENDED", "TERMINATED", "COMPLETED", "WITHDRAWN", "UNKNOWN", "AVAILABLE",
                   "NO_LONGER_AVAILABLE", "TEMPORARILY_NOT_AVAILABLE", "APPROVED_FOR_MARKETING", "WITHHELD"]


def clinical_trial_search(condition: str, intervention: str = "", status: str = "", max_results: int = 5):
    condition = _query(condition)
    _count(max_results, "max_results", 50)
    if not isinstance(intervention, str) or not isinstance(status, str):
        raise ValueError("intervention and status must be strings")
    status = status.strip().upper()
    if status and status not in _TRIAL_STATUSES:
        raise ValueError("Unsupported trial status; use a ClinicalTrials.gov status such as RECRUITING")
    params = {"query.cond": condition, "pageSize": max_results, "format": "json"}
    if intervention.strip():
        params["query.intr"] = intervention.strip()
    if status:
        params["filter.overallStatus"] = status
    try:
        studies = json.loads(_get(f"{TRIALS}?{urlencode(params)}"))["studies"]
        results = []
        for study in studies[:max_results]:
            protocol = study["protocolSection"]
            identification = protocol["identificationModule"]
            nct_id = identification["nctId"]
            state = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            locations = protocol.get("contactsLocationsModule", {}).get("locations", [])
            results.append({"nct_id": nct_id, "title": identification.get("briefTitle", ""),
                            "url": f"https://clinicaltrials.gov/study/{nct_id}",
                            "status": state.get("overallStatus", ""), "phase": design.get("phases", []),
                            "study_type": design.get("studyType", ""),
                            "last_update": state.get("lastUpdatePostDateStruct", {}).get("date", ""),
                            "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
                            "interventions": [{"name": item.get("name", ""), "type": item.get("type", "")}
                                              for item in protocol.get("armsInterventionsModule", {}).get("interventions", [])],
                            "summary": protocol.get("descriptionModule", {}).get("briefSummary", "")[:1000],
                            "eligibility": {"criteria": eligibility.get("eligibilityCriteria", "")[:3000],
                                            "minimum_age": eligibility.get("minimumAge", ""),
                                            "maximum_age": eligibility.get("maximumAge", ""),
                                            "sex": eligibility.get("sex", ""),
                                            "healthy_volunteers": eligibility.get("healthyVolunteers")},
                            "locations": [{key: place.get(key, "") for key in ("facility", "city", "state", "country")}
                                          for place in locations[:5]],
                            "truncated": {"summary": len(protocol.get("descriptionModule", {}).get("briefSummary", "")) > 1000,
                                          "eligibility": len(eligibility.get("eligibilityCriteria", "")) > 3000,
                                          "locations": len(locations) > 5}})
        return results
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"error": "ClinicalTrials.gov retrieval failed; no reliable results are available"}


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
    Tool("paper_fulltext",
         "Fetch an openly available article's full-text XML from Europe PMC using its PMCID. "
         "Returns title, source URL, available section IDs/titles, and section text. "
         "Use section to select an exact section ID or match a heading (e.g. Methods, Results, Discussion). "
         "max_chars limits total returned section text; inspect truncated and request specific sections as needed. "
         "Not every PubMed article has accessible full text. Failures return an error object.",
         _schema({"pmcid": {"type": "string", "description": "PMC identifier, e.g. PMC7092803; not a PMID or DOI."},
                  "section": {"type": "string", "default": "", "description": "Optional section ID or heading; empty returns sections in order."},
                  "max_chars": {"type": "integer", "minimum": 1, "maximum": 50000, "default": 20000}},
                 ["pmcid"]), paper_fulltext),
    Tool("clinical_trial_search",
         "Search ClinicalTrials.gov online for registered studies by condition, optional intervention and recruitment status. "
         "Returns NCT IDs, titles, source URLs, status, phase, last update, summaries, eligibility and up to five locations. "
         "Summary and eligibility text are capped at 1000 and 3000 characters; truncation is reported. "
         "Registry records do not establish treatment effectiveness or individual eligibility. Failures return an error object.",
         _schema({"condition": {"type": "string", "description": "Disease or condition, e.g. psoriasis."},
                  "intervention": {"type": "string", "default": "", "description": "Optional drug or intervention name."},
                  "status": {"type": "string", "enum": [""] + _TRIAL_STATUSES, "default": "",
                             "description": "Optional registry status, e.g. RECRUITING or COMPLETED; empty means all statuses."},
                  "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5}},
                 ["condition"]), clinical_trial_search),
]


def make_tools(dataset_dir=None, episode_id=None):
    from ama.image_analysis_tools import make_image_analysis_tool
    return list(TOOLS) + [make_image_analysis_tool()]
