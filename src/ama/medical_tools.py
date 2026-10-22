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
from ama.guideline_retrieval import contains, expression, local_guidelines, rank_guidelines, topic_groups, INTENTS

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


def _identifiers(pmid: str = "", pmcid: str = "", doi: str = "", links=None) -> dict:
    pmid, doi = str(pmid or ""), doi or ""
    pmcid = pmcid or ""
    pmcid = pmcid.upper() if re.fullmatch(r"PMC[1-9]\d*", pmcid, re.I) else ""
    fulltext_url = f"https://europepmc.org/articles/{pmcid}" if pmcid else ""
    fulltext_links = [dict(item) for item in (links or [])
                      if item.get("availabilityCode") in {"OA", "F"} and item.get("url")]
    if fulltext_url and not any(item["url"] == fulltext_url for item in fulltext_links):
        fulltext_links.insert(0, {"url": fulltext_url, "documentStyle": "html", "site": "Europe_PMC"})
    return {"pmid": str(pmid), "pmcid": pmcid, "doi": doi, "fulltext_url": fulltext_url,
            "fulltext_links": fulltext_links,
            "fulltext_status": "pmcid_available_unverified" if pmcid else "no_pmcid"}


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
        ids = {item.get("IdType"): _xml_text(item) for item in record.findall("./PubmedData/ArticleIdList/ArticleId")}
        result.append({**_identifiers(pmid, ids.get("pmc", ""), ids.get("doi", "")),
                       "title": _xml_text(article.find("ArticleTitle")),
                       "journal": _xml_text(article.find("./Journal/Title")), "year": year,
                       "abstract": " ".join(sections), "abstract_truncated": False,
                       "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
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
        results = []
        for pmid in ids:
            row = summary[pmid]
            identifiers = {item.get("idtype"): item.get("value", "") for item in row.get("articleids", [])}
            results.append({**_identifiers(pmid, identifiers.get("pmc", ""), identifiers.get("doi", "")),
                            "title": _clean(row.get("title", "")),
                            "journal": row.get("fulljournalname", row.get("source", "")),
                            "year": row.get("pubdate", "")[:4], "abstract": "", "abstract_truncated": False,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
        return results
    except (ValueError, KeyError, TypeError):
        return {"error": "PubMed retrieval failed; no reliable results are available"}


def _local_guidelines(query: str) -> list[dict]:
    configured = os.environ.get("AMA_GUIDELINES_PATH")
    path = Path(configured) if configured else Path(__file__).with_name("guidelines.json")
    return local_guidelines(query, path)


def guideline_search(query: str, max_results: int = 5, live: bool = True, *,
                     population: str = "", subtype: str = "", intent: str = "",
                     year_to: int | None = None, search_expression: str = ""):
    query = _query(query)
    _count(max_results, "max_results", 50)
    if type(live) is not bool:
        raise ValueError("live must be a boolean")
    if not isinstance(population, str) or population not in {"", "adult", "pediatric", "older_adult"}:
        raise ValueError("population must be adult, pediatric, older_adult or empty")
    if not isinstance(subtype, str) or subtype not in {"", "acute", "chronic", "autoimmune"}:
        raise ValueError("subtype must be acute, chronic, autoimmune or empty")
    if not isinstance(intent, str) or intent not in {"", *INTENTS}:
        raise ValueError("intent must be diagnosis, investigation, treatment or empty")
    if year_to is not None and (type(year_to) is not int or not 1000 <= year_to <= 9999):
        raise ValueError("year_to must be an integer between 1000 and 9999")
    if not isinstance(search_expression, str):
        raise ValueError("search_expression must be a string")
    if not subtype:
        for name, aliases in {"autoimmune": ["autoimmune", "自身免疫性"], "chronic": ["chronic", "慢性"],
                              "acute": ["acute", "急性"]}.items():
            if any(contains(query, alias) for alias in aliases):
                subtype = name
                break
    topic_groups(query)  # Validate topic before reading files or contacting a service.
    results, errors = [], []
    try:
        results = _local_guidelines(query)
    except (OSError, ValueError, TypeError):
        errors.append("Could not read the local guideline index")
    if live:
        try:
            # Fetch a bounded candidate pool, then filter/rank before applying max_results.
            params = urlencode({"query": expression(query, search_expression, year_to),
                                "format": "json", "pageSize": min(100, max(20, max_results * 4)),
                                "resultType": "core"})
            data = json.loads(_get(f"{EPMC}?{params}"))
            for article in data["resultList"]["result"]:
                pmid, doi = article.get("pmid", ""), article.get("doi", "")
                pmcid = article.get("pmcid") or (article.get("id", "") if article.get("source") == "PMC" else "")
                url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else
                       f"https://doi.org/{doi}" if doi else
                       f"https://europepmc.org/article/{article['source']}/{article['id']}")
                results.append({**_identifiers(pmid, pmcid, doi,
                                article.get("fullTextUrlList", {}).get("fullTextUrl", [])),
                                "source": "europepmc", "org": "", "org_source": "",
                                "journal": article.get("journalInfo", {}).get("journal", {}).get("title", ""),
                                "year": article.get("pubYear", ""),
                                "title": _clean(article.get("title", "")), "url": url,
                                "pub_type": article.get("pubTypeList", {}).get("pubType", []),
                                "abstract": _clean(article.get("abstractText", ""))})
        except (ValueError, KeyError, TypeError, AttributeError):
            errors.append("Europe PMC retrieval failed")
    results = rank_guidelines(results, query, population=population, subtype=subtype,
                              intent=intent, year_to=year_to, max_results=max_results)
    for row in results:
        # Relevance/applicability checks see the complete abstract, not a truncated prefix.
        abstract = row.get("abstract", "")
        row["abstract_truncated"] = len(abstract) > 800
        row["abstract"] = abstract[:800]
        row.update(_identifiers(row.get("pmid", ""), row.get("pmcid", ""), row.get("doi", ""),
                                row.get("fulltext_links", [])))
        row.setdefault("journal", "")
        row.setdefault("org", "")
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


def paper_fulltext(pmcid: str = "", section: str = "", max_chars: int = 20000, *, pmid: str = ""):
    if not isinstance(pmcid, str) or not isinstance(pmid, str) or bool(pmcid) == bool(pmid):
        raise ValueError("Supply exactly one pmcid (PMC identifier) or pmid (PubMed identifier)")
    if pmcid and not re.fullmatch(r"PMC[1-9]\d*", pmcid.strip(), re.I):
        raise ValueError("pmcid must be a PMC identifier, e.g. PMC7092803")
    if pmid and not re.fullmatch(r"[1-9]\d*", pmid.strip()):
        raise ValueError("pmid must be a numeric PubMed identifier")
    if not isinstance(section, str):
        raise ValueError("section must be a string")
    _count(max_chars, "max_chars", 50000)
    if pmid:
        pmid = pmid.strip()
        try:
            params = urlencode({"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json", "resultType": "core"})
            records = json.loads(_get(f"{EPMC}?{params}"))["resultList"]["result"]
            pmcid = next((r.get("pmcid", "") for r in records if str(r.get("pmid", "")) == pmid
                          and re.fullmatch(r"PMC[1-9]\d*", r.get("pmcid") or "", re.I)), "")
        except (ValueError, KeyError, TypeError):
            return {"error": "PMID to PMCID resolution failed; do not guess a PMCID", "pmid": pmid}
        if not pmcid:
            return {"error": "No PMCID available for this PMID; accessible full-text XML could not be established",
                    "pmid": pmid, "pmcid": "", "fulltext_status": "no_pmcid"}
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
        # Include nested sections for targeted diagnosis/investigation reads. Default
        # whole-article output still uses top-level sections to avoid duplicate text.
        selectable = list(sections)
        known = {id(node) for node in top_sections}
        for index, node in enumerate(body.iter("sec"), 1):
            if id(node) not in known:
                selectable.append({"id": node.get("id") or f"nested-section-{index}",
                                   "title": _xml_text(node.find("title")) or node.get("sec-type") or f"Section {index}",
                                   "text": _article_text(node)})
        available = [{"id": item["id"], "title": item["title"]} for item in selectable]
        if section.strip():
            requested = section.strip().casefold()
            exact = [item for item in selectable if item["id"].casefold() == requested]
            terms = INTENTS.get(requested, [requested])
            matches = [item for item in selectable if any(term in item["title"].casefold() for term in terms)]
            # A matched parent includes its descendants; do not return them twice.
            sections = exact or [item for item in matches if not any(
                len(other["text"]) > len(item["text"]) and item["text"] in other["text"] for other in matches)]
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
            anchors = {node.get("id") for node in body.iter("sec")}
            returned.append(dict(item, text=text, truncated=len(text) < len(item["text"]),
                                 source_url=f"{url}#{item['id']}" if item["id"] in anchors else url))
            remaining -= len(text)
        return {"pmcid": pmcid, "pmid": pmid, "title": _xml_text(article.find("./front/article-meta/title-group/article-title")),
                "url": url, "available_sections": available, "sections": returned,
                "truncated": total_chars > max_chars}
    except (ValueError, ET.ParseError) as exc:
        return {"error": f"Full text retrieval failed: {exc}. Full text may not be openly available.",
                "pmcid": pmcid, "pmid": pmid, "url": url, "fulltext_status": "xml_unavailable"}


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


def _schema(properties: dict, required: list[str], **constraints) -> dict:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False, **constraints}


TOOLS = [
    Tool("pubmed_search",
         "Search PubMed (NCBI E-utilities) for biomedical literature. Returns PMIDs, titles, journals, "
         "years, abstracts, PMCIDs/DOIs when supplied by PubMed, and full-text links/status. "
         "Use returned PMCID or PMID with paper_fulltext; do not invent IDs. Failures return an error object.",
         _schema({"query": {"type": "string", "description": "PubMed query; supports PubMed search syntax."},
                  "retmax": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                  "abstract": {"type": "boolean", "default": True}}, ["query"]), pubmed_search),
    Tool("guideline_search",
         "Find topic-matched clinical guidelines and consensus statements in a source-checked local index "
         "and Europe PMC. Filters reporting guidelines/lab protocols using transparent lexical rules. "
         "Returns classification/applicability evidence, publication year (latest version not verified), "
         "PMID/PMCID/DOI, full-text links and abstract truncation flags. Unknown organizations stay empty. "
         "Use population/subtype/intent/year_to to restrict scope; unspecified metadata cannot meet explicit filters. "
         "Use paper_fulltext for diagnostic criteria, not the abstract alone. Set live=false for offline search. "
         "Partial failures return an error object with available results.",
         _schema({"query": {"type": "string", "description": "Disease/topic in plain language; English/Chinese aliases for the indexed diseases are supported. Put raw syntax in search_expression."},
                  "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5,
                                  "description": "Maximum combined ranked results, including local records."},
                  "live": {"type": "boolean", "default": True},
                  "population": {"type": "string", "enum": ["", "adult", "pediatric", "older_adult"], "default": ""},
                  "subtype": {"type": "string", "enum": ["", "acute", "chronic", "autoimmune"], "default": ""},
                  "intent": {"type": "string", "enum": ["", "diagnosis", "investigation", "treatment"], "default": ""},
                  "year_to": {"type": "integer", "minimum": 1000, "maximum": 9999,
                              "description": "Optional inclusive publication-year cutoff, applied locally and online."},
                  "search_expression": {"type": "string", "default": "", "description": "Additional raw Europe PMC expression; applies online only and never substitutes for the disease/topic filter."}}, ["query"]), guideline_search),
    Tool("paper_fulltext",
         "Fetch accessible full-text XML from Europe PMC. Supply exactly one PMCID or PMID; "
         "PMID is resolved to PMCID online and missing mappings return an explicit error. "
         "Returns title, source/section URLs, nested section IDs/titles, and section text. "
         "Use section for an exact ID, heading substring, or diagnosis/investigation/treatment. "
         "max_chars limits total returned section text; inspect truncated and request specific sections as needed. "
         "Not every PubMed article has accessible full text. Failures return an error object.",
         _schema({"pmcid": {"type": "string", "minLength": 1, "description": "PMC identifier, e.g. PMC7386163. Supply this or pmid."},
                  "pmid": {"type": "string", "minLength": 1, "description": "Numeric PubMed identifier; resolves to PMCID. Supply this or pmcid."},
                  "section": {"type": "string", "default": "", "description": "Optional section ID or heading; empty returns sections in order."},
                  "max_chars": {"type": "integer", "minimum": 1, "maximum": 50000, "default": 20000}},
                 [], oneOf=[{"required": ["pmcid"]}, {"required": ["pmid"]}]), paper_fulltext),
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
