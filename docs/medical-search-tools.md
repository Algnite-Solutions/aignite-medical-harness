# Medical fact and evidence search tools

Load `--tools src/ama/medical_tools.py` with either `ama chat` or `ama run`.
On `agent-faithful-reason`, the module's factory registers four retrieval tools plus
`analyze_images`, using the existing function-calling interface. The four retrieval
tools return JSON text without a separate LLM or access to dataset targets.
`analyze_images` uses an isolated visual expert and enables image delegation as
described in the [README](../README.md#optional-tools). Without `--tools`, no tools are loaded.

```bash
ama chat datasets/symptom2disease_demo --episode s2d-000001 --model qwen36 --tools src/ama/medical_tools.py
```

Choose any configured model that supports function calling. The four public data APIs
require no extra API key; main-model and visual-expert calls are separate and may incur costs.
PubMed accepts its source query syntax. Guideline search separates the disease/topic
from optional raw Europe PMC syntax and supports English/Chinese aliases for the indexed diseases.

## `pubmed_search(query, retmax=5, abstract=True)`

Searches PubMed through NCBI ESearch in relevance order, then retrieves article XML
with EFetch. Returns PMID, title, journal, year, abstract, PubMed URL, PMCID and DOI
when present, full-text links/status, and an `abstract_truncated` flag. If abstract
retrieval fails, ESummary provides metadata with an empty abstract. An empty list
means no matches; an `error` object means retrieval failed.

`query` must be nonempty; `retmax` is an integer from 1 to 50. Optional `NCBI_EMAIL`
and `NCBI_API_KEY` identify the caller. Requests are serialized within the process
with a minimum interval of 0.35 seconds without a key or 0.11 seconds with one.
This does not coordinate separate processes. Abstracts are not full texts or quality assessments.

## `guideline_search(query, max_results=5, live=True, population="", subtype="", intent="", year_to=None, search_expression="")`

Combines a local JSON index with a bounded Europe PMC candidate pool. Disease terms
and aliases are scoped to `TITLE_ABS`; additional guideline/consensus cues broaden
candidate types. Europe PMC relevance is used instead of citation ordering. Results
then require a disease match in the title or curated keywords: a passing abstract
mention in an unrelated guideline is insufficient. This conservative rule favors
precision and may omit relevant broad-scope guidelines.

Transparent title/publication-type rules distinguish clinical practice guidelines,
consensus statements, reporting guidelines, laboratory protocols, and other records.
Only the first two are returned. Studies/comments about guidelines are excluded,
including source-frozen examples from the live response. Classification evidence
is returned; these heuristics do not replace expert review.

`population` accepts `adult`, `pediatric`, or `older_adult`; `subtype` accepts `acute`,
`chronic`, or `autoimmune`; `intent` accepts `diagnosis`, `investigation`, or `treatment`.
Explicit acute/chronic/autoimmune terms in the topic also constrain the subtype.
Empty filters leave scope unrestricted. Explicit filters require matching metadata:
unspecified population does not satisfy an adult filter. Applicability fields are
curated metadata and/or lexical cues, not verified patient-specific eligibility.
By default, diagnosis-oriented references and broad populations rank ahead of
specialized subgroups, followed by publication year. `year_to` is an inclusive
cutoff, applied to local and online results. Unknown publication years cannot meet
an explicit cutoff. Every result identifies its publication year and marks latest
version status as `not_verified`; newer does not automatically mean applicable.

`query` is the topic; use `search_expression` for extra raw Europe PMC constraints
(online only). Simple legacy `TITLE:appendicitis` input still works, but field names
never become local keywords. Matching searches only approved field values, never
JSON keys or URLs. Known disease names and additional meaningful topic terms must
match; alias support is curated, not arbitrary translation.

Records preserve PMID, PMCID, DOI, available free/open full-text links and status,
journal, organization, and classification/applicability evidence. A journal is never
reported as the issuing organization. Online organizations remain unknown unless
merged with source-checked local metadata. Abstracts are limited to 800 characters
with `abstract_truncated`; relevance checks see the complete abstract first.
Duplicate publications retain available identifiers and free full-text links from
both sources, even when a sparse local record ranks first.
`max_results` (1-50) caps the combined deduplicated result list, not just online results.
`live=False` disables network access for this tool.

The bundled index contains five metadata records: KDIGO 2024 CKD and four historical
WSES references for appendicitis, cholecystitis, diverticulitis, and **severe acute**
pancreatitis. Their public IDs and diagnostic section anchors were checked against
the source on 2026-10-21; this is not physician review or certification of the latest
version. The WSES references date from 2019/2020 and do not establish suitable
knowledge cutoffs for a clinical experiment. They include scope, organization source,
and diagnosis/investigation section IDs, not guideline full text. Set
`AMA_GUIDELINES_PATH` to a custom JSON list; each record needs `title` and `url`.
Partial failures return `{"error": "...", "results": [...]}` and retain available results.
There is no general guideline-webpage fetch tool.

## `paper_fulltext(pmcid="", section="", max_chars=20000, pmid="")`

Supply exactly one PMCID or PMID. PMID is resolved through an exact PubMed-source
Europe PMC lookup; absent mappings are reported explicitly and no identifier is
guessed. A PMCID is an identifier, not a guarantee that the XML endpoint is accessible.
Returns title, source URL, available nested section IDs/titles, section text and
section URLs. An empty `section` returns top-level sections without duplicating nested
text; otherwise it matches an exact section ID or heading substring. The intent aliases
`diagnosis`, `investigation`, and `treatment` match related English heading cues.

`max_chars` (1-50000) caps the total returned section text; per-section and overall
`truncated` flags identify omitted content. Select a section to inspect later text
when an article exceeds the budget. Not every article provides accessible XML.
This tool does not read arbitrary PDFs or bypass access restrictions. Use identifiers
returned by search to read diagnostic criteria, differential diagnosis, and test-selection
sections with their source anchors. An abstract alone is not evidence of having read
these sections. Missing/inaccessible full text returns an error.

## `clinical_trial_search(condition, intervention="", status="", max_results=5)`

Queries ClinicalTrials.gov API v2 by condition, optional intervention, and optional
overall status, such as `RECRUITING` or `COMPLETED`. Empty status means all statuses.
`condition` must be nonempty; `max_results` is an integer from 1 to 50.

Returns NCT ID, title, URL, status, phase, study type, last update, conditions,
interventions, summary (1000 characters), eligibility criteria (3000 characters),
age/sex eligibility fields, and up to five study locations. Truncation flags describe
these limits. Registry records do not establish treatment effectiveness or personal
eligibility; the tool does not return complete posted results tables.

## Network behavior and tests

Each retrieval HTTP request has a 20-second timeout and a 4 MiB response cap. Retrieval failures
return errors; argument errors are handled by the Agent's existing tool-error path.
Tools do not automatically retry requests. Returned source text enters the normal
conversation history, with no automatic summary or trimming.

Deterministic tests, including Agent function-calling integration:

```bash
python -m pytest -q tests/test_medical_tools.py tests/test_guideline_retrieval.py tests/test_image_analysis_tools.py tests/test_medical_tools_api.py
```

Public endpoint smoke tests (no model calls, opt-in; PowerShell):

```powershell
$env:RUN_MEDICAL_TOOLS_LIVE_TESTS = "1"
python -m pytest -q tests/test_medical_tools_api.py -k 'not deepseek'
Remove-Item Env:RUN_MEDICAL_TOOLS_LIVE_TESTS
```

Optional real model function-calling tests use the `deepseek-flash` alias in `ama.json`.
They load `DEEPSEEK_API_KEY` from the environment
or local ignored `.env`, call the retrieval APIs, and consume model-provider credits:

```powershell
$env:RUN_API_TESTS = "1"
python -m pytest -q tests/test_medical_tools_api.py -k deepseek
Remove-Item Env:RUN_API_TESTS
```

All network tests skip by default. The four-disease tests use an empty local index
to independently check live retrieval, then separately read the source-checked local
references' diagnostic sections online. Raw search responses, URLs, retrieval times,
and returned sections are saved under ignored `tmp/guideline-review/live/` using only
public disease queries. Fixtures under `tests/fixtures/guideline_retrieval/` freeze
minimal metadata for the reference set and misleading live candidates.

Tests validate source/ID preservation, topic relevance, scope filters and protocol
integration, not medical answer accuracy. The four source references and fixed
distractors constitute a narrow regression set, not a comprehensive precision/recall
benchmark. Clinical benefit requires a separate `ama run`/`ama eval` experiment with
fixed model, cohort, tool budget and guideline cutoff. The factory continues to create
fresh visual-expert state per episode; retrieval calls do not release image evidence.

### Verification on `agent-faithful-reason` (2026-10-22)

- Focused retrieval, image-analysis and Agent regression: 134 passed. The five-tool
  factory is preserved, with fresh image state per episode. New duplicate-record
  tests cover full-text identifier/link preservation and curated organization metadata.
- Broader regression: 231 passed, 17 opt-in network tests skipped, 1 existing Windows
  CRLF-sensitive shell-wrapper test deselected (`test_shell_sample_and_separate_runs_work_without_raw_csv`).
- Separately enabled public endpoint tests: 12 passed, including online-only topic
  retrieval and four reference searches followed by diagnostic-section XML reads.
- Paid model tests were not run. These checks do not measure clinical diagnostic
  accuracy or establish a general retrieval precision/recall score.
