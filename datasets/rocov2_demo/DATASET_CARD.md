# rocov2_demo — Dataset Card

## Source and license

Source: ROCOv2 radiology (https://github.com/sctg-development/ROCOv2-radiology)

License: ROCOv2: CC BY-NC-SA 4.0; each image retains its source attribution

## Research use

Single-turn radiology image captioning and UMLS concept prediction research.

## Construction

Each selected image becomes one Episode with one Turn and one image Evidence item. Captions and CUIs remain evaluator-only targets.

## Size

- Episodes: 3
- Turns: 3
- Evidence items: 3

## Scoring and targets

Scorer: `rocov2`. Target records: 3.

## Artifacts

JPEG images are copied under artifacts/. Per-image PMC links and attributions are recorded in provenance.jsonl and import_report.json.

## Limitations and reporting

This derived subset is not an official ROCOv2 benchmark result. Captions have one reference and token overlap does not measure clinical correctness.

## Included-image provenance

| Image | PMCID | Attribution | Source |
|---|---|---|---|
| ROCOv2_2023_test_000001 | PMC8762516 | CC BY-NC Al Mulhim et al. (2022) | [article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8762516) |
| ROCOv2_2023_test_000009 | PMC9198419 | CC BY-NC Trowbridge et al. (2022) | [article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9198419) |
| ROCOv2_2023_test_000015 | PMC8931810 | CC BY Hishikawa et al. (2022) | [article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8931810) |
