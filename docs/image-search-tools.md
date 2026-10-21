# MedHEB similar-image retrieval

The optional tool file `src/ama/image_search_tools.py` registers one function:

```python
image_search(image_path, top_k=5)
```

The service encodes a server-prepared MIMIC-CXR JPEG with `google/medsiglip-448`
(1152-dimensional, L2-normalized image embeddings) and retrieves similar training
images from Qdrant. Reports are paired content, not whole-report embeddings. AMA
only calls the retrieval API; no encoder or database client is installed locally.

## Connection and credentials

Set these in the environment or ignored local `.env`:

- `MEDHEB_SEARCH_URL`: default `http://127.0.0.1:8766` when AMA shares the service host/network.
- `MEDHEB_SEARCH_KEY`: API key supplied securely by the service maintainer.
- `MEDHEB_SEARCH_KEY_FILE`: optional protected UTF-8 file containing the key. A nonempty
  environment key takes precedence. Do not commit the file or its contents.

When running on a separate machine, an authorized user can create a tunnel in a
separate terminal and leave it running:

```bash
ssh -N -L 18766:127.0.0.1:8766 aignitelab@10.214.105.106
```

Then set `MEDHEB_SEARCH_URL=http://127.0.0.1:18766`. The DGX service listens on its
loopback address, so `http://10.214.105.106:8766` is not the client endpoint. In a
container, localhost refers to that container; on the same Docker network the
service may be reachable as `http://api:8766`.

An optional manual startup check is:

```bash
curl --noproxy '*' http://127.0.0.1:18766/health
```

`GET /health` needs no key. Its `status`, `device`, `collection`, and startup
`gallery_size` indicate startup readiness; it does not probe NAS/database health
on each call or test retrieval quality. The tool does not automatically call it
before every search.

## Inputs and loading

`image_path` must be an actual path obtained from the server's prepared manifest:

```text
files/pXX/pPATIENT_ID/sSTUDY_ID/DICOM_ID.jpg
```

This is a format placeholder, not a real case. The server resolves the path under
`/mnt/nas/data/incoming/MedEmbed/prepared/chest-20261018-160202/MIMIC-CXR-JPG-2.1.0-medheb`
and validates existence/permission against its prepared data. The client accepts
1-512 characters and checks the relative JPEG layout. It does not read the image,
upload bytes, accept image URLs/Base64, or map AMA-local artifact filenames to
server paths. Provide the authorized query's actual server path in the observation
or user input; loading the tool alone does not create this mapping.

`top_k` must be an integer from 1 to 10; the default is 5. Booleans, strings, and
floats are rejected. Only these two fields go into `POST /search/image`, with
`Content-Type: application/json`, `Content-Length`, and `X-API-Key` headers.

For a processed MIMIC-CXR dataset, replace these placeholders with actual paths,
episode ID, and a configured model supporting tools/images:

```bash
ama chat /path/to/processed/mimic_cxr --episode STUDY_ID --model MODEL --tools src/ama/image_search_tools.py
```

No flag means no tools. This module works with the existing Agent and does not
change its image input, history, scoring, or model configuration.

## Outputs and interpretation

The JSON tool result preserves `results`, `similarity`, `elapsed_seconds`, `model`,
`collection`, `same_patient_excluded`, `identical_image_excluded`, and the free-text
`scope`. Each match preserves its string study/image IDs, original report and
newlines, image/report server paths, train split, evidence ID, and cosine score.

- There may be 0 to `top_k` matches; fewer candidates are not padded.
- Multiple images may share `mimic-train:<study_id>`. Image-level results remain
  separate, while `ToolResult.evidence_ids` releases each citeable ID once.
- Remote image/report paths are provenance, not downloadable URLs or local file
  attachments. `ToolResult.images` is empty.
- Cosine similarity is not diagnostic confidence. Similar reports are external
  reference cases, not the query's ground truth or verified findings.
- The service requires train-only results and excludes the query patient and
  SHA256-identical images. The client validates the returned flags, train split,
  patient IDs in paths, identity consistency, score bounds/order, and result count
  before releasing evidence. It cannot independently recompute SHA256 exclusions
  without the server's image contents/hashes.
- Inspect `collection` and `scope`. The supplied service contract describes a
  32-patient demonstration gallery; a separately building batch collection is not
  automatically selected. No gallery size or collection name is hardcoded here.

Reports become ordinary tool text in the Agent's saved conversation. Use the
project's restricted storage arrangements for credentialed MIMIC data and runs.
Credentials are never included in tool definitions, results, or client errors.

## Failure behavior and dummy tests

Each request has a 180-second client timeout. Proxy environment variables are
ignored, and redirects are not followed so the key is not forwarded to another
endpoint. There are no automatic retries. The client imposes an 8 MiB response cap.

HTTP status is checked before JSON parsing. Known statuses map to the documented
`invalid_request_or_image` (400), `unauthorized` (401), `not_found` (404), and
`retrieval_failed` (503) errors, retaining `http_status`. Other statuses use
`http_error`; error bodies may be HTML and are not echoed into the conversation.
Connection failures/timeouts return `transport_error`; malformed or inconsistent
successful responses return `invalid_response`. Oversized responses return
`response_too_large`. Missing/bad client settings return `client_configuration`.
Invalid tool arguments use the existing Agent tool-error handling. Errors release
no evidence IDs.

```bash
python -m pytest -q tests/test_image_search_tools.py tests/test_agent.py
```

The tests start a dummy HTTP server on an ephemeral loopback port, use synthetic
paths/reports and a fake credential, and drive the tool through a deterministic
FakeModel. They check request framing/authentication, proxy bypass, zero/short
results, duplicate study citations, train/patient exclusions, malformed payloads,
HTTP errors including HTML and redirects, timeout handling, key-file loading, and
the Agent call/result round trip. No DGX, SSH session, Qdrant connection, real
credential, model-provider call, or clinical image is required. They establish
client protocol behavior, not real-service connectivity or retrieval quality.
