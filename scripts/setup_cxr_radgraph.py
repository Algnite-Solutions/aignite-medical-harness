"""Download and verify pinned official weights; HF_ENDPOINT may select a mirror."""
import argparse
import tarfile
import os
import urllib.request
from pathlib import Path

from ama.cxr_radgraph import (WEIGHTS_REVISION, WEIGHTS_SHA256, CONFIG_SHA256, ENCODER_ALIAS,
                             ENCODER_REVISION, ENCODER_FILES, git_blob_digest)
from ama.importers.mimic_cxr import digest, staging

ARCHIVE_SHA256 = "7808499d3dc6c37859d3283a4258be0364104420224aeae7395f2f0f9c28bbb9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("local_data/radgraph"))
    parser.add_argument("--archive", type=Path, help="optional predownloaded pinned archive")
    args = parser.parse_args()
    dest = args.cache / "radgraph"
    if dest.exists():
        if digest(dest / "weights.th") != WEIGHTS_SHA256:
            raise ValueError("existing weights differ from pinned version")
        print("Pinned RadGraph weights already present")
    else:
        archive = args.archive
        if archive is None:
            from huggingface_hub import hf_hub_download
            archive = Path(hf_hub_download("StanfordAIMI/RRG_scorers", "radgraph.tar.gz",
                                          revision=WEIGHTS_REVISION, cache_dir=str(args.cache / "downloads")))
        if digest(archive) != ARCHIVE_SHA256:
            raise ValueError("official archive hash mismatch")
        with staging(dest) as stage:
            with tarfile.open(archive) as stream:
                stream.extractall(stage, filter="data")
            if digest(stage / "weights.th") != WEIGHTS_SHA256:
                raise ValueError("extracted weights hash mismatch")
        print("Pinned RadGraph weights verified")
    if digest(dest / "config.json") != CONFIG_SHA256:
        raise ValueError("model config differs from pinned version")
    repo = args.cache / "encoder-hub" / ("models--" + ENCODER_ALIAS.replace("/", "--"))
    snapshot = repo / "snapshots" / ENCODER_REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    canonical = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    for name, checksum in ENCODER_FILES.items():
        path = snapshot / name
        if not path.exists():
            data = urllib.request.urlopen(f"{endpoint}/{canonical}/resolve/{ENCODER_REVISION}/{name}", timeout=60).read()
            if git_blob_digest(data) != checksum:
                raise ValueError(f"encoder file hash mismatch: {name}")
            path.write_bytes(data)
        if git_blob_digest(path.read_bytes()) != checksum:
            raise ValueError(f"existing encoder file differs: {name}")
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(ENCODER_REVISION)
    print("Pinned encoder config and tokenizer verified (no base weights needed)")


if __name__ == "__main__":
    main()
