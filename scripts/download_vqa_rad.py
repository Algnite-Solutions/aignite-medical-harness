#!/usr/bin/env python3
"""Download an immutable public VQA-RAD snapshot; never overwrite a destination.

Only Python's standard library is needed. HTTPS_PROXY can select a local proxy.
No credentials or model APIs are used. This script does not mount network disks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "flaviagiammarino/vqa-rad"
REVISION = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"


def fetch(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ama-vqa-rad-download/1"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            partial.rename(destination)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    fetch(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true",
          root / "upstream-metadata.json")
    metadata = json.loads((root / "upstream-metadata.json").read_text())
    if metadata["sha"] != REVISION:
        raise ValueError("upstream returned an unexpected revision")
    records = []
    for entry in sorted(metadata["siblings"], key=lambda row: row["rfilename"]):
        name = entry["rfilename"]
        path = root / "snapshot" / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe upstream filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} ({entry['size']} bytes)", flush=True)
        fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}", path)
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if len(content) != entry["size"]:
            raise ValueError(f"size mismatch: {name}")
        if entry.get("lfs"):
            verified = sha == entry["lfs"]["sha256"]
        else:
            blob = b"blob " + str(len(content)).encode() + b"\0" + content
            verified = hashlib.sha1(blob).hexdigest() == entry["blobId"]
        if not verified:
            raise ValueError(f"upstream checksum mismatch: {name}")
        records.append({"path": name, "bytes": len(content), "sha256": sha})
    manifest = {"repo_id": REPO, "revision": REVISION,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "upstream_checksums_verified": True,
                "files": records}
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    (root / "source-manifest.json").write_bytes(payload)
    (root / "source-manifest.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "  source-manifest.json\n")
    print(f"Complete: {len(records)} files, {sum(row['bytes'] for row in records)} bytes", flush=True)


if __name__ == "__main__":
    main()
