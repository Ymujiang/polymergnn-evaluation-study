"""Materialize the pinned upstream runtime from a local checkout of PolymerGNN.

    git clone https://github.com/owencqueen/PolymerGNN.git runtime/upstream
    git -C runtime/upstream checkout c0345c1476f137721039c3d7d8e6772d93075794
    python3 scripts/materialize_runtime.py

This copies each file listed in runtime_source_manifest.json out of runtime/upstream, verifying its Git
blob hash, its SHA-256 digest and its size first, and writes runtime_compatible_manifest.json with the
same records. Nothing is downloaded: upstream carries no licence grant, so the pinned source has to come
from your own checkout.
"""
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "runtime_source_manifest.json"
CHECKOUT = ROOT / "runtime" / "upstream"
OUT = ROOT / "runtime_compatible_manifest.json"


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def read_record(record):
    path = CHECKOUT / record["path"]
    if not path.is_file():
        return None, "missing from checkout: " + record["path"]
    data = path.read_bytes()
    if len(data) != record["bytes"]:
        return None, "size mismatch: " + record["path"]
    if blob_sha(data) != record["git_blob"]:
        return None, "git blob mismatch: " + record["path"]
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        return None, "sha256 mismatch: " + record["path"]
    return {"path": record["path"], "git_blob": record["git_blob"],
            "sha256": record["sha256"], "bytes": record["bytes"]}, None


def main():
    if not (CHECKOUT / "polymerlearn").is_dir():
        sys.exit("No upstream checkout at " + str(CHECKOUT) + ". Clone upstream at the pinned commit "
                 "there first; see README, section Rebuilding the upstream runtime.")
    manifest = json.loads(MANIFEST.read_text())
    records, errors = [], []
    for record in manifest["files"]:
        out, err = read_record(record)
        if err:
            errors.append(err)
        else:
            records.append(out)
    if errors:
        sys.exit("Pinned runtime does not match the manifest:\n  " + "\n  ".join(errors[:10]))
    OUT.write_text(json.dumps({"commit": manifest["commit"], "files": records}, indent=1) + "\n")
    print("PASS pinned runtime files: " + str(len(records)))


if __name__ == "__main__":
    main()
