#!/usr/bin/env python3
"""Build full/delta Windows app packages from Git's exported HEAD, without Git history."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile


SCHEMA = 1
RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)


def validate_path(path: str) -> None:
    if not isinstance(path, str) or not path.startswith("app/"):
        raise ValueError(f"Not an app file: {path!r}")
    for part in path.split("/"):
        if (part in ("", ".", "..") or part.endswith((".", " "))
                or re.search(r'[\x00-\x1f<>:"\\|?*]', part) or RESERVED.match(part)):
            raise ValueError(f"Unsafe Windows path: {path!r}")


def validate_state(state: dict) -> dict[str, dict]:
    if (not isinstance(state, dict) or type(state.get("schema")) is not int or state.get("schema") != SCHEMA
            or not re.fullmatch(r"[A-Za-z0-9._-]+", state.get("release_id", ""))
            or not re.fullmatch(r"[0-9a-f]{40,64}", state.get("commit", ""))
            or not isinstance(state.get("files"), list)):
        raise ValueError("Invalid deployment state")
    files = {}
    names = set()
    for entry in state["files"]:
        path = entry["path"]
        validate_path(path)
        if path.lower() in names:
            raise ValueError(f"Duplicate Windows path: {path}")
        if (not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
                or type(entry.get("size")) is not int or entry["size"] < 0):
            raise ValueError(f"Invalid file identity: {path}")
        names.add(path.lower())
        files[path] = entry
    for path in names:
        if any("/".join(path.split("/")[:i]) in names for i in range(2, len(path.split("/")))):
            raise ValueError(f"File/directory collision: {path}")
    if "app/__init__.py" not in files:
        raise ValueError("Deployment state has no app/__init__.py")
    return files


def json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def build_package(repo: Path, output: Path, release_id: str, base_path: Path | None = None,
                  revision: str = 'HEAD') -> dict:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", release_id):
        raise ValueError("Invalid release ID")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--verify", revision + '^{commit}'], text=True).strip()
    # git archive preserves export-ignore/export-subst and never reads dirty app files.
    exported = subprocess.check_output(["git", "-C", str(repo), "archive", "--format=tar", commit, "app"])
    payload = {}
    with tarfile.open(fileobj=io.BytesIO(exported)) as archive:
        for member in archive:
            if member.isdir():
                continue
            validate_path(member.name)
            if not member.isfile():
                raise ValueError(f"Only regular files can be deployed: {member.name}")
            payload[member.name] = archive.extractfile(member).read()
    state = dict(schema=SCHEMA, release_id=release_id, commit=commit, files=[
        dict(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        for path, data in sorted(payload.items())])
    target = validate_state(state)
    baseline = None
    base_sha256 = None
    if base_path is not None:
        raw = base_path.read_bytes().strip()
        baseline = json.loads(raw)
        if baseline is not None:
            baseline = validate_state(baseline)
            base_sha256 = hashlib.sha256(raw).hexdigest()
    # Case-only renames on Windows are explicitly delete + add, never two payload files.
    upserts = [path for path, entry in target.items()
               if baseline is None or path not in baseline or
               any(baseline[path][field] != entry[field] for field in ('sha256', 'size'))]
    deletes = sorted(set(baseline or {}) - set(target))
    manifest = dict(schema=SCHEMA, kind="full" if baseline is None else "delta",
                    base_sha256=base_sha256, upserts=upserts, deletes=deletes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json_bytes(manifest))
        archive.writestr("state.json", json_bytes(state))
        for path in upserts:
            archive.writestr("payload/" + path, payload[path])
    return dict(kind=manifest["kind"], upserts=len(upserts), deletes=len(deletes),
                total_files=len(target), bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--base-state", type=Path)
    parser.add_argument("--revision", default='HEAD')
    args = parser.parse_args()
    summary = build_package(args.repo, args.output, args.release_id, args.base_state, args.revision)
    print(f"Package: {summary['kind']}; upserts={summary['upserts']}; deletes={summary['deletes']}; total={summary['total_files']}")
    print(f"Archive size: {summary['bytes'] / 1048576:.2f} MiB ({summary['bytes']} bytes)")
    print(f"SHA-256: {summary['sha256']}")


if __name__ == "__main__":
    main()
