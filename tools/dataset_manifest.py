"""Record what a dataset directory contained, so a result can name its inputs.

The datasets live on /project and are deliberately not in git: they are hundreds
of gigabytes, several are trial data under a DUA, and a repository is the wrong
place for either. What git can carry is the answer to "which bytes was this
result computed from", which is a manifest: for every file, its size and its
SHA-256, plus one digest over the whole directory.

That digest is what a paper, a config or an experiment record cites. It costs
kilobytes, it is diffable, and it turns "which split did this run use" from a
conversation into a lookup. It also catches the failure this repository is most
exposed to today -- a dataset changing under a finished result, silently.

Reading is streamed, so memory does not scale with file size. Hashing the full
datasets_raw tree is not login-node work: use tools/manifest.slurm.

    python tools/dataset_manifest.py write  datasets_processed/DCLP3
    python tools/dataset_manifest.py verify datasets_processed/DCLP3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MANIFEST = "MANIFEST.sha256.json"
CHUNK = 1 << 20
SKIP = {MANIFEST, ".DS_Store"}


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def scan(root: Path) -> dict:
    """Every regular file under root, in one canonical order.

    Symlinks are recorded by target and never followed. Following them would
    hash whatever they point at today, which is the thing a manifest exists to
    pin down -- and on this cluster a results directory symlinked into /scratch
    is a normal layout, not an exotic one.
    """

    files: dict[str, dict] = {}
    links: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            if name in SKIP:
                continue
            full = Path(dirpath) / name
            rel = str(full.relative_to(root))
            if full.is_symlink():
                links[rel] = os.readlink(full)
                continue
            sha, size = file_digest(full)
            files[rel] = {"sha256": sha, "bytes": size}

    # One digest for the whole directory: the sorted "sha256  path" lines,
    # hashed. Independent of walk order, filesystem and platform.
    rollup = hashlib.sha256()
    for rel in sorted(files):
        rollup.update(("%s  %s\n" % (files[rel]["sha256"], rel)).encode("utf-8"))
    for rel in sorted(links):
        rollup.update(("symlink %s  %s\n" % (links[rel], rel)).encode("utf-8"))

    return {
        "root": root.name,
        "files": files,
        "symlinks": links,
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "directory_sha256": rollup.hexdigest(),
    }


def write(root: Path) -> int:
    manifest = scan(root)
    (root / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "%s: %d files, %.2f GB, directory_sha256=%s"
        % (
            manifest["root"],
            manifest["file_count"],
            manifest["total_bytes"] / 1e9,
            manifest["directory_sha256"],
        )
    )
    return 0


def verify(root: Path) -> int:
    path = root / MANIFEST
    if not path.exists():
        print("%s: no %s; run `write` first" % (root, MANIFEST), file=sys.stderr)
        return 2

    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = scan(root)
    if current["directory_sha256"] == recorded.get("directory_sha256"):
        print("%s: unchanged (%d files)" % (current["root"], current["file_count"]))
        return 0

    was, now = recorded.get("files", {}), current["files"]
    for rel in sorted(set(was) - set(now)):
        print("  removed  %s" % rel)
    for rel in sorted(set(now) - set(was)):
        print("  added    %s" % rel)
    for rel in sorted(set(now) & set(was)):
        if now[rel]["sha256"] != was[rel]["sha256"]:
            print("  CHANGED  %s" % rel)
    print("%s: DIFFERS from the recorded manifest" % current["root"], file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["write", "verify"])
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args(argv)

    status = 0
    for root in args.roots:
        if not root.is_dir():
            print("%s: not a directory" % root, file=sys.stderr)
            status = max(status, 2)
            continue
        status = max(status, write(root) if args.action == "write" else verify(root))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
