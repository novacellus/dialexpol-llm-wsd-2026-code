#!/usr/bin/env python3
"""Fetch and verify the pinned DiaLexPoL dataset archive.

The release maintainer must set ``DIALEXPOL_DATA_URL`` to the immutable
versioned archive URL and ``DIALEXPOL_ARCHIVE_SHA256`` to that archive's
SHA-256.  A local ``file://`` URL is supported for clean-clone testing.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PINNED_DATA_URL = "https://zenodo.org/records/21625867/files/dialexpol-wsd-datasets-v1.0.0.zip?download=1"
PINNED_ARCHIVE_SHA256 = "5b0d61b49a0108e4c78d6d0e6905ac1eea6bb6d7c168f8c822358607a267c44a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_checksums(root: Path) -> None:
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        raise SystemExit("Dataset archive does not contain checksums.sha256")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip(" *")
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"Checksum mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("DIALEXPOL_DATA_URL", PINNED_DATA_URL))
    parser.add_argument("--archive-sha256", default=os.environ.get("DIALEXPOL_ARCHIVE_SHA256", PINNED_ARCHIVE_SHA256))
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("DIALEXPOL_DATA_DIR", Path.cwd() / "data")))
    parser.add_argument("--force", action="store_true",
                        help="replace a previous installation containing checksums.sha256")
    args = parser.parse_args()
    if not args.url or not args.archive_sha256:
        raise SystemExit("Provide the immutable archive URL and DIALEXPOL_ARCHIVE_SHA256")

    archive = args.data_dir.parent / ".dialexpol-dataset-download"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if args.url.startswith("file://"):
        shutil.copyfile(Path(args.url[7:]), archive)
    else:
        urllib.request.urlretrieve(args.url, archive)
    if sha256(archive) != args.archive_sha256.lower():
        raise SystemExit("Dataset archive checksum mismatch")

    staging = args.data_dir.parent / ".dialexpol-dataset-unpack"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(staging)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            handle.extractall(staging, filter="data")
    else:
        raise SystemExit("Unsupported dataset archive format; expected zip or tar")

    roots = [staging] + [p for p in staging.iterdir() if p.is_dir()]
    verified = next((root for root in roots if (root / "checksums.sha256").is_file()), None)
    if verified is None:
        raise SystemExit("Unpacked archive has no checksums.sha256")
    verify_release_checksums(verified)
    if args.data_dir.exists():
        if any(args.data_dir.iterdir()):
            if not args.force or not (args.data_dir / "checksums.sha256").is_file():
                raise SystemExit(
                    f"Refusing to replace non-empty data directory: {args.data_dir}; "
                    "use --force only for a previous verified installation"
                )
            shutil.rmtree(args.data_dir)
    shutil.move(str(verified), str(args.data_dir))
    shutil.rmtree(staging, ignore_errors=True)
    archive.unlink(missing_ok=True)
    print(f"Verified dataset installed at {args.data_dir}")


if __name__ == "__main__":
    main()
