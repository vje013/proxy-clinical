#!/usr/bin/env python3
"""Pack a run folder into one zip the moment the run finishes.

    python scripts/pack_run.py runs/pilot-qwen2.5-3b-v2-20260919            # -> runs/pilot-qwen2.5-3b-v2-20260919.zip
    python scripts/pack_run.py <run_dir> --out /content/run.zip

Includes everything the run wrote (adapter, manifests, predictions, eval,
determinism, pip freeze, config) and excludes checkpoint-* directories, which
are only useful for resuming inside the same runtime. Files are added in a
fixed order with fixed timestamps, so packing the same folder twice gives the
same bytes. Writes <zip>.sha256 next to the zip and prints the manifest.

Why this exists: Colab's /content is wiped when the runtime is reclaimed, and
a run whose artifacts were never downloaded is a run that did not happen.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

EXCLUDE_DIR_PREFIXES = ("checkpoint-",)
FIXED_DATE = (1980, 1, 1, 0, 0, 0)  # earliest zip timestamp; keeps the archive byte-stable across packs
REQUIRED = ("adapter/adapter_model.safetensors", "run_manifest.json", "predictions.jsonl", "eval_report.md",
            "eval.json", "determinism.json", "pip_freeze.txt")


def iter_files(run_dir: Path):
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(EXCLUDE_DIR_PREFIXES))
        for name in sorted(files):
            p = Path(root) / name
            yield p, p.relative_to(run_dir.parent).as_posix()


def pack(run_dir: Path, out: Path | None = None, require_complete: bool = True) -> tuple[Path, list[str]]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run folder {run_dir} does not exist")
    missing = [r for r in REQUIRED if not (run_dir / r).exists()]
    if missing and require_complete:
        raise SystemExit(f"run folder {run_dir} is incomplete, missing: {missing} (pass --partial to pack anyway)")
    out = out or run_dir.with_suffix(".zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    names: list[str] = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in iter_files(run_dir):
            info = zipfile.ZipInfo(arc, date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(p, "rb") as fh:
                z.writestr(info, fh.read())
            names.append(arc)
    os.replace(tmp, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    return out, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="zip a run folder (checkpoints excluded)")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="zip path (default: <run_dir>.zip next to the folder)")
    ap.add_argument("--partial", action="store_true", help="pack even if required artifacts are missing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out, names = pack(Path(args.run_dir), Path(args.out) if args.out else None, require_complete=not args.partial)
    if not args.quiet:
        for n in names:
            print(f"  {n}", file=sys.stderr)
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB, {len(names)} files, sha256 {Path(str(out) + '.sha256').read_text().split()[0][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
