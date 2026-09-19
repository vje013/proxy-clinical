"""Command line entry point.

    synthgen generate --n 500 --seed <hex> --out data/pilot/
    synthgen validate --corpus data/pilot/corpus.jsonl
    synthgen check-determinism --n 100 --seed <hex>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import GENERATOR_VERSION
from .cast import parse_master_seed
from .emit import DEFAULT_INSTRUCTION_VERSION, INSTRUCTIONS, dumps_record, is_val, read_jsonl, training_pair, write_jsonl
from .generate import Composition, Generator, build_plan, corpus_meta
from .validate import GateResult, distribution_report, run_all_gates


def _composition_from_args(args: argparse.Namespace) -> Composition:
    kwargs = {}
    for key in ("single", "multi", "listing", "bundle_docs", "hard"):
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = val
    return Composition(**kwargs)


def _generate_records(n: int, seed_bytes: bytes, comp: Composition, prefix: str) -> list[dict]:
    plan = build_plan(n, comp, seed_bytes, prefix=prefix)
    return Generator(seed_bytes).generate(plan)


def cmd_generate(args: argparse.Namespace) -> int:
    t0 = time.time()
    seed_bytes = parse_master_seed(args.seed)
    comp = _composition_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = _generate_records(args.n, seed_bytes, comp, args.prefix)
    print(f"generated {len(records)} samples in {time.time() - t0:.1f}s", file=sys.stderr)

    paraphrase_stats = None
    if args.paraphrase:
        from .paraphrase import anthropic_paraphraser, apply_paraphrase
        records, stats = apply_paraphrase(records, anthropic_paraphraser(args.paraphrase_model))
        paraphrase_stats = stats.as_dict()
        print(f"paraphrase: kept {stats.kept}/{stats.attempted}, discarded {stats.discarded}", file=sys.stderr)

    gates = run_all_gates(records)

    determinism = "SKIPPED"
    if args.determinism_check and not args.paraphrase:
        again = _generate_records(args.n, seed_bytes, comp, args.prefix)
        a = "\n".join(dumps_record(r) for r in records)
        b = "\n".join(dumps_record(r) for r in again)
        determinism = "PASS" if a == b else "FAIL"
        gates.append(GateResult("6 determinism", [] if a == b else ["second run differs from first"]))
        print(f"determinism: {determinism}", file=sys.stderr)
    elif args.paraphrase:
        determinism = "SKIPPED (paraphrase pass is non-deterministic)"

    counts = comp.counts(args.n)
    meta = corpus_meta(args.n, seed_bytes.hex(), comp, counts, args.instruction_version)
    meta["paraphrase"] = paraphrase_stats
    meta["determinism"] = determinism

    n_corpus = write_jsonl(out / "corpus.jsonl", records)
    train = [training_pair(r, args.instruction_version) for r in records if not is_val(r)]
    val = [training_pair(r, args.instruction_version) for r in records if is_val(r)]
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    meta["split_counts"] = {"train": len(train), "val": len(val)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = Path(args.report) if args.report else out / "pilot_report.md"
    report = distribution_report(records, meta, [g for g in gates if not g.gate.startswith("6")],
                                 determinism=determinism)
    report_path.write_text(report, encoding="utf-8")

    for g in gates:
        print(f"gate {g.gate}: {'PASS' if g.ok else 'FAIL'} ({len(g.failures)} failures)", file=sys.stderr)
    print(f"wrote {n_corpus} records to {out} (train {len(train)}, val {len(val)}); report at {report_path}",
          file=sys.stderr)
    return 0 if all(g.ok for g in gates) else 1


def cmd_validate(args: argparse.Namespace) -> int:
    records = read_jsonl(args.corpus)
    gates = run_all_gates(records)
    meta_path = Path(args.corpus).parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for g in gates:
        print(f"gate {g.gate}: {'PASS' if g.ok else 'FAIL'} ({len(g.failures)} failures)")
        for f in g.failures[:10]:
            print(f"   - {f}")
    if args.report:
        Path(args.report).write_text(distribution_report(records, meta, gates), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0 if all(g.ok for g in gates) else 1


def cmd_check_determinism(args: argparse.Namespace) -> int:
    seed_bytes = parse_master_seed(args.seed)
    comp = _composition_from_args(args)
    a = [dumps_record(r) for r in _generate_records(args.n, seed_bytes, comp, args.prefix)]
    b = [dumps_record(r) for r in _generate_records(args.n, seed_bytes, comp, args.prefix)]
    if a == b:
        print(f"determinism: PASS ({len(a)} records byte-identical)")
        return 0
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"determinism: FAIL ({len(diffs)} differing records, first at index {diffs[:5]})")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synthgen", description=f"Proxy Clinical synthetic generator {GENERATOR_VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--n", type=int, default=500, help="number of samples (the one config value that scales the corpus)")
        sp.add_argument("--seed", required=True, help="master seed (hex)")
        sp.add_argument("--prefix", default="pilot", help="sample_id prefix")
        sp.add_argument("--single", type=float, default=None, help="share of single-patient narratives (default 0.52)")
        sp.add_argument("--multi", type=float, default=None, help="share of multi-patient narratives (default 0.24)")
        sp.add_argument("--listing", type=float, default=None, help="share of standalone listings (default 0.16)")
        sp.add_argument("--bundle-docs", dest="bundle_docs", type=float, default=None, help="share of bundle documents (default 0.08)")
        sp.add_argument("--hard", type=float, default=None, help="hard-case flagged share (default 0.10)")

    g = sub.add_parser("generate", help="generate a corpus, run the gates, write the report")
    add_common(g)
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--report", default=None, help="report path (default <out>/pilot_report.md)")
    g.add_argument("--instruction-version", dest="instruction_version", default=DEFAULT_INSTRUCTION_VERSION,
                   choices=sorted(INSTRUCTIONS), help="training-view output contract (default v2: text anchors)")
    g.add_argument("--paraphrase", action="store_true", help="optional LLM paraphrase pass (network; OFF by default)")
    g.add_argument("--paraphrase-model", default="claude-sonnet-4-5")
    g.add_argument("--no-determinism-check", dest="determinism_check", action="store_false",
                   help="skip the second run / byte comparison")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="run the gates on an existing corpus.jsonl")
    v.add_argument("--corpus", required=True)
    v.add_argument("--report", default=None)
    v.set_defaults(func=cmd_validate)

    d = sub.add_parser("check-determinism", help="generate twice in memory and compare bytes")
    add_common(d)
    d.set_defaults(func=cmd_check_determinism)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
