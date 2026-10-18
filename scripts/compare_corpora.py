#!/usr/bin/env python
"""TinyStories against a second corpus (PREREGISTRATION.md Amendment A6).

    ISOMEM_DATASET=fineweb_edu python scripts/compare_corpora.py [--json PATH] [--fig PATH]

Answers the one question the second corpus exists for: does the optimum precision hold on
harder text? Scores P8-P10 and applies A6's interpretation rules exactly as written.

Losses are compared *within* a corpus in nats per token, as everywhere else. Anything
compared *across* corpora is converted to bits per byte, because the two corpora have
different tokenizers and a nat per token means a different amount of text in each. The
conversion uses each corpus' measured val tokens-per-byte, recomputed here from files whose
hashes are verified against their committed configs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from isomem import corpus as C  # noqa: E402
from isomem.analysis import load_cells, null_verdict, pairwise_tests  # noqa: E402

LN2 = math.log(2)
TS_COLOR, OTHER_COLOR = "#2a78d6", "#eb6834"      # categorical slots 1-2, validated


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def tokens_per_byte(c: C.Corpus, cache: Path) -> float:
    """Val tokens per UTF-8 byte under the corpus' own tokenizer, hash-verified.

    Byte-level BPE maps every byte to one character of the token string, so a token's byte
    length is its string length. The EOT separator carries no bytes. Cached, because the
    token files exist only on the compute host.
    """
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    if c.name in cached:
        return cached[c.name]
    from tokenizers import Tokenizer

    val = c.tokens_dir / "val.bin"
    if _sha(c.tokenizer_path) != c.tokenizer_hash:
        raise AssertionError(f"{c.name}: tokenizer hash mismatch")
    if _sha(val) != c.split_hashes["val"]:
        raise AssertionError(f"{c.name}: val split hash mismatch")
    tok = Tokenizer.from_file(str(c.tokenizer_path))
    eot = tok.token_to_id("<|endoftext|>")
    blen = np.array([0 if i == eot else len(tok.id_to_token(i))
                     for i in range(tok.get_vocab_size())])
    ids = np.memmap(val, dtype=np.uint16, mode="r")
    tpb = float(len(ids) / blen[ids].sum())
    cached[c.name] = tpb
    cache.write_text(json.dumps(cached, indent=2) + "\n")
    return tpb


def bpb(nats: float, tpb: float) -> float:
    return nats * tpb / LN2


def rule(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--fig", type=Path, default=None)
    args = ap.parse_args()

    other = C.ACTIVE
    if other.is_default:
        print("set ISOMEM_DATASET to the second corpus")
        return 1
    out_dir = other.results_csv.parent
    blob = compare(C.load(C.DEFAULT), other, out_dir,
                   args.fig or out_dir / "figures" / "corpora.png")
    if args.json:
        args.json.write_text(json.dumps(blob, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")
    return 0


def compare(ts: C.Corpus, other: C.Corpus, out_dir: Path, fig_path: Path) -> dict:
    """The whole report, for two explicit corpus profiles. Returns the numbers."""
    cells = {ts.name: load_cells(ts.results_csv, sweep=ts.sweep_label),
             other.name: load_cells(other.results_csv, sweep=other.sweep_label)}
    floor = {c.name: json.loads(c.bigram_path.read_text())["val_loss"] for c in (ts, other)}
    cache = out_dir / "tokens_per_byte.json"
    tpb = {c.name: tokens_per_byte(c, cache) for c in (ts, other)}

    print(f"TinyStories vs {other.name} — PREREGISTRATION.md Amendment A6")
    print(f"tokens/byte: {ts.name} {tpb[ts.name]:.6f}, {other.name} {tpb[other.name]:.6f}")
    print(f"bigram floor: {ts.name} {floor[ts.name]:.4f} nats ({bpb(floor[ts.name], tpb[ts.name]):.3f} "
          f"bits/byte), {other.name} {floor[other.name]:.4f} nats "
          f"({bpb(floor[other.name], tpb[other.name]):.3f} bits/byte)")

    blob = {"tokens_per_byte": tpb, "bigram_floor_nats": floor, "budgets": {}}
    for name in other.arm["budgets"]:
        rule(f"{name}")
        row = {k: sorted([c for c in v if c.budget_name == name and c.usable],
                         key=lambda c: c.bits) for k, v in cells.items()}
        if not row[other.name]:
            print("  no usable cells yet")
            continue
        o_by = {c.bits: c for c in row[other.name]}
        t_by = {c.bits: c for c in row[ts.name]}
        print(f"  {'b':>5} | {ts.name:^22} | {other.name:^30}")
        print(f"  {'':>5} | {'nats':>9} {'sd':>6} {'bpb':>5} | {'nats':>9} {'sd':>6} "
              f"{'bpb':>5} {'n':>2} {'floor':>5}")
        for b in sorted(set(o_by) | set(t_by)):
            t, o = t_by.get(b), o_by.get(b)
            tl = f"{t.mean:>9.4f} {t.std:>6.4f} {bpb(t.mean, tpb[ts.name]):>5.3f}" if t else " " * 22
            ol = (f"{o.mean:>9.4f} {o.std:>6.4f} {bpb(o.mean, tpb[other.name]):>5.3f} {o.n:>2} "
                  f"{'ok' if o.mean < floor[other.name] else 'FAIL':>5}") if o else ""
            print(f"  {b:>5g} | {tl} | {ol}")

        best = {k: min(v, key=lambda c: c.mean) for k, v in row.items()}
        verdict = {k: null_verdict(cells[k], name) for k in row}
        for k in row:
            print(f"  {k}: {verdict[k].describe().split(': ', 1)[1]}")
        tests = pairwise_tests(cells[other.name], name)
        if tests:
            print(f"  {other.name} Welch tests against its argmin:")
            for t in tests:
                print(f"    vs b={t['other_bits']:<5g} gap {t['gap']:+.4f}  sd {t['pooled_sd']:.4f}"
                      f"  p={t['p']:.3g}{'' if t['detectable'] else '  (below power threshold)'}")

        pen = {k: (next((c.mean for c in row[k] if c.bits == 1.58), None), best[k].mean)
               for k in row}
        pen_bpb = {k: bpb(v[0] - v[1], tpb[k]) if v[0] is not None else None
                   for k, v in pen.items()}
        print(f"  ternary penalty: {ts.name} {pen_bpb[ts.name]:.4f} bits/byte, "
              f"{other.name} {pen_bpb[other.name]:.4f} bits/byte")

        same = best[other.name].bits == best[ts.name].bits
        if verdict[other.name].is_null:
            reading = f"NULL on {other.name}: no argmin is claimed at this budget"
        elif same:
            reading = f"same argmin (b={best[other.name].bits:g}): the result holds on harder text"
        else:
            reading = (f"argmin MOVES from b={best[ts.name].bits:g} to b={best[other.name].bits:g}: "
                       f"the optimum depends on the text")
        print(f"  A6 reading: {reading}")

        blob["budgets"][name] = {
            "argmin": {k: best[k].bits for k in row},
            "null": {k: verdict[k].is_null for k in row},
            "gap": {k: verdict[k].gap for k in row},
            "s_max": {k: verdict[k].s_max for k in row},
            "ternary_penalty_bpb": pen_bpb,
            "all_beat_floor": all(c.mean < floor[other.name] for c in row[other.name]),
            "reading": reading,
            "cells": {k: [{"bits": c.bits, "mean": c.mean, "std": c.std, "n": c.n}
                          for c in v] for k, v in row.items()},
        }

    rule("Predictions (A6, recorded before any result)")
    b = blob["budgets"]
    p8 = {n: v["argmin"][other.name] <= 6 for n, v in b.items()}
    p9 = {n: v["ternary_penalty_bpb"][other.name] < v["ternary_penalty_bpb"][ts.name]
          for n, v in b.items()}
    p10 = {n: v["all_beat_floor"] for n, v in b.items()}
    for label, res, text in (("P8", p8, "argmin at b <= 6"),
                             ("P9", p9, "ternary penalty smaller in bits/byte"),
                             ("P10", p10, "every feasible cell beats the floor")):
        detail = ", ".join(f"{n}: {'yes' if ok else 'NO'}" for n, ok in res.items())
        print(f"  {label} ({text}): {'HOLDS' if res and all(res.values()) else 'FAILS'}  [{detail}]")
    blob["predictions"] = {"P8": p8, "P9": p9, "P10": p10}

    check = out_dir / "residual_check.txt"
    rule("Residual check (A6)")
    print(check.read_text() if check.exists() else "  not run yet")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    if b:
        figure(b, tpb, ts.name, other.name, fig_path)
        print(f"\nwrote {fig_path}")
    return blob


def figure(budgets: dict, tpb: dict, ts: str, other: str, path: Path) -> None:
    """One panel per budget. y is each corpus' excess over its OWN optimum, in bits per
    byte: absolute losses differ by corpus, so the comparison that means something is
    where each curve bottoms out and how steeply it rises away from that point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#1f1f1e", "#6b6a64"
    names = list(budgets)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.2), dpi=160,
                             sharey=True, squeeze=False)
    styles = {ts: (TS_COLOR, "o", "TinyStories"), other: (OTHER_COLOR, "s", "FineWeb-Edu")}
    for ax, name in zip(axes[0], names):
        ends = []
        for j, (k, (colour, marker, label)) in enumerate(styles.items()):
            cs = budgets[name]["cells"][k]
            best = min(c["mean"] for c in cs)
            x = [c["bits"] for c in cs]
            y = [bpb(c["mean"] - best, tpb[k]) for c in cs]
            e = [bpb(c["std"], tpb[k]) for c in cs]
            ax.errorbar(x, y, yerr=e, color=colour, marker=marker, markersize=6,
                        linewidth=2, capsize=3, label=label, zorder=3,
                        markeredgecolor="white", markeredgewidth=1.2)
            argmin = budgets[name]["argmin"][k]
            # the argmin names its corpus: two identical "b*=6" labels would not say which
            ax.annotate(f"{label}: b*={argmin:g}", (argmin, 0), textcoords="offset points",
                        xytext=(0, -14 - 12 * j), ha="center", fontsize=8, color=ink)
            ends.append([y[-1], x[-1], label])
        # end-of-line labels, pushed apart when the two curves finish close together
        ends.sort()
        lo, hi = ax.get_ylim()
        min_gap = 0.06 * (hi - lo)
        if len(ends) == 2 and ends[1][0] - ends[0][0] < min_gap:
            mid = (ends[0][0] + ends[1][0]) / 2
            ends[0][0], ends[1][0] = mid - min_gap / 2, mid + min_gap / 2
        for ty, tx, label in ends:
            ax.annotate(label, (tx, ty), textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=8, color=ink)
        ax.axhline(0, color=muted, linewidth=0.8)
        ticks = sorted({c["bits"] for v in budgets[name]["cells"].values() for c in v})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(name, fontsize=10, loc="left", color=ink)
        ax.set_xlabel("bits per weight (b)", color=ink)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.08 * (hi - lo), hi)
    axes[0][0].set_ylabel("val loss above the corpus' own optimum (bits/byte)", color=ink)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper center")
    fig.suptitle("Does the optimum precision hold on harder text?  ±1 seed sd, n=5",
                 fontsize=10.5, x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
