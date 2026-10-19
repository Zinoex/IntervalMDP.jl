#!/usr/bin/env python3
"""Check that `lean/AxiomCheck.lean` lists every theorem of the Lean library.

Usage (from anywhere):

    python3 lean/scripts/check_axiom_check.py [--root <lean-root>] [--list]

Collects every `theorem`/`lemma` declared in `IntervalMDPProofs/**/*.lean` (after attributes
`@[...]` and modifiers such as `protected`), with its fully qualified name (the enclosing
`namespace` blocks, tracked through `namespace`/`section`/`end`; a leading `_root_.` resets the
prefix), and compares the set against the `#print axioms <name>` entries of `AxiomCheck.lean`.

Prints the number of declared theorems, of `#print axioms` entries, and of missing theorems
(listed with `--list`), plus entries that name no declared theorem (informational: they may be
`def`s or Mathlib names). Exit status 0 iff no theorem is missing. `private` theorems cannot be
named in `#print axioms` and are reported separately; they do not fail the check.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MODIFIERS = ("noncomputable", "private", "protected", "nonrec")
DECL = re.compile(r"^(theorem|lemma)\s+([^\s:({\[]+)")
PRINT = re.compile(r"^#print\s+axioms\s+(\S+)")


def strip_prefix(line: str) -> tuple[str, bool]:
    s = line.strip()
    private = False
    changed = True
    while changed:
        changed = False
        if s.startswith("@["):
            depth = 0
            for i, ch in enumerate(s):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        s = s[i + 1 :].strip()
                        changed = True
                        break
        for mod in MODIFIERS:
            if s.startswith(mod + " "):
                private = private or mod == "private"
                s = s[len(mod) :].strip()
                changed = True
    return s, private


def strip_comments(text: str) -> str:
    # block comments / docstrings (nesting-aware), then line comments
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("/-", i):
            depth += 1
            i += 2
        elif depth and text.startswith("-/", i):
            depth -= 1
            i += 2
        else:
            if not depth:
                out.append(text[i])
            elif text[i] == "\n":
                out.append("\n")
            i += 1
    return re.sub(r"--.*", "", "".join(out))


def declared(root: Path) -> tuple[set[str], set[str]]:
    names, privates = set(), set()
    for path in sorted((root / "IntervalMDPProofs").rglob("*.lean")):
        stack: list[tuple[str, list[str]]] = []  # (kind, components)
        for raw in strip_comments(path.read_text(encoding="utf-8")).splitlines():
            line = raw.strip()
            m = re.match(r"^namespace\s+(\S+)", line)
            if m:
                stack.append(("namespace", m.group(1).split(".")))
                continue
            m = re.match(r"^(noncomputable\s+)?section(\s+(\S+))?\s*$", line)
            if m:
                stack.append(("section", []))
                continue
            m = re.match(r"^end(\s+(\S+))?\s*$", line)
            if m and stack:
                stack.pop()
                continue
            s, private = strip_prefix(line)
            m = DECL.match(s)
            if not m:
                continue
            name = m.group(2)
            if name.startswith("_root_."):
                full = name[len("_root_.") :]
            else:
                prefix = [c for _, comps in stack for c in comps]
                full = ".".join(prefix + [name])
            (privates if private else names).add(full)
    return names, privates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    names, privates = declared(root)
    entries = []
    for line in (root / "AxiomCheck.lean").read_text(encoding="utf-8").splitlines():
        m = PRINT.match(line.strip())
        if m:
            entries.append(m.group(1))
    listed = set(entries)
    missing = sorted(names - listed)
    extra = sorted(listed - names)
    dups = sorted({e for e in entries if entries.count(e) > 1})
    print(f"declared theorems: {len(names)}")
    print(f"#print axioms entries: {len(entries)} ({len(listed)} distinct)")
    print(f"missing: {len(missing)}")
    if args.list:
        for name in missing:
            print(f"  MISSING {name}")
    if extra:
        print(f"entries naming no declared theorem: {len(extra)}")
        for name in extra:
            print(f"  EXTRA {name}")
    if dups:
        print(f"duplicate entries: {len(dups)}: {', '.join(dups)}")
    if privates:
        print(f"private theorems (cannot be listed): {len(privates)}")
    print("RESULT: " + ("PASS" if not missing else "FAIL"))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
