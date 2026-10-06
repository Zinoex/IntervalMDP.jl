#!/usr/bin/env python3
"""Check that every Lean declaration docstring names its Julia counterpart.

Usage (from anywhere):

    python3 lean/scripts/check_julia_refs.py [--root <lean-root>] [--verbose]

Rules, for every declaration docstring (`/-- ... -/` immediately followed by a declaration:
`def`, `theorem`, `lemma`, `structure`, `class`, `inductive`, `abbrev`, `instance`, `opaque`,
`axiom`, `example`, possibly after attributes `@[...]` and modifiers such as `noncomputable`,
`private`, `protected`) in `IntervalMDPProofs/**/*.lean` and `IntervalMDPProofs.lean`:

1. the docstring contains a line starting with `Julia counterpart:`;
2. that paragraph (from the `Julia counterpart:` line up to the next blank line or the end of the
   docstring) either starts with `Julia counterpart: none (Lean-side proof device).` or cites at
   least one Julia source file as `src/<path>.jl`;
3. every `src/<path>.jl` cited anywhere in the Lean sources (docstrings, module docs, comments)
   exists in the repository (the parent of the Lean root).

Structure fields and inductive constructors (indented docstrings that are not followed by a
declaration keyword) are not checked.

Exit status: 0 if every docstring passes, 1 otherwise (offenders are listed).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DECL_KEYWORDS = (
    "def",
    "theorem",
    "lemma",
    "structure",
    "class",
    "inductive",
    "abbrev",
    "instance",
    "opaque",
    "axiom",
    "example",
)
MODIFIERS = ("noncomputable", "private", "protected", "partial", "unsafe", "nonrec")
NONE_LINE = "Julia counterpart: none (Lean-side proof device)."
SRC_PATH = re.compile(r"\bsrc/[A-Za-z0-9_./-]+\.jl\b")
DECL_NAME = re.compile(r"^\s*(?:" + "|".join(DECL_KEYWORDS) + r")\s+([^\s:({\[]+)")


def strip_prefix(line: str) -> str:
    """Remove leading attributes and modifiers from a declaration line."""
    s = line.strip()
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
                        s = s[i + 1 :].lstrip()
                        changed = True
                        break
            else:
                return s
        for mod in MODIFIERS:
            if s.startswith(mod + " "):
                s = s[len(mod) + 1 :].lstrip()
                changed = True
    return s


def is_declaration(line: str) -> bool:
    s = strip_prefix(line)
    return any(s == kw or s.startswith(kw + " ") for kw in DECL_KEYWORDS)


def docstrings(text: str):
    """Yield (start_line, docstring_text, following_code_line) for each `/-- ... -/`."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        idx = lines[i].find("/--")
        if idx == -1 or lines[i][:idx].strip():
            i += 1
            continue
        start = i
        buf = [lines[i][idx + 3 :]]
        if "-/" in buf[0]:
            buf[0] = buf[0][: buf[0].index("-/")]
            end = i
        else:
            j = i + 1
            while j < n and "-/" not in lines[j]:
                buf.append(lines[j])
                j += 1
            if j < n:
                buf.append(lines[j][: lines[j].index("-/")])
            end = j
        k = end + 1
        while k < n and not lines[k].strip():
            k += 1
        following = lines[k] if k < n else ""
        # A declaration may also start on the docstring's closing line: `-/ def foo`.
        tail = lines[end][lines[end].index("-/") + 2 :] if "-/" in lines[end] else ""
        if tail.strip():
            following = tail
        yield start + 1, "\n".join(buf), following
        i = end + 1


def counterpart_paragraph(doc: str) -> str | None:
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Julia counterpart:"):
            para = [line.strip()]
            for nxt in lines[i + 1 :]:
                if not nxt.strip():
                    break
                para.append(nxt.strip())
            return " ".join(para)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--root", type=Path, default=default_root, help="Lean root (default: lean/)")
    ap.add_argument("--verbose", action="store_true", help="list every checked declaration")
    args = ap.parse_args()

    lean_root: Path = args.root.resolve()
    repo_root = lean_root.parent
    files = sorted((lean_root / "IntervalMDPProofs").rglob("*.lean"))
    top = lean_root / "IntervalMDPProofs.lean"
    if top.exists():
        files.insert(0, top)
    if not files:
        print(f"error: no Lean sources under {lean_root}", file=sys.stderr)
        return 1

    offenders: list[str] = []
    missing_paths: list[str] = []
    checked = 0
    with_src = 0
    with_none = 0

    for f in files:
        rel = f.relative_to(lean_root)
        text = f.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in SRC_PATH.finditer(line):
                if not (repo_root / m.group(0)).is_file():
                    missing_paths.append(f"{rel}:{lineno}: cited path does not exist: {m.group(0)}")
        for lineno, doc, following in docstrings(text):
            if not is_declaration(following):
                continue
            checked += 1
            m = DECL_NAME.match(strip_prefix(following))
            name = m.group(1) if m else following.strip()
            where = f"{rel}:{lineno}: {name}"
            para = counterpart_paragraph(doc)
            if para is None:
                offenders.append(f"{where}: no 'Julia counterpart:' line")
                continue
            if para.startswith(NONE_LINE):
                with_none += 1
                status = "none"
            elif SRC_PATH.search(para):
                with_src += 1
                status = ", ".join(sorted(set(SRC_PATH.findall(para))))
            else:
                offenders.append(
                    f"{where}: 'Julia counterpart:' paragraph cites no src/*.jl file and is not "
                    f"'{NONE_LINE}'"
                )
                continue
            if args.verbose:
                print(f"ok  {where}  [{status}]")

    for o in offenders + missing_paths:
        print(o)
    print(
        f"checked {checked} declaration docstrings in {len(files)} files: "
        f"{with_src} cite src/*.jl, {with_none} are Lean-side proof devices, "
        f"{len(offenders)} offenders, {len(missing_paths)} missing src paths"
    )
    return 1 if offenders or missing_paths else 0


if __name__ == "__main__":
    sys.exit(main())
