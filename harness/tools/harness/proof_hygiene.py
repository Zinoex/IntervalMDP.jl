#!/usr/bin/env python3
"""Static Lean proof-hygiene pre-check used by the Formal Verification gate.

THIS IS A PRE-CHECK, NOT A PROOF CHECK. Passing it is necessary but never
sufficient: the verifier must still run the pinned `lake build` (and the
`#print axioms` check) and Lean must accept the proofs. If Lean is unavailable
the gate is a BLOCKER, not a pass.

What it checks (comments, string literals -- incl. raw strings -- and char
literals such as `'"'` are stripped first; `{...}` holes of interpolated
strings are kept as code):
  * `sorry`, `admit`, `stop` tactics and explicit `sorryAx`        -> FAIL
  * `axiom` declarations whose FULLY QUALIFIED name is not in the
    approved list (`axiom propext` inside `namespace Fixture` is
    `Fixture.propext`, not core `propext`)                         -> FAIL
  * `native_decide` (adds the Lean.ofReduceBool axiom) unless the
    proof policy approves `Lean.ofReduceBool`                      -> FAIL
  * kernel bypass / trusted-code constructs `set_option
    debug.skipKernelTC`, `@[implemented_by]`, `@[extern]`, `unsafe`,
    `run_cmd`/`run_elab`/`run_meta` unless approved via
    --approve-hazard NAME                                          -> FAIL
  * each expected theorem/lemma (fully qualified name) is declared;
    the name may be on the line after `theorem`                    -> FAIL if missing
  * an expected theorem whose statement type (parsed across lines up
    to the top-level `:=`, after binders) is literally `True`      -> FAIL (placeholder)
Optionally parses captured `#print axioms <name>` output and fails on
`sorryAx` or any axiom outside the approved list.

Usage:
  proof_hygiene.py <lean-root> [--theorem NAME ...] [--approved-axiom NAME ...]
                   [--approve-hazard NAME ...] [--axioms-output FILE]
                   [--emit-axiom-check MODULE]

Exit code 0 = all static obligations PASS, 1 = at least one FAIL, 2 = usage/IO.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

DEFAULT_APPROVED = ["propext", "Classical.choice", "Quot.sound"]
SKIP_DIRS = {".lake", "lake-packages", ".git", "build"}


def _is_idchar(ch):
    """Lean identifier character (letters incl. Greek/letterlike, digits, `_`, `'`,
    `!`, `?`, subscripts). Symbols such as `⟨`, `→`, `(` are not."""
    return ch.isalnum() or ch in "_'!?" or 0x2080 <= ord(ch) <= 0x209C or 0x1D49C <= ord(ch) <= 0x1D59F


def _char_literal_end(src, i):
    """If src[i] == "'" starts a Lean char literal return the index just past
    it, else None. Handles `'a'`, `'"'`, and escapes `'\\''`, `'\\\\'`, `'\\n'`,
    `'\\x41'`, `'\\u{3B1}'`. An apostrophe right after an identifier character
    (`h'`, `x''`) is part of the identifier, not a literal."""
    n = len(src)
    if i > 0 and _is_idchar(src[i - 1]):
        return None
    if i + 1 >= n or src[i + 1] == "\n":
        return None
    if src[i + 1] == "\\":
        j = i + 2
        if j < n and src[j] == "x":
            j += 3
        elif j < n and src[j] == "u" and j + 1 < n and src[j + 1] == "{":
            k = src.find("}", j)
            if k < 0:
                return None
            j = k + 1
        else:
            j += 1
        return j + 1 if j < n and src[j] == "'" else None
    if src[i + 1] != "'" and i + 2 < n and src[i + 2] == "'":
        return i + 3
    return None


def strip_comments_and_strings(src):
    """Remove Lean line comments, (nested) block comments, string literals
    (incl. raw strings `r"..."`/`r#"..."#`) and char literals, preserving
    newlines so line numbers stay meaningful. The `{...}` holes of interpolated
    strings (`s!"..."`, `m!"..."`, `f!"..."`) are kept as code, since they can
    contain terms such as `sorry`."""
    out = []
    i, n, depth = 0, len(src), 0

    def blank(seg):
        out.append("".join("\n" if ch == "\n" else " " for ch in seg))

    while i < n:
        c = src[i]
        two = src[i:i + 2]
        if depth > 0:
            if two == "/-":
                depth += 1
                i += 2
            elif two == "-/":
                depth -= 1
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
            continue
        if two == "/-":
            depth = 1
            i += 2
            continue
        if two == "--":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "r" and (i == 0 or not _is_idchar(src[i - 1])):
            m = re.match(r'r(#*)"', src[i:])
            if m:
                close = '"' + m.group(1)
                k = src.find(close, i + len(m.group(0)))
                k = n if k < 0 else k + len(close)
                blank(src[i:k])
                out.append('""')
                i = k
                continue
        if c == "'":
            k = _char_literal_end(src, i)
            if k is not None:
                blank(src[i:k])
                out.append("'c'")
                i = k
                continue
        if c == '"':
            interp = i > 0 and src[i - 1] == "!"
            i += 1
            out.append('"')
            while i < n and src[i] != '"':
                if src[i] == "\\":
                    out.append("  ")
                    i += 2
                    continue
                if interp and src[i] == "{":
                    # keep the interpolated term as code (brace-balanced)
                    bd = 0
                    while i < n:
                        ch = src[i]
                        bd += (ch == "{") - (ch == "}")
                        out.append(ch)
                        i += 1
                        if bd == 0:
                            break
                    continue
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            i += 1
            out.append('"')
            continue
        out.append(c)
        i += 1
    return "".join(out)


IDENT = r"[A-Za-z_\u00C0-\uFFFF][A-Za-z0-9_'!?.\u00C0-\uFFFF]*"
MODS = r"(?:(?:private|protected|noncomputable|nonrec|partial|unsafe)\s+)*"
DECL_KW_RE = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*" + MODS + r"(theorem|lemma|axiom)\b")
NAME_AFTER_KW = re.compile(r"\s+(" + IDENT + r")")
NEXT_DECL_RE = re.compile(r"^\s*(?:@\[|(?:private|protected|noncomputable|nonrec|partial|unsafe)\s|"
                          r"(?:theorem|lemma|axiom|def|abbrev|instance|example|structure|inductive|class|"
                          r"namespace|section|end|opaque|open|variable)\b|#)")
NS_RE = re.compile(r"^\s*namespace\s+(" + IDENT + r")")
END_RE = re.compile(r"^\s*end(?:\s+(" + IDENT + r"))?\s*$")
SECTION_RE = re.compile(r"^\s*(?:noncomputable\s+)?section\b(?:\s+(" + IDENT + r"))?")
BAD_TACTICS = [("sorry", re.compile(r"(?<![A-Za-z0-9_'.])sorry(?![A-Za-z0-9_'!?])")),
               ("admit", re.compile(r"(?<![A-Za-z0-9_'.])admit(?![A-Za-z0-9_'!?])")),
               ("stop", re.compile(r"(?<![A-Za-z0-9_'.])stop(?![A-Za-z0-9_'!?])")),
               ("sorryAx", re.compile(r"(?<![A-Za-z0-9_'])sorryAx(?![A-Za-z0-9_'])"))]
NATIVE_DECIDE = re.compile(r"(?<![A-Za-z0-9_'.])native_decide(?![A-Za-z0-9_'])")
# Kernel-bypass / trust-extending constructs: FAIL unless the proof policy approves them
# (pass the hazard name via --approve-hazard).
KERNEL_HAZARDS = [
    ("debug.skipKernelTC", re.compile(r"\bdebug\.skipKernelTC\b")),
    ("implemented_by", re.compile(r"(?<![A-Za-z0-9_'.])implemented_by(?![A-Za-z0-9_'])")),
    ("extern", re.compile(r"(?<![A-Za-z0-9_'.])extern(?![A-Za-z0-9_'])")),
    ("unsafe", re.compile(r"(?<![A-Za-z0-9_'.])unsafe(?![A-Za-z0-9_'])")),
    ("run_cmd/run_elab/run_meta", re.compile(r"(?<![A-Za-z0-9_'.])(?:run_cmd|run_elab|run_meta)(?![A-Za-z0-9_'])")),
]
_OPEN, _CLOSE = "([{⦃⟨", ")]}⦄⟩"


def statement_of(text, pos, line_starts, lines):
    """Text of a declaration's signature starting at `pos` (just after the
    name) up to the first top-level `:=` / `|` equation / `where`, or the next
    declaration line. Spans multiple lines."""
    depth, i, n = 0, pos, len(text)
    while i < n:
        ch = text[i]
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        elif depth == 0:
            if text.startswith(":=", i):
                return text[pos:i]
            if ch == "\n":
                nxt = text[i + 1:text.find("\n", i + 1) if text.find("\n", i + 1) >= 0 else n]
                if NEXT_DECL_RE.match(nxt) or re.match(r"^\s*\|", nxt):
                    return text[pos:i]
            if re.match(r"\bwhere\b", text[i:i + 6]) and (i == 0 or not _is_idchar(text[i - 1])):
                return text[pos:i]
        i += 1
    return text[pos:]


def type_of_statement(stmt):
    """The type after the first top-level `:` (i.e. after the binders)."""
    depth = 0
    for i, ch in enumerate(stmt):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0 and not stmt.startswith(":=", i):
            return stmt[i + 1:].strip()
    return None


def is_placeholder_type(ty):
    """Literally `True` (possibly parenthesised) -- a placeholder statement."""
    if ty is None:
        return False
    t = re.sub(r"\s+", "", ty)
    while t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    return t in ("True", "_root_.True")


def lean_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            if fn.endswith(".lean") and not fn.startswith("lakefile"):
                yield os.path.join(dirpath, fn)


def scan(root, theorems, approved, approved_hazards=()):
    results = []  # (status, obligation, detail)
    declared = {}  # full name -> (file, line, statement)
    hazards = []
    kernel = []
    proj_axioms = []
    files = list(lean_files(root))
    if not files:
        results.append(("FAIL", "lean-sources", "no .lean source files found under " + root))
    for path in files:
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            text = strip_comments_and_strings(fh.read())
        lines = text.split("\n")
        line_starts, off = [], 0
        for ln in lines:
            line_starts.append(off)
            off += len(ln) + 1
        scopes = []  # stack of ("ns"|"sec", name)
        for idx, line in enumerate(lines):
            lineno = idx + 1
            for tag, rx in BAD_TACTICS:
                if rx.search(line):
                    hazards.append("%s:%d uses `%s`" % (rel, lineno, tag))
            if NATIVE_DECIDE.search(line) and "Lean.ofReduceBool" not in approved:
                hazards.append("%s:%d uses `native_decide` (axiom Lean.ofReduceBool not approved)" % (rel, lineno))
            for tag, rx in KERNEL_HAZARDS:
                if rx.search(line) and tag not in approved_hazards:
                    kernel.append("%s:%d uses `%s` (kernel bypass / trusted code; needs proof-policy approval)"
                                  % (rel, lineno, tag))
            m = NS_RE.match(line)
            if m:
                scopes.append(("ns", m.group(1)))
                continue
            m = SECTION_RE.match(line)
            if m:
                scopes.append(("sec", m.group(1) or ""))
                continue
            m = END_RE.match(line)
            if m and scopes:
                scopes.pop()
                continue
            m = DECL_KW_RE.match(line)
            if not m:
                continue
            kind = m.group(1)
            nm = NAME_AFTER_KW.match(text, line_starts[idx] + m.end())
            if not nm:
                hazards.append("%s:%d `%s` without a parsable name" % (rel, lineno, kind))
                continue
            name = nm.group(1)
            ns = ".".join(s[1] for s in scopes if s[0] == "ns")
            full = name[len("_root_."):] if name.startswith("_root_.") else (ns + "." + name if ns else name)
            if kind == "axiom":
                proj_axioms.append((full, rel, lineno))
            else:
                stmt = statement_of(text, nm.end(), line_starts, lines)
                declared[full] = (rel, lineno, stmt)
    results.append(("FAIL" if hazards else "PASS", "no-sorry-admit-placeholder",
                    "; ".join(hazards) if hazards else "no sorry/admit/stop/sorryAx/native_decide in %d file(s)" % len(files)))
    results.append(("FAIL" if kernel else "PASS", "no-kernel-bypass",
                    "; ".join(kernel) if kernel else
                    "no debug.skipKernelTC/implemented_by/extern/unsafe/run_cmd"
                    + (" (approved: %s)" % ", ".join(approved_hazards) if approved_hazards else "")))
    # Fully qualified names only: `axiom propext` inside `namespace Fixture` declares
    # `Fixture.propext`, which is NOT the approved core axiom `propext`.
    bad_ax = [a for a in proj_axioms if a[0] not in approved]
    results.append(("FAIL" if bad_ax else "PASS", "no-unapproved-axioms",
                    "; ".join("%s:%d declares axiom %s (not approved)" % (f, l, a) for a, f, l in bad_ax)
                    if bad_ax else "approved axioms: %s" % ", ".join(approved)))
    for thm in theorems:
        hit = declared.get(thm)
        if not hit:
            results.append(("FAIL", "theorem-present:" + thm,
                            "no theorem/lemma with fully qualified name %s declared" % thm))
            continue
        rel, lineno, stmt = hit
        if is_placeholder_type(type_of_statement(stmt)):
            results.append(("FAIL", "theorem-present:" + thm, "%s:%d statement is literally `True` (placeholder)" % (rel, lineno)))
        else:
            results.append(("PASS", "theorem-present:" + thm, "declared at %s:%d" % (rel, lineno)))
    return results


AX_LINE = re.compile(r"'([^']+)' depends on axioms: \[([^\]]*)\]")
AX_NONE = re.compile(r"'([^']+)' does not depend on any axioms")


def check_axioms_output(text, theorems, approved):
    """Parse `#print axioms` output (Lean 4 format)."""
    seen = {}
    for m in AX_LINE.finditer(text):
        seen[m.group(1)] = [a.strip() for a in m.group(2).split(",") if a.strip()]
    for m in AX_NONE.finditer(text):
        seen[m.group(1)] = []
    res = []
    for thm in theorems:
        if thm not in seen:
            res.append(("FAIL", "print-axioms:" + thm, "no `#print axioms` output for %s" % thm))
            continue
        axs = seen[thm]
        bad = [a for a in axs if a not in approved]
        if bad:
            res.append(("FAIL", "print-axioms:" + thm, "depends on unapproved axioms %s" % bad))
        else:
            res.append(("PASS", "print-axioms:" + thm, "axioms %s all approved" % (axs or "none")))
    return res


def lean_pipeline(lean_root, modules, theorems, approved, toolchain_bin, approved_hazards=(), timeout=1800):
    """Full gate evidence on an (already prepared) Lean root:
    static scan + `lake build` + `#print axioms` for every expected theorem.

    `toolchain_bin` must be the `bin/` directory of an ALREADY INSTALLED
    toolchain (e.g. discover.py's `lean.toolchain_bin`). Its `lake`/`lean` are
    invoked directly -- never the elan proxy -- so no toolchain is ever
    downloaded or switched. Returns a list of (status, obligation, detail)."""
    results = list(scan(lean_root, theorems, approved, approved_hazards))
    lake = os.path.join(toolchain_bin, "lake")
    if not os.path.isfile(lake):
        results.append(("FAIL", "lake-build", "BLOCKER: no lake in %s (toolchain not installed)" % toolchain_bin))
        return results
    env = dict(os.environ)
    env["PATH"] = toolchain_bin + os.pathsep + env.get("PATH", "")
    for k in ("ELAN_TOOLCHAIN", "LEAN_PATH", "LEAN_SYSROOT"):
        env.pop(k, None)
    p = subprocess.run([lake, "build"], cwd=lean_root, capture_output=True, text=True, env=env, timeout=timeout)
    out = (p.stdout + p.stderr).strip()
    results.append(("PASS" if p.returncode == 0 else "FAIL", "lake-build",
                    "exit %d: %s" % (p.returncode, out[-600:].replace("\n", " | "))))
    if p.returncode != 0:
        return results
    with tempfile.TemporaryDirectory(prefix="axcheck-") as tmp:
        f = os.path.join(tmp, "AxiomCheck.lean")
        with open(f, "w") as fh:
            fh.write("".join("import %s\n" % m for m in modules))
            fh.write("".join("#print axioms %s\n" % t for t in theorems))
        q = subprocess.run([lake, "env", "lean", f], cwd=lean_root, capture_output=True, text=True, env=env,
                           timeout=timeout)
    ax_out = q.stdout + q.stderr
    if q.returncode != 0:
        results.append(("FAIL", "print-axioms", "exit %d: %s" % (q.returncode, ax_out.strip()[-600:])))
    results += check_axioms_output(ax_out, theorems, approved)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--theorem", action="append", default=[])
    ap.add_argument("--approved-axiom", action="append", default=None)
    ap.add_argument("--approve-hazard", action="append", default=[],
                    help="kernel-bypass construct approved by the proof policy (e.g. implemented_by)")
    ap.add_argument("--axioms-output")
    ap.add_argument("--run-lake", metavar="TOOLCHAIN_BIN",
                    help="also run `lake build` + `#print axioms` with this INSTALLED toolchain bin dir "
                         "(never the elan proxy; never downloads). Requires --module.")
    ap.add_argument("--module", action="append", default=[], help="module(s) to import for #print axioms")
    ap.add_argument("--emit-axiom-check", metavar="MODULE", action="append")
    a = ap.parse_args(argv)
    approved = a.approved_axiom if a.approved_axiom is not None else list(DEFAULT_APPROVED)
    if a.emit_axiom_check:
        # Feed this to: (cd <lean-root> && lake env lean --stdin)
        for mod in a.emit_axiom_check:
            print("import " + mod)
        for t in a.theorem:
            print("#print axioms " + t)
        return 0
    if not os.path.isdir(a.root):
        print("ERROR: lean root not found: " + a.root, file=sys.stderr)
        return 2
    if a.run_lake:
        if not a.module:
            ap.error("--run-lake requires --module")
        results = lean_pipeline(a.root, a.module, a.theorem, approved, a.run_lake, a.approve_hazard)
    else:
        results = scan(a.root, a.theorem, approved, a.approve_hazard)
    if a.axioms_output:
        with open(a.axioms_output, encoding="utf-8") as fh:
            results += check_axioms_output(fh.read(), a.theorem, approved)
    for status, ob, detail in results:
        print("%s %s: %s" % (status, ob, detail))
    failed = any(r[0] == "FAIL" for r in results)
    if a.run_lake:
        print("RESULT: %s (static scan + lake build + #print axioms with %s)" % ("FAIL" if failed else "PASS", a.run_lake))
    else:
        print("RESULT: %s (static pre-check only; Lean must still accept `lake build` with the pinned toolchain)"
              % ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
