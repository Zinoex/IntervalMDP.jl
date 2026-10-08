#!/usr/bin/env python3
"""Format, or check the formatting of, the Julia files a change touched.

Uses the same JuliaFormatter version as the repository's format-check CI
(read from `.github/workflows/FormatCheck.yml`, default 2.1.6) and the
target's `.JuliaFormatter.toml`, so a file that passes here passes CI.

Only the files the change touched are considered: `*.jl` files added,
copied, modified or renamed relative to `--base`, plus untracked `*.jl`
files. Files under `harness/` (intentionally unformatted fixtures) are
skipped. Pre-existing unformatted files elsewhere are not touched, so
specs that require `src/` to stay byte-identical are not violated.

JuliaFormatter is installed (once) into the shared environment
`@juliaformatter`, never into the target project.

Usage:
  julia_format.py check <root> [--base REF] [--files F ...]
  julia_format.py fix   <root> [--base REF] [--files F ...]

Exit codes:
  check: 0 = all touched files formatted (or none touched),
         1 = unformatted files (listed), 2 = error (formatter unavailable, git error).
  fix:   0 = files formatted in place (rewritten ones listed), 2 = error.
"""
import argparse
import os
import re
import subprocess
import sys

DEFAULT_VERSION = "2.1.6"
SKIP_PREFIXES = ("harness/",)

JULIA_SCRIPT = r"""
import Pkg
v = VersionNumber(ARGS[1])
have = any(d -> d.name == "JuliaFormatter" && d.version == v, values(Pkg.dependencies()))
if !have
    try
        Pkg.add(Pkg.PackageSpec(name = "JuliaFormatter", version = v); io = devnull)
    catch err
        println("JULIA_FORMAT_ERROR: cannot install JuliaFormatter $v: ", sprint(showerror, err))
        exit(2)
    end
end
using JuliaFormatter
mode = ARGS[2]
for f in ARGS[3:end]
    # format(f; overwrite = false) returns true iff f is already formatted.
    ok = mode == "check" ? format(f; overwrite = false) : format(f)
    ok || println("JULIA_FORMAT_CHANGED: ", f)
end
"""


def formatter_version(root):
    path = os.path.join(root, ".github", "workflows", "FormatCheck.yml")
    try:
        with open(path, encoding="utf-8") as fh:
            m = re.search(r'name\s*=\s*"JuliaFormatter"\s*,\s*version\s*=\s*"([^"]+)"', fh.read())
            if m:
                return m.group(1)
    except OSError:
        pass
    return DEFAULT_VERSION


def git_lines(root, *args):
    out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return [line for line in out.stdout.splitlines() if line]


def touched_files(root, base):
    files = git_lines(root, "diff", "--name-only", "--diff-filter=ACMR", base, "--", "*.jl")
    files += git_lines(root, "ls-files", "--others", "--exclude-standard", "--", "*.jl")
    return sorted({f for f in files if not f.startswith(SKIP_PREFIXES)})


def run(mode, root, files, version):
    if not files:
        return 0, []
    cmd = ["julia", "--startup-file=no", "--project=@juliaformatter", "-e", JULIA_SCRIPT,
           version, mode, *[os.path.join(root, f) for f in files]]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    changed = [os.path.relpath(line.split(": ", 1)[1], root)
               for line in out.stdout.splitlines() if line.startswith("JULIA_FORMAT_CHANGED: ")]
    if out.returncode != 0:
        msg = (out.stdout + out.stderr).strip().splitlines()
        print("\n".join(msg[-15:]), file=sys.stderr)
        return 2, changed
    return 0, changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["check", "fix"])
    ap.add_argument("root")
    ap.add_argument("--base", default="origin/main", help="base ref of the change (default origin/main)")
    ap.add_argument("--files", nargs="+", help="explicit files (relative to root) instead of the git diff")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    version = formatter_version(root)
    try:
        files = sorted(a.files) if a.files else touched_files(root, a.base)
    except RuntimeError as err:
        print(f"RESULT: ERROR ({err})")
        return 2
    print(f"JuliaFormatter {version}; {len(files)} touched Julia file(s) vs {a.base}")
    for f in files:
        print(f"  {f}")
    code, changed = run(a.mode, root, files, version)
    if code == 2:
        print("RESULT: ERROR (formatter did not run; see stderr) — this is not a pass")
        return 2
    if a.mode == "check":
        if changed:
            print(f"RESULT: FAIL ({len(changed)} unformatted file(s))")
            for f in changed:
                print(f"  UNFORMATTED {f}")
            return 1
        print("RESULT: PASS (all touched Julia files formatted)")
        return 0
    for f in changed:
        print(f"  REFORMATTED {f}")
    print(f"RESULT: FORMATTED ({len(changed)} file(s) rewritten)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
