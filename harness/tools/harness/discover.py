#!/usr/bin/env python3
"""Target project / toolchain discovery for the Claude Code harness.

Given a TARGET ROOT (independent of the harness repository) this prints the
detected project kinds (julia, lean), the commands each stage must run,
tool availability, and any BLOCKERS. It never installs, downloads or switches
toolchains; it only inspects files and runs `--version` style probes.

Precedence for every command/path value (highest first):
  1. spec      -- `--set key=value` (values from the task spec's
                  "Commands / Toolchain" section, passed by the orchestrator)
  2. config    -- harness.config.toml (`--config PATH`, else
                  $HARNESS_CONFIG, else <target>/harness.config.toml)
  3. discovered-- inferred from Project.toml / Manifest.toml / lean-toolchain /
                  lakefile.{lean,toml}
  4. blocker   -- if a required value cannot be inferred safely

Usage:
  discover.py <target-root> [--config PATH] [--set key=value ...]
              [--require julia,lean,gpu] [--json] [--no-probe]

Exit codes: 0 = no blockers, 2 = blockers present, 1 = usage error.

Output (default): one `key=value` per line, sorted, plus `blocker=...`,
`unavailable=...` and `warning=...` lines. `source.<key>=spec|config|discovered`
records where each value came from.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

try:
    import tomllib  # Python >= 3.11
except ImportError:  # pragma: no cover
    tomllib = None

SKIP_DIRS = {".git", ".lake", "lake-packages", ".julia", "build", "dist"}
DEFAULT_APPROVED_AXIOMS = ["propext", "Classical.choice", "Quot.sound"]
GPU_PACKAGES = ["CUDA", "AMDGPU", "Metal", "oneAPI", "KernelAbstractions"]
TOOLS = ["julia", "lean", "lake", "elan", "nvidia-smi"]


def load_toml(path):
    if tomllib is None:
        raise RuntimeError("python3 >= 3.11 (tomllib) required to parse " + path)
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Julia compat handling (subset of Pkg semantics: caret default, ~, >=, =, -)
# ---------------------------------------------------------------------------
def _parse_ver(s):
    parts = [int(p) for p in re.findall(r"\d+", s)[:3]]
    n = len(parts)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts), n


def _caret_upper(v, n):
    major, minor, patch = v
    if major > 0 or n == 1:
        return (major + 1, 0, 0)
    if minor > 0 or n == 2:
        return (0, minor + 1, 0)
    return (0, 0, patch + 1)


def compat_ranges(spec):
    ranges = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if " - " in item:
            lo, hi = [x.strip() for x in item.split(" - ", 1)]
            lov, _ = _parse_ver(lo)
            hiv, hn = _parse_ver(hi)
            upper = (hiv[0] + 1, 0, 0) if hn == 1 else (hiv[0], hiv[1] + 1, 0) if hn == 2 else (hiv[0], hiv[1], hiv[2] + 1)
            ranges.append((lov, upper))
        elif item.startswith(">="):
            v, _ = _parse_ver(item[2:])
            ranges.append((v, (10**9, 0, 0)))
        elif item.startswith("="):
            v, _ = _parse_ver(item[1:])
            ranges.append((v, (v[0], v[1], v[2] + 1)))
        elif item.startswith("~"):
            v, n = _parse_ver(item[1:])
            upper = (v[0] + 1, 0, 0) if n == 1 else (v[0], v[1] + 1, 0)
            ranges.append((v, upper))
        else:
            v, n = _parse_ver(item.lstrip("^"))
            ranges.append((v, _caret_upper(v, n)))
    return ranges


def version_satisfies(version, spec):
    v, _ = _parse_ver(version)
    rs = compat_ranges(spec)
    if not rs:
        return None
    return any(lo <= v < hi for lo, hi in rs)


def elan_home():
    return os.environ.get("ELAN_HOME") or os.path.join(os.path.expanduser("~"), ".elan")


def toolchain_dir(toolchain):
    """Directory of an *already installed* elan toolchain (never installs).

    elan stores `leanprover/lean4:v4.15.0` as `toolchains/leanprover--lean4---v4.15.0`."""
    if not toolchain:
        return None
    name = toolchain.strip().replace("/", "--").replace(":", "---")
    d = os.path.join(elan_home(), "toolchains", name)
    return d if os.path.isfile(os.path.join(d, "bin", "lake")) else None


# ---------------------------------------------------------------------------
class Discovery:
    def __init__(self, root, config_path=None, sets=None, require=None, probe=True):
        self.root = os.path.abspath(root)
        self.values = {}
        self.source = {}
        self.blockers = []
        self.warnings = []
        self.unavailable = []
        self.require = set(require or [])
        self.probe = probe
        self.sets = dict(sets or {})
        self.config = {}
        self.config_path = None
        cp = config_path or os.environ.get("HARNESS_CONFIG")
        if not cp and os.path.isfile(os.path.join(self.root, "harness.config.toml")):
            cp = os.path.join(self.root, "harness.config.toml")
        if cp:
            if not os.path.isfile(cp):
                self.blockers.append("config file not found: " + cp)
            else:
                self.config_path = os.path.abspath(cp)
                try:
                    self.config = load_toml(cp)
                except Exception as exc:  # noqa: BLE001
                    self.blockers.append("config file unparseable: %s (%s)" % (cp, exc))

    # -- precedence helpers -------------------------------------------------
    def cfg(self, dotted):
        cur = self.config
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def put(self, key, discovered=None):
        """Resolve key by precedence spec > config > discovered."""
        if key in self.sets:
            val, src = self.sets[key], "spec"
        elif self.cfg(key) is not None:
            val, src = self.cfg(key), "config"
        elif discovered is not None:
            val, src = discovered, "discovered"
        else:
            return None
        self.values[key] = val
        self.source[key] = src
        return val

    def path_in_root(self, rel):
        return os.path.normpath(os.path.join(self.root, rel))

    # -- tools ------------------------------------------------------------
    def tool_status(self):
        for t in TOOLS:
            self.values["tools." + t] = "available" if shutil.which(t) else "UNAVAILABLE"

    @staticmethod
    def _is_elan_proxy(path):
        real = os.path.realpath(path)
        return os.path.basename(real) in ("elan", "elan-init") or os.sep + ".elan" + os.sep + "bin" in path

    def run_probe(self, cmd):
        if not self.probe:
            return None
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return (out.stdout + out.stderr).strip()
        except Exception:  # noqa: BLE001
            return None

    # -- julia ------------------------------------------------------------
    def julia(self):
        proj_rel = self.put("julia.project", ".") if self.cfg("julia.project") or "julia.project" in self.sets else None
        proj_dir = self.path_in_root(proj_rel) if proj_rel else self.root
        ptoml = os.path.join(proj_dir, "Project.toml")
        if not os.path.isfile(ptoml) and not os.path.isfile(os.path.join(proj_dir, "JuliaProject.toml")):
            return False
        if not os.path.isfile(ptoml):
            ptoml = os.path.join(proj_dir, "JuliaProject.toml")
        self.values["julia.project"] = proj_dir
        self.source.setdefault("julia.project", "discovered")
        try:
            proj = load_toml(ptoml)
        except Exception as exc:  # noqa: BLE001
            self.blockers.append("Project.toml unparseable: %s" % exc)
            return True
        self.values["julia.package"] = proj.get("name", "")
        manifest = os.path.join(proj_dir, "Manifest.toml")
        self.values["julia.manifest"] = manifest if os.path.isfile(manifest) else "absent"
        compat = (proj.get("compat") or {}).get("julia")
        compat = self.put("julia.compat", compat)
        if compat is None:
            self.warnings.append("Project.toml declares no [compat] julia; version requirement must come from spec/config")
        channel = self.put("julia.channel")
        exe = "julia" + (" +" + str(channel) if channel else "")
        q = shlex.quote(proj_dir)
        self.put("julia.instantiate", "%s --project=%s -e 'using Pkg; Pkg.instantiate()'" % (exe, q))
        self.put("julia.test", "%s --project=%s -e 'using Pkg; Pkg.test()'" % (exe, q))
        # GPU code detection: GPU packages in deps / weakdeps / extensions
        deps = set((proj.get("deps") or {}).keys()) | set((proj.get("weakdeps") or {}).keys())
        backends = [p for p in GPU_PACKAGES if p in deps]
        ext_dir = os.path.join(proj_dir, "ext")
        if os.path.isdir(ext_dir):
            for fn in os.listdir(ext_dir):
                for p in GPU_PACKAGES:
                    if p in fn and p not in backends:
                        backends.append(p)
        self.values["gpu.code_present"] = "yes" if backends else "no"
        self.values["gpu.backends"] = ",".join(backends)
        self.put("gpu.test")  # never inferred: GPU runner must be configured
        # The probe runs in a TEMPORARY environment (develop target + add CUDA from the depot) so it
        # works when CUDA is only a weakdep / test extra (IntervalMDP.jl layout); `using CUDA` inside
        # the target project would error there. A probe ERROR is not evidence of "no GPU": it is a
        # setup problem (FAIL / blocker to investigate), never an UNAVAILABLE pass-through.
        probe_tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_check.py")
        self.put("gpu.probe", ("python3 %s probe %s" % (shlex.quote(probe_tool), q)) if "CUDA" in backends else None)
        if "CUDA" in backends:
            self.values["gpu.classifier"] = "python3 %s run %s --cmd <gpu.test>" % (shlex.quote(probe_tool), q)
        self.put("benchmark.cmd")
        # Version check against compat, never switching versions.
        if shutil.which("julia"):
            out = self.run_probe(["julia"] + (["+" + str(channel)] if channel else []) + ["--version"])
            m = re.search(r"(\d+\.\d+\.\d+)", out or "")
            if m:
                self.values["julia.version_available"] = m.group(1)
                if compat:
                    ok = version_satisfies(m.group(1), str(compat))
                    self.values["julia.version_ok"] = {True: "yes", False: "NO", None: "unknown"}[ok]
                    if ok is False:
                        self.blockers.append(
                            "julia %s does not satisfy compat '%s' - configure julia.channel; never switch silently"
                            % (m.group(1), compat))
        else:
            self.unavailable.append("julia")
        return True

    # -- lean -------------------------------------------------------------
    @staticmethod
    def _is_lean_dir(d):
        return any(os.path.isfile(os.path.join(d, f)) for f in ("lean-toolchain", "lakefile.lean", "lakefile.toml"))

    def lean(self):
        root_rel = self.put("lean.root")
        if root_rel is not None:
            lean_root = self.path_in_root(root_rel)
            if not self._is_lean_dir(lean_root):
                self.blockers.append("configured lean.root has no lean-toolchain/lakefile: " + lean_root)
                return True
        else:
            cands = []
            if self._is_lean_dir(self.root):
                cands = [self.root]
            else:
                for dirpath, dirnames, _ in os.walk(self.root):
                    depth = os.path.relpath(dirpath, self.root).count(os.sep)
                    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                    if dirpath != self.root and self._is_lean_dir(dirpath):
                        cands.append(dirpath)
                        dirnames[:] = []
                    if depth >= 2:
                        dirnames[:] = []
            if not cands:
                return False
            if len(cands) > 1:
                self.blockers.append("ambiguous Lean project roots %s - set lean.root" % cands)
                return True
            lean_root = cands[0]
            self.source["lean.root"] = "discovered"
        self.values["lean.root"] = lean_root
        tc_file = os.path.join(lean_root, "lean-toolchain")
        pinned = None
        if os.path.isfile(tc_file):
            with open(tc_file) as fh:
                pinned = fh.read().strip() or None
        asserted = self.sets.get("lean.toolchain") or self.cfg("lean.toolchain")
        if asserted and pinned and str(asserted).strip() != pinned:
            self.blockers.append("configured lean.toolchain %r differs from pinned lean-toolchain %r: never switch toolchains"
                                 % (asserted, pinned))
        if pinned:
            self.values["lean.toolchain"] = pinned
            self.source["lean.toolchain"] = "discovered"
        if not pinned:
            self.blockers.append("Lean project has no pinned lean-toolchain: refusing to pick a toolchain")
        lakefile = next((f for f in ("lakefile.lean", "lakefile.toml") if os.path.isfile(os.path.join(lean_root, f))), None)
        self.values["lean.lakefile"] = lakefile or "absent"
        build = self.put("lean.build", "lake build" if lakefile else None)
        if not build:
            self.blockers.append("no lakefile.lean/lakefile.toml and no configured lean.build: cannot infer Lean build command")
        self.put("lean.test")  # optional, project specific (e.g. `lake test`)
        axioms = self.put("lean.approved_axioms", list(DEFAULT_APPROVED_AXIOMS))
        self.values["lean.approved_axioms"] = ",".join(axioms) if isinstance(axioms, list) else str(axioms)
        # Installed toolchain check (never install).
        installed = "unknown"
        tc_dir = toolchain_dir(pinned) if pinned else None
        if tc_dir:
            # Pinned toolchain present locally: its own bin/lake never triggers a download
            # (unlike the elan proxy). Record it so stages can call it directly.
            installed = "yes"
            self.values["lean.toolchain_bin"] = os.path.join(tc_dir, "bin")
        elif pinned and shutil.which("lean") and not self._is_elan_proxy(shutil.which("lean")):
            # A standalone (non-elan) lean binary: safe to ask its version.
            out = self.run_probe(["lean", "--version"]) or ""
            short = pinned.split(":")[-1].lstrip("v")
            installed = "yes" if short and short in out else "NO"
        elif pinned:
            # Never run `elan` or an elan proxy (`lean`/`lake` under ~/.elan/bin) here: outside a
            # project with an installed pin the proxy resolves `default_toolchain` (e.g. "stable")
            # and may DOWNLOAD it. Presence of $ELAN_HOME/toolchains/<pin> is the only check.
            installed = "NO"
        self.values["lean.toolchain_installed"] = installed
        if not shutil.which("lake") and not tc_dir:
            self.unavailable.append("lake")
        if installed == "NO":
            self.unavailable.append("lean-toolchain:" + str(pinned))
        return True

    # ---------------------------------------------------------------------
    def run(self):
        if not os.path.isdir(self.root):
            self.blockers.append("target root does not exist: " + self.root)
            return self
        self.values["target_root"] = self.root
        self.values["config_file"] = self.config_path or "none"
        self.tool_status()
        kinds_cfg = self.sets.get("target.kinds") or self.cfg("target.kinds")
        if isinstance(kinds_cfg, str):
            kinds_cfg = [k.strip() for k in kinds_cfg.split(",") if k.strip()]
        kinds = []
        for k, fn in (("julia", self.julia), ("lean", self.lean)):
            if kinds_cfg is not None and k not in kinds_cfg:
                continue
            if fn():
                kinds.append(k)
        if kinds_cfg is not None:
            for k in kinds_cfg:
                if k not in kinds:
                    self.blockers.append("configured kind '%s' not detected at target root" % k)
        self.values["kinds"] = ",".join(kinds) if kinds else "none"
        if not kinds:
            self.blockers.append("no Julia/Lean project detected: cannot infer commands")
        # Required capabilities -> blockers when unavailable / uninferable.
        for req in sorted(self.require):
            if req == "julia" and ("julia" not in kinds or "julia" in self.unavailable):
                self.blockers.append("required julia unavailable or not detected")
            elif req == "lean" and ("lean" not in kinds or any(u == "lake" or u.startswith("lean-toolchain") for u in self.unavailable)):
                self.blockers.append("required Lean verification cannot run: Lean project/toolchain UNAVAILABLE")
            elif req == "gpu" and not self.values.get("gpu.test"):
                self.blockers.append("required GPU check has no configured gpu.test runner: GPU UNAVAILABLE")
        return self

    def as_dict(self):
        return {"values": self.values, "source": self.source, "blockers": self.blockers,
                "unavailable": self.unavailable, "warnings": self.warnings}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--require", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args(argv)
    sets = {}
    for kv in a.set:
        if "=" not in kv:
            ap.error("--set expects KEY=VALUE")
        k, v = kv.split("=", 1)
        sets[k.strip()] = v.strip()
    req = [r.strip() for r in a.require.split(",") if r.strip()]
    d = Discovery(a.root, a.config, sets, req, probe=not a.no_probe).run()
    if a.json:
        print(json.dumps(d.as_dict(), indent=2, sort_keys=True))
    else:
        for k in sorted(d.values):
            print("%s=%s" % (k, d.values[k]))
        for k in sorted(d.source):
            print("source.%s=%s" % (k, d.source[k]))
        for u in d.unavailable:
            print("unavailable=" + u)
        for w in d.warnings:
            print("warning=" + w)
        for b in d.blockers:
            print("blocker=" + b)
    return 2 if d.blockers else 0


if __name__ == "__main__":
    sys.exit(main())
