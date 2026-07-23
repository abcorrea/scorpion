#! /usr/bin/env python

"""
Lama-first agile score across four translator generations.

Each run translates AND searches under one competition-style budget
(30 minutes, 8 GiB), like experiment F, but with four translators instead
of two:

  00-python          the Python translator (src/translate). It is unchanged
                     from the C++ port's merge through our branch tip, so it
                     is extracted from the same pin as 02.
  01-cpp-initial     66a77f191, "Add C++ version of translator." (the merge
                     that introduced src/translate-cpp).
  02-cpp-revised     4fd73aea3, "Revise C++ translator." (Jendrik's last
                     commit; the modernize-translator branch point).
  03-cpp-modernized  6472e21e1, the modernize-translator tip.

The headline attribute is the IPC agile score on the end-to-end wall time t
(translate + search, as printed by the wrapper): 0 if no plan was found,
1 if t <= 1 s, else 1 - log(t)/log(1800). Coverage is reported alongside.

All four algorithms share ONE search binary, so any difference comes from
translation. Between 66a77f191 and our tip every src/ change is inside
src/translate-cpp, so the shared binary is byte-for-byte the search code of
every C++ revision; it is taken from the search-bin cache pin used by
experiment F. C++ translators come from the same per-SHA binary cache as
experiments A/E; the Python translator runs from an extracted source tree
behind a small shim so the wrapper interface stays uniform.
DOWNWARD_BENCHMARKS and HTG_BENCHMARKS_FLATTENED must be set.
"""

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import custom_parser
import project

from downward import suites
from lab import tools
from lab.experiment import Experiment
from lab.reports import Attribute

REPO = project.get_repo_base()
WRAPPER = str(project.DIR / "run-lama-rev.sh")
BIN_CACHE = project.DIR / "data" / "translate-bin-cache"
SEARCH_CACHE = project.DIR / "data" / "search-bin-cache"

COLLECTIONS = [
    ("downward", "DOWNWARD_BENCHMARKS"),
    ("htg", "HTG_BENCHMARKS_FLATTENED"),
]

# (nick, git SHA, implementation), oldest first; see the module docstring.
REVISIONS = [
    ("00-python", "4fd73aea3", "py"),
    ("01-cpp-initial", "66a77f191", "cpp"),
    ("02-cpp-revised", "4fd73aea3", "cpp"),
    ("03-cpp-modernized", "6472e21e1", "cpp"),
]
# Search code is identical across all pins (only src/translate-cpp differs),
# so reuse experiment F's cached search binary.
SEARCH_REV = "d99d83aa3"

# Overall per-run budget; also the upper bound of the agile score.
TIME_LIMIT = 1800

if project.REMOTE:
    ENV = project.BaselSlurmEnvironment(
        partition="infai_2",
        memory_per_cpu="6G",
        cpus_per_task=2,
        setup=project.BaselSlurmEnvironment.DEFAULT_SETUP,
    )
else:
    ENV = project.LocalEnvironment(processes=4)

AGILE_SCORE = Attribute("agile_score", min_wins=False, function=sum, digits=2)
ATTRIBUTES = [
    "error",
    "coverage",
    AGILE_SCORE,
    "overall_wall_time",
    "translator_completed",
    "translator_time",
    "translator_wall_time",
    "peak_memory_kb",
    "search_time",
    "total_time",
    "search_peak_memory_kb",
    "plan_length",
    "plan_cost",
    "sas_sha256",
]


def build_translate_binary(sha):
    """Build (once) revision *sha*'s translate-cpp; return its binary path."""
    out_bin = BIN_CACHE / sha / "translate-cpp"
    if out_bin.is_file():
        return str(out_bin)
    print(f"Building translate-cpp for {sha} ...", flush=True)
    src = BIN_CACHE / sha / "src"
    build = BIN_CACHE / sha / "build"
    shutil.rmtree(BIN_CACHE / sha, ignore_errors=True)
    src.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", f"{sha}:src/translate-cpp"],
        check=True, stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, check=True)
    # Force <cstdint> into every translation unit: older revisions use
    # uint32_t without including it, and GCC >= 13 no longer provides the
    # header transitively.
    subprocess.check_call(
        ["cmake", "-S", str(src), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CXX_FLAGS=-include cstdint"],
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["cmake", "--build", str(build), "-j", "8"], stdout=subprocess.DEVNULL
    )
    shutil.copy2(build / "translate-cpp", out_bin)
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(build, ignore_errors=True)
    return str(out_bin)


def build_python_translator(sha):
    """Extract (once) revision *sha*'s src/translate; return a shim that runs
    it with the wrapper's TRANSLATE_BINARY interface (DOMAIN PROBLEM ->
    ./output.sas)."""
    cache = BIN_CACHE / f"py-{sha}"
    shim = cache / "translate-py"
    if shim.is_file():
        return str(shim)
    print(f"Extracting Python translator for {sha} ...", flush=True)
    src = cache / "translate"
    shutil.rmtree(cache, ignore_errors=True)
    src.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", f"{sha}:src/translate"],
        check=True, stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, check=True)
    shim.write_text(
        "#!/bin/bash\n"
        f'exec python3 "{src}/translate.py" "$@"\n'
    )
    shim.chmod(0o755)
    return str(shim)


def collection_domains(root: Path) -> list[str]:
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and any(p.glob("*.pddl"))
    )


def build_search_binary(sha):
    """Build (once) revision *sha*'s search component; return the binary path."""
    out_bin = SEARCH_CACHE / sha / "downward"
    if out_bin.is_file():
        return str(out_bin)
    print(f"Building search binary for {sha} ...", flush=True)
    work = SEARCH_CACHE / sha / "work"
    shutil.rmtree(SEARCH_CACHE / sha, ignore_errors=True)
    work.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", sha],
        check=True, stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(work)], input=archive.stdout, check=True)
    subprocess.check_call(
        [sys.executable, "build.py", "release"], cwd=work,
        stdout=subprocess.DEVNULL,
    )
    shutil.copy2(work / "builds" / "release" / "bin" / "downward", out_bin)
    shutil.rmtree(work, ignore_errors=True)
    return str(out_bin)


TRANSLATORS = {
    nick: (build_python_translator(sha) if impl == "py"
           else build_translate_binary(sha))
    for nick, sha, impl in REVISIONS
}
SEARCH_BINARY = build_search_binary(SEARCH_REV)

exp = Experiment(environment=ENV)

n_tasks = 0
for prefix, env in COLLECTIONS:
    try:
        root = Path(os.environ[env]).resolve()
    except KeyError:
        sys.exit(f"{env} must point to the {prefix} benchmark collection")
    assert root.is_dir(), f"{root} is not a directory"
    for task in suites.build_suite(root, collection_domains(root)):
        domain = f"{prefix}-{task.domain}"
        n_tasks += 1
        for nick, _, _ in REVISIONS:
            run = exp.add_run()
            run.add_command(
                "translate-and-search",
                [WRAPPER, TRANSLATORS[nick], SEARCH_BINARY,
                 task.domain_file, task.problem_file],
                time_limit=TIME_LIMIT,
                memory_limit=8000,
            )
            run.set_property("domain", domain)
            run.set_property("problem", task.problem)
            run.set_property("algorithm", nick)
            run.set_property("id", [nick, domain, task.problem])
print(f"{n_tasks} tasks x {len(REVISIONS)} revisions = {n_tasks * len(REVISIONS)} runs")

parser = custom_parser.get_parser()
parser.add_pattern("sas_sha256", r"SAS_SHA256 ([0-9a-f]+)", type=str)
# Search-side attributes; the search's own log is echoed into run.log.
parser.add_pattern("plan_length", r"Plan length: (\d+) step\(s\)", type=int)
parser.add_pattern("plan_cost", r"Plan cost: (\d+)", type=int)
parser.add_pattern("search_time", r"Search time: (.+?)s", type=float)
parser.add_pattern("total_time", r"Total time: (.+?)s", type=float)
parser.add_pattern("overall_wall_time", r"OVERALL_WALL_TIME (.+)", type=float)


def fix_coverage_and_score(content, props):
    """Coverage = plan found; agile score = IPC log score of the end-to-end
    wall time within the overall budget."""
    props["translator_completed"] = 1 if "translator_task_size" in props else 0
    props["coverage"] = 1 if "\nPLAN_FOUND" in content else 0
    props["agile_score"] = tools.compute_log_score(
        props["coverage"], props.get("overall_wall_time"),
        lower_bound=1.0, upper_bound=TIME_LIMIT,
    )


def search_peak_memory(content, props):
    """Max RSS of the search phase (its own /usr/bin/time section)."""
    import re
    m = re.search(
        r"=== search usrtime ===.*?Maximum resident set size \(kbytes\): (\d+)",
        content, re.S)
    if m:
        props["search_peak_memory_kb"] = int(m.group(1))


parser.add_function(fix_coverage_and_score)
parser.add_function(search_peak_memory)
exp.add_parser(parser)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")

project.add_absolute_report(exp, attributes=ATTRIBUTES)
project.add_comparative_report(
    exp,
    [("00-python", "03-cpp-modernized"),
     ("02-cpp-revised", "03-cpp-modernized")],
    attributes=["coverage", AGILE_SCORE, "overall_wall_time"],
    name=f"{exp.name}-pairs",
)


def agile_summary():
    """Total coverage and agile score per revision, per-domain differences,
    and a cross-revision output.sas byte-equivalence check."""
    props = json.load(open(Path(exp.eval_dir) / "properties"))
    nicks = [nick for nick, _, _ in REVISIONS]
    by_task = defaultdict(dict)
    for run in props.values():
        by_task[(run["domain"], run["problem"])][run["algorithm"]] = run

    totals = {nick: [0, 0.0] for nick in nicks}  # coverage, agile
    per_domain = defaultdict(lambda: {nick: [0, 0.0] for nick in nicks})
    divergent = []
    for (domain, problem), runs in sorted(by_task.items()):
        hashes = set()
        for nick in nicks:
            run = runs.get(nick, {})
            cov = run.get("coverage", 0)
            score = run.get("agile_score", 0.0)
            totals[nick][0] += cov
            totals[nick][1] += score
            per_domain[domain][nick][0] += cov
            per_domain[domain][nick][1] += score
            if run.get("sas_sha256"):
                hashes.add(run["sas_sha256"])
        if len(hashes) > 1:
            divergent.append(f"  {domain}:{problem}")

    lines = [
        "lama-first coverage and agile score, four translator generations",
        "=" * 72,
        f"{'algorithm':<20} {'coverage':>9} {'agile score':>12}",
    ]
    for nick in nicks:
        cov, agile = totals[nick]
        lines.append(f"{nick:<20} {cov:>9} {agile:>12.2f}")

    lines += ["", "Domains where coverage differs between revisions:"]
    for domain, stats in sorted(per_domain.items()):
        coverages = [stats[nick][0] for nick in nicks]
        if len(set(coverages)) > 1:
            cells = "  ".join(f"{c:>4}" for c in coverages)
            lines.append(f"  {domain:<48} {cells}")

    lines += [
        "",
        f"Tasks where the revisions' output.sas differs: {len(divergent)}",
    ]
    lines.extend(divergent)
    report = "\n".join(lines)
    print(report)
    out = Path(exp.eval_dir) / "agile-summary.txt"
    out.write_text(report + "\n")
    print(f"\nWrote {out}")


exp.add_step("agile-summary", agile_summary)

exp.run_steps()
