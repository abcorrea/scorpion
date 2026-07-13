#! /usr/bin/env python

"""
End-to-end impact of the modernized translator on planner coverage.

Same two translator revisions as experiment E ("base" is the revision the
modernize-translator branch started from, "modernized" is its tip), but each
run now translates AND searches: the translated task is handed to a fixed
search binary running lama-first, and translator plus search share one
competition-style budget (30 minutes, 8 GiB). Coverage here means "plan
found", so translator time and memory savings show up as end-to-end coverage
and not just as translator statistics.

The search binary is built once, from the modernized revision, and shared by
both algorithms: the branch only touches src/translate-cpp, so the search
code is identical across revisions and any coverage difference comes from
translation. Translator binaries come from the same per-SHA cache scheme as
experiment E. DOWNWARD_BENCHMARKS and HTG_BENCHMARKS_FLATTENED must be set.
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
from lab.experiment import Experiment

REPO = project.get_repo_base()
WRAPPER = str(project.DIR / "run-lama-rev.sh")
BIN_CACHE = project.DIR / "data" / "translate-bin-cache"
SEARCH_CACHE = project.DIR / "data" / "search-bin-cache"

COLLECTIONS = [
    ("downward", "DOWNWARD_BENCHMARKS"),
    ("htg", "HTG_BENCHMARKS_FLATTENED"),
]

# (nick, git SHA), oldest first; same pins as experiment E.
REVISIONS = [
    ("00-base", "4fd73aea3"),
    ("01-modernized", "d99d83aa3"),
]
# The search binary is shared by both algorithms (search code is identical
# across the two revisions); build it from the newest pin.
SEARCH_REV = REVISIONS[-1][1]

if project.REMOTE:
    ENV = project.BaselSlurmEnvironment(
        partition="infai_2",
        memory_per_cpu="6G",
        cpus_per_task=2,
        setup=project.BaselSlurmEnvironment.DEFAULT_SETUP,
    )
else:
    ENV = project.LocalEnvironment(processes=4)

ATTRIBUTES = [
    "error",
    "coverage",
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
    # Force <cstdint> into every translation unit: the base revision uses
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


def collection_domains(root: Path) -> list[str]:
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and any(p.glob("*.pddl"))
    )


TRANSLATE_BINARIES = {sha: build_translate_binary(sha) for _, sha in REVISIONS}
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
        for nick, sha in REVISIONS:
            run = exp.add_run()
            run.add_command(
                "translate-and-search",
                [WRAPPER, TRANSLATE_BINARIES[sha], SEARCH_BINARY,
                 task.domain_file, task.problem_file],
                time_limit=1800,
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


def fix_coverage(content, props):
    """Coverage = plan found; the translator-only verdict moves aside."""
    props["translator_completed"] = 1 if "translator_task_size" in props else 0
    props["coverage"] = 1 if "\nPLAN_FOUND" in content else 0


def search_peak_memory(content, props):
    """Max RSS of the search phase (its own /usr/bin/time section)."""
    import re
    m = re.search(
        r"=== search usrtime ===.*?Maximum resident set size \(kbytes\): (\d+)",
        content, re.S)
    if m:
        props["search_peak_memory_kb"] = int(m.group(1))


parser.add_function(fix_coverage)
parser.add_function(search_peak_memory)
exp.add_parser(parser)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")

project.add_absolute_report(exp, attributes=ATTRIBUTES)
project.add_comparative_report(
    exp,
    [("00-base", "01-modernized")],
    attributes=["coverage", "total_time", "plan_cost"],
    name=f"{exp.name}-base-vs-modernized",
)


def coverage_summary():
    """Per-task coverage diff with the reason for every flipped task."""
    props = json.load(open(Path(exp.eval_dir) / "properties"))
    by_task = defaultdict(dict)
    for run in props.values():
        by_task[(run["domain"], run["problem"])][run["algorithm"]] = run

    both = neither = only_base = only_new = 0
    flipped = []
    per_domain = defaultdict(lambda: [0, 0])
    for (domain, problem), runs in sorted(by_task.items()):
        b, n = runs.get("00-base", {}), runs.get("01-modernized", {})
        cb, cn = b.get("coverage", 0), n.get("coverage", 0)
        per_domain[domain][0] += cb
        per_domain[domain][1] += cn
        if cb and cn:
            both += 1
        elif not cb and not cn:
            neither += 1
        else:
            if cb:
                only_base += 1
                loser = n
                tag = "only base solves"
            else:
                only_new += 1
                loser = b
                tag = "only modernized solves"
            reason = ("translator failed" if not loser.get("translator_completed")
                      else "search failed")
            flipped.append(f"  {tag}: {domain}:{problem} ({reason})")

    lines = [
        "lama-first coverage, base vs modernized translator",
        "=" * 62,
        f"solved by both        : {both}",
        f"solved by neither     : {neither}",
        f"only base solves      : {only_base}",
        f"only modernized solves: {only_new}",
        "",
        "Domains with a coverage difference:",
    ]
    for domain, (cb, cn) in sorted(per_domain.items()):
        if cb != cn:
            lines.append(f"  {domain:<48} {cb:>4} -> {cn:>4}")
    if flipped:
        lines.append("")
        lines.append("Flipped tasks:")
        lines.extend(flipped)
    report = "\n".join(lines)
    print(report)
    out = Path(exp.eval_dir) / "coverage-summary.txt"
    out.write_text(report + "\n")
    print(f"\nWrote {out}")


exp.add_step("coverage-summary", coverage_summary)

exp.run_steps()
