#! /usr/bin/env python

"""
Compare the C++ translator before and after the modernize-translator branch
on all of downward-benchmarks and htg-domains (flattened).

"base" is the revision the branch started from ("Revise C++ translator");
"modernized" is the branch tip. Each revision's translate-cpp is built
standalone into a per-SHA binary cache (the directory is self-contained), so
the full Fast Downward build is not needed. Both revisions run on
byte-identical input; per task we record the translator's own CPU time, peak
RSS (external /usr/bin/time, uniform across revisions), and the sha256 of
output.sas. The equivalence step then checks, per task, whether the two
revisions produced byte-identical SAS output.

Domain names can collide across the two collections, so every domain is
prefixed with its collection to keep tasks distinct. DOWNWARD_BENCHMARKS and
HTG_BENCHMARKS_FLATTENED must be set in the environment.
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
WRAPPER = str(project.DIR / "run-translate-rev.sh")
BIN_CACHE = project.DIR / "data" / "translate-bin-cache"

# (prefix, env var) for each benchmark collection; both must come from the
# environment.
COLLECTIONS = [
    ("downward", "DOWNWARD_BENCHMARKS"),
    ("htg", "HTG_BENCHMARKS_FLATTENED"),
]

# (nick, git SHA), oldest first. "00-base" is the commit the
# modernize-translator branch starts from; "01-modernized" is its tip.
REVISIONS = [
    ("00-base", "4fd73aea3"),
    ("01-modernized", "5cf72e322"),
]

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
    "translator_time",
    "translator_wall_time",
    "peak_memory_kb",
    "translator_self_memory_kb",
    "translator_task_size",
    "translator_variables",
    "translator_operators",
    "translator_axioms",
    "sas_sha256",
]


def build_binary(sha):
    """Build (once) revision *sha*'s translate-cpp; return its binary path."""
    out_bin = BIN_CACHE / sha / "translate-cpp"
    if out_bin.is_file():
        return str(out_bin)
    print(f"Building translate-cpp for {sha} ...", flush=True)
    src = BIN_CACHE / sha / "src"
    build = BIN_CACHE / sha / "build"
    shutil.rmtree(BIN_CACHE / sha, ignore_errors=True)
    src.mkdir(parents=True)
    # Extract only src/translate-cpp at this revision (self-contained CMake).
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", f"{sha}:src/translate-cpp"],
        check=True, stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, check=True)
    # Force <cstdint> into every translation unit: the base revision uses
    # uint32_t without including it, and GCC >= 13 no longer provides the
    # header transitively. The modernized revision includes it properly; the
    # flag is harmless there and keeps both builds uniform.
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


def collection_domains(root: Path) -> list[str]:
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and any(p.glob("*.pddl"))
    )


BINARIES = {sha: build_binary(sha) for _, sha in REVISIONS}

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
                "translate",
                [WRAPPER, BINARIES[sha], task.domain_file, task.problem_file],
                time_limit=1800,
                memory_limit=8000,
            )
            run.set_property("domain", domain)
            run.set_property("problem", task.problem)
            run.set_property("algorithm", nick)
            run.set_property("id", [nick, domain, task.problem])
print(f"{n_tasks} tasks x {len(REVISIONS)} revisions = {n_tasks * len(REVISIONS)} runs")

parser = custom_parser.get_parser()
# custom_parser has no pattern for the wrapper's sha256 line; add it here.
parser.add_pattern("sas_sha256", r"SAS_SHA256 ([0-9a-f]+)", type=str)
exp.add_parser(parser)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")

# Runtime and peak-memory comparison, base vs modernized (no domain grouping:
# our domain names are collection-prefixed and not in project's rename table).
project.add_absolute_report(exp, attributes=ATTRIBUTES)
project.add_comparative_report(
    exp,
    [("00-base", "01-modernized")],
    attributes=["translator_time", "peak_memory_kb", "translator_task_size"],
    name=f"{exp.name}-base-vs-modernized",
)

# Relative scatter plots for the end-to-end effect.
project.RELATIVE = True
project.add_scatter_plot_reports(
    exp,
    [("00-base", "01-modernized")],
    ["translator_time", "peak_memory_kb"],
)


def check_byte_equivalence():
    """Compare base vs modernized output.sas sha256 per task; write a report."""
    props = json.load(open(Path(exp.eval_dir) / "properties"))
    by_task = defaultdict(dict)
    for run in props.values():
        by_task[(run["domain"], run["problem"])][run["algorithm"]] = run

    identical = differ = only_base = only_new = neither = 0
    diffs = []
    for (domain, problem), runs in sorted(by_task.items()):
        base = runs.get("00-base", {})
        new = runs.get("01-modernized", {})
        bh, nh = base.get("sas_sha256"), new.get("sas_sha256")
        if bh and nh:
            if bh == nh:
                identical += 1
            else:
                differ += 1
                diffs.append(f"{domain}:{problem}")
        elif bh and not nh:
            only_base += 1
        elif nh and not bh:
            only_new += 1
        else:
            neither += 1

    lines = [
        "Byte-equivalence of base vs modernized output.sas",
        "=" * 62,
        f"both translated, byte-identical : {identical}",
        f"both translated, DIFFER         : {differ}",
        f"only base translated            : {only_base}",
        f"only modernized translated      : {only_new}",
        f"neither translated              : {neither}",
    ]
    if diffs:
        lines.append("")
        lines.append("Tasks with differing output:")
        lines.extend(f"  {d}" for d in diffs)
    report = "\n".join(lines)
    print(report)
    out = Path(exp.eval_dir) / "byte-equivalence.txt"
    out.write_text(report + "\n")
    print(f"\nWrote {out}")


exp.add_step("equivalence", check_byte_equivalence)

exp.run_steps()
