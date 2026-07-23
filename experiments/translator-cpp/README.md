# C++ translator experiments

Downward Lab experiments evaluating the C++ translator port (`src/translate-cpp`)
against the Python translator (`src/translate`). All scripts share the boilerplate
in this directory instead of each carrying its own copy.

## Shared boilerplate

- `project.py` — Lab environment/report helpers (Tetralith/Basel/local).
- `pyproject.toml`, `uv.lock` — pinned dependencies; run scripts with `uv run`.
- `custom_parser.py` — one parser for all experiments: translator time/memory,
  output-shape attributes, and `sas_sha256` (the byte-equivalence key printed by
  `run-translate-cmp.sh`); unmatched patterns are simply absent per experiment.
- `run-translate.sh` — run one translator binary under `/usr/bin/time` (peak RSS).
- `run-translate-cmp.sh` — run one translator via `fast-downward.py --translator`
  (py or cpp) and print the output.sas sha256 for byte-equivalence checks.
- `run-translate-rev.sh` — run one cached per-revision `translate-cpp` binary
  and print the output.sas sha256.
- `run-lama-rev.sh` — translate with one revision's binary, then run a fixed
  search binary with lama-first; one shared 30 min / 8 GiB budget, per-phase
  `/usr/bin/time`, and the end-to-end wall time for agile scores.

## Experiments

- `2026-07-04-A-translator-revisions.py` — compare kept `translate-cpp` git
  revisions (runtime + peak memory), building each binary once into a SHA-keyed
  `data/translate-bin-cache`.
- `2026-07-04-B-translator-revisions-2.py` — follow-up revisions run; reuses A's
  binary cache.
- `2026-07-04-C-py-vs-cpp.py` — py vs cpp over downward-benchmarks, htg-domains,
  and Autoscale-Sat, with per-task byte-equivalence.
- `2026-07-05-D-ipc2023-learning.py` — py vs cpp over the IPC 2023 learning-track
  test set (10 domains x 90 tasks), with byte-equivalence.
- `2026-07-12-E-base-vs-modernized.py` — translator-only A/B of the
  modernize-translator branch point vs its tip (runtime, peak RSS,
  byte-equivalence) over downward-benchmarks + htg-domains.
- `2026-07-13-F-lama-first-coverage.py` — end-to-end lama-first coverage with
  the same two revisions; translator and search share one 30 min / 8 GiB
  budget.
- `2026-07-23-G-lama-first-agile.py` — lama-first coverage and IPC agile score
  across four translator generations: Python, the initial C++ port, Jendrik's
  revised C++ translator, and the modernize-translator tip.

## Running

```
uv run ./2026-07-04-C-py-vs-cpp.py            # list steps
uv run ./2026-07-04-C-py-vs-cpp.py build start parse fetch equivalence
```

On a Slurm cluster the environment is detected automatically and the steps are
submitted as a dependency chain.
