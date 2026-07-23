#!/bin/bash
# Wrapper for end-to-end translator-revision comparisons.
#
# Args: TRANSLATE_BINARY SEARCH_BINARY DOMAIN_PDDL PROBLEM_PDDL
#
# Translates the task with one revision's translate-cpp, then runs the fixed
# search binary with the lama-first configuration on the result. Both phases
# share one competition-style budget (OVERALL seconds, 8 GiB virtual), so
# time and memory saved by the translator are available to the search. Each
# phase runs under its own /usr/bin/time -v, so per-phase wall time and peak
# RSS are recorded uniformly for every revision. The sha256 of output.sas is
# printed for the byte-equivalence check; the file itself is deleted after
# the search to keep run directories small.
set -uo pipefail

TRANSLATE="$1"; SEARCH="$2"; DOMAIN="$3"; PROBLEM="$4"

# Leave margin under the experiment's 1800 s hard kill for logging.
OVERALL=1770
start=$(date +%s)
start_hr=$(date +%s.%N)

# lama-first, copied from driver/aliases.py and flattened to one line.
LAMA_FIRST='let(hlm, landmark_sum(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false), let(hff, ff(transform=adapt_costs(one)), lazy_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one,reopen_closed=false)))'

(
    ulimit -v 8388608 2>/dev/null || true
    /usr/bin/time -v -o translate-time.log \
        timeout "$OVERALL" "$TRANSLATE" "$DOMAIN" "$PROBLEM"
) > translate.log 2>&1
tstatus=$?

cat translate.log
echo "=== translate usrtime ==="
cat translate-time.log 2>/dev/null
echo "translate exit status: $tstatus"

if [ -f output.sas ]; then
    echo "SAS_SHA256 $(sha256sum output.sas | cut -d' ' -f1)"
    elapsed=$(( $(date +%s) - start ))
    remaining=$(( OVERALL - elapsed ))
    if [ "$remaining" -gt 0 ]; then
        (
            ulimit -v 8388608 2>/dev/null || true
            /usr/bin/time -v -o search-time.log \
                timeout "$remaining" "$SEARCH" --search "$LAMA_FIRST" \
                < output.sas
        ) > search.log 2>&1
        sstatus=$?
        cat search.log
        echo "=== search usrtime ==="
        cat search-time.log 2>/dev/null
        echo "search exit status: $sstatus"
        if [ -f sas_plan ]; then
            echo "PLAN_FOUND"
        fi
    else
        echo "search skipped: no time left in the overall budget"
    fi
    rm -f output.sas
fi

# End-to-end wall time (translate + search + bookkeeping) for time scores.
echo "OVERALL_WALL_TIME $(awk -v s="$start_hr" -v e="$(date +%s.%N)" 'BEGIN{printf "%.2f", e - s}')"
