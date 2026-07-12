#include "split.h"

#include "../utils/graph.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;
namespace translate::grounding {
namespace {
/* Variables sharing across atoms induce connected components. */
vector<vector<Atom>> get_connected_conditions(const vector<Atom> &conditions) {
    if (conditions.empty())
        return {};
    // Build var -> list-of-condition-indices.
    unordered_map<string, vector<int>> var_to_conds;
    for (size_t i = 0; i < conditions.size(); ++i) {
        for (const auto &arg : conditions[i].args) {
            if (arg.is_symbol()) {
                const string &s = arg.name();
                if (!s.empty() && s.front() == '?')
                    var_to_conds[s].push_back(static_cast<int>(i));
            }
        }
    }
    vector<pair<int, int>> edges;
    for (const auto &[v, idxs] : var_to_conds) {
        for (size_t k = 1; k < idxs.size(); ++k)
            edges.emplace_back(idxs[0], idxs[k]);
    }
    auto comp = utils::connected_components(conditions.size(), edges);
    int n_comp = 0;
    for (int c : comp)
        n_comp = max(n_comp, c + 1);
    vector<vector<Atom>> result(n_comp);
    for (size_t i = 0; i < conditions.size(); ++i)
        result[comp[i]].push_back(conditions[i]);
    // Sort each component by atom for deterministic output.
    for (auto &c : result)
        sort(c.begin(), c.end());
    ranges::sort(result);
    return result;
}

Rule project_rule(
    const Atom &target_effect, const vector<Atom> &conditions, Program &prog) {
    auto cond_vars = get_variables(conditions);
    auto eff_vars = get_variables(target_effect);
    vector<string> retained;
    for (const auto &v : eff_vars)
        if (cond_vars.contains(v))
            retained.push_back(v);
    ranges::sort(retained);
    ArgList args;
    args.reserve(retained.size());
    for (const auto &v : retained)
        args.emplace_back(v);
    Atom effect(prog.new_predicate_name(), move(args));
    return Rule{conditions, effect};
}

/* ----- Greedy binary join ----- */

class OccurrencesTracker {
public:
    unordered_map<string, int> occ;
    void update(const Atom &a, int delta) {
        for (const auto &arg : a.args) {
            if (arg.is_symbol()) {
                const string &s = arg.name();
                if (!s.empty() && s.front() == '?') {
                    occ[s] += delta;
                    if (occ[s] == 0)
                        occ.erase(s);
                }
            }
        }
    }
    unordered_set<string> variables() const {
        unordered_set<string> out;
        for (const auto &[v, _] : occ)
            out.insert(v);
        return out;
    }
};

using Cost = tuple<int, int, int>;

Cost compute_join_cost(const Atom &left, const Atom &right) {
    auto lv = get_variables(left);
    auto rv = get_variables(right);
    if (lv.size() > rv.size())
        swap(lv, rv);
    int common = 0;
    for (const auto &v : lv)
        if (rv.contains(v))
            ++common;
    return {
        static_cast<int>(lv.size()) - common,
        static_cast<int>(rv.size()) - common, -common};
}

/*
  Size/selectivity profile of one joinee for size-aware join ordering:
  estimated tuple count plus estimated distinct values per variable.
*/
struct JoineeStats {
    double est = 0.0;
    unordered_map<string, double> distinct;
};

JoineeStats base_stats(const Atom &atom, const ExtensionStats &stats) {
    JoineeStats out;
    auto sz = stats.size.find(atom.predicate);
    out.est = sz == stats.size.end() ? 0.0 : static_cast<double>(sz->second);
    auto ds = stats.distinct.find(atom.predicate);
    for (size_t i = 0; i < atom.args.size(); ++i) {
        const Arg &a = atom.args[i];
        if (!a.is_symbol())
            continue;
        const string &name = a.name();
        if (name.empty() || name.front() != '?')
            continue;
        double d = out.est;
        if (ds != stats.distinct.end() && i < ds->second.size())
            d = static_cast<double>(ds->second[i]);
        auto [it, inserted] = out.distinct.emplace(name, d);
        // The same variable at several positions: the tighter bound wins.
        if (!inserted)
            it->second = min(it->second, d);
    }
    return out;
}

// System-R estimate of |a >< b|: sizes divided, per shared variable, by the
// larger distinct count. Cross products fall out naturally (no divisor).
double estimate_join(const JoineeStats &a, const JoineeStats &b) {
    double est = a.est * b.est;
    for (const auto &[v, da] : a.distinct) {
        auto it = b.distinct.find(v);
        if (it != b.distinct.end())
            est /= max({da, it->second, 1.0});
    }
    return est;
}

JoineeStats join_stats(
    const JoineeStats &a, const JoineeStats &b, double est) {
    JoineeStats out;
    out.est = est;
    for (const auto &[v, d] : a.distinct)
        out.distinct[v] = min(d, est);
    for (const auto &[v, d] : b.distinct) {
        auto [it, inserted] = out.distinct.emplace(v, min(d, est));
        if (!inserted)
            it->second = min(it->second, min(d, est));
    }
    return out;
}

vector<Rule> greedy_join(
    const Rule &rule, Program &prog, const ExtensionStats *stats) {
    vector<Atom> joinees = rule.conditions;
    OccurrencesTracker occ;
    occ.update(rule.effect, +1);
    for (const auto &c : rule.conditions)
        occ.update(c, +1);

    // Per-joinee size/selectivity profiles (size-aware mode only), kept
    // aligned with `joinees`.
    vector<JoineeStats> jstats;
    if (stats) {
        jstats.reserve(joinees.size());
        for (const auto &c : joinees)
            jstats.push_back(base_stats(c, *stats));
    }
    vector<Rule> result;
    while (joinees.size() >= 2) {
        size_t bi = 0, bj = 0;
        JoineeStats next_stats;
        if (stats) {
            /*
              Bushy, estimate-driven order: join the globally cheapest pair
              by estimated result size, cross products included. This lets
              selective one-tuple relations combine "across" (at_lander in
              Rovers' communicate schemas) and lets type filters pre-reduce
              base relations BEFORE a fanout join -- a left-deep chain
              would re-materialize the full output once per trailing
              filter (logistics' drive-truck: 3 x 2M tuples). Ties break
              on the lower pair indices (deterministic).
            */
            double best_est = 0.0;
            bool have = false;
            for (size_t j = 1; j < joinees.size(); ++j) {
                for (size_t i = 0; i < j; ++i) {
                    double e = estimate_join(jstats[i], jstats[j]);
                    if (!have || e < best_est) {
                        have = true;
                        best_est = e;
                        bi = j;
                        bj = i;
                    }
                }
            }
            next_stats = join_stats(jstats[bj], jstats[bi], best_est);
        } else {
            // Find min-cost pair.
            Cost best{INT_MAX, INT_MAX, INT_MAX};
            for (size_t i = 0; i < joinees.size(); ++i) {
                for (size_t j = 0; j < i; ++j) {
                    Cost c = compute_join_cost(joinees[i], joinees[j]);
                    if (c < best) {
                        best = c;
                        bi = i;
                        bj = j;
                    }
                }
            }
        }
        Atom left = joinees[bi];
        Atom right = joinees[bj];
        // Remove larger index first.
        joinees.erase(joinees.begin() + bi);
        joinees.erase(joinees.begin() + bj);
        if (stats) {
            jstats.erase(jstats.begin() + bi);
            jstats.erase(jstats.begin() + bj);
        }
        occ.update(left, -1);
        occ.update(right, -1);

        auto lv = get_variables(left);
        auto rv = get_variables(right);
        unordered_set<string> common_vars;
        for (const auto &v : lv)
            if (rv.contains(v))
                common_vars.insert(v);
        unordered_set<string> condition_vars = lv;
        for (const auto &v : rv)
            condition_vars.insert(v);
        auto live = occ.variables();
        unordered_set<string> effect_vars;
        for (const auto &v : live)
            if (condition_vars.contains(v))
                effect_vars.insert(v);

        auto maybe_project = [&](const Atom &joinee) -> Atom {
            auto jv = get_variables(joinee);
            unordered_set<string> retained;
            for (const auto &v : jv)
                if (effect_vars.contains(v) || common_vars.contains(v))
                    retained.insert(v);
            if (retained == jv)
                return joinee;
            vector<string> sorted_ret(retained.begin(), retained.end());
            ranges::sort(sorted_ret);
            ArgList args;
            args.reserve(sorted_ret.size());
            for (const auto &v : sorted_ret)
                args.emplace_back(v);
            Atom effect(prog.new_predicate_name(), move(args));
            Rule pr{{joinee}, effect, RuleKind::PROJECT};
            result.push_back(pr);
            return effect;
        };
        Atom new_left = maybe_project(left);
        Atom new_right = maybe_project(right);

        vector<string> sorted_eff(effect_vars.begin(), effect_vars.end());
        ranges::sort(sorted_eff);
        ArgList join_args;
        join_args.reserve(sorted_eff.size());
        for (const auto &v : sorted_eff)
            join_args.emplace_back(v);
        Atom join_effect(prog.new_predicate_name(), move(join_args));
        Rule join_rule{{new_left, new_right}, join_effect, RuleKind::JOIN};
        result.push_back(join_rule);
        joinees.push_back(join_effect);
        occ.update(join_effect, +1);
        if (stats)
            jstats.push_back(move(next_stats));
    }
    // Final result rule: replace last result's effect with the rule's
    // original effect.
    if (!result.empty()) {
        result.back().effect = rule.effect;
    } else {
        // 0 or 1 condition: last result is empty; nothing to do.
    }
    return result;
}

vector<Rule> split_into_binary_rules(
    Rule rule, Program &prog, const ExtensionStats *stats) {
    if (rule.conditions.size() <= 1) {
        rule.kind = RuleKind::PROJECT;
        return {rule};
    }
    return greedy_join(rule, prog, stats);
}

/*
  Append a canonical encoding of `atom` to `key`: variables are numbered by
  first occurrence (negative codes), constants keep their symbol id. Two
  rules get equal keys iff they are identical up to variable renaming.
*/
void append_canonical(
    const Atom &atom, bool include_predicate, unordered_map<int, int> &var_ids,
    vector<int> &key) {
    if (include_predicate)
        key.push_back(atom.predicate);
    key.push_back(static_cast<int>(atom.args.size()));
    for (const auto &a : atom.args) {
        bool is_var = false;
        if (a.is_symbol()) {
            const string &s = a.name();
            is_var = !s.empty() && s.front() == '?';
        }
        if (is_var) {
            auto [it, _] =
                var_ids.emplace(a.v, static_cast<int>(var_ids.size()));
            key.push_back(-1 - it->second);
        } else {
            key.push_back(a.v);
        }
    }
}

/*
  Merge auxiliary (p$) rules whose bodies are identical up to variable
  renaming. Rules that share join subexpressions arise routinely once
  effect rules carry their action's precondition body (deferred action
  grounding): every effect of a schema re-splits the same precondition,
  so whole join subtrees repeat per effect and across schemas. Every p$
  predicate has exactly one producing rule, so dropping a duplicate and
  rewriting its consumers to the surviving predicate preserves the
  model. Iterated to a fixpoint: a rewrite can make more rules identical.
*/
void merge_duplicate_rules(Program &prog) {
    size_t merged = 0;
    for (;;) {
        map<vector<int>, int> canon_to_head;
        unordered_map<int, int> replace;
        vector<char> drop(prog.rules.size(), 0);
        for (size_t i = 0; i < prog.rules.size(); ++i) {
            const Rule &r = prog.rules[i];
            const string &head = symbols().name(r.effect.predicate);
            if (head.rfind("p$", 0) != 0)
                continue;
            vector<int> key;
            unordered_map<int, int> var_ids;
            append_canonical(r.effect, false, var_ids, key);
            key.push_back(static_cast<int>(r.conditions.size()));
            for (const auto &c : r.conditions)
                append_canonical(c, true, var_ids, key);
            auto [it, inserted] =
                canon_to_head.emplace(move(key), r.effect.predicate);
            if (!inserted && it->second != r.effect.predicate) {
                replace[r.effect.predicate] = it->second;
                drop[i] = 1;
            }
        }
        if (replace.empty())
            break;
        merged += replace.size();
        vector<Rule> kept;
        kept.reserve(prog.rules.size());
        for (size_t i = 0; i < prog.rules.size(); ++i) {
            if (drop[i])
                continue;
            Rule &r = prog.rules[i];
            for (auto &c : r.conditions) {
                auto it = replace.find(c.predicate);
                if (it != replace.end())
                    c.predicate = it->second;
            }
            kept.push_back(move(r));
        }
        prog.rules = move(kept);
    }
    if (merged > 0)
        cout << "Merged " << merged << " duplicate rules." << endl;
}

vector<Rule> split_rule(
    const Rule &rule, Program &prog, const ExtensionStats *stats) {
    vector<Atom> important, trivial;
    for (const auto &c : rule.conditions) {
        bool has_var = false;
        for (const auto &a : c.args) {
            if (a.is_symbol()) {
                const string &s = a.name();
                if (!s.empty() && s.front() == '?') {
                    has_var = true;
                    break;
                }
            }
        }
        (has_var ? important : trivial).push_back(c);
    }
    auto components = get_connected_conditions(important);
    if (components.size() == 1 && trivial.empty()) {
        return split_into_binary_rules(rule, prog, stats);
    }
    vector<Rule> projected_rules;
    projected_rules.reserve(components.size());
    for (auto &comp : components)
        projected_rules.push_back(project_rule(rule.effect, comp, prog));
    vector<Rule> result;
    for (auto &pr : projected_rules) {
        auto sub = split_into_binary_rules(pr, prog, stats);
        for (auto &r : sub)
            result.push_back(move(r));
    }
    vector<Atom> combining_conds;
    combining_conds.reserve(projected_rules.size() + trivial.size());
    for (auto &pr : projected_rules)
        combining_conds.push_back(pr.effect);
    for (auto &t : trivial)
        combining_conds.push_back(t);
    Rule combining{combining_conds, rule.effect};
    combining.kind =
        (combining_conds.size() >= 2) ? RuleKind::PRODUCT : RuleKind::PROJECT;
    result.push_back(combining);
    return result;
}
}

ExtensionStats ExtensionStats::of(const vector<Atom> &facts) {
    ExtensionStats out;
    unordered_map<int, vector<unordered_set<int>>> values;
    for (const Atom &a : facts) {
        ++out.size[a.predicate];
        auto &vals = values[a.predicate];
        if (vals.size() < a.args.size())
            vals.resize(a.args.size());
        for (size_t i = 0; i < a.args.size(); ++i)
            vals[i].insert(a.args[i].v);
    }
    for (auto &[pred, vals] : values) {
        auto &d = out.distinct[pred];
        d.reserve(vals.size());
        for (const auto &v : vals)
            d.push_back(v.size());
    }
    return out;
}

void split_rules(Program &prog, const ExtensionStats *stats) {
    vector<Rule> new_rules;
    for (const auto &r : prog.rules) {
        auto sub = split_rule(r, prog, stats);
        for (auto &nr : sub)
            new_rules.push_back(move(nr));
    }
    prog.rules = move(new_rules);
    merge_duplicate_rules(prog);
}
}
