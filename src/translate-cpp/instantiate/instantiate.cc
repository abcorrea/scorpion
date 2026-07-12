#include "instantiate.h"

#include "../translate_options.h"

#include "../pddl/action.h"
#include "../pddl/axiom.h"
#include "../pddl/condition.h"
#include "../pddl/effect.h"
#include "../pddl/f_expression.h"
#include "../pddl/task.h"
#include "../utils/hash.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

using namespace std;
namespace translate::instantiate {
using namespace pddl;

namespace {
// Interned ids of the predicates that can appear as (action/axiom) effects.
// build_fluent_facts uses these to select the fluent atoms from the model with
// an int lookup instead of hashing each atom's predicate name (equal names
// always intern to the same id).
unordered_set<int> get_fluent_predicates(const Task &task) {
    unordered_set<int> out;
    for (const auto &a : task.actions) {
        for (const auto &eff : a.effects) {
            if (eff.literal) {
                const auto &lit = static_cast<const Literal &>(*eff.literal);
                out.insert(grounding::symbols().intern(lit.predicate));
            }
        }
    }
    for (const auto &x : task.axioms)
        out.insert(grounding::symbols().intern(x.name));
    return out;
}

// Three shared shapes of the reachable fluent facts (one Atom per fact): the
// AtomSet in the Result (fact_groups), the FactMap GroundKey->FactId (the
// instantiation probe + building the FactId->(var,val) table), and fact_by_id
// FactId->Atom (rebuilding the few axiom/goal literals as atoms). Each distinct
// fluent fact gets a dense FactId in model order.
struct FluentFacts {
    AtomSet set; // for fact_groups
    FactMap ids; // GroundKey -> FactId
    vector<shared_ptr<const Atom>> fact_by_id; // FactId -> Atom
};

FluentFacts build_fluent_facts(
    const vector<grounding::Atom> &model,
    const unordered_set<int> &fluent_preds) {
    FluentFacts out;
    for (const auto &a : model) {
        if (!fluent_preds.contains(a.predicate))
            continue;
        GroundKey key;
        key.predicate = a.predicate;
        for (const auto &x : a.args)
            key.args.push_back(x.v);
        // The model has no duplicate atoms, so assign the next FactId.
        FactId id = static_cast<FactId>(out.fact_by_id.size());
        vector<string> args;
        args.reserve(a.args.size());
        for (const auto &x : a.args)
            args.push_back(grounding::arg_to_string(x));
        auto atom = make_shared<const Atom>(a.predicate_name(), move(args));
        out.set.insert(atom);
        out.ids.insert(move(key), id);
        out.fact_by_id.push_back(move(atom));
    }
    return out;
}

// Rebuild a ground literal as an Atom/NegatedAtom (for the few axiom and goal
// literals, which stay ConditionPtr-based downstream).
ConditionPtr to_condition(
    const GroundLiteral &lit,
    const vector<shared_ptr<const Atom>> &fact_by_id) {
    const auto &atom = fact_by_id[lit.fact];
    if (!lit.negated)
        return atom;
    return make_shared<NegatedAtom>(atom->predicate, atom->args);
}

// Add static-true init facts to the fact map: each init atom whose ground key
// is not already a reachable fluent fact is a static fact, marked STATIC_FACT.
// (Fluent facts were inserted with their FactId first and take priority.)
void add_static_init_facts(const Task &task, FactMap &facts) {
    for (const auto &elem : task.init) {
        auto *ap = get_if<shared_ptr<const Atom>>(&elem);
        if (!ap || !*ap)
            continue;
        GroundKey key;
        key.predicate = grounding::symbols().intern((*ap)->predicate);
        for (const auto &arg : (*ap)->args)
            key.args.push_back(grounding::symbols().intern(arg));
        facts.insert(move(key), STATIC_FACT);
    }
}

// Identity of a primitive numeric expression: its function symbol and argument
// names. A pair keyed map replaces the old symbol + '\x1f' + args string key --
// no separator-byte convention, and the cost lookup no longer rebuilds a joined
// string per ground action.
using PneKey = pair<string, vector<string>>;
struct PneKeyHash {
    size_t operator()(const PneKey &k) const noexcept {
        size_t h = hash<string>{}(k.first);
        for (const auto &a : k.second)
            utils::hash_combine(h, hash<string>{}(a));
        return h;
    }
};
using InitAssignments =
    unordered_map<PneKey, shared_ptr<const FunctionalExpression>, PneKeyHash>;

// PNE-to-expression map for init assignments.
InitAssignments build_init_assignments(const Task &task) {
    InitAssignments out;
    for (const auto &elem : task.init) {
        if (auto *as = get_if<shared_ptr<Assign>>(&elem)) {
            if (*as && (*as)->fluent)
                out[{(*as)->fluent->symbol, (*as)->fluent->args}] =
                    (*as)->expression;
        }
    }
    return out;
}

// Objects grouped by (super)type, as interned object ids -- so binding a
// parameter during instantiation stores an id directly, feeding the integer
// ground-fact probe without any string work.
unordered_map<string, vector<int>> get_objects_by_type(const Task &task) {
    unordered_map<string, vector<int>> result;
    unordered_map<string, vector<string>> supertypes;
    for (const auto &t : task.types)
        supertypes[t.name] = t.supertype_names;
    for (const auto &obj : task.objects) {
        int id = grounding::symbols().intern(obj.name);
        result[obj.type_name].push_back(id);
        for (const auto &sup : supertypes[obj.type_name])
            result[sup].push_back(id);
    }
    return result;
}

// Recursively iterate over the cartesian product of objects-by-type for
// each parameter, calling `fn(var_mapping)` for each assignment.
void for_each_assignment(
    const vector<TypedObject> &parameters, VarMapping &var_mapping,
    const unordered_map<string, vector<int>> &objects_by_type,
    const function<void()> &fn, size_t depth = 0) {
    if (depth == parameters.size()) {
        fn();
        return;
    }
    const auto &par = parameters[depth];
    auto it = objects_by_type.find(par.type_name);
    if (it == objects_by_type.end())
        return;
    for (const auto &obj : it->second) {
        var_mapping[par.name] = obj;
        for_each_assignment(
            parameters, var_mapping, objects_by_type, fn, depth + 1);
    }
}

void instantiate_effect(
    const Effect &eff, VarMapping &var_mapping, const FactMap &fluent_facts,
    const unordered_map<string, vector<int>> &objects_by_type,
    vector<GroundEffect> &result) {
    auto inst_once = [&]() {
        vector<GroundLiteral> condition;
        if (eff.condition &&
            !eff.condition->instantiate(var_mapping, fluent_facts, condition))
            return;
        vector<GroundLiteral> lit_out;
        if (eff.literal &&
            !eff.literal->instantiate(var_mapping, fluent_facts, lit_out))
            return;
        if (!lit_out.empty()) {
            result.emplace_back(move(condition), move(lit_out[0]));
        }
    };
    if (eff.parameters.empty()) {
        inst_once();
    } else {
        for_each_assignment(
            eff.parameters, var_mapping, objects_by_type, inst_once);
    }
}

long long evaluate_constant(const FunctionalExpression &expr) {
    if (expr.kind() == FunctionalExpression::Kind::CONSTANT) {
        return static_cast<const NumericConstant &>(expr).value;
    }
    throw runtime_error("cost expression is not a numeric constant");
}

// Ground the action's cost under `var_mapping`: 1 without a metric; otherwise
// evaluate its cost expression (a constant directly, or a PNE looked up in the
// initial assignments), or 0 if the action has no cost expression.
long long resolve_action_cost(
    const Action &action, const VarMapping &var_mapping,
    const InitAssignments &init_assignments, bool use_metric) {
    if (!use_metric)
        return 1;
    if (!action.cost || !action.cost->expression)
        return 0;
    // Instantiate the cost expression: if it's a PNE, look up in
    // init_assignments; else if it's a constant, use it.
    if (action.cost->expression->kind() != FunctionalExpression::Kind::PNE)
        return evaluate_constant(*action.cost->expression);
    const auto &pne = static_cast<const PrimitiveNumericExpression &>(
        *action.cost->expression);
    vector<string> resolved_args;
    resolved_args.reserve(pne.args.size());
    for (const auto &a : pne.args) {
        auto it = var_mapping.find(a);
        resolved_args.push_back(
            it == var_mapping.end() ? a
                                    : grounding::symbols().name(it->second));
    }
    auto it = init_assignments.find(PneKey{pne.symbol, move(resolved_args)});
    if (it == init_assignments.end())
        throw runtime_error("Could not find PNE initialization for cost");
    return evaluate_constant(*it->second);
}

/*
  Compiled form of an action schema. Instantiating a schema for every ground
  action re-walks its condition tree (a virtual call per node) and resolves
  every argument name through a string lookup in VarMapping -- all of which
  is invariant across the schema's ground actions. Compilation resolves each
  literal once into an integer predicate id plus per-arg codes (>= 0:
  interned constant id; < 0: ~code = slot in the parameter-value array), so
  binding a ground action just copies interned ids from the model atom into
  the slots and probes the fact map.

  Schemas whose conditions use shapes the compiler does not handle
  (disjunctions, universals, falsity) keep `usable == false` and fall back
  to the original tree walk.
*/
struct CompiledLiteral {
    int predicate;
    bool negated;
    small_vector::SmallVector<int, 4> args; // >= 0: const id; < 0: ~slot
};

struct CompiledEffect {
    // Object-id domains for the effect's own parameters (their slots follow
    // the action's), resolved from objects_by_type once. An effect whose
    // parameter type has no objects can never fire.
    vector<const vector<int> *> param_domains;
    bool dead = false;
    vector<CompiledLiteral> condition;
    CompiledLiteral literal;
};

struct CompiledAction {
    bool usable = false;
    vector<CompiledLiteral> precondition;
    vector<CompiledEffect> effects;
    size_t num_slots = 0;
};

/*
  Resolve an argument name to its parameter slot, mirroring VarMapping's
  one-entry-per-name update semantics: effect parameters shadow action
  parameters of the same name, and among same-named parameters the last
  binding wins. Names bound to no parameter are treated as constants and
  interned (exactly what the string path's failed map lookup did).
*/
int resolve_slot(
    const string &name, const vector<TypedObject> &action_params,
    const vector<TypedObject> *eff_params) {
    if (eff_params) {
        for (int i = static_cast<int>(eff_params->size()) - 1; i >= 0; --i)
            if ((*eff_params)[i].name == name)
                return static_cast<int>(action_params.size()) + i;
    }
    for (int i = static_cast<int>(action_params.size()) - 1; i >= 0; --i)
        if (action_params[i].name == name)
            return i;
    return -1;
}

CompiledLiteral compile_literal(
    const Literal &lit, bool negated, const vector<TypedObject> &action_params,
    const vector<TypedObject> *eff_params) {
    CompiledLiteral out;
    out.predicate = lit.predicate_id;
    out.negated = negated;
    out.args.reserve(lit.args.size());
    for (const auto &a : lit.args) {
        int slot = resolve_slot(a, action_params, eff_params);
        out.args.push_back(
            slot >= 0 ? ~slot : grounding::symbols().intern(a));
    }
    return out;
}

// Flatten `cond` into literals (in tree order, matching the instantiate()
// walk). Returns false on a shape the compiler does not handle.
// With `fluent_preds` set (deferred action grounding only), positive
// literals over non-fluent (static) predicates are omitted: the ground
// action was derived by joining exactly those relations, so the probe
// cannot fail, and a satisfied static literal contributes no
// GroundLiteral anyway.
bool compile_condition(
    const ConditionPtr &cond, const vector<TypedObject> &action_params,
    const vector<TypedObject> *eff_params, vector<CompiledLiteral> &out,
    const unordered_set<int> *fluent_preds = nullptr) {
    if (!cond)
        return true;
    switch (cond->kind()) {
    case Condition::Kind::TRUTH:
        return true;
    case Condition::Kind::ATOM: {
        const auto &lit = static_cast<const Literal &>(*cond);
        if (fluent_preds && !fluent_preds->contains(lit.predicate_id))
            return true;
        out.push_back(
            compile_literal(lit, false, action_params, eff_params));
        return true;
    }
    case Condition::Kind::NEGATED_ATOM:
        out.push_back(compile_literal(
            static_cast<const Literal &>(*cond), true, action_params,
            eff_params));
        return true;
    case Condition::Kind::CONJUNCTION:
        for (const auto &c : cond->parts())
            if (c && !compile_condition(
                          c, action_params, eff_params, out, fluent_preds))
                return false;
        return true;
    case Condition::Kind::EXISTENTIAL:
        // Mirrors ExistentialCondition::instantiate: only body[0] is
        // walked; the quantified parameters get no bindings of their own.
        return cond->parts().empty() ||
               compile_condition(
                   cond->parts()[0], action_params, eff_params, out,
                   fluent_preds);
    default:
        return false;
    }
}

CompiledAction compile_action(
    const Action &action,
    const unordered_map<string, vector<int>> &objects_by_type,
    const unordered_set<int> *fluent_preds_if_deferred) {
    CompiledAction ca;
    if (!compile_condition(
            action.precondition, action.parameters, nullptr, ca.precondition,
            fluent_preds_if_deferred))
        return ca;
    size_t max_eff_params = 0;
    for (const auto &eff : action.effects) {
        CompiledEffect ce;
        if (!eff.literal)
            continue; // no literal -> the effect can never emit anything
        auto lk = eff.literal->kind();
        if (lk != Condition::Kind::ATOM && lk != Condition::Kind::NEGATED_ATOM)
            return ca;
        if (!compile_condition(
                eff.condition, action.parameters, &eff.parameters,
                ce.condition))
            return ca;
        ce.literal = compile_literal(
            static_cast<const Literal &>(*eff.literal),
            lk == Condition::Kind::NEGATED_ATOM, action.parameters,
            &eff.parameters);
        ce.param_domains.reserve(eff.parameters.size());
        for (const auto &par : eff.parameters) {
            auto it = objects_by_type.find(par.type_name);
            if (it == objects_by_type.end()) {
                ce.dead = true;
                break;
            }
            ce.param_domains.push_back(&it->second);
        }
        max_eff_params = max(max_eff_params, eff.parameters.size());
        ca.effects.push_back(move(ce));
    }
    ca.num_slots = action.parameters.size() + max_eff_params;
    ca.usable = true;
    return ca;
}

// Probe one compiled literal under `slots`, appending fluent literals to
// `result`. Same single-probe classification as Atom/NegatedAtom::
// instantiate: fluent -> real literal, static-true -> satisfied (positive) /
// falsified (negative), absent -> falsified (positive) / satisfied (negative).
bool probe_compiled(
    const CompiledLiteral &lit, const vector<int> &slots,
    const FactMap &facts, vector<GroundLiteral> &result) {
    static thread_local GroundKey key;
    key.predicate = lit.predicate;
    key.args.clear();
    for (int a : lit.args)
        key.args.push_back(a < 0 ? slots[~a] : a);
    const FactId *id = facts.find(key);
    if (!lit.negated) {
        if (!id)
            return false;
        if (*id != STATIC_FACT)
            result.push_back({*id, false});
        return true;
    }
    if (!id)
        return true;
    if (*id != STATIC_FACT) {
        result.push_back({*id, true});
        return true;
    }
    return false;
}

// Enumerate assignments of effect parameters `depth..` (product of the
// compiled domains, same order as for_each_assignment) and emit one ground
// effect per satisfied assignment.
void emit_compiled_effect(
    const CompiledEffect &ce, vector<int> &slots, size_t base, size_t depth,
    const FactMap &fluent_facts, vector<GroundEffect> &result) {
    if (depth == ce.param_domains.size()) {
        vector<GroundLiteral> condition;
        for (const auto &lit : ce.condition)
            if (!probe_compiled(lit, slots, fluent_facts, condition))
                return;
        vector<GroundLiteral> lit_out;
        if (probe_compiled(ce.literal, slots, fluent_facts, lit_out) &&
            !lit_out.empty())
            result.emplace_back(move(condition), lit_out[0]);
        return;
    }
    for (int obj : *ce.param_domains[depth]) {
        slots[base + depth] = obj;
        emit_compiled_effect(
            ce, slots, base, depth + 1, fluent_facts, result);
    }
}

optional<PropositionalAction> instantiate_action_compiled(
    const Action &action, const CompiledAction &ca,
    const grounding::Atom &atom, const vector<string> &args,
    const InitAssignments &init_assignments, const FactMap &fluent_facts,
    bool use_metric) {
    static thread_local vector<int> slots;
    slots.assign(ca.num_slots, 0);
    for (size_t i = 0; i < action.parameters.size(); ++i)
        slots[i] = atom.args[i].v;

    vector<GroundLiteral> precondition;
    for (const auto &lit : ca.precondition)
        if (!probe_compiled(lit, slots, fluent_facts, precondition))
            return nullopt;

    vector<GroundEffect> effects;
    for (const auto &ce : ca.effects) {
        if (ce.dead)
            continue;
        emit_compiled_effect(
            ce, slots, action.parameters.size(), 0, fluent_facts, effects);
    }
    if (effects.empty() && !get_options().keep_no_ops)
        return nullopt;

    // Grounded name: same format as instantiate_action (see the trailing-
    // space note there).
    string name = "(" + action.name + " ";
    for (int i = 0; i < action.num_external_parameters; ++i) {
        if (i > 0)
            name.push_back(' ');
        name += args[i];
    }
    name.push_back(')');

    long long cost = 1;
    if (use_metric) {
        // resolve_action_cost only reads the mapping for PNE costs; rebuild
        // it just for that (rare) case.
        static thread_local VarMapping var_mapping;
        var_mapping.clear();
        for (size_t i = 0; i < action.parameters.size(); ++i)
            var_mapping[action.parameters[i].name] =
                static_cast<int>(slots[i]);
        cost = resolve_action_cost(
            action, var_mapping, init_assignments, use_metric);
    }
    return PropositionalAction(
        name, move(precondition), move(effects), static_cast<int>(cost));
}

optional<PropositionalAction> instantiate_action(
    const Action &action, const vector<string> &args,
    const InitAssignments &init_assignments, const FactMap &fluent_facts,
    const unordered_map<string, vector<int>> &objects_by_type,
    bool use_metric) {
    if (args.size() != action.parameters.size())
        return nullopt;
    // Reused across ground actions (instantiate_action is never re-entrant):
    // clear() keeps the backing buffer, so rebinding per ground action neither
    // frees nor re-allocates in the dominant instantiation phase. Parameterised
    // effects still take their own copy before binding extra parameters.
    static thread_local VarMapping var_mapping;
    var_mapping.clear();
    for (size_t i = 0; i < action.parameters.size(); ++i)
        var_mapping[action.parameters[i].name] =
            grounding::symbols().intern(args[i]);

    // Build the grounded name using only external parameters.
    //
    // We mirror Python's `"(%s %s)" % (action.name, " ".join(args))`
    // exactly -- that format always emits a space after action.name,
    // even when the args list is empty, producing "(name )" (with a
    // space before the close paren). After SAS-output paren-stripping
    // this becomes a trailing-space in the operator name, which is
    // load-bearing for byte-identical output and stable sort key.
    string name = "(" + action.name + " ";
    for (int i = 0; i < action.num_external_parameters; ++i) {
        if (i > 0)
            name.push_back(' ');
        name += args[i];
    }
    name.push_back(')');

    vector<GroundLiteral> precondition;
    if (action.precondition && !action.precondition->instantiate(
                                   var_mapping, fluent_facts, precondition))
        return nullopt;

    vector<GroundEffect> effects;
    for (const auto &eff : action.effects) {
        if (eff.parameters.empty()) {
            // A parameterless effect adds no bindings, and instantiate()
            // only reads var_mapping, so share the action's mapping
            // directly instead of copying the whole map per effect.
            instantiate_effect(
                eff, var_mapping, fluent_facts, objects_by_type, effects);
        } else {
            VarMapping local_mapping = var_mapping;
            instantiate_effect(
                eff, local_mapping, fluent_facts, objects_by_type, effects);
        }
    }
    if (!effects.empty() || get_options().keep_no_ops) {
        long long cost = resolve_action_cost(
            action, var_mapping, init_assignments, use_metric);
        return PropositionalAction(
            name, move(precondition), move(effects), static_cast<int>(cost));
    }
    return nullopt;
}

shared_ptr<PropositionalAxiom> instantiate_axiom(
    const Axiom &axiom, const vector<string> &args, const FactMap &fluent_facts,
    const vector<shared_ptr<const Atom>> &fact_by_id) {
    if (args.size() != axiom.parameters.size())
        return nullptr;
    VarMapping var_mapping;
    for (size_t i = 0; i < axiom.parameters.size(); ++i)
        var_mapping[axiom.parameters[i].name] =
            grounding::symbols().intern(args[i]);

    vector<string> name_args;
    name_args.push_back(axiom.name);
    for (int i = 0; i < axiom.num_external_parameters; ++i)
        name_args.push_back(args[i]);
    string name = "(";
    for (size_t i = 0; i < name_args.size(); ++i) {
        if (i)
            name.push_back(' ');
        name += name_args[i];
    }
    name.push_back(')');

    vector<GroundLiteral> condition_lits;
    if (axiom.condition && !axiom.condition->instantiate(
                               var_mapping, fluent_facts, condition_lits))
        return nullptr;
    // Axioms are few: keep the downstream (axiom_rules) atom-based.
    vector<ConditionPtr> condition;
    condition.reserve(condition_lits.size());
    for (const auto &l : condition_lits)
        condition.push_back(to_condition(l, fact_by_id));

    vector<string> eff_args;
    eff_args.reserve(axiom.num_external_parameters);
    for (int i = 0; i < axiom.num_external_parameters; ++i) {
        const auto &n = axiom.parameters[i].name;
        auto it = var_mapping.find(n);
        eff_args.push_back(
            it == var_mapping.end() ? n
                                    : grounding::symbols().name(it->second));
    }
    auto effect = make_shared<const Atom>(axiom.name, move(eff_args));
    return make_shared<PropositionalAxiom>(
        move(name), move(condition), move(effect));
}

optional<vector<ConditionPtr>> instantiate_goal(
    const ConditionPtr &goal, const FactMap &fluent_facts,
    const vector<shared_ptr<const Atom>> &fact_by_id) {
    vector<GroundLiteral> lits;
    VarMapping empty;
    if (goal && !goal->instantiate(empty, fluent_facts, lits))
        return nullopt;
    vector<ConditionPtr> result;
    result.reserve(lits.size());
    for (const auto &l : lits)
        result.push_back(to_condition(l, fact_by_id));
    return result;
}
}

Result instantiate(
    const Task &task, const vector<grounding::Atom> &model,
    const grounding::PredicateRoles &roles,
    const unordered_map<int, vector<grounding::Atom>> *negated_statics) {
    Result out;
    // Extensions of the static predicates referenced by negated
    // preconditions, probed to skip statically inapplicable ground actions
    // with one lookup. The equality facts =(o, o) are ordinary init facts
    // in the model, so inequality constraints need no special case.
    unordered_set<grounding::Atom, grounding::AtomHash> static_atoms;
    if (negated_statics && !negated_statics->empty()) {
        // Equality is handled by value comparison below; only other static
        // predicates need their extensions materialized.
        const int eq = grounding::symbols().intern("=");
        unordered_set<int> referenced;
        for (const auto &[pred, negs] : *negated_statics)
            for (const auto &n : negs)
                if (n.predicate != eq)
                    referenced.insert(n.predicate);
        if (!referenced.empty())
            for (const auto &a : model)
                if (referenced.contains(a.predicate))
                    static_atoms.insert(a);
    }
    size_t statically_inapplicable = 0;
    const int eq_pred = grounding::symbols().intern("=");
    grounding::Atom probe("", {});
    auto violates_negated_static = [&](const grounding::Atom &atom) {
        if (!negated_statics || negated_statics->empty())
            return false;
        auto ni = negated_statics->find(atom.predicate);
        if (ni == negated_statics->end())
            return false;
        for (const grounding::Atom &neg : ni->second) {
            // Inequality constraints (the common case: orgsyn, GED,
            // caldera) resolve by comparing the two bound values -- no
            // probe. The generic path handles any other static predicate
            // via the =(o, o)-style extension probe.
            if (neg.predicate == eq_pred && neg.args.size() == 2) {
                const grounding::Arg &x = neg.args[0];
                const grounding::Arg &y = neg.args[1];
                int xv = x.is_position() ? atom.args[x.position()].v : x.v;
                int yv = y.is_position() ? atom.args[y.position()].v : y.v;
                if (xv == yv)
                    return true;
                continue;
            }
            probe.predicate = neg.predicate;
            probe.args.clear();
            for (const grounding::Arg &arg : neg.args)
                probe.args.push_back(
                    arg.is_position() ? atom.args[arg.position()] : arg);
            if (static_atoms.contains(probe))
                return true;
        }
        return false;
    };
    out.reachable_action_parameters.resize(task.actions.size());
    auto fluent_preds = get_fluent_predicates(task);
    auto fluent = build_fluent_facts(model, fluent_preds);
    out.fluent_facts = move(fluent.set);
    out.fluent_fact_ids = move(fluent.ids);
    out.fact_by_id = move(fluent.fact_by_id);
    const FactMap &fluent_facts = out.fluent_fact_ids;
    const auto &fact_by_id = out.fact_by_id;
    add_static_init_facts(task, out.fluent_fact_ids);
    auto init_assignments = build_init_assignments(task);
    auto objects_by_type = get_objects_by_type(task);

    // One compiled form per schema; ground actions of unsupported schemas
    // take the original tree-walking path. (objects_by_type outlives the
    // compiled domain pointers; it is not mutated below.)
    vector<CompiledAction> compiled_actions;
    compiled_actions.reserve(task.actions.size());
    for (const auto &action : task.actions)
        compiled_actions.push_back(compile_action(
            action, objects_by_type,
            negated_statics ? &fluent_preds : nullptr));

    for (const auto &atom : model) {
        switch (roles.role_of(atom.predicate)) {
        case grounding::PredicateRole::GOAL_REACHABLE:
            out.relaxed_reachable = true;
            break;
        case grounding::PredicateRole::ACTION: {
            int action_idx = roles.index_of(atom.predicate);
            const Action &action = task.actions[action_idx];
            if (atom.args.size() < action.parameters.size())
                break;
            vector<string> args;
            args.reserve(action.parameters.size());
            // Interned object ids for the reachable-parameters table (kept for
            // invariant finding). Storing ids instead of the argument strings
            // shrinks this long-lived table ~8x and lets its only consumer
            // (BalanceChecker's "ever equal?" test) compare ints; equal object
            // names always intern to the same id, so the test is unchanged.
            vector<int> arg_ids;
            arg_ids.reserve(action.parameters.size());
            for (size_t i = 0; i < action.parameters.size(); ++i)
                arg_ids.push_back(atom.args[i].v);
            // A statically inapplicable action (violated negated static
            // precondition) skips instantiation entirely, but its
            // parameters are still recorded below: the invariant finder
            // must see the same reachable-parameter sets either way.
            optional<PropositionalAction> inst;
            if (violates_negated_static(atom)) {
                ++statically_inapplicable;
            } else {
                for (size_t i = 0; i < action.parameters.size(); ++i)
                    args.push_back(grounding::arg_to_string(atom.args[i]));
                const CompiledAction &ca = compiled_actions[action_idx];
                inst = ca.usable
                           ? instantiate_action_compiled(
                                 action, ca, atom, args, init_assignments,
                                 fluent_facts, task.use_min_cost_metric)
                           : instantiate_action(
                                 action, args, init_assignments,
                                 fluent_facts, objects_by_type,
                                 task.use_min_cost_metric);
            }
            out.reachable_action_parameters[action_idx].push_back(
                move(arg_ids));
            if (inst)
                out.instantiated_actions.push_back(move(*inst));
            break;
        }
        case grounding::PredicateRole::AXIOM: {
            int axiom_idx = roles.index_of(atom.predicate);
            const Axiom &axiom = task.axioms[axiom_idx];
            if (atom.args.size() < axiom.parameters.size())
                break;
            vector<string> args;
            args.reserve(axiom.parameters.size());
            for (size_t i = 0; i < axiom.parameters.size(); ++i)
                args.push_back(grounding::arg_to_string(atom.args[i]));
            auto inst =
                instantiate_axiom(axiom, args, fluent_facts, fact_by_id);
            if (inst)
                out.instantiated_axioms.push_back(move(inst));
            break;
        }
        case grounding::PredicateRole::OTHER:
            break;
        }
    }
    if (statically_inapplicable > 0)
        cout << statically_inapplicable
             << " actions violate negated static preconditions." << endl;
    out.instantiated_goal =
        instantiate_goal(task.goal, fluent_facts, fact_by_id);
    return out;
}
}
