#include "build.h"

#include "model.h"
#include "split.h"

#include "../translate_options.h"

#include "../pddl/action.h"
#include "../pddl/axiom.h"
#include "../pddl/condition.h"
#include "../pddl/f_expression.h"
#include "../pddl/task.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

using namespace std;
namespace translate::grounding {
using namespace pddl;

namespace {
/*
  Build the Datalog rule body for a condition and a set of parameters.
  Mirrors normalize.condition_to_rule_body.

  - For each `parameter`, requires `type@T(par.name)`.
  - For an ExistentialCondition wrapper: requires `type@T(var)` for each
    bound var and then descends.
  - For a Conjunction, recurses over parts. For literals, only positive
    ones become part of the body.
  - For Falsity, returns a body of [@always-false] to make the rule
    unsatisfiable.
  - If a PNE is provided (only for action costs), require its definition
    predicate.
*/
vector<Atom> condition_to_rule_body(
    const vector<TypedObject> &parameters, const ConditionPtr &condition,
    const PrimitiveNumericExpression *pne) {
    vector<Atom> result;
    result.reserve(parameters.size());
    for (const auto &par : parameters) {
        result.emplace_back(
            type_predicate_name(par.type_name), ArgList{Arg(par.name)});
    }
    if (condition && condition->kind() != Condition::Kind::TRUTH) {
        ConditionPtr cur = condition;
        if (cur->kind() == Condition::Kind::EXISTENTIAL) {
            const auto &q = static_cast<const ExistentialCondition &>(*cur);
            for (const auto &par : q.parameters) {
                result.emplace_back(
                    type_predicate_name(par.type_name), ArgList{Arg(par.name)});
            }
            cur = q.body[0];
        }
        vector<ConditionPtr> parts;
        if (cur->kind() == Condition::Kind::CONJUNCTION)
            parts = cur->parts();
        else
            parts = {cur};
        for (const auto &part : parts) {
            if (!part)
                continue;
            if (part->kind() == Condition::Kind::FALSITY) {
                return {Atom("@always-false", {})};
            }
            if (part->kind() != Condition::Kind::ATOM &&
                part->kind() != Condition::Kind::NEGATED_ATOM) {
                throw runtime_error(
                    "Condition not normalized: cannot build rule body");
            }
            const auto &lit = static_cast<const Literal &>(*part);
            if (!lit.negated()) {
                ArgList args;
                args.reserve(lit.args.size());
                for (const auto &a : lit.args)
                    args.emplace_back(a);
                result.emplace_back(lit.predicate, move(args));
            }
        }
    }
    if (pne) {
        // @def-<symbol>(pne.args...)
        ArgList args;
        args.reserve(pne->args.size());
        for (const auto &a : pne->args)
            args.emplace_back(a);
        result.emplace_back("@def-" + pne->symbol, move(args));
    }
    return result;
}

/*
  Collect the negated static atoms of a normalized precondition whose
  arguments are all action parameters or constants, with parameter args
  resolved to their positions (position Args index the deferred head, whose
  args are exactly the parameters in order). Fluent predicates are excluded:
  only static truth is state-independent, so only static atoms may prune
  ground actions.
*/
void collect_negated_statics(
    const ConditionPtr &condition, const unordered_set<int> &fluents,
    const unordered_map<string, int> &param_index, vector<Atom> &out) {
    if (!condition)
        return;
    switch (condition->kind()) {
    case Condition::Kind::EXISTENTIAL:
        if (!condition->parts().empty())
            collect_negated_statics(
                condition->parts()[0], fluents, param_index, out);
        return;
    case Condition::Kind::CONJUNCTION:
        for (const auto &part : condition->parts())
            collect_negated_statics(part, fluents, param_index, out);
        return;
    case Condition::Kind::NEGATED_ATOM: {
        const auto &lit = static_cast<const Literal &>(*condition);
        ArgList args;
        args.reserve(lit.args.size());
        for (const auto &a : lit.args) {
            if (!a.empty() && a.front() == '?') {
                auto it = param_index.find(a);
                if (it == param_index.end())
                    return; // bound by an existential: cannot post-filter
                args.emplace_back(Arg(it->second));
            } else {
                args.emplace_back(a);
            }
        }
        Atom atom(lit.predicate, move(args));
        if (!fluents.contains(atom.predicate))
            out.push_back(move(atom));
        return;
    }
    default:
        return;
    }
}

/*
  Datalog head predicate names. Python uses the Action/Axiom object as
  the predicate; in C++ we encode the (action/axiom) index so the
  instantiate pass can map back to the source action/axiom even when
  multiple actions share a name (e.g., after split_disjunctions).
*/
string action_head_predicate(int action_index) {
    return "@a$" + to_string(action_index);
}
string axiom_head_predicate(int axiom_index) {
    return "@x$" + to_string(axiom_index);
}

// Auxiliary head atom for an action/axiom: its parameters, plus any variables
// bound by an outermost existential in its condition, under `head_predicate`.
Atom make_head(
    const string &head_predicate, const vector<TypedObject> &parameters,
    const ConditionPtr &condition) {
    ArgList variables;
    variables.reserve(parameters.size());
    for (const auto &p : parameters)
        variables.emplace_back(p.name);
    if (condition && condition->kind() == Condition::Kind::EXISTENTIAL) {
        const auto &q = static_cast<const ExistentialCondition &>(*condition);
        for (const auto &p : q.parameters)
            variables.emplace_back(p.name);
    }
    return Atom(head_predicate, move(variables));
}

Atom action_head(const Action &action, int action_index) {
    return make_head(
        action_head_predicate(action_index), action.parameters,
        action.precondition);
}

Atom axiom_head(const Axiom &axiom, int axiom_index) {
    return make_head(
        axiom_head_predicate(axiom_index), axiom.parameters, axiom.condition);
}

void add_typed_object(
    Program &prog, const TypedObject &obj,
    const unordered_map<string, const Type *> &type_dict) {
    auto it = type_dict.find(obj.type_name);
    vector<string> chain;
    chain.push_back(obj.type_name);
    if (it != type_dict.end())
        for (const auto &sup : it->second->supertype_names)
            chain.push_back(sup);
    for (const auto &t : chain) {
        prog.add_fact(Atom(type_predicate_name(t), ArgList{Arg(obj.name)}));
    }
}

void translate_facts(Program &prog, const Task &task) {
    unordered_map<string, const Type *> type_dict;
    for (const auto &t : task.types)
        type_dict[t.name] = &t;
    for (const auto &obj : task.objects)
        add_typed_object(prog, obj, type_dict);
    for (const auto &elem : task.init) {
        if (auto *atom = get_if<shared_ptr<const pddl::Atom>>(&elem)) {
            if (*atom) {
                ArgList args;
                args.reserve((*atom)->args.size());
                for (const auto &a : (*atom)->args)
                    args.emplace_back(a);
                prog.add_fact(Atom((*atom)->predicate, move(args)));
            }
        } else if (auto *as = get_if<shared_ptr<Assign>>(&elem)) {
            if (*as && (*as)->fluent) {
                ArgList args;
                args.reserve((*as)->fluent->args.size());
                for (const auto &a : (*as)->fluent->args)
                    args.emplace_back(a);
                prog.add_fact(
                    Atom("@def-" + (*as)->fluent->symbol, move(args)));
            }
        }
    }
}

void build_exploration_rules(
    Program &prog, const Task &task, Program *deferred,
    unordered_map<int, vector<Atom>> *negatives) {
    unordered_set<int> fluents;
    if (deferred) {
        for (const auto &a : task.actions)
            for (const auto &eff : a.effects)
                if (eff.literal)
                    fluents.insert(symbols().intern(
                        static_cast<const Literal &>(*eff.literal).predicate));
        for (const auto &x : task.axioms)
            fluents.insert(symbols().intern(x.name));
    }
    for (size_t i = 0; i < task.actions.size(); ++i) {
        const Action &action = task.actions[i];
        Atom head = action_head(action, static_cast<int>(i));
        prog.predicate_roles.set(
            head.predicate, PredicateRole::ACTION, static_cast<int>(i));
        const PrimitiveNumericExpression *pne = nullptr;
        if (action.cost && action.cost->expression &&
            action.cost->expression->kind() ==
                FunctionalExpression::Kind::PNE) {
            pne = static_cast<const PrimitiveNumericExpression *>(
                action.cost->expression.get());
        }
        auto body =
            condition_to_rule_body(action.parameters, action.precondition, pne);
        if (deferred) {
            deferred->predicate_roles.set(
                head.predicate, PredicateRole::ACTION, static_cast<int>(i));
            const Atom &deferred_head = head;
            unordered_map<string, int> param_index;
            for (size_t k = 0; k < action.parameters.size(); ++k)
                param_index[action.parameters[k].name] = static_cast<int>(k);
            vector<Atom> negs;
            collect_negated_statics(
                action.precondition, fluents, param_index, negs);
            if (!negs.empty())
                (*negatives)[deferred_head.predicate] = move(negs);
            // Parameters precede any existential-witness variables in the
            // head, so the filters' position args stay valid.
            deferred->add_rule(Rule{body, deferred_head});
        } else {
            prog.add_rule(Rule{body, head});
        }

        for (const auto &eff : action.effects) {
            if (!eff.literal)
                continue;
            const auto &lit = static_cast<const Literal &>(*eff.literal);
            if (lit.negated())
                continue;
            // With deferred action grounding, the effect rule carries the
            // full precondition body instead of the action atom: the least
            // model on the fluents is unchanged (the action atom was just a
            // materialized shared subexpression), but rule splitting can now
            // project intermediates down to the effect's own variables
            // instead of dragging every action parameter to the wide head.
            vector<Atom> rule_body;
            if (deferred)
                rule_body = body;
            else
                rule_body = {head};
            auto sub = condition_to_rule_body({}, eff.condition, nullptr);
            for (auto &c : sub)
                rule_body.push_back(move(c));
            ArgList eff_args;
            eff_args.reserve(lit.args.size());
            for (const auto &a : lit.args)
                eff_args.emplace_back(a);
            prog.add_rule(Rule{rule_body, Atom(lit.predicate, move(eff_args))});
        }
    }
    for (size_t i = 0; i < task.axioms.size(); ++i) {
        const Axiom &axiom = task.axioms[i];
        Atom app_head = axiom_head(axiom, static_cast<int>(i));
        prog.predicate_roles.set(
            app_head.predicate, PredicateRole::AXIOM, static_cast<int>(i));
        auto app_body =
            condition_to_rule_body(axiom.parameters, axiom.condition, nullptr);
        prog.add_rule(Rule{app_body, app_head});
        // External params head.
        ArgList eff_args;
        for (int j = 0; j < axiom.num_external_parameters; ++j)
            eff_args.emplace_back(axiom.parameters[j].name);
        Atom eff_head(axiom.name, move(eff_args));
        prog.add_rule(Rule{{app_head}, eff_head});
    }
    // Goal rule.
    if (task.goal) {
        Atom head("@goal-reachable", {});
        prog.predicate_roles.set(head.predicate, PredicateRole::GOAL_REACHABLE);
        auto body = condition_to_rule_body({}, task.goal, nullptr);
        prog.add_rule(Rule{body, head});
    }
}
}

BuiltProgram build_program(const Task &task) {
    BuiltProgram out;
    out.has_deferred = get_options().defer_action_grounding;
    Program &prog = out.program;
    cout << "Generating Datalog program..." << endl;
    translate_facts(prog, task);
    build_exploration_rules(
        prog, task, out.has_deferred ? &out.deferred_actions : nullptr,
        out.has_deferred ? &out.deferred_negatives : nullptr);
    cout << "Normalizing Datalog program..." << endl;
    prog.normalize();
    return out;
}

void ground_deferred_actions(
    Program &deferred, const PredicateRoles &phase1_roles,
    vector<Atom> &model) {
    // Seed the applicability rules with the complete extension of every
    // ordinary predicate: everything in the model except auxiliary (p$)
    // atoms and phase-1 role atoms (axiom heads, @goal-reachable).
    const size_t num_symbols = symbols().size();
    vector<char> keep(num_symbols, 0);
    for (size_t id = 0; id < num_symbols; ++id) {
        keep[id] =
            (symbols().name(id).find('$') == string::npos &&
             phase1_roles.role_of(static_cast<int>(id)) ==
                 PredicateRole::OTHER)
                ? 1
                : 0;
    }
    for (const Atom &a : model)
        if (keep[a.predicate])
            deferred.add_fact(a);
    deferred.normalize();
    // Relations are complete here, so their statistics can drive the join
    // order.
    ExtensionStats stats = ExtensionStats::of(deferred.facts);
    split_rules(deferred, &stats);
    auto action_model = compute_model(deferred, /*sort_facts=*/false);
    // Precompute action predicates by id; split_rules interned fresh aux
    // names, so re-size against the current symbol table.
    vector<char> is_action(symbols().size(), 0);
    for (size_t id = 0; id < is_action.size(); ++id)
        is_action[id] = deferred.predicate_roles.role_of(
                            static_cast<int>(id)) == PredicateRole::ACTION
                            ? 1
                            : 0;
    size_t appended = 0;
    for (auto &a : action_model) {
        if (is_action[a.predicate]) {
            model.push_back(move(a));
            ++appended;
        }
    }
    cout << appended << " reachable ground actions." << endl;
}
}
