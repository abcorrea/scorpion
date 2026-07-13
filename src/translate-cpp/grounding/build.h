#ifndef GROUNDING_BUILD_H
#define GROUNDING_BUILD_H

#include "program.h"

#include "../pddl/task.h"

#include <vector>

namespace translate::grounding {
/*
  Build the Datalog Program that represents reachability for the given
  task. The task must already be normalized (see normalize::normalize).
  The returned Program is also normalized (Program::normalize() has been
  called) but not yet split (split_rules() applies later).

  With options.defer_action_grounding (the default), the reachability
  program contains no wide action-applicability atoms: each effect rule
  carries the action's precondition body directly, and the applicability
  rules are returned separately in `deferred_actions` for a second grounding
  pass (ground_deferred_actions) once the model is complete. Schemas whose
  head has fewer than two arguments are grounded in the main pass regardless
  (deferral cannot help them), so has_deferred is false for fully pre-ground
  inputs and the second pass is skipped.
*/
struct BuiltProgram {
    Program program;
    Program deferred_actions;
    // Negated static precondition atoms per deferred action-head predicate
    // id, args resolved to parameter positions (constants stay symbols). A
    // ground action whose parameter binding satisfies such an atom
    // positively is inapplicable in every state (static truth is
    // state-independent); instantiate uses these to skip the full
    // instantiation of such actions with one probe, while still recording
    // their parameters for the invariant finder (whose search must see the
    // same reachable-parameter sets as the Python translator).
    std::unordered_map<int, std::vector<Atom>> deferred_negatives;
    bool has_deferred = false;
};

BuiltProgram build_program(const pddl::Task &task);

/*
  Ground the deferred action-applicability rules against the completed
  model: seed them with the complete extension of every non-auxiliary,
  role-free predicate in `model`, run the usual normalize/split/model
  pipeline, and append the reachable action atoms to `model` (instantiate
  consumes them from there). `phase1_roles` identifies phase-1 role atoms
  (axiom heads, @goal-reachable) that must not be re-seeded as facts.
*/
void ground_deferred_actions(
    Program &deferred, const PredicateRoles &phase1_roles,
    std::vector<Atom> &model);
}

#endif
