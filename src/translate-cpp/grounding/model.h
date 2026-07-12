#ifndef GROUNDING_MODEL_H
#define GROUNDING_MODEL_H

#include "program.h"

#include <vector>

namespace translate::grounding {
/*
  Run semi-naive evaluation of the Datalog program. Returns the full
  list of derived ground atoms in the order they were derived.

  The program must have been split (split_rules()) so that each rule has
  a kind of JOIN, PRODUCT, or PROJECT.
*/
/*
  Compute the canonical model. Consumes prog.facts (moved into the queue).
  `sort_facts` keeps the initial facts in Python's name-sorted order, which
  fixes the derivation order the Python translator produces; pass false
  when the caller does not need that order (deferred action grounding),
  saving a string-comparison sort over the complete seeded model.
*/
std::vector<Atom> compute_model(Program &prog, bool sort_facts = true);
}

#endif
