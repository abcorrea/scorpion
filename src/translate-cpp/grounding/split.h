#ifndef GROUNDING_SPLIT_H
#define GROUNDING_SPLIT_H

#include "program.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace translate::grounding {
// Per-predicate statistics of a complete set of ground facts: extension
// size and, per argument position, the number of distinct values. Used to
// order joins by estimated result size.
struct ExtensionStats {
    std::unordered_map<int, std::size_t> size;
    std::unordered_map<int, std::vector<std::size_t>> distinct;

    static ExtensionStats of(const std::vector<Atom> &facts);
};

/*
  Split rules whose conditions fall into disjoint variable-connected
  components, then split each k-ary join (k>=2) into binary joins via
  greedy_join. Each output rule has type JOIN, PRODUCT, or PROJECT.

  Without `stats` (the default), join pairs are ordered by the
  variable-overlap heuristic shared with the Python translator, keeping the
  split byte-compatible. When the program's relations are already complete
  (deferred action grounding evaluates against the finished model), `stats`
  enables a size-aware left-deep order instead: joins are chained onto the
  running intermediate in ascending order of estimated result size
  (|I| * |R| / prod over shared vars of max(V_I, V_R), the classic
  System-R estimate), which keeps intermediates near the size of the most
  selective prefix instead of exploding on wide rules.
*/
void split_rules(Program &prog, const ExtensionStats *stats = nullptr);
}

#endif
