# Hints for Minimum spanning tree

## Hint 1

Sort the edges by weight. Which edges should a cheapest tree prefer?

## Hint 2

Walk the sorted edges. An edge whose ends are already connected would close a cycle: skip it. Otherwise take it.

## Hint 3

Keep components in a union-find with path halving and union by size (the union-find exercise). Count the edges you take; if it is less than `n - 1` at the end, the graph is disconnected.
