# Hints for Union-find

## Hint 1

Keep `parent = list(range(n))`. The root of `x` is found by following `parent` until `parent[r] == r`. Two items are together exactly when their roots are equal; `union` sets one root's parent to the other root.

## Hint 2

Track `size[root]` and always attach the smaller tree under the larger one. Trees can then never be deeper than about log2(n), even on a stream of unions that would otherwise build a chain.

## Hint 3

Add path compression to `find`, and write it as a loop rather than recursion: first walk up to find the root, then walk the same path again setting each `parent[x] = root`. Path halving (`parent[x] = parent[parent[x]]` while walking) is a one-pass alternative.

In C++, two `vector<int>`s (`parent` and `size`) are all you need; `std::iota` fills `parent` with `0 .. n-1`.
