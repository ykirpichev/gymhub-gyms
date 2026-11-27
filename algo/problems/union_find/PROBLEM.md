# Union-find

**Level:** medium   **Teaches:** disjoint-set union, path compression, union by size or rank

## Task

There are `n` items numbered `0 .. n-1`, each starting in its own group. Process `ops` in order. A `union` joins the groups of two items. A `same` asks whether two items are currently in the same group. Return the answers to all `same` queries, in order.

## Arguments

| Name    | Type         | Role                                                   |
| ------- | ------------ | ------------------------------------------------------ |
| `n`     | `int`        | number of items, `>= 1`                                |
| `ops`   | `list[list]` | `["union", a, b]` or `["same", a, b]` with `0 <= a, b < n` |
| returns | `list[bool]` | one answer per `same`, in order                        |

Solve it in Python or C++. In C++ the function is:

```cpp
struct Op {
    bool is_union;  // true for ["union", a, b], false for ["same", a, b]
    int a, b;
};

vector<bool> solve(int n, vector<Op>& ops);
```

The starter file defines `Op`; keep it as it is, because the tests build the `ops` vector from it.

## Rules

- `union` of two items already in the same group, or of an item with itself, changes nothing.
- `same(a, a)` is always `True`.
- Groups only grow; nothing is ever split.

## Example

```
n = 5
union 0 1
same 0 1   -> True
same 1 2   -> False
union 1 2
same 0 2   -> True
same 3 4   -> False
union 3 4
same 2 4   -> False
result = [True, False, True, False, False]
```

## How to think about it

Store each group as a tree, where every item points to a parent and the root names the group. Two items are in the same group when they have the same root. A naive version that hangs one root under the other can build a chain of length `n`, so every lookup walks O(n) steps. Two fixes, each simple:

- **Union by size (or rank):** hang the smaller tree under the larger, so depth stays O(log n).
- **Path compression:** after finding a root, point every node you walked through straight at it.

Together they make each operation effectively constant time (inverse Ackermann).

## Tests

Empty op lists, a single item, self-unions, repeated unions, merging groups, a long chain, and random streams. Benchmark input: `n = 100,000` with about 300,000 ops, starting with unions that form a chain if roots are linked naively. `bench` fails if the median is over **200 ms** in Python or **6 ms** in C++. Recursion in `find` can also hit Python's recursion limit on deep trees.
