# A CuTe layout

Implement CuTe's layout function. A layout is a `shape` and a `stride` with the same nesting: a number or a list of
layouts. Its size is the product of all the numbers in the shape. A 1D coordinate `i` (from 0 to size − 1) maps to an
offset like this:

- For a number shape `s` with stride `d`: offset `= i * d` (here `i < s`).
- For a list of modes, split `i` **colexicographically** (the first mode changes fastest): for each mode in turn,
  `sub = i % size(mode)`, `i = i // size(mode)`; the offset is the sum of each mode's offset for its `sub`.

Return the list of offsets for `i = 0, 1, …, size − 1`.

Example: `solve([4, 2], [1, 4])` (column-major 4 × 2) → `[0, 1, 2, 3, 4, 5, 6, 7]`; `solve([4, 2], [2, 1])` (row-major)
→ `[0, 2, 4, 6, 1, 3, 5, 7]`.
