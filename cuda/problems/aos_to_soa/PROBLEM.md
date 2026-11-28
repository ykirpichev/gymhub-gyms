# AoS to SoA

Particles are stored as an array of structures: a flat list `aos = [x0, y0, z0, m0, x1, y1, z1, m1, …]` with `fields`
values per particle. Convert it to a structure of arrays: a list of `fields` lists, the first holding every particle's
first field, and so on.

Then return also the stride, in elements, between `x[i]` and `x[i+1]` in each layout, as `[soa, aos_stride, soa_stride]`.

Example: `solve([1, 2, 3, 4, 5, 6], 3)` → `[[[1, 4], [2, 5], [3, 6]], 3, 1]`.
