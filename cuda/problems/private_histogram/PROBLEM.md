# Privatized histogram

Model a histogram kernel that uses per-block private histograms. `values` are integers in `[0, bins)`, processed by
blocks of `block` consecutive values (the last block may be shorter). Each block counts its values in a shared-memory
histogram (one shared atomic per value), then adds every **non-zero** bin of its private histogram to the global
histogram (one global atomic per non-zero bin).

Return `[histogram, shared_atomics, global_atomics]`. Compare: a naive kernel does `len(values)` global atomics.

Example: `values = [0, 1, 1, 3, 3, 3]`, `bins = 4`, `block = 3`: block 0 sees `[0, 1, 1]` (2 non-zero bins), block 1
sees `[3, 3, 3]` (1 bin). Histogram `[1, 2, 0, 3]`, 6 shared atomics, 3 global atomics.
