# Which loads coalesce?

`A` is a row-major `float` matrix with `N = 4096` columns, and `t = threadIdx.x` runs over a warp's lanes 0–31. `A` starts on a 128-byte boundary and `row` and `col` are the same for every lane. Which loads are fully coalesced (4 sectors per warp)? **Select all that apply.**
