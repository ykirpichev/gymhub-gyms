#### Idea
The unit of issue is the warp. Count, per warp, which sides have at least one active lane. Each side issued costs its
instruction count once for the warp, whatever the number of active lanes; useful work counts lanes.

#### What it tells you
Efficiency `useful / (32 × issued)` drops to 50% for a balanced two-way branch inside every warp, and to 1/32 in the
worst case of a 32-way `switch`. Two fixes are common: reorganize the data so that threads with the same path sit in
the same warp (sort or bin work items first), or make the branch cheap and branch-free (`y = cond ? a : b`, which
compiles to a select, not a jump).

#### Pitfalls
- Counting a side for a warp even when no lane takes it: the hardware skips a side no lane needs.
- Using `on * then_cost` for `issued`: issue is per warp, not per lane.
