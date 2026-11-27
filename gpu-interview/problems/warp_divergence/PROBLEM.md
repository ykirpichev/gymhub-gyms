# What a warp does at a branch

A block of 256 threads runs:

```cuda
if (threadIdx.x % 2 == 0) { y[i] = expensive_a(x[i]); }
else                      { y[i] = expensive_b(x[i]); }
```

`expensive_a` and `expensive_b` take about the same time. Which statement is right?
