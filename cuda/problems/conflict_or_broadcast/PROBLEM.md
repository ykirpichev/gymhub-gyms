# Bank conflict or broadcast?

`__shared__ float s[1024];` and `t = threadIdx.x` (one warp). Which accesses are free of bank conflicts?
