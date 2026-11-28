# Simulate a stream pipeline

A job copies data to the GPU, runs a kernel and copies the result back. It is split into `chunks` equal chunks, one
stream each. Copy in takes `h2d` time units for the whole array, the kernel `kernel` units, copy out `d2h` units; a
chunk takes `1/chunks` of each.

There are three engines: a host-to-device copy engine, the SMs (one kernel at a time) and a device-to-host copy engine.
Each engine runs its tasks one at a time, in chunk order: chunk `c`'s copy in, then chunk `c + 1`'s, and so on. A chunk's
kernel starts when its copy in has finished and the SMs are free; its copy out when its kernel has finished and the
copy-out engine is free.

Return the time at which the last copy out finishes, rounded to 3 decimals.

Example: `solve(1, 8, 8, 8)` is `24.0` (no overlap). `solve(4, 8, 8, 8)` is `12.0`: each stage of a chunk takes 2;
copy-ins end at 2, 4, 6, 8, kernels at 4, 6, 8, 10, copy-outs at 6, 8, 10, 12.
