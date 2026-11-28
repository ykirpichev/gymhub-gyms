# Simulate a launch

Model a 1D launch in Python. Given `grid` blocks of `block` threads and an array of `n` elements, return a list of
length `n` where entry `j` is the global thread index that processes element `j`, or `-1` if no thread does.

`mode` is `"single"` (each thread with `i < n` processes element `i` only) or `"stride"` (a grid-stride loop: thread `i`
processes `i`, `i + grid*block`, `i + 2*grid*block`, ... while below `n`).

Example: `solve(2, 4, 10, "single")` is `[0, 1, 2, 3, 4, 5, 6, 7, -1, -1]`, and with `"stride"` it is
`[0, 1, 2, 3, 4, 5, 6, 7, 0, 1]`.
