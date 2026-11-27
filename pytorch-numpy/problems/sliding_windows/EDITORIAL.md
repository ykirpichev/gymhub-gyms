#### The idea
A window view is a 2-D array over the same buffer with strides `(step · itemsize, itemsize)`: moving one row
advances `step` elements, moving one column advances one. `sliding_window_view` builds the stride-1 version safely;
slicing it with `[::step]` multiplies the row stride by `step`. Nothing is copied.

#### A solution
```python
import numpy as np


def solve(x, window, step):
    if window < 1 or step < 1:
        raise ValueError("window and step must be >= 1")
    a = np.asarray(x)
    if window > len(a):
        return (np.empty((0, window), dtype=a.dtype), np.empty(0))
    windows = np.lib.stride_tricks.sliding_window_view(a, window)[::step]
    return (windows, windows.mean(axis=1))
```

#### Why a view
For a signal of a million samples and windows of 1,000, a copy would be a billion elements; the view is free. The
same idea is behind `torch.Tensor.unfold`, and behind im2col for convolutions (which then *does* copy, to feed a
matmul). Overlapping windows share memory, so writing into the view changes several windows at once; that is why
`sliding_window_view` returns a read-only view.

#### Pitfalls
- `as_strided` with strides in elements instead of bytes (NumPy wants bytes; PyTorch wants elements).
- `as_strided` with a shape that runs past the buffer: no error, just garbage from neighbouring memory.
