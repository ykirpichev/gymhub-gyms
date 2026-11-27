# Sliding windows without copies

Write `solve(x, window, step)` that returns a 2-D NumPy array whose row `i` is `x[i*step : i*step + window]`, for every window that fits entirely in `x`, and the mean of each window. Build the windows as a **view**: `np.lib.stride_tricks.sliding_window_view(x, window)[::step]` or `np.lib.stride_tricks.as_strided` with strides `(step * itemsize, itemsize)`. Return `(windows, means)`.

If `window > len(x)`, return an empty `(0, window)` array and an empty means array. Raise `ValueError` if `window < 1` or `step < 1`.

```
solve([1, 2, 3, 4, 5], 3, 2)  ->  (array([[1, 2, 3], [3, 4, 5]]), array([2., 4.]))
```

#### What interviewers look for

- The number of windows: `(len(x) - window) // step + 1`.
- Knowing that the result is a view with overlapping rows, so writing into it is dangerous (`as_strided` will let you; `sliding_window_view` returns it read-only).

#### Follow-ups

- How is `unfold` in PyTorch (`x.unfold(0, window, step)`) the same idea, and how does it give convolution as a matmul (im2col)?
- What is the memory of the view, and of `windows.copy()`, for `len(x) = 10^6` and `window = 1000`?
- Compute the window means in O(n) without the view. (Cumulative sums.)
