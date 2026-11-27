# Classifier-free guidance

Write `solve(cond, uncond, w)`.

- `cond`: the conditional noise prediction, a list or array of any shape.
- `uncond`: the unconditional prediction of the **same shape**, or `None`.
- `w`: the guidance scale, any finite float. `w = 0` returns the unconditional prediction. `w = 1` returns the conditional one. Other values, including negatives, extrapolate.

Return a float64 array

`uncond + w * (cond - uncond)`.

If `uncond` is `None`, `cond` is a stacked batch: the first half of axis 0 is conditional and the second half is unconditional. Split there, guide each row, and return **only the guided conditional half**. Axis 0 must have even length, otherwise raise `ValueError`.

Shapes that differ, including shapes that would broadcast, raise `ValueError`. Do not broadcast.

```
solve([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], 2.0)
-> array([2., 3., 5.])

solve([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0], [1.0, 0.0]], None, 2.0)
-> array([[2., 3.],
          [5., 8.]])
```

The stacked call splits into `[[1, 2], [3, 4]]` and `[[0, 1], [1, 0]]`.

**Interview focus:** Conditional/unconditional predictions, guidance scale trade-off.

**Follow-ups** to answer out loud once your solution passes:

- What happens to diversity and image quality as the guidance scale grows?
- How is the model trained so one network gives both predictions?
- How many network evaluations does guidance add per step, and how could you avoid them?
