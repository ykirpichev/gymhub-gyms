# How an ML coding round works

In an ML coding round you implement a model or a metric from its definition, usually in Python and NumPy, in 30–45 minutes, without scikit-learn or PyTorch. The interviewer is checking three things: that you know the math, that you can turn it into correct, vectorized code, and that you handle the edge cases a real implementation must.

**A routine that works for every exercise in this gym:**

1. **Restate the inputs and outputs with shapes.** "`X` is `N × D`, `y` is `N`, I return weights of length `D` and a scalar bias." Ask about anything unspecified: ties, empty inputs, the dtype of the labels.
2. **Write the math first**, as a comment: the prediction, the loss, the gradient. For logistic regression: $p = \sigma(Xw + b)$, $\nabla_w = X^\top (p - y) / N$.
3. **Write the shortest correct version**, vectorized where it is natural: matrix products instead of loops over rows, broadcasting instead of loops over features.
4. **Run the example by hand** or in your head, then try an edge case: one row, a constant feature, all one class, a cluster that loses all its points.
5. **Say the complexity** and what you would change at scale.

**Habits that earn points:**

- Numerical stability without being asked: subtract the max in softmax, clip probabilities before `log`, add a small epsilon to a variance.
- Validating inputs that would otherwise fail silently (mismatched lengths, `k > N`).
- Deterministic tie-breaking, stated out loud.
- Separating `fit` from `predict` and never fitting preprocessing on test data.

> [!TIP]
> Each exercise here ends with an **Interview focus** line: the points an interviewer would note. After you pass, read it and the follow-ups, and answer the follow-ups out loud.

The tracks go from classic models to data and evaluation, then language-model mechanics, generative-AI building blocks and the control loop around agents. The PyTorch and NumPy course drills the tensor skills these exercises lean on.
