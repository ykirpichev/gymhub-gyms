# What learning from data means

A program written by hand turns rules into answers. A learned model turns **examples** into rules: you show it pairs $(x_i, y_i)$ and it finds a function $f$ that maps one to the other.

Three things define the learning problem:

1. **A model family**, such as linear functions $f(x) = w^\top x + b$ or trees of a given depth. Learning picks one member.
2. **A loss** $\ell(f(x), y)$: how wrong one prediction is. Squared error for numbers, cross-entropy for classes.
3. **Data**: the $n$ examples you learn from.

Training chooses the member with the smallest average loss on the training examples, the *empirical risk*:

$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} \ell\big(f(x_i), y_i\big)$$

> [!WARNING]
> A low training loss is not the goal. The goal is a low loss on examples the model has **not** seen. A model can memorize the training set and still fail on the next example. That gap is what *generalization* measures.

This is why data is split. The model learns from the **training** set, you compare models and settings on the **validation** set, and the **test** set is scored once at the end, to estimate how the chosen model will do in use.

> [!TIP]
> The questions in this course come back to these pieces: which model family, which loss, and whether a result says anything about unseen data.
