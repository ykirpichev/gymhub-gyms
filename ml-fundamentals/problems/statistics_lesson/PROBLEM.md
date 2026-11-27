# The statistics ML interviews assume

**Bayes' rule** turns a likelihood into a posterior:

$$P(\text{class} \mid x) = \frac{P(x \mid \text{class}) \, P(\text{class})}{P(x)}.$$

**Naive Bayes** assumes features are independent given the class, so $P(x \mid c) = \prod_j P(x_j \mid c)$. The assumption is rarely true, but the classifier often ranks classes well anyway. Work in log space to avoid underflow, and smooth counts (add one) so an unseen word does not zero out a class.

**MLE and MAP.** Maximum likelihood picks parameters that make the data most probable, $\arg\max_\theta P(\text{data} \mid \theta)$. MAP adds a prior, $\arg\max_\theta P(\text{data} \mid \theta) P(\theta)$. A Gaussian prior on weights gives L2 regularization; a Laplace prior gives L1. With lots of data, MAP and MLE agree.

**R²** for regression is $1 - SS_\text{res} / SS_\text{tot}$: the fraction of the target's variance the model explains. It is 0 for predicting the mean and can be negative on held-out data for a model worse than the mean.

**p-values.** A p-value is the probability, *assuming the null hypothesis is true*, of seeing a result at least as extreme as the one observed. It is not the probability that the null is true, and a small p-value says nothing about how large or important the effect is. Running many tests inflates false positives; correct for it (Bonferroni, or control the false discovery rate).

**Distances** encode what "similar" means:

| Distance | Use when |
|---|---|
| Euclidean | magnitudes matter and features are on comparable scales |
| Manhattan (L1) | robust to single large differences; grid-like features |
| Cosine similarity, $\frac{a \cdot b}{\|a\|\|b\|}$ | direction matters, not length: text vectors, embeddings |
| Hamming / Jaccard | binary vectors or sets |

> [!TIP]
> If embeddings are normalized to unit length, ranking by cosine similarity, by dot product and by Euclidean distance gives the same order, since $\|a - b\|^2 = 2 - 2\cos(a, b)$.
