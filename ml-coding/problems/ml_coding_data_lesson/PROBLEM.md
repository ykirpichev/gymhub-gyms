# Data handling and metrics in code

Half of real ML code is not the model: it is splitting, transforming and scoring data. Interviewers use these exercises to see whether you avoid the subtle bugs.

**Metrics from counts.** Build the confusion matrix first, then every metric is a ratio of its cells. Decide what to return when a denominator is zero (no predicted positives means precision is undefined; libraries return 0 with a warning). **ROC AUC** is the probability that a random positive outscores a random negative, so it can be computed by sorting the scores once and counting, $O(N \log N)$, with ties counting one half. Multi-class metrics come in **macro** (average per class) and **micro** (pool counts) flavours.

**Sampling.** **Reservoir sampling** keeps a uniform sample of $k$ items from a stream of unknown length in one pass: keep the first $k$, then replace a random slot with item $i$ with probability $k / i$. **Stratified** sampling keeps class proportions; **weighted** sampling draws with given probabilities (`np.random.default_rng(seed).choice(..., p=w)`). Seed the generator you are given; never the global one.

**Splits without leakage.** Shuffle with a seed, then split. Rows that belong together (the same user, the same session, near-duplicate texts) go to the same side, or the validation score measures memorization. Time series split by time.

**Fit on train, apply everywhere.** Standardization, imputation and category vocabularies are *fitted*: compute the mean and standard deviation (or the category list) on the training rows only, then apply those numbers to validation and test. Handle what training never saw: a zero variance column (divide by 1 instead), a category that appears only at test time (an "unknown" bucket).

**Streaming statistics.** Means and variances over a stream use running updates (Welford's algorithm) instead of storing everything; a weighted mean keeps a running weighted sum and a running total weight.
