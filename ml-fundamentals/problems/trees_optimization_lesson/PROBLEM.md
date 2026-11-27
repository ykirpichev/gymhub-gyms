# Trees, ensembles and gradient descent

**A decision tree** splits the data again and again on one feature and a threshold. At each node it picks the split that most reduces **impurity**. For classification, the Gini impurity of a node with class proportions $p_k$ is $G = 1 - \sum_k p_k^2$, which is 0 for a pure node; entropy, $-\sum_k p_k \log p_k$, behaves similarly. A split's quality is the parent's impurity minus the size-weighted impurity of its children. For regression, the impurity is the variance. Trees are easy to read and need no feature scaling, but a deep tree memorizes its training set: limit its depth or minimum leaf size, or prune it.

**Ensembles** fix the variance of single trees:

- **Bagging** trains many trees on bootstrap samples and averages them. A **random forest** also picks a random subset of features at each split, so the trees are less alike and averaging cancels more of the noise. Bagging lowers variance.
- **Boosting** trains trees one after another, each fitting what the ensemble so far gets wrong. In **gradient boosting**, each new tree fits the negative gradient of the loss (the residuals, for squared error), and its predictions are added with a small learning rate. Boosting lowers bias; too many rounds overfit, so use early stopping.

**Gradient descent** moves the parameters against the gradient: $\theta \leftarrow \theta - \eta \nabla L(\theta)$. The learning rate $\eta$ is the knob that matters most: too large diverges, too small crawls. Variants:

| Optimizer | Idea |
|---|---|
| SGD | a gradient from a mini-batch: noisy but cheap |
| Momentum | keep a running average of gradients and step along it, smoothing zig-zags |
| AdaGrad / RMSProp | divide each coordinate's step by the root of its recent squared gradients |
| Adam | momentum plus RMSProp-style scaling, with bias correction for the first steps |

> [!NOTE]
> Trees are invariant to monotone transformations of a feature, so scaling and log transforms change nothing for them, while they matter a lot for gradient descent on linear models and networks.
