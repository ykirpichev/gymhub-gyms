# Linear models and their losses

A **linear model** scores an input with a weighted sum, $z = w^\top x + b$. What changes between models is what $z$ means and which loss judges it.

**Linear regression** predicts $\hat{y} = z$ and minimizes the sum of squared errors, $\sum_i (y_i - \hat{y}_i)^2$. Squaring punishes large errors heavily, so outliers pull the fit. The minimizer has a closed form, $w = (X^\top X)^{-1} X^\top y$, and it is also the maximum-likelihood estimate if the noise is Gaussian.

**Logistic regression** turns the score into a probability with the sigmoid, $p = \sigma(z) = 1 / (1 + e^{-z})$, and minimizes the **binary cross-entropy**:

$$\ell = -\big[y \log p + (1 - y) \log (1 - p)\big].$$

The gradient is beautifully simple, $\partial \ell / \partial z = p - y$, which is why the same pattern (prediction minus target) appears in softmax regression and in neural network output layers. There is no closed form; fit it with gradient descent or Newton's method.

**Support vector machines** use the **hinge loss** $\max(0, 1 - y z)$ with labels $y \in \{-1, +1\}$ plus an L2 penalty on $w$. Points beyond the margin cost nothing, so the solution depends only on the *support vectors* near the boundary. The **kernel trick** replaces dot products $x_i^\top x_j$ with a kernel $k(x_i, x_j)$, fitting a linear model in a feature space you never build.

| Task | Typical output | Loss |
|---|---|---|
| Regression | a number | squared error; absolute error or Huber when outliers matter |
| Binary classification | a probability | binary cross-entropy |
| Multi-class, one label | a distribution over classes | softmax cross-entropy |
| Multi-label | a probability per label | a sigmoid and BCE per label |
| Ranking | an order | pairwise or listwise losses |

> [!TIP]
> In interviews, connect each loss to a likelihood: squared error is Gaussian noise, cross-entropy is a Bernoulli or categorical model. It explains why the losses look the way they do and makes the next question ("what if the noise has heavy tails?") easy.
