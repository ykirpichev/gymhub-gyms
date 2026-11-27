# A broadcasting bug in a loss

A model returns `pred` of shape `(N, 1)`. The targets `y` have shape `(N,)`. The loss is `((pred - y) ** 2).mean()`. Training runs without errors but the model barely learns. Why?
