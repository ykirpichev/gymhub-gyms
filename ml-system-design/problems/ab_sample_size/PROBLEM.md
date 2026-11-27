# A/B test sample size

The current checkout conversion rate is $p_1 = 0.10$. You want to detect an increase to $p_2 = 0.105$ with a two-sided test at $\alpha = 0.05$ ($z = 1.96$) and 80% power ($z = 0.8416$). Using

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \, \big(p_1(1 - p_1) + p_2(1 - p_2)\big)}{(p_2 - p_1)^2}$$

how many users do you need **per arm**?
