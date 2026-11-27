## Hint 1
A reference policy is allowed. A network trained to predict reward, and a network trained to predict value, are not.

## Hint 2
Separate offline pair losses from PPO. GRPO's baseline is a statistic of the group, not a second network.
