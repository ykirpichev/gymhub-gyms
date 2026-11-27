# Pairs without a reward model

You already have a supervised fine-tune and a set of human pairs: a prompt, a chosen response, and a rejected response. You do not want to fit a reward model or run an online RL loop. You can keep a frozen copy of the current policy in memory.

Which method is built for that setup?
