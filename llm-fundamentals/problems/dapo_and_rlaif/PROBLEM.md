# AI labels and long reasoning RL

Two teams describe their setup.

Team A stops paying human raters. A strong model writes the preference labels, and Team A trains on those labels.
Team B runs group-relative RL on math. Long chain-of-thought runs collapse: entropy dies, and many prompts come back with a whole group all correct or all wrong, so the advantage is zero. They switch to an asymmetric clip (a higher ceiling than floor), they drop groups with no reward variance, and they average the policy gradient over tokens rather than giving each sequence equal weight.

Which names match?
