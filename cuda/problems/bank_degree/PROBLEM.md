# Bank conflict degree

Given the 4-byte word index each lane of a warp reads from shared memory (`words`, up to 32 entries), return the
conflict degree: the largest number of *distinct* words that fall in the same bank (bank = `word % 32`). Lanes reading
the same word share a broadcast and count once. An empty list has degree 0.

Example: `[0, 2, 4, …, 62]` (stride 2): banks 0, 2, …, 30, each holding two distinct words → `2`.
