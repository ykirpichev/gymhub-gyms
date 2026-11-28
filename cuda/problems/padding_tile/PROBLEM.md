# Padding the tile

A warp reads column `c` of `__shared__ float tile[32][32]`: lane `r` reads `tile[r][c]`. How many ways is the bank
conflict? And if the tile is declared `float tile[32][33]` instead?

Type the conflict degree of the unpadded tile.
