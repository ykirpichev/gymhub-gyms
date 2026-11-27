# Why the tile is 32 by 33

A transpose kernel loads a 32 × 32 tile of the input into `__shared__ float tile[32][33]` by rows, synchronizes, then reads it by columns to write the output by rows. Why 33 columns?
