## Hint 1
Each named optimizer adds a different piece of memory: nothing but the current batch, a velocity, a scale per coordinate, or both plus a correction.

## Hint 2
The normal equation (XᵀX)w = Xᵀy is not a gradient step. Count the cost of building a d×d matrix from n rows.
