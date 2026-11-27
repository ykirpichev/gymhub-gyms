## Hint 1
The negative gradient of `½(y − F)²` is `y − F`. Fit the stump to those residuals. Fitting `y`, or fitting `F − y`, moves the model the wrong way.

## Hint 2
`np.unique` on a column is the sorted distinct values. Thresholds are halfway between neighbours. The left leaf is the mean residual where `column <= t`, the right leaf is the mean of the rest, and the score is the sum of squared errors around those means.

## Hint 3
Scan features from index 0 and thresholds from low to high. Keep the smallest score; a later split wins only when its score is lower by more than `1e-9`. If no threshold exists, predict the mean residual on every row with feature `-1` and threshold `0`. Then `F_new = F + learning_rate * prediction`.
