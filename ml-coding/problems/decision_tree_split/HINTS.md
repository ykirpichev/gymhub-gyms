## Hint 1
Gini of a group is `1 - sum(p**2)` from its class counts. Weight the two sides by how many rows they hold, and divide by `N`. Compare that with the Gini of all `N` labels. If nothing is lower by more than `1e-12`, return `None`.

## Hint 2
`np.unique` on a column gives the sorted distinct values. Thresholds are halfway between neighbours, not the values themselves and not midpoints between duplicate rows. The left mask is `column <= threshold`.

## Hint 3
Walk features from index 0 and thresholds from low to high. Keep a split only when its weighted Gini is smaller than the best so far by more than `1e-12`, so an equal score leaves the earlier feature and the earlier threshold in place.
