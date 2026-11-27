## Hint 1
Build the boolean vector `pred = scores >= threshold` and count TP, FP and FN with it. Guard each division: `x / d if d else 0.0`.

## Hint 2
Counting all pairs is O(P·N). Sort instead: if you rank all scores from 1 (lowest) to N, the number of negatives each positive beats is its rank minus its rank among the positives. Summed over positives that gives U = (sum of positive ranks) − P(P+1)/2, and AUC = U / (P·N).

## Hint 3
Ties: give every member of a group of equal scores the **average** of the ranks the group spans (e.g. ranks 4, 5, 6 all become 5). Then a tied positive/negative pair contributes exactly 1/2 to U. Sort once, walk runs of equal values, assign the mean rank to each run.
