## Hint 1
Fit every column on the training rows, then transform train and test with those saved numbers. A test value must not change a median, a mode, or the list of categories.

## Hint 2
Numeric fill is the median of the observed training values (the average of the two middle ones when the count is even), or 0 when the training column is entirely missing. Categorical fill is the most common training string; if several tie, take the one that sorts first. An all-missing categorical column has no categories.

## Hint 3
One-hot columns are the sorted training categories, then one final unknown column. Fill a missing category with the mode before encoding. A value that is still not in the training set, including every value of an all-missing column, lands in unknown.
