## Hint 1
Build the confusion matrix first; everything else comes from it. TP for class c is the diagonal entry, the column sum is how often c was predicted (TP + FP), the row sum is how often c was the truth (TP + FN).

## Hint 2
For macro averages, pick the classes that occur at all: row sum + column sum > 0. A class that is true but never predicted still counts (precision 0.0); dividing by K instead silently drags the average down with classes that never occurred.

## Hint 3
Micro averaging pools TP, FP and FN across classes. In single-label classification every mistake is one FP (for the predicted class) and one FN (for the true class), so micro precision, recall and F1 all equal accuracy.
