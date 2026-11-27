# Hints for Count inversions

## Hint 1

Split the list in half. Every inversion has both ends in the left half, both in the right half, or one in each. The first two kinds come from recursion; the third is what the merge step must count.

## Hint 2

Merge the two sorted halves with two pointers `i` (left) and `j` (right). When `right[j] < left[i]`, `right[j]` is smaller than every remaining left element, so add `len(left) - i`. Take from the left on ties so equal values are not counted.

## Hint 3

Python recursion and slicing add overhead. A bottom-up merge sort that ping-pongs between two buffers is fast and avoids recursion. Alternatively: map values to ranks with `sorted(set(nums))`, then for each value query a Fenwick tree for the count of seen ranks greater than it and add its rank.

In C++, keep the count in `long long`, and allocate the merge buffer once outside the loops instead of creating vectors in every merge.
