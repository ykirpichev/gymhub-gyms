## Hint 1
Lengths come first. `floor(n * fraction + 1e-12)` for train, validation and test, then hand the leftover indices out one at a time in that same order until the three lengths sum to `n`.

## Hint 2
Build `rng = random.Random(seed)`. With no labels, shuffle `range(n)` and slice it. With labels, shuffle inside each class (classes in sorted order, same rng) so the seed changes which rows move, not the counts.

## Hint 3
Give each class a count triple that sums to the class size and stays within 1 of `class_size * split_size / n`. A class of 1 is only in train, a class of 2 is in train and validation, and a class of 3 or more has at least one row in every split. Move a single row between splits until the class counts add up to the global lengths.
