# Hints for Prefix suggestions

## Hint 1

Remove duplicates and sort the dictionary. Where in the sorted list do the words that start with `"mo"` sit?

## Hint 2

They form one contiguous block, and it starts at `bisect_left(sorted_words, prefix)`.

## Hint 3

From that position, take words while they start with the prefix, at most three. Each query costs one binary search plus three string checks.
