## Hint 1
Write one recursive helper that walks a value (dict values, list items, strings) and either collects the referenced ids or substitutes outputs. Use it twice: once to validate, once to resolve.

## Hint 2
Validation needs only one pass: keep a set of ids seen so far; every reference in a step must already be in it, and the step's own id must not be.

## Hint 3
Keep a dict `id -> trace entry`. For each step, the blocked ids are the referenced ids whose entry status is not `"ok"`. A skipped step is not `"ok"` either, so skips propagate down a dependency chain without extra code.
