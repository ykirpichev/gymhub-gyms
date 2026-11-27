# Prefix suggestions

**Level:** medium   **Teaches:** tries, binary search over sorted strings, prefix ranges

## Task

You have a dictionary of `words` and a list of `prefixes` a user typed. For each prefix, return up to three dictionary words that start with it, smallest first in plain string order. Duplicates in the dictionary count once.

## Arguments

| Name       | Type              | Role                                                        |
| ---------- | ----------------- | ----------------------------------------------------------- |
| `words`    | `list[str]`       | the dictionary, lowercase letters, possibly with duplicates |
| `prefixes` | `list[str]`       | queries, lowercase letters; a prefix may be empty           |
| returns    | `list[list[str]]` | one list of at most 3 words per prefix                      |

## Rules

- A word starts with a prefix when `word.startswith(prefix)`. Every word starts with the empty prefix, and a word starts with itself.
- Order the suggestions by Python string comparison (`"app" < "apple" < "apply"`).
- Fewer than three matches: return them all. None: return `[]`.

## Example

```
words    = ["mouse", "mobile", "moneypot", "monitor", "mousepad", "money", "monitor"]
prefixes = ["mo", "mou", "mous", "x", ""]
"mo"   -> ["mobile", "money", "moneypot"]
"mou"  -> ["mouse", "mousepad"]
"mous" -> ["mouse", "mousepad"]
"x"    -> []
""     -> ["mobile", "money", "moneypot"]
```

## How to think about it

In a sorted list, all words with a given prefix sit next to each other, and the first of them is where the prefix itself would be inserted. So sort once, binary-search each prefix, and read up to three words forward while they still match. A trie that stores the three smallest words at every node answers each query by walking the prefix, which is the shape autocomplete services use.

## Tests

An empty dictionary, an empty prefix, a prefix longer than every word, a prefix equal to a word, duplicates, and random inputs up to 2,000 words. Benchmark input: 100,000 words and 100,000 prefixes. `bench` fails if the median is over **400 ms** in Python; scanning the whole dictionary for each prefix takes minutes.
