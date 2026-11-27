# Hints for Largest rectangle in a histogram

## Hint 1

Fix the bar whose height limits the rectangle. How far can the rectangle extend to the left and to the right before it hits a lower bar?

## Hint 2

Keep a stack of indices whose heights increase from bottom to top. When bar `i` is lower than the top of the stack, the top bar cannot extend past `i`: pop it and compute its rectangle now.

## Hint 3

For a popped bar of height `h`, the right boundary is `i` and the left boundary is the new top of the stack (or `-1` if the stack is empty), so the width is `i - left - 1`. Add a bar of height 0 at the end so every bar gets popped.
