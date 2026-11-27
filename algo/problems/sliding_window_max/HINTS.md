# Hints for Sliding window maximum

## Hint 1

If a value arrives that is at least as large as some earlier value still in the window, that earlier value can never be the answer again. You can throw it away.

## Hint 2

Keep a `collections.deque` of indices whose values are decreasing from front to back. Before pushing index `i`, pop from the back while the value there is `<= nums[i]`.

## Hint 3

The front of the deque is the window maximum. Drop it from the front when its index has slid out, that is when `front <= i - k`. Start emitting answers once `i >= k - 1`.

In C++, `std::deque<int>` of indices does the same job. Call `out.reserve(nums.size() - k + 1)` before filling the result (when `nums` is not empty).
