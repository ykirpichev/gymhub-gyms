#include <vector>
using namespace std;

// Bottom-up merge sort that counts, for each element taken from the right half,
// how many left-half elements are still waiting (all of them are larger).
long long solve(vector<int>& nums) {
    int n = nums.size();
    vector<int> a(nums), buf(n);
    long long count = 0;
    for (int width = 1; width < n; width *= 2) {
        for (int lo = 0; lo < n; lo += 2 * width) {
            int mid = min(lo + width, n), hi = min(lo + 2 * width, n);
            int i = lo, j = mid, t = lo;
            while (i < mid && j < hi) {
                if (a[j] < a[i]) {
                    buf[t++] = a[j++];
                    count += mid - i;
                } else {
                    buf[t++] = a[i++];
                }
            }
            while (i < mid) buf[t++] = a[i++];
            while (j < hi) buf[t++] = a[j++];
        }
        swap(a, buf);
    }
    return count;
}
