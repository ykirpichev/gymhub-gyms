#include <deque>
#include <vector>
using namespace std;

vector<int> solve(vector<int>& nums, int k) {
    vector<int> out;
    if (nums.empty()) return out;
    out.reserve(nums.size() - k + 1);
    deque<int> window;  // indices, values decreasing from front to back
    for (int i = 0; i < (int)nums.size(); ++i) {
        while (!window.empty() && nums[window.back()] <= nums[i]) window.pop_back();
        window.push_back(i);
        if (window.front() <= i - k) window.pop_front();
        if (i >= k - 1) out.push_back(nums[window.front()]);
    }
    return out;
}
