#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> solve(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;
    for (auto& interval : intervals) {
        if (!merged.empty() && interval[0] <= merged.back()[1])
            merged.back()[1] = max(merged.back()[1], interval[1]);
        else
            merged.push_back(interval);
    }
    return merged;
}
