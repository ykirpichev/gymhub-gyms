#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

vector<string> solve(vector<string>& words, int k) {
    unordered_map<string, int> counts;
    counts.reserve(words.size());
    for (const string& word : words) counts[word]++;
    vector<pair<int, const string*>> ranked;
    ranked.reserve(counts.size());
    for (const auto& [word, count] : counts) ranked.push_back({count, &word});
    auto better = [](const auto& x, const auto& y) {
        return x.first != y.first ? x.first > y.first : *x.second < *y.second;
    };
    partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(), better);
    vector<string> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i) out.push_back(*ranked[i].second);
    return out;
}
