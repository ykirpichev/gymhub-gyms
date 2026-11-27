#include <functional>
#include <queue>
#include <utility>
#include <vector>
using namespace std;

vector<long long> solve(int n, vector<vector<int>>& edges, int src) {
    // Compressed adjacency lists (CSR): offsets per node, then (target, weight) pairs.
    vector<int> start(n + 1, 0);
    for (const auto& e : edges) start[e[0] + 1]++;
    for (int u = 0; u < n; ++u) start[u + 1] += start[u];
    vector<pair<int, int>> adj(edges.size());
    vector<int> fill(start.begin(), start.end() - 1);
    for (const auto& e : edges) adj[fill[e[0]]++] = {e[1], e[2]};

    const long long UNSETTLED = -1;
    vector<long long> dist(n, UNSETTLED);
    vector<long long> best(n, -1);  // best tentative distance pushed so far
    using Item = pair<long long, int>;
    priority_queue<Item, vector<Item>, greater<Item>> heap;
    heap.push({0, src});
    best[src] = 0;
    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (dist[u] != UNSETTLED) continue;  // stale entry
        dist[u] = d;
        for (int i = start[u]; i < start[u + 1]; ++i) {
            auto [v, w] = adj[i];
            long long nd = d + w;
            if (dist[v] == UNSETTLED && (best[v] < 0 || nd < best[v])) {
                best[v] = nd;
                heap.push({nd, v});
            }
        }
    }
    return dist;
}
