#include <numeric>
#include <utility>
#include <vector>
using namespace std;

struct Op {
    bool is_union;
    int a, b;
};

vector<bool> solve(int n, vector<Op>& ops) {
    vector<int> parent(n), size(n, 1);
    iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    };
    vector<bool> answers;
    for (const Op& op : ops) {
        int ra = find(op.a), rb = find(op.b);
        if (!op.is_union) {
            answers.push_back(ra == rb);
        } else if (ra != rb) {
            if (size[ra] < size[rb]) swap(ra, rb);
            parent[rb] = ra;
            size[ra] += size[rb];
        }
    }
    return answers;
}
