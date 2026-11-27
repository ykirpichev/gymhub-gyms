#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
using namespace std;

int solve(string& a, string& b) {
    const string& outer = a.size() >= b.size() ? a : b;
    const string& inner = a.size() >= b.size() ? b : a;
    int m = inner.size();
    vector<int> row(m + 1);  // one rolling row plus a saved diagonal value
    iota(row.begin(), row.end(), 0);
    for (int i = 1; i <= (int)outer.size(); ++i) {
        int diagonal = row[0];
        row[0] = i;
        char ca = outer[i - 1];
        for (int j = 1; j <= m; ++j) {
            int up = row[j];
            row[j] = ca == inner[j - 1] ? diagonal : 1 + min({diagonal, up, row[j - 1]});
            diagonal = up;
        }
    }
    return row[m];
}
