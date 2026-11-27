// Runs solution.cpp on the test cases. You do not need to edit this file.
#include <unordered_map>

#include "gym_io.hpp"
#include "solution.cpp"

namespace {

struct Op {
    bool is_get;
    int key;
    int value;
};

// Keys arrive as ints, or as strings in a few tests. Strings get stable int ids
// that do not collide with the int keys used by the tests.
std::vector<Op> parse_ops(const gym::Json& json) {
    std::unordered_map<std::string, int> ids;
    std::vector<Op> ops;
    ops.reserve(json.size());
    for (const gym::Json& op : json.items) {
        auto name = gym::as<std::string>(op[0]);
        int key;
        if (op[1].is_string()) {
            auto [it, inserted] = ids.try_emplace(op[1].text, 1'000'000'000 + static_cast<int>(ids.size()));
            key = it->second;
        } else {
            key = gym::as<int>(op[1]);
        }
        if (name == "get")
            ops.push_back({true, key, 0});
        else if (name == "put")
            ops.push_back({false, key, gym::as<int>(op[2])});
        else
            throw std::runtime_error("unknown op " + name);
    }
    return ops;
}

}  // namespace

int main() {
    return gym::serve([](const gym::Json& args) {
        auto capacity = gym::as<int>(args[0]);
        auto ops = parse_ops(args[1]);
        return gym::timed([&] {
            LRUCache cache(capacity);
            std::vector<int> results;
            for (const Op& op : ops) {
                if (op.is_get)
                    results.push_back(cache.get(op.key));
                else
                    cache.put(op.key, op.value);
            }
            return results;
        });
    });
}
