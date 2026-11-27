// Runs solution.cpp on the test cases. You do not need to edit this file.
#include "gym_io.hpp"
#include "solution.cpp"

int main() {
    return gym::serve([](const gym::Json& args) {
        auto n = gym::as<int>(args[0]);
        std::vector<Op> ops;
        ops.reserve(args[1].size());
        for (const gym::Json& op : args[1].items) {
            auto name = gym::as<std::string>(op[0]);
            if (name != "union" && name != "same") throw std::runtime_error("unknown op " + name);
            ops.push_back(Op{name == "union", gym::as<int>(op[1]), gym::as<int>(op[2])});
        }
        return gym::timed([&] { return solve(n, ops); });
    });
}
