// Runs solution.cpp on the test cases. You do not need to edit this file.
#include "gym_io.hpp"
#include "solution.cpp"

int main() {
    return gym::serve([](const gym::Json& args) {
        auto n = gym::as<int>(args[0]);
        auto edges = gym::as<std::vector<std::vector<int>>>(args[1]);
        auto src = gym::as<int>(args[2]);
        return gym::timed([&] { return solve(n, edges, src); });
    });
}
