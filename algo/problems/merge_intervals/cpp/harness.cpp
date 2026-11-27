// Runs solution.cpp on the test cases. You do not need to edit this file.
#include "gym_io.hpp"
#include "solution.cpp"

int main() {
    return gym::serve([](const gym::Json& args) {
        auto intervals = gym::as<std::vector<std::vector<int>>>(args[0]);
        return gym::timed([&] { return solve(intervals); });
    });
}
