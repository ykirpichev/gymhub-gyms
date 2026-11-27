// Runs solution.cpp on the test cases. You do not need to edit this file.
#include "gym_io.hpp"
#include "solution.cpp"

int main() {
    return gym::serve([](const gym::Json& args) {
        auto words = gym::as<std::vector<std::string>>(args[0]);
        auto k = gym::as<int>(args[1]);
        return gym::timed([&] { return solve(words, k); });
    });
}
