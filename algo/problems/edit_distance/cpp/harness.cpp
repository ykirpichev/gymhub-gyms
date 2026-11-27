// Runs solution.cpp on the test cases. You do not need to edit this file.
#include "gym_io.hpp"
#include "solution.cpp"

int main() {
    return gym::serve([](const gym::Json& args) {
        auto a = gym::as<std::string>(args[0]);
        auto b = gym::as<std::string>(args[1]);
        return gym::timed([&] { return solve(a, b); });
    });
}
