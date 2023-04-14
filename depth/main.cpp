#include <fmt/core.h>
#include <stdexcept>

#include "app.h"

int main() try {
    App app;
    app.run();
    return EXIT_SUCCESS;
} catch (std::exception const &e) {
    fmt::print(stderr, "{}\n", e.what());
    return EXIT_FAILURE;
}
