#pragma once

#include <cstdlib>
#include <iostream>
#include <source_location>

namespace gquery {

// Asserts a condition and prints a detailed message if the assertion fails.
// The function uses std::source_location to capture file, line, and function information automatically.
inline void assert_with_message(bool                        condition,
                                const char                 *msg      = "Assertion failed",
                                const std::source_location &location = std::source_location::current()) {
    if (!condition) {
        std::cerr << location.file_name() << ":" << location.line()
                  << " in function " << location.function_name()
                  << " - " << msg << std::endl;
        std::abort();
    }
}

// Macro wrapper for convenience
#define ASSERT(cond, msg) assert_with_message((cond), (msg), std::source_location::current())

} // namespace gquery
