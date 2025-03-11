/*
 * Copyright 2024 gQuery Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include <memory>

// Forward declaration for spdlog
namespace spdlog {
    class logger;
    namespace level {
        enum level_enum : int;
    }
}

namespace gquery {

// LogLevel Definition
enum class LogLevel { 
    Verbose,  // Detailed information (development)
    Error,    // Error conditions
    Fatal,    // Fatal errors that cause termination
    Invalid   // Invalid log level (for parsing errors)
};

// Convert between LogLevel and string representations
std::string ToString(LogLevel level);
LogLevel LogLevelFromString(const std::string &s);

// Initialize and shutdown logging
void InitLogging(LogLevel level, const std::string &logFile = "", 
                 bool logUtilization = false, bool useGPU = false);
void ShutdownLogging();

// Core logging functions
void Log(LogLevel level, const char *file, int line, const char *message);
[[noreturn]] void LogFatal(LogLevel level, const char *file, int line, const char *message);

// Template declarations for formatted logging
template <typename... Args>
void Log(LogLevel level, const char *file, int line, const char *format_str, Args&&... args);

template <typename... Args>
[[noreturn]] void LogFatal(LogLevel level, const char *file, int line, const char *format_str, Args&&... args);

// Logging macros
#define LOG_VERBOSE(...) \
    (gquery::internal::ShouldLog(gquery::LogLevel::Verbose) && \
     (gquery::Log(gquery::LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...) \
    (gquery::internal::ShouldLog(gquery::LogLevel::Error) && \
     (gquery::Log(gquery::LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...) \
    gquery::LogFatal(gquery::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

// Internal implementation details
namespace internal {
    bool ShouldLog(LogLevel level);
    std::shared_ptr<spdlog::logger> GetLogger();
    spdlog::level::level_enum ConvertToSpdlogLevel(LogLevel level);
}

} // namespace gquery

// Template implementations
#include <spdlog/fmt/fmt.h>

namespace gquery {

template <typename... Args>
void Log(LogLevel level, const char *file, int line, const char *format_str, Args&&... args) {
    try {
        std::string message = fmt::format(format_str, std::forward<Args>(args)...);
        Log(level, file, line, message.c_str());
    } catch (const fmt::format_error& e) {
        Log(LogLevel::Error, file, line, 
            fmt::format("Format error in log message: {}", e.what()).c_str());
        Log(level, file, line, format_str);
    }
}

template <typename... Args>
[[noreturn]] void LogFatal(LogLevel level, const char *file, int line, const char *format_str, Args&&... args) {
    try {
        std::string message = fmt::format(format_str, std::forward<Args>(args)...);
        LogFatal(level, file, line, message.c_str());
    } catch (const fmt::format_error& e) {
        Log(LogLevel::Error, file, line, 
            fmt::format("Format error in log message: {}", e.what()).c_str());
        LogFatal(level, file, line, format_str);
    }
}

} // namespace gquery
