#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace gquery {

// Define the log levels.
enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};

class Logger {
public:
    // Sets the global log level threshold.
    static void setLogLevel(LogLevel level) {
        getInstance().logLevel = level;
    }

    // Gets the current log level.
    static LogLevel getLogLevel() {
        return getInstance().logLevel;
    }

    // Logs a message if the log level is above the threshold.
    static void log(LogLevel level, const std::string &message,
                    const char *file, int line, const char *function) {
        if (level < getInstance().logLevel)
            return;

        // Get current time.
        auto    now   = std::chrono::system_clock::now();
        auto    timeT = std::chrono::system_clock::to_time_t(now);
        std::tm localTime;
#ifdef _WIN32
        localtime_s(&localTime, &timeT);
#else
        localtime_r(&timeT, &localTime);
#endif

        std::ostringstream oss;
        oss << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");
        std::string timeStr = oss.str();

        // Ensure thread-safe logging.
        std::lock_guard<std::mutex> lock(getInstance().mutex);

        // Print to std::cerr for errors, std::cout for other levels.
        std::ostream &out = (level == LogLevel::Error) ? std::cerr : std::cout;
        out << timeStr << " [" << levelToString(level) << "] "
            << "(" << file << ":" << line << " " << function << ") "
            << message << "\n";
    }

    // Converts the log level to a string.
    static const char *levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::Debug:
                return "DEBUG";
            case LogLevel::Info:
                return "INFO";
            case LogLevel::Warning:
                return "WARNING";
            case LogLevel::Error:
                return "ERROR";
        }
        return "UNKNOWN";
    }

private:
    Logger() : logLevel(LogLevel::Debug) {} // Default log level is Debug.
    static Logger &getInstance() {
        static Logger instance;
        return instance;
    }

    LogLevel   logLevel;
    std::mutex mutex;
};

// If C++20's std::source_location is available, define USE_SOURCE_LOCATION before including this header.
// Otherwise, the fallback macros use __FILE__, __LINE__, and __func__.
#ifdef USE_SOURCE_LOCATION
#include <source_location>
inline void logWithSourceLocation(LogLevel level, const std::string &message,
                                  const std::source_location &location = std::source_location::current()) {
    Logger::log(level, message, location.file_name(), location.line(), location.function_name());
}
#else
// Fallback macro if std::source_location is unavailable.
#define logWithSourceLocation(level, message) \
    Logger::log(level, message, __FILE__, __LINE__, __func__)
#endif

// Convenience macros for logging at different levels.
#define LOG_DEBUG(message)   logWithSourceLocation(LogLevel::Debug, message)
#define LOG_INFO(message)    logWithSourceLocation(LogLevel::Info, message)
#define LOG_WARNING(message) logWithSourceLocation(LogLevel::Warning, message)
#define LOG_ERROR(message)   logWithSourceLocation(LogLevel::Error, message)

} // namespace gquery
