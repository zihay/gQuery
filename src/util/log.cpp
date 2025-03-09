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

#include <util/log.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <iostream>
#include <mutex>
#include <chrono>
#include <atomic>
#include <thread>

namespace gquery {

namespace {
    std::shared_ptr<spdlog::logger> logger;
    LogLevel currentLogLevel = LogLevel::Error;
    std::atomic<bool> shuttingDown{false};
    std::thread utilization_thread;
    std::mutex logger_mutex;
}

// Convert LogLevel to string
std::string ToString(LogLevel level) {
    switch (level) {
        case LogLevel::Verbose: return "VERBOSE";
        case LogLevel::Error:   return "ERROR";
        case LogLevel::Fatal:   return "FATAL";
        default:                return "UNKNOWN";
    }
}

// Parse string to LogLevel
LogLevel LogLevelFromString(const std::string &s) {
    if (s == "verbose") return LogLevel::Verbose;
    else if (s == "error") return LogLevel::Error;
    else if (s == "fatal") return LogLevel::Fatal;
    return LogLevel::Invalid;
}

// Initialize the logging system
void InitLogging(LogLevel level, const std::string &logFile, bool logUtilization, bool useGPU) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    currentLogLevel = level;
    
    // Create a vector of sinks
    std::vector<spdlog::sink_ptr> sinks;
    
    // Always add console output
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
    sinks.push_back(console_sink);
    
    // Add file output if requested
    if (!logFile.empty()) {
        try {
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFile, true);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%s:%#] %v");
            sinks.push_back(file_sink);
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Failed to open logfile: " << ex.what() << std::endl;
        }
    }
    
    // Create and register logger
    logger = std::make_shared<spdlog::logger>("gquery", sinks.begin(), sinks.end());
    
    // Set log level
    logger->set_level(internal::ConvertToSpdlogLevel(level));
    
    // Register as default logger
    spdlog::set_default_logger(logger);
    
    // Log initialization message
    logger->info("Logging system initialized with level: {}", ToString(level));
    
    // Start utilization monitoring if requested
    if (logUtilization) {
        shuttingDown = false;
        
        // Create a thread to monitor and log system utilization
        bool useGPUCopy = useGPU;  // Copy for lambda capture
        utilization_thread = std::thread([useGPUCopy]() {
            while (!shuttingDown) {
                // Simple memory usage reporting (platform-independent parts)
                if (internal::ShouldLog(LogLevel::Verbose)) {
                    logger->info("System utilization: Memory stats placeholder");
                    
                    // GPU stats if requested and available
                    if (useGPUCopy) {
                        logger->info("GPU utilization placeholder");
                    }
                }
                
                // Sleep for a while
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        });
    }
}

// Shut down the logging system
void ShutdownLogging() {
    // Signal thread to stop
    shuttingDown = true;
    
    // Wait for thread to finish if it's running
    if (utilization_thread.joinable()) {
        utilization_thread.join();
    }
    
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    // Flush all log messages
    if (logger) {
        logger->info("Logging system shutting down");
        logger->flush();
        spdlog::drop_all(); // Release all loggers
        logger.reset();
    }
}

// Core logging function
void Log(LogLevel level, const char *file, int line, const char *message) {
    if (!internal::ShouldLog(level))
        return;
        
    // Extract filename from path
    const char* filename = file;
    const char* lastSlash = strrchr(file, '/');
    if (lastSlash) {
        filename = lastSlash + 1;
    }
    
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    if (logger) {
        // Log with appropriate level
        switch (level) {
            case LogLevel::Verbose:
                logger->debug("[{}:{}] {}", filename, line, message);
                break;
            case LogLevel::Error:
                logger->error("[{}:{}] {}", filename, line, message);
                break;
            case LogLevel::Fatal:
                logger->critical("[{}:{}] {}", filename, line, message);
                break;
            default:
                logger->warn("[{}:{}] UNKNOWN LEVEL: {}", filename, line, message);
                break;
        }
    } else {
        // Fallback if logger is not initialized
        std::cerr << ToString(level) << " [" << filename << ":" << line << "] " 
                  << message << std::endl;
    }
}

// Fatal error logging function
[[noreturn]] void LogFatal(LogLevel level, const char *file, int line, const char *message) {
    Log(level, file, line, message);
    
    // Ensure all logs are flushed
    if (logger) {
        logger->flush();
    }
    
    // Terminate the program
    std::abort();
}

namespace internal {

// Check if a message at the given level should be logged
bool ShouldLog(LogLevel level) {
    return level >= currentLogLevel;
}

// Get the current logger
std::shared_ptr<spdlog::logger> GetLogger() {
    std::lock_guard<std::mutex> lock(logger_mutex);
    return logger;
}

// Convert LogLevel to spdlog level
spdlog::level::level_enum ConvertToSpdlogLevel(LogLevel level) {
    switch (level) {
        case LogLevel::Verbose: return spdlog::level::debug;
        case LogLevel::Error:   return spdlog::level::err;
        case LogLevel::Fatal:   return spdlog::level::critical;
        default:                return spdlog::level::info;
    }
}

} // namespace internal

} // namespace gquery
