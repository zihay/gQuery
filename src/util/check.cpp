/*
 * Copyright 2024 gQuery Contributors
 * Copyright 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys
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

// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <string.h>
#include <util/check.h>

#include <cstdlib>
#include <iostream>

#ifdef PBRT_IS_OSX
#include <cxxabi.h>
#include <execinfo.h>
#endif
#ifdef PBRT_IS_LINUX
#include <cxxabi.h>
#include <execinfo.h>
#endif
#ifdef PBRT_IS_WINDOWS
// clang-format off
#include <windows.h>
#include <tchar.h>
#include <process.h>
#include <dbghelp.h>
// clang-format on
#endif

namespace gquery {

void PrintStackTrace() {
#if defined(__APPLE__) || defined(__linux__)
    std::cerr << "Stack trace not implemented for this platform\n";
#elif defined(_WIN32)
    std::cerr << "Stack trace not implemented for this platform\n";
#else
    std::cerr << "Stack trace not implemented for this platform\n";
#endif
}

static std::vector<std::function<std::string(void)>> callbacks;

void CheckCallbackScope::Fail() {
    PrintStackTrace();

    std::string message;
    for (auto iter = callbacks.rbegin(); iter != callbacks.rend(); ++iter)
        message += (*iter)();
    fprintf(stderr, "%s\n\n", message.c_str());

    abort();
}

std::vector<std::function<std::string(void)>> CheckCallbackScope::callbacks;

CheckCallbackScope::CheckCallbackScope(std::function<std::string(void)> callback) {
    callbacks.push_back(std::move(callback));
}

CheckCallbackScope::~CheckCallbackScope() {
    CHECK_GT(callbacks.size(), 0);
    callbacks.pop_back();
}

} // namespace gquery
