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

#ifndef GQUERY_UTIL_CHECK_H
#define GQUERY_UTIL_CHECK_H

#include <util/log.h>
#include <functional>
#include <string>
#include <vector>

namespace gquery {

void PrintStackTrace();

// CHECK Macro Definitions
#define CHECK(x) (!(!(x) && (LOG_FATAL("Check failed: %s", #x), true)))

#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

// CHECK_IMPL Macro Definition
#define CHECK_IMPL(a, b, op)                                                           \
    do {                                                                               \
        auto va = a;                                                                   \
        auto vb = b;                                                                   \
        if (!(va op vb))                                                               \
            LOG_FATAL("Check failed: %s " #op " %s with %s = %s, %s = %s", #a, #b, #a, \
                      va, #b, vb);                                                     \
    } while (false) /* swallow semicolon */

#ifdef GQUERY_DEBUG_BUILD
#define DCHECK(x)       CHECK(x)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)
#else
#define EMPTY_CHECK \
    do {            \
    } while (false) /* swallow semicolon */

#define DCHECK(x) EMPTY_CHECK
#define DCHECK_EQ(a, b) EMPTY_CHECK
#define DCHECK_NE(a, b) EMPTY_CHECK
#define DCHECK_GT(a, b) EMPTY_CHECK
#define DCHECK_GE(a, b) EMPTY_CHECK
#define DCHECK_LT(a, b) EMPTY_CHECK
#define DCHECK_LE(a, b) EMPTY_CHECK
#endif

// CheckCallbackScope Definition
class CheckCallbackScope {
public:
    // CheckCallbackScope Public Methods
    CheckCallbackScope(std::function<std::string(void)> callback);
    ~CheckCallbackScope();

    CheckCallbackScope(const CheckCallbackScope &)            = delete;
    CheckCallbackScope &operator=(const CheckCallbackScope &) = delete;

    static void Fail();

private:
    // CheckCallbackScope Private Members
    static std::vector<std::function<std::string(void)>> callbacks;
};

} // namespace gquery

#endif // GQUERY_UTIL_CHECK_H
