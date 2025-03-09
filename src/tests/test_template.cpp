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

#include <gtest/gtest.h>
#include <util/check.h>
#include <util/log.h>

// Test fixture (for tests that share setup/teardown)
class MyComponentTest : public ::testing::Test {
protected:
    // Setup called before each test
    void SetUp() override {
        // Initialize test resources
        gquery::InitLogging(gquery::LogLevel::Verbose);
    }

    // Teardown called after each test
    void TearDown() override {
        // Clean up test resources
        gquery::ShutdownLogging();
    }

    // Common test objects can be defined here
};

// Simple test without fixture
TEST(SimpleTest, BasicAssertions) {
    // Arrange
    int x = 42;
    
    // Act & Assert
    EXPECT_EQ(x, 42);
    EXPECT_GT(x, 0);
}

// Test using the fixture
TEST_F(MyComponentTest, TestWithFixture) {
    // This test has access to anything defined in the fixture
    EXPECT_TRUE(true);
}

// Parameterized test example
class ParameterizedTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(ParameterizedTest, Addition) {
    // Get parameters
    auto [a, b, expected] = GetParam();
    
    // Test
    EXPECT_EQ(a + b, expected);
}

// Define parameters for the parameterized test
INSTANTIATE_TEST_SUITE_P(
    BasicMath,
    ParameterizedTest,
    ::testing::Values(
        std::make_tuple(1, 1, 2),
        std::make_tuple(2, 3, 5),
        std::make_tuple(10, -5, 5)
    )
); 