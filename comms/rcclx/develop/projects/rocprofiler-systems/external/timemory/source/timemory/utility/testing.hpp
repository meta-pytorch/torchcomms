//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file utility/testing.hpp
 * \headerfile utility/testing.hpp "timemory/utility/testing.hpp"
 * This is used for C++ testing of the timemory package
 *
 */

#pragma once

// C headers
#include <cassert>
#include <cstdint>
#include <cstdio>

// C++ headers
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// timemory headers
#include "timemory/backends/dmp.hpp"
#include "timemory/utility/macros.hpp"

//--------------------------------------------------------------------------------------//

// ASSERT_NEAR
// EXPECT_EQ
// EXPECT_FLOAT_EQ
// EXPECT_DOUBLE_EQ

#if !defined(EXPECT_EQ)
#    define EXPECT_EQ(lhs, rhs)                                                          \
        if(lhs != rhs)                                                                   \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << #lhs << " != " << #rhs << " @ line " << __LINE__ << " of "             \
               << __FILE__;                                                              \
            std::cerr << ss.str() << std::endl;                                          \
            throw std::runtime_error(ss.str());                                          \
        }
#endif

#if !defined(ASSERT_FALSE)
#    define ASSERT_FALSE(expr)                                                           \
        if(expr)                                                                         \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << "Expression: ( " << #expr << " ) "                                     \
               << "failed @ line " << __LINE__ << " of " << __FILE__;                    \
            std::cerr << ss.str() << std::endl;                                          \
            throw std::runtime_error(ss.str());                                          \
        }
#endif

#if !defined(ASSERT_TRUE)
#    define ASSERT_TRUE(expr)                                                            \
        if(!(expr))                                                                      \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << "Expression: !( " << #expr << " ) "                                    \
               << "failed @ line " << __LINE__ << " of " << __FILE__;                    \
            std::cerr << ss.str() << std::endl;                                          \
            throw std::runtime_error(ss.str());                                          \
        }
#endif

inline std::string
rank_prefix()
{
    std::stringstream ss;
    if(tim::dmp::is_initialized())
        ss << "[" << tim::dmp::rank() << "] ";
    return ss.str();
}

#define rank_cout std::cout << rank_prefix()

//--------------------------------------------------------------------------------------//
#define TEST_SUMMARY(argv_0, ntest_counter, nfail_counter)                               \
    {                                                                                    \
        std::stringstream rank_sout;                                                     \
        std::stringstream filler;                                                        \
        filler.fill('=');                                                                \
        filler << "#" << std::setw(78) << ""                                             \
               << "#";                                                                   \
        rank_sout << "\n... [\e[1;33mTESTING COMPLETED\e[0m] ... \n" << std::endl;       \
        rank_sout << filler.str() << "\n#\n";                                            \
        rank_sout << "#\t"                                                               \
                  << "[" << argv_0 << "] ";                                              \
        if(num_fail > 0)                                                                 \
            rank_sout << "\e[1;31mTESTS FAILED\e[0m: " << nfail_counter << '/'           \
                      << ntest_counter << std::endl;                                     \
        else                                                                             \
            rank_sout << "\e[1;36mTESTS PASSED\e[0m: "                                   \
                      << (ntest_counter - nfail_counter) << '/' << ntest_counter         \
                      << std::endl;                                                      \
        rank_sout << "#\n" << filler.str() << "\n" << std::endl;                         \
        rank_cout << rank_sout.str();                                                    \
    }

//--------------------------------------------------------------------------------------//
//  Usage:
//      CONFIGURE_TEST_SELECTOR(8)
//
//  Required for RUN_TEST
//
#define CONFIGURE_TEST_SELECTOR(total_tests)                                             \
    int           total_num_tests = total_tests;                                         \
    std::set<int> tests;                                                                 \
    if(argc == 1)                                                                        \
        for(int i = 0; i < total_tests; ++i)                                             \
            tests.insert(i + 1);                                                         \
    for(int i = 1; i < argc; ++i)                                                        \
        tests.insert(atoi(argv[i]));

//--------------------------------------------------------------------------------------//
//  Usage:
//
//      int num_fail = 0;   // tracks the number of failed tests
//      int num_test = 0;   // tracks the number of tests executed
//
//      try
//      {
//          RUN_TEST(test_serialize, num_test, num_fail);
//      }
//      catch(std::exception& e)
//      {
//          std::cerr << e.what() << std::endl;
//      }
//
#define RUN_TEST(test_num, func, ntest_counter, nfail_counter)                           \
    {                                                                                    \
        if(test_num > total_num_tests || tests.count(test_num) != 0)                     \
        {                                                                                \
            if(test_num > total_num_tests)                                               \
                printf("Warning! Test %i is greater than the specified number of "       \
                       "tests: "                                                         \
                       "%i\n",                                                           \
                       test_num, total_num_tests);                                       \
            try                                                                          \
            {                                                                            \
                ntest_counter += 1;                                                      \
                func();                                                                  \
            } catch(std::exception & e)                                                  \
            {                                                                            \
                std::cerr << e.what() << std::endl;                                      \
                nfail_counter += 1;                                                      \
            }                                                                            \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            printf("\n... Skipping test #%i ...\n\n", test_num);                         \
        }                                                                                \
    }

//--------------------------------------------------------------------------------------//
