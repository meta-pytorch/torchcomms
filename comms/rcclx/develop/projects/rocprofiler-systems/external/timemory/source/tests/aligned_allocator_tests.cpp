// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/backends.hpp"
#include "timemory/config.hpp"
#include "timemory/environment/definition.hpp"
#include "timemory/ert/aligned_allocator.hpp"
#include "timemory/settings.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include <x86intrin.h>

template <typename Tp>
using aligned_vector_t = std::vector<Tp, tim::ert::aligned_allocator<Tp>>;

static const std::size_t niter    = 5;
static const float       fepsilon = std::numeric_limits<float>::epsilon();
static const double      depsilon = std::numeric_limits<double>::epsilon();

//--------------------------------------------------------------------------------------//
//  all types, unless overloaded, are treated as non-intrinsic
//
template <typename Tp>
struct is_intrinsic : std::false_type
{};

// overload for intrinsic type __m128 (four packed 32-bit floats)
template <>
struct is_intrinsic<__m128> : std::true_type
{
    static constexpr std::size_t entries = 4;
};

// overload for intrinsic type __m128 (eight packed 32-bit floats)
template <>
struct is_intrinsic<__m256> : std::true_type
{
    static constexpr std::size_t entries = 8;
};

// overload for intrinsic type __m128d (two packed 64-bit floats)
template <>
struct is_intrinsic<__m128d> : std::true_type
{
    static constexpr std::size_t entries = 2;
};

// overload for intrinsic type __m256d (four packed 64-bit floats)
template <>
struct is_intrinsic<__m256d> : std::true_type
{
    static constexpr std::size_t entries = 4;
};

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}
//--------------------------------------------------------------------------------------//
// shorthand for check if type is an intrinsic type
//
template <typename Tp, bool _Val = true>
using enable_intrinsic_t =
    typename std::enable_if<(is_intrinsic<Tp>::value == _Val), int>::type;

//--------------------------------------------------------------------------------------//
// print function for containers with non-intrinsic types
//
template <typename Tp, typename... _ImplicitArgs,
          template <typename, typename...> class _Vec, enable_intrinsic_t<Tp, false> = 0>
void
print(std::ostream& os, const std::string& label, const _Vec<Tp, _ImplicitArgs...>& vec)
{
    using vec_t = _Vec<Tp, _ImplicitArgs...>;
    std::stringstream ss;
    ss.precision(1);
    ss << std::setw(16) << std::left << label
       << " (alignment: " << (8 * std::alignment_of<vec_t>::value) << ") : " << std::fixed
       << std::right;
    auto nsize = std::distance(vec.begin(), vec.end());
    for(auto itr = vec.begin(); itr != vec.end(); ++itr)
    {
        auto n = std::distance(vec.begin(), itr);
        ss << "[" << n << "] = " << std::setw(4) << *itr;
        if(n + 1 < nsize)
            ss << ", ";
    }
    os << ss.str() << std::endl;
}

//--------------------------------------------------------------------------------------//
// print function for containers with intrinsic types
//
template <typename Tp, typename... _ImplicitArgs,
          template <typename, typename...> class _Vec, enable_intrinsic_t<Tp> = 0>
void
print(std::ostream& os, const std::string& label, const _Vec<Tp, _ImplicitArgs...>& vec)
{
    using vec_t = _Vec<Tp, _ImplicitArgs...>;
    std::stringstream ss;
    ss.precision(1);
    ss << std::setw(16) << std::left << label
       << " (alignment: " << (8 * std::alignment_of<vec_t>::value) << ") : " << std::fixed
       << std::right;
    auto entries = is_intrinsic<Tp>::entries;
    auto nsize   = std::distance(vec.begin(), vec.end()) * entries;
    for(auto itr = vec.begin(); itr != vec.end(); ++itr)
    {
        auto n = std::distance(vec.begin(), itr) * entries;
        for(std::size_t i = 0; i < is_intrinsic<Tp>::entries; ++i)
        {
            ss << "[" << n << "] = " << std::setw(4) << (*itr)[i];
            if(n + 1 < nsize)
                ss << ", ";
            n += 1;
        }
    }
    os << ss.str() << std::endl;
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class aligned_allocator_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(aligned_allocator_tests, m128)
{
    aligned_vector_t<__m128> lhs;
    aligned_vector_t<__m128> rhs;
    aligned_vector_t<float>  factors(8);
    std::vector<float>       intrin_check(4, 0.0);
    std::vector<float>       solution(niter + 3, 0.0);

    // generate the factors that will be added together
    float init = 0.0;
    std::generate(factors.begin(), factors.end(), [&] {
        float tmp = init;
        init += 1.0f;
        return tmp;
    });

    // generate the expected solution
    init = 4.0;
    std::generate(solution.begin(), solution.end(), [&] {
        float tmp = init;
        init += 2.0f;
        return tmp;
    });

    // print the solution and initial factors for record-keeping
    // std::cout << "\nsizeof(__m128) = " << sizeof(__m128) << std::endl;
    // details::print(std::cout, "initial factors", factors);
    // details::print(std::cout, "solution", solution);

    for(std::size_t i = 0; i < niter; ++i)
    {
        // print current iteration
        // std::cout << "\niteration " << i << ":\n";

        // create two __m128 packed vectors to add together
        lhs.push_back(_mm_set_ps(factors[3], factors[2], factors[1], factors[0]));
        rhs.push_back(_mm_set_ps(factors[7], factors[6], factors[5], factors[4]));

        // print the current __m128 packed vectors being added
        details::print(std::cout, "\t  lhs", lhs);
        details::print(std::cout, "\t+ rhs", rhs);

        // do the intrinsic operation
        __m128 intrin_result = _mm_add_ps(lhs.back(), rhs.back());

        // convert to standard vector
        for(uint64_t j = 0; j < 4; ++j)
            intrin_check[j] = intrin_result[j];

        // print the check
        details::print(std::cout, "\t= intrin_check", intrin_check);

        // verify the operation succeeded
        for(uint64_t j = 0; j < 4; ++j)
            ASSERT_NEAR(intrin_check[j], *(solution.begin() + i + j), fepsilon);

        // add 1.0 to all the factors
        for(auto& itr : factors)
            itr += 1.0f;

        // erase the previous data so we can print the whole vector in next iteration
        lhs.erase(lhs.begin());
        rhs.erase(rhs.begin());
    }
    // std::cout << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(aligned_allocator_tests, m128d)
{
    aligned_vector_t<__m128d> lhs;
    aligned_vector_t<__m128d> rhs;
    aligned_vector_t<double>  factors(4);
    std::vector<double>       intrin_check(2, 0.0);
    std::vector<double>       solution(niter + 1, 0.0);

    // generate the factors that will be added together
    double init = 0.0;
    std::generate(factors.begin(), factors.end(), [&] {
        double tmp = init;
        init += 1.0;
        return tmp;
    });

    // generate the expected solution
    init = 2.0;
    std::generate(solution.begin(), solution.end(), [&] {
        double tmp = init;
        init += 2.0;
        return tmp;
    });

    // print the solution and initial factors for record-keeping
    // std::cout << "\nsizeof(__m128d) = " << sizeof(__m128d) << std::endl;
    // details::print(std::cout, "initial factors", factors);
    // details::print(std::cout, "solution", solution);

    for(std::size_t i = 0; i < niter; ++i)
    {
        // print current iteration
        // std::cout << "\niteration " << i << ":\n";

        // create two __m128 packed vectors to add together
        lhs.push_back(_mm_set_pd(factors[1], factors[0]));
        rhs.push_back(_mm_set_pd(factors[3], factors[2]));

        // print the current __m128 packed vectors being added
        details::print(std::cout, "\t  lhs", lhs);
        details::print(std::cout, "\t+ rhs", rhs);

        // do the intrinsic operation
        __m128d intrin_result = _mm_add_pd(lhs.back(), rhs.back());

        // convert to standard vector
        for(uint64_t j = 0; j < 2; ++j)
            intrin_check[j] = intrin_result[j];

        // print the check
        details::print(std::cout, "\t= intrin_check", intrin_check);

        // verify the operation succeeded
        for(uint64_t j = 0; j < 2; ++j)
            ASSERT_NEAR(intrin_check[j], *(solution.begin() + i + j), depsilon);

        // add 1.0 to all the factors
        for(auto& itr : factors)
            itr += 1.0;

        // erase the previous data so we can print the whole vector in next iteration
        lhs.erase(lhs.begin());
        rhs.erase(rhs.begin());
    }
    // std::cout << std::endl;
}

//--------------------------------------------------------------------------------------//
