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

#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

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

inline std::string
demangle(const std::string& _inst)
{
    std::stringstream ss;
    ss << "\n"
       << "original :: \t" << _inst << "\n"
       << "demangle :: \t" << tim::demangle(_inst.c_str()) << "\n"
       << std::endl;
    return ss.str();
}

template <typename Tp>
inline std::string
demangle()
{
    std::stringstream ss;
    ss << "\n"
       << "original :: \t" << typeid(Tp).name() << "\n"
       << "demangle :: \t" << tim::demangle<Tp>() << "\n"
       << std::endl;
    return ss.str();
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class mangle_tests : public ::testing::Test
{};

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, sample)
{
    std::string function_name =
        "_ZN5amrex13launch_globalIZNS_3ForIlZ27doEsirkepovDepositionShapeNILi3EEvPKdS4_"
        "S4_S4_S4_S4_S4_PKiRKNS_6Array4IdEESA_SA_ldRKSt5arrayIdLm3EESC_NS_4Dim3EdlEUllE_"
        "vEEvT_OT0_mEUlvE_EEvSH_";

    // verify the demangling did not fail
    EXPECT_NE(function_name, tim::demangle(function_name))
        << details::demangle(function_name);
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, auto_tuple)
{
    using namespace tim::component;
    std::string demangled_name =
        "tim::auto_tuple<tim::component::wall_clock, tim::component::cuda_event>";
    std::string runtime_name = tim::demangle<tim::auto_tuple<wall_clock, cuda_event>>();
    EXPECT_EQ(demangled_name, runtime_name);
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, cuda_kernel)
{
    std::string function_name =
        "_ZN5amrex13launch_globalIZNS_3ForIlZ27doEsirkepovDepositionShapeNILi3EEvPKdS4_"
        "S4_S4_S4_S4_S4_PKiRKNS_6Array4IdEESA_SA_ldRKSt5arrayIdLm3EESC_NS_4Dim3EdlEUllE_"
        "vEEvT_OT0_mEUlvE_EEvSH_";

    // verify the demangling did not fail
    EXPECT_NE(function_name, tim::demangle(function_name))
        << details::demangle(function_name);
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, data_type)
{
    auto runtime_value  = tim::demangle<double>();
    auto demangled_name = std::string{ "double" };
    EXPECT_EQ(demangled_name, runtime_value) << details::demangle<double>();
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, struct_type)
{
    using namespace tim::component;

    auto runtime_value  = tim::demangle<wall_clock>();
    auto demangled_name = std::string{ "tim::component::wall_clock" };
    EXPECT_EQ(demangled_name, runtime_value) << details::demangle<wall_clock>();
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, variadic_struct_type)
{
    auto runtime_value  = tim::demangle<tim::auto_tuple<>>();
    auto demangled_name = std::string{ "tim::auto_tuple<>" };
    EXPECT_EQ(demangled_name, runtime_value) << details::demangle<tim::auto_tuple<>>();
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, function)
{
    using Type          = decltype(details::get_test_name);
    auto runtime_value  = tim::demangle<Type>();
    auto demangled_name = tim::demangle<std::string>() + " ()";
    EXPECT_EQ(demangled_name, runtime_value) << details::demangle<Type>();
}

//--------------------------------------------------------------------------------------//

namespace Foo
{
template <typename T>
class Bar;
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, undefined_type)
{
    auto runtime_value  = tim::try_demangle<Foo::Bar<double>>();
    auto demangled_name = std::string{ "Foo::Bar<double>" };
    EXPECT_EQ(demangled_name, runtime_value);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    tim::settings::verbose()     = 0;
    tim::settings::debug()       = false;
    tim::settings::json_output() = true;
    tim::timemory_init(&argc, &argv);
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    tim::settings::dart_type() = "peak_rss";
    // TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE("PEAK_RSS", ::tim::component::peak_rss);
    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
