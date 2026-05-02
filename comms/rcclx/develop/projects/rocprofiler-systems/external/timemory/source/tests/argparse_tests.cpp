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

#include "timemory/config.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/variadic/macros.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

static int         test_argc = 0;
static char**      test_argv = nullptr;
static int         _margc    = 0;
static char**      _margv    = nullptr;
static int         _log_once = 0;
static const char* _arg0     = "./test";

using mutex_t    = std::mutex;
using lock_t     = std::unique_lock<mutex_t>;
using argparse_t = tim::argparse::argument_parser;
using argerror_t = typename argparse_t::result_type;

//--------------------------------------------------------------------------------------//

namespace std
{
std::ostream&
operator<<(std::ostream& os, const std::vector<std::string>& v)
{
    for(size_t i = 0; i < v.size(); ++i)
    {
        os << v.at(i);
        if(i + 1 < v.size())
            os << ", ";
    }
    return os;
}
}  // namespace std
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
//
//--------------------------------------------------------------------------------------//
//
inline auto&
parse_function()
{
    using function_t           = std::function<argerror_t(argparse_t&, int&, char**&)>;
    static function_t _functor = [](argparse_t& parser, int& _ac, char**& _av) {
        return parser.parse(_ac, _av, 0);
    };
    return _functor;
}
//
//--------------------------------------------------------------------------------------//
//
inline void
cleanup()
{
    if(test_argv)
    {
        for(int i = 0; i < test_argc; ++i)
            free(test_argv[i]);
        delete[] test_argv;
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Args>
inline auto
parse(argparse_t& parser, Args&&... args)
{
    cleanup();

    parser.enable_help();
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output")
        .max_count(1);
    parser.add_argument().names({ "--debug" }).description("Debug output").count(0);
    parser.add_argument()
        .names({ "-e", "--error" })
        .description("All warnings produce runtime errors")
        .count(0);
    parser.add_argument()
        .names({ "-o", "--output" })
        .description("Enable binary-rewrite to new executable")
        .count(1);
    parser.add_argument()
        .names({ "-I", "-R", "--function-include" })
        .description("Regex for selecting functions");
    parser.add_argument()
        .names({ "-E", "--function-exclude" })
        .description("Regex for excluding functions");
    parser.add_argument()
        .names({ "-MI", "-MR", "--module-include" })
        .description("Regex for selecting modules/files/libraries");
    parser.add_argument()
        .names({ "-ME", "--module-exclude" })
        .description("Regex for excluding modules/files/libraries");
    parser.add_argument()
        .names({ "-m", "--main-function" })
        .description("The primary function to instrument around, e.g. 'main'")
        .count(1);
    parser.add_argument()
        .names({ "-l", "--instrument-loops" })
        .description("Instrument at the loop level")
        .count(0);
    parser.add_argument()
        .names({ "-C", "--collection" })
        .description("Include text file(s) listing function to include or exclude "
                     "(prefix with '!' to exclude)");
    parser.add_argument()
        .names({ "-P", "--collection-path" })
        .description("Additional path(s) to folders containing collection files");
    parser.add_argument()
        .names({ "-s", "--stubs" })
        .description("Instrument with library stubs for LD_PRELOAD")
        .count(0);
    parser.add_argument()
        .names({ "-L", "--library" })
        .description("Library with instrumentation routines (default: \"libtimemory\")")
        .count(1);
    parser.add_argument()
        .names({ "-S", "--stdlib" })
        .description("Enable instrumentation of C++ standard library functions. Use with "
                     "caution because timemory uses the STL internally!")
        .max_count(1);
    parser.add_argument()
        .names({ "-p", "--pid" })
        .description("Connect to running process")
        .count(1);
    parser.add_argument()
        .names({ "-d", "--default-components" })
        .description("Default components to instrument");
    parser.add_argument()
        .names({ "-M", "--mode" })
        .description("Instrumentation mode. 'trace' mode is immutable, 'region' mode is "
                     "mutable by timemory library interface")
        .choices({ "trace", "region" })
        .count(1);
    parser.add_argument()
        .names({ "--prefer" })
        .description("Prefer this library types when available")
        .choices({ "shared", "static" })
        .count(1);
    parser
        .add_argument({ "--mpi" },
                      "Enable MPI support (requires timemory built w/ MPI and GOTCHA "
                      "support)")
        .count(0);
    parser.add_argument({ "--mpip" }, "Enable MPI profiling via GOTCHA")
        .count(0)
        .max_count(1);
    parser.add_argument({ "--ompt" }, "Enable OpenMP profiling via OMPT")
        .count(0)
        .max_count(1);
    parser.add_argument({ "--load" }, "Extra libraries to load");

    test_argc    = sizeof...(Args) + 1;
    test_argv    = new char*[test_argc];
    test_argv[0] = strdup(_arg0);
    int i        = 1;
    TIMEMORY_FOLD_EXPRESSION((test_argv[i] = strdup(std::forward<Args>(args)), ++i));

    auto err = parse_function()(parser, test_argc, test_argv);

    // only log the help function once
    auto _log_id = _log_once++;

    if(err)
    {
        std::cerr << err << std::endl;
        parser.print_help("-- <ERROR MESSAGE>");
        return -1;
    }

    if(_log_id == 0)
        parser.print_help("-- <LOGGING>");

    return 0;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace details
//
//--------------------------------------------------------------------------------------//
//
class argparse_tests : public ::testing::Test
{
protected:
    void        SetUp() override {}
    std::string extra_help = "-- <CMD> <ARGS>";

    static void SetUpTestSuite() { tim::timemory_init(_margc, _margv); }
    static void TearDownTestSuite() { tim::timemory_finalize(); }
};

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, basic)
{
    argparse_t parser(details::get_test_name());
    auto       ret = details::parse(parser, "-I", "one", "char", "short", "option");
    auto       arg = parser.get<std::vector<std::string>>("I");

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(arg.size(), 4);
    EXPECT_TRUE(arg.at(0) == "one");
    EXPECT_TRUE(arg.at(1) == "char");
    EXPECT_TRUE(arg.at(2) == "short");
    EXPECT_TRUE(arg.at(3) == "option");
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, use_long_short_option)
{
    argparse_t parser(details::get_test_name());
    auto       ret = details::parse(parser, "-MI", "two", "char", "short", "option");
    auto       arg = parser.get<std::vector<std::string>>("MI");

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(arg.size(), 4);
    EXPECT_TRUE(arg.at(0) == "two");
    EXPECT_TRUE(arg.at(1) == "char");
    EXPECT_TRUE(arg.at(2) == "short");
    EXPECT_TRUE(arg.at(3) == "option");
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, use_module_include_long_option)
{
    argparse_t parser(details::get_test_name());
    auto       ret =
        details::parse(parser, "--module-include", "two", "char", "short", "option");
    auto arg = parser.get<std::vector<std::string>>("module-include");

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(arg.size(), 4);
    EXPECT_TRUE(arg.at(0) == "two");
    EXPECT_TRUE(arg.at(1) == "char");
    EXPECT_TRUE(arg.at(2) == "short");
    EXPECT_TRUE(arg.at(3) == "option");
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, use_module_exclude_long_option)
{
    argparse_t parser(details::get_test_name());
    auto       ret =
        details::parse(parser, "--module-exclude", "two", "char", "short", "option");
    auto arg = parser.get<std::vector<std::string>>("module-exclude");

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(arg.size(), 4);
    EXPECT_TRUE(arg.at(0) == "two");
    EXPECT_TRUE(arg.at(1) == "char");
    EXPECT_TRUE(arg.at(2) == "short");
    EXPECT_TRUE(arg.at(3) == "option");
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, unused_long_short_option)
{
    argparse_t parser(details::get_test_name());
    auto       ret = details::parse(parser, "-M", "region", "char", "short", "option");
    auto       arg = parser.get<std::vector<std::string>>("M");

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(arg.size(), 1);
    EXPECT_TRUE(arg.at(0) == "region");
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, combined_short)
{
    argparse_t parser(details::get_test_name());
    auto       ret = details::parse(parser, "-sSl");

    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(parser.exists("stubs"));
    EXPECT_TRUE(parser.exists("stdlib"));
    EXPECT_TRUE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, combined_short_long)
{
    argparse_t parser(details::get_test_name());
    auto ret = details::parse(parser, "-sSl", "-R", "combined", "short", "and", "long");
    auto arg = parser.get<std::vector<std::string>>("function-include");

    EXPECT_EQ(ret, 0);

    EXPECT_TRUE(parser.exists("stubs"));
    EXPECT_TRUE(parser.exists("stdlib"));
    EXPECT_TRUE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));

    EXPECT_TRUE(parser.get<bool>("stubs"));
    EXPECT_TRUE(parser.get<bool>("stdlib"));
    EXPECT_TRUE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));

    EXPECT_TRUE(arg.at(0) == "combined");
    EXPECT_TRUE(arg.at(1) == "short");
    EXPECT_TRUE(arg.at(2) == "and");
    EXPECT_TRUE(arg.at(3) == "long");
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, ompt_exclude)
{
    argparse_t parser(details::get_test_name());
    auto       ret = details::parse(parser, "-o", "./ex_optional_off.inst", "-E",
                              "'impl::fibonacci'", "--ompt", "--", "ex_optional_off");
    auto       exc = parser.get<std::vector<std::string>>("function-exclude");
    auto       out = parser.get<std::vector<std::string>>("output");

    std::cout << "exc: " << exc << std::endl;
    std::cout << "out: " << out << std::endl;

    EXPECT_EQ(ret, 0);

    EXPECT_FALSE(parser.exists("stubs"));
    EXPECT_FALSE(parser.exists("stdlib"));
    EXPECT_FALSE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));
    EXPECT_TRUE(parser.exists("ompt"));

    EXPECT_FALSE(parser.get<bool>("stubs"));
    EXPECT_FALSE(parser.get<bool>("stdlib"));
    EXPECT_FALSE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));
    EXPECT_TRUE(parser.get<bool>("ompt"));

    EXPECT_EQ(exc.size(), 1);
    EXPECT_TRUE(exc.at(0) == "'impl::fibonacci'");

    EXPECT_EQ(out.size(), 1);
    EXPECT_TRUE(out.at(0) == "./ex_optional_off.inst");
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, boolean_with_on)
{
    argparse_t parser(details::get_test_name());
    auto ret = details::parse(parser, "-sS", "-l", "ON", "-R", "combined", "short", "and",
                              "long");
    auto arg = parser.get<std::vector<std::string>>("R");

    EXPECT_EQ(ret, 0);

    EXPECT_TRUE(parser.exists("stubs"));
    EXPECT_TRUE(parser.exists("stdlib"));
    EXPECT_TRUE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));

    EXPECT_TRUE(parser.get<bool>("stubs"));
    EXPECT_TRUE(parser.get<bool>("stdlib"));
    EXPECT_TRUE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));

    EXPECT_TRUE(arg.at(0) == "combined");
    EXPECT_TRUE(arg.at(1) == "short");
    EXPECT_TRUE(arg.at(2) == "and");
    EXPECT_TRUE(arg.at(3) == "long");
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, boolean_with_yes)
{
    argparse_t parser(details::get_test_name());
    auto       ret = details::parse(parser, "-sS", "-l", "YES", "-R", "combined", "short",
                              "and", "long");
    auto       arg = parser.get<std::vector<std::string>>("R");

    EXPECT_EQ(ret, 0);

    EXPECT_TRUE(parser.exists("stubs"));
    EXPECT_TRUE(parser.exists("stdlib"));
    EXPECT_TRUE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));

    EXPECT_TRUE(parser.get<bool>("stubs"));
    EXPECT_TRUE(parser.get<bool>("stdlib"));
    EXPECT_TRUE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));

    EXPECT_TRUE(arg.at(0) == "combined");
    EXPECT_TRUE(arg.at(1) == "short");
    EXPECT_TRUE(arg.at(2) == "and");
    EXPECT_TRUE(arg.at(3) == "long");
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, boolean_with_true)
{
    argparse_t parser(details::get_test_name());
    auto ret = details::parse(parser, "-sS", "-l", "T", "-R", "combined", "short", "and",
                              "long");
    auto arg = parser.get<std::vector<std::string>>("R");

    EXPECT_EQ(ret, 0);

    EXPECT_TRUE(parser.exists("stubs"));
    EXPECT_TRUE(parser.exists("stdlib"));
    EXPECT_TRUE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));

    EXPECT_TRUE(parser.get<bool>("stubs"));
    EXPECT_TRUE(parser.get<bool>("stdlib"));
    EXPECT_TRUE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));

    EXPECT_TRUE(arg.at(0) == "combined");
    EXPECT_TRUE(arg.at(1) == "short");
    EXPECT_TRUE(arg.at(2) == "and");
    EXPECT_TRUE(arg.at(3) == "long");
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, boolean_with_numeric)
{
    argparse_t parser(details::get_test_name());
    auto       ret = details::parse(parser, "-s", "-S", "0", "-l", "5", "-R", "combined",
                              "short", "and", "long");
    auto       arg = parser.get<std::vector<std::string>>("R");

    EXPECT_EQ(ret, 0);

    EXPECT_TRUE(parser.exists("stubs"));
    EXPECT_TRUE(parser.exists("stdlib"));
    EXPECT_TRUE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));

    EXPECT_TRUE(parser.get<bool>("stubs"));
    EXPECT_FALSE(parser.get<bool>("stdlib"));
    EXPECT_TRUE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));

    EXPECT_TRUE(arg.at(0) == "combined");
    EXPECT_TRUE(arg.at(1) == "short");
    EXPECT_TRUE(arg.at(2) == "and");
    EXPECT_TRUE(arg.at(3) == "long");
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, parse_known_options)
{
    argparse_t parser(details::get_test_name());
    auto       orig           = details::parse_function();
    details::parse_function() = [](argparse_t& _parser, int& _ac, char**& _av) {
        argerror_t err;
        std::tie(err, _ac, _av) = _parser.parse_known_args(_ac, _av, "--", 2);
        return err;
    };
    auto ret = details::parse(parser, "-sS", "-l", "T", "-R", "combined", "short", "and",
                              "long", "--", "10", "20");
    auto arg = parser.get<std::vector<std::string>>("R");

    EXPECT_EQ(ret, 0);

    EXPECT_TRUE(parser.exists("stubs"));
    EXPECT_TRUE(parser.exists("stdlib"));
    EXPECT_TRUE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));

    EXPECT_TRUE(parser.get<bool>("stubs"));
    EXPECT_TRUE(parser.get<bool>("stdlib"));
    EXPECT_TRUE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));

    EXPECT_TRUE(arg.at(0) == "combined");
    EXPECT_TRUE(arg.at(1) == "short");
    EXPECT_TRUE(arg.at(2) == "and");
    EXPECT_TRUE(arg.at(3) == "long");

    EXPECT_EQ(test_argc, 3);
    EXPECT_EQ(std::string(test_argv[0]), std::string(_arg0));
    EXPECT_EQ(std::string(test_argv[1]), std::string("10"));
    EXPECT_EQ(std::string(test_argv[2]), std::string("20"));
    EXPECT_TRUE(test_argv[3] == nullptr);

    details::parse_function() = orig;
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, parse_known_options_without_options)
{
    argparse_t parser(details::get_test_name());
    auto       orig           = details::parse_function();
    details::parse_function() = [](argparse_t& _parser, int& _ac, char**& _av) {
        argerror_t err;
        std::tie(err, _ac, _av) = _parser.parse_known_args(_ac, _av, "--", 2);
        return err;
    };
    auto ret = details::parse(parser, "10", "20");

    EXPECT_EQ(ret, 0);

    EXPECT_FALSE(parser.exists("stubs"));
    EXPECT_FALSE(parser.exists("stdlib"));
    EXPECT_FALSE(parser.exists("instrument-loops"));
    EXPECT_FALSE(parser.exists("load"));
    EXPECT_FALSE(parser.exists("p"));
    EXPECT_FALSE(parser.exists("pid"));

    EXPECT_FALSE(parser.get<bool>("stubs"));
    EXPECT_FALSE(parser.get<bool>("stdlib"));
    EXPECT_FALSE(parser.get<bool>("instrument-loops"));
    EXPECT_FALSE(parser.get<bool>("load"));
    EXPECT_FALSE(parser.get<bool>("p"));
    EXPECT_FALSE(parser.get<bool>("pid"));

    EXPECT_EQ(test_argc, 3);
    EXPECT_EQ(std::string(test_argv[0]), std::string(_arg0));
    EXPECT_EQ(std::string(test_argv[1]), std::string("10"));
    EXPECT_EQ(std::string(test_argv[2]), std::string("20"));

    details::parse_function() = orig;
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, timemory_argparse_vec)
{
    auto _restore_debug    = tim::settings::debug();
    tim::settings::debug() = false;

    auto _enabled = tim::settings::enabled();
    auto _verbose = tim::settings::verbose();
    auto _debug   = tim::settings::debug();
    auto _python  = tim::settings::python_exe();

    std::vector<std::string> args = {
        _margv[0],          "--timemory-enabled=false", "--timemory-verbose", "10",
        "--timemory-debug", "--timemory-python-exe",    "/fake/python",
    };

    std::cout << "Argument: ";
    for(const auto& arg : args)
        std::cout << arg << " ";
    std::cout << std::endl;

    tim::timemory_argparse(args);

    EXPECT_EQ(tim::settings::verbose(), 10);
    EXPECT_NE(tim::settings::enabled(), _enabled);
    EXPECT_NE(tim::settings::verbose(), _verbose);
    EXPECT_NE(tim::settings::debug(), _debug);
    EXPECT_NE(tim::settings::python_exe(), _python);
    EXPECT_EQ(tim::settings::python_exe(), std::string("/fake/python"));

    tim::settings::enabled()    = _enabled;
    tim::settings::verbose()    = _verbose;
    tim::settings::debug()      = _debug;
    tim::settings::python_exe() = _python;

    if(_restore_debug)
        tim::settings::debug() = true;
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, timemory_init_argparse_vec)
{
    auto _restore_debug    = tim::settings::debug();
    tim::settings::debug() = false;

    auto _enabled = tim::settings::enabled();
    auto _verbose = tim::settings::verbose();
    auto _debug   = tim::settings::debug();
    auto _python  = tim::settings::python_exe();

    std::vector<std::string> args = {
        _margv[0],
        "--timemory-enabled=false",
        "--timemory-verbose",
        "10",
        "--timemory-debug",
        "--timemory-python-exe",
        "/fake/python",
        "--",
        "./test",
        "--foo",
    };

    std::cout << "Argument: ";
    for(const auto& arg : args)
        std::cout << arg << " ";
    std::cout << std::endl;

    tim::argparse::argument_parser p{ _margv[0] };
    tim::timemory_init(args, p);

    EXPECT_EQ(tim::settings::verbose(), 10);
    EXPECT_NE(tim::settings::enabled(), _enabled);
    EXPECT_NE(tim::settings::verbose(), _verbose);
    EXPECT_NE(tim::settings::debug(), _debug);
    EXPECT_NE(tim::settings::python_exe(), _python);
    EXPECT_EQ(tim::settings::python_exe(), std::string("/fake/python"));

    ASSERT_EQ(args.size(), 3);
    EXPECT_EQ(args.at(0), std::string{ _margv[0] });
    EXPECT_EQ(args.at(1), std::string{ "./test" });
    EXPECT_EQ(args.at(2), std::string{ "--foo" });

    tim::settings::enabled()    = _enabled;
    tim::settings::verbose()    = _verbose;
    tim::settings::debug()      = _debug;
    tim::settings::python_exe() = _python;

    if(_restore_debug)
        tim::settings::debug() = true;
}

//--------------------------------------------------------------------------------------//

TEST_F(argparse_tests, timemory_argparse_ptr)
{
    auto _restore_debug    = tim::settings::debug();
    tim::settings::debug() = false;

    auto _enabled = tim::settings::enabled();
    auto _verbose = tim::settings::verbose();
    auto _debug   = tim::settings::debug();
    auto _python  = tim::settings::python_exe();

    std::vector<std::string> args = {
        _margv[0],          "--timemory-enabled=false", "--timemory-verbose", "10",
        "--timemory-debug", "--timemory-python-exe",    "/fake/python",       "--",
        "some-argument"
    };

    int    argc = args.size();
    char** argv = new char*[argc];

    for(int i = 0; i < argc; ++i)
    {
        argv[i] = new char[args.at(i).length() + 1];
        strncpy(argv[i], args.at(i).c_str(), args.at(i).length() + 1);
        argv[i][args.at(i).length()] = '\0';
    }

    std::cout << "Argument: ";
    for(int i = 0; i < argc; ++i)
        std::cout << argv[i] << " ";
    std::cout << std::endl;

    tim::timemory_argparse(&argc, &argv);

    EXPECT_EQ(argc, 2);
    EXPECT_EQ(std::string(argv[0]), args.front());
    EXPECT_EQ(std::string(argv[1]), args.back());

    EXPECT_EQ(tim::settings::verbose(), 10);
    EXPECT_NE(tim::settings::enabled(), _enabled);
    EXPECT_NE(tim::settings::verbose(), _verbose);
    EXPECT_NE(tim::settings::debug(), _debug);
    EXPECT_NE(tim::settings::python_exe(), _python);
    EXPECT_EQ(tim::settings::python_exe(), std::string("/fake/python"));

    tim::settings::enabled()    = _enabled;
    tim::settings::verbose()    = _verbose;
    tim::settings::debug()      = _debug;
    tim::settings::python_exe() = _python;

    for(int i = 0; i < argc; ++i)
        delete[] argv[i];
    delete[] argv;

    if(_restore_debug)
        tim::settings::debug() = true;
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    _margc = argc;
    _margv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    return ret;
}

//--------------------------------------------------------------------------------------//
