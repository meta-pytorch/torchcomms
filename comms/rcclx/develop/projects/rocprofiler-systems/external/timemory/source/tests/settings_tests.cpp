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

#include "timemory/timemory.hpp"

//--------------------------------------------------------------------------------------//

static uint64_t reference_value = 10;
static uint64_t copied_value    = 10;

//--------------------------------------------------------------------------------------//

namespace details
{
//  Get the current tests name
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class settings_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(settings_tests, insert_via_args)
{
    using settings_t = tim::settings;
    using strvec_t   = std::vector<std::string>;
    using strset_t   = std::set<std::string>;

    reference_value = 10;
    copied_value    = 10;

    settings_t* _settings = settings_t::instance();

    std::string _ref_env  = "SETTINGS_TEST_REFERENCE";
    std::string _ref_name = "test_reference";
    std::string _ref_cmd  = "--test-reference";
    std::string _ref_cat  = "custom";
    std::string _ref_desc = TIMEMORY_JOIN("", "Reference value for settings_tests.",
                                          details::get_test_name());

    auto _ins = _settings->insert<uint64_t, uint64_t&>(
        _ref_env, _ref_name, _ref_desc, reference_value, strset_t{ _ref_cat },
        strvec_t{ _ref_cmd }, 1, -2);

    EXPECT_TRUE(_ins.second);

    auto tidx_check = std::type_index(typeid(uint64_t));
    auto vidx_check = std::type_index(typeid(uint64_t&));
    auto _env_check = tim::get_env<uint64_t>(_ref_env, 0);
    auto _odr_check = _settings->ordering().back();
    auto _itr_check = _settings->find(_ref_env);

    EXPECT_EQ(_env_check, reference_value);
    EXPECT_EQ(_odr_check, _ref_env);
    ASSERT_NE(_itr_check, _settings->end());
    EXPECT_EQ(_itr_check->first, _ref_env);

    auto itr = _itr_check->second;

    EXPECT_EQ(itr->get_type_index(), tidx_check);
    EXPECT_EQ(itr->get_value_index(), vidx_check);

    EXPECT_EQ(itr->get_count(), 1);
    EXPECT_EQ(itr->get_max_count(), -2);

    EXPECT_EQ(itr->get_name(), _ref_name);
    EXPECT_EQ(itr->get_env_name(), _ref_env);
    EXPECT_EQ(itr->get_command_line().front(), _ref_cmd);

    EXPECT_TRUE(itr->matches(_ref_env, true));
    EXPECT_TRUE(itr->matches(_ref_name, true));
    EXPECT_TRUE(itr->matches(_ref_cmd, true));

    EXPECT_TRUE(itr->matches(_ref_env, false));
    EXPECT_TRUE(itr->matches(_ref_name, false));
    EXPECT_TRUE(itr->matches(_ref_cmd, false));

    EXPECT_TRUE(itr->matches(_ref_env, "custom"));
    EXPECT_TRUE(itr->matches(_ref_name, "custom"));
    EXPECT_TRUE(itr->matches(_ref_cmd, "custom"));

    EXPECT_FALSE(itr->matches(_ref_env, "native"));
    EXPECT_FALSE(itr->matches(_ref_name, "native"));
    EXPECT_FALSE(itr->matches(_ref_cmd, "native"));

    EXPECT_TRUE(itr->matches(_ref_env, "custom", true));
    EXPECT_TRUE(itr->matches(_ref_name, "custom", true));
    EXPECT_TRUE(itr->matches(_ref_cmd, "custom", true));

    EXPECT_FALSE(itr->matches(_ref_env, "native", true));
    EXPECT_FALSE(itr->matches(_ref_name, "native", true));
    EXPECT_FALSE(itr->matches(_ref_cmd, "native", true));

    EXPECT_TRUE(itr->matches("", "custom", true));
    EXPECT_TRUE(itr->matches("", "custom", false));

    EXPECT_FALSE(itr->matches("", "native", true));
    EXPECT_FALSE(itr->matches("", "native", false));

    uint64_t _tmp = 0;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    auto _disp = itr->get_display();
    EXPECT_EQ(_disp["name"], _ref_name);
    EXPECT_EQ(_disp["count"], "1");
    EXPECT_EQ(_disp["max_count"], "-2");
    EXPECT_EQ(_disp["env_name"], _ref_env);
    EXPECT_EQ(_disp["description"], _ref_desc);
    EXPECT_EQ(_disp["command_line"], _ref_cmd);
    EXPECT_EQ(_disp["categories"], _ref_cat);

    reference_value = 5;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    itr->set<uint64_t>(4);
    EXPECT_EQ(reference_value, 4);
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    itr->set("3");
    EXPECT_EQ(reference_value, 3);
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    auto citr = itr->clone();
    EXPECT_EQ(citr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(citr->as_string(), std::to_string(reference_value));

    reference_value = 2;
    EXPECT_NE(reference_value, citr->get<uint64_t>().second);
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));
    EXPECT_NE(citr->get<uint64_t>().second, reference_value);
    EXPECT_NE(citr->as_string(), std::to_string(reference_value));

    citr->set<uint64_t>(1);
    copied_value = citr->get<uint64_t>().second;
    EXPECT_TRUE(citr->get(_tmp));
    EXPECT_EQ(reference_value, 2);
    EXPECT_EQ(citr->get<uint64_t>().second, 1);
    EXPECT_EQ(copied_value, 1);
    EXPECT_EQ(_tmp, 1);
    EXPECT_EQ(_tmp, copied_value);
    EXPECT_EQ(citr->as_string(), std::to_string(copied_value));
}

//--------------------------------------------------------------------------------------//

TEST_F(settings_tests, insert_via_pointer)
{
    using settings_t = tim::settings;
    using strvec_t   = std::vector<std::string>;
    using strset_t   = std::set<std::string>;

    reference_value = 10;
    copied_value    = 10;

    settings_t* _settings = settings_t::instance();

    std::string _ref_env  = "SETTINGS_TEST_COPY";
    std::string _ref_name = "test_copy";
    std::string _ref_cmd  = "--test-copy";
    std::string _ref_cat  = "custom";
    std::string _ref_desc =
        TIMEMORY_JOIN("", "Copied value for settings_tests.", details::get_test_name());

    auto _ptr = std::make_shared<tim::tsettings<uint64_t>>(
        reference_value, _ref_name, _ref_env, _ref_desc, strvec_t({ _ref_cmd }), 1, -2,
        strvec_t{}, strset_t{ _ref_cat });
    auto _ins = _settings->insert(_ptr);

    EXPECT_TRUE(_ins.second);

    auto tidx_check = std::type_index(typeid(uint64_t));
    auto vidx_check = std::type_index(typeid(uint64_t));
    auto _env_check = tim::get_env<uint64_t>(_ref_env, 0);
    auto _odr_check = _settings->ordering().back();
    auto _itr_check = _settings->find(_ref_env);

    EXPECT_EQ(_env_check, reference_value);
    EXPECT_EQ(_odr_check, _ref_env);
    ASSERT_NE(_itr_check, _settings->end());
    EXPECT_EQ(_itr_check->first, _ref_env);

    auto itr = _itr_check->second;

    EXPECT_EQ(itr->get_type_index(), tidx_check);
    EXPECT_EQ(itr->get_value_index(), vidx_check);

    EXPECT_EQ(itr->get_count(), 1);
    EXPECT_EQ(itr->get_max_count(), -2);

    EXPECT_EQ(itr->get_name(), _ref_name);
    EXPECT_EQ(itr->get_env_name(), _ref_env);
    EXPECT_EQ(itr->get_command_line().front(), _ref_cmd);

    EXPECT_TRUE(itr->matches(_ref_env, true));
    EXPECT_TRUE(itr->matches(_ref_name, true));
    EXPECT_TRUE(itr->matches(_ref_cmd, true));

    EXPECT_TRUE(itr->matches(_ref_env, false));
    EXPECT_TRUE(itr->matches(_ref_name, false));
    EXPECT_TRUE(itr->matches(_ref_cmd, false));

    EXPECT_TRUE(itr->matches(_ref_env, "custom"));
    EXPECT_TRUE(itr->matches(_ref_name, "custom"));
    EXPECT_TRUE(itr->matches(_ref_cmd, "custom"));

    EXPECT_FALSE(itr->matches(_ref_env, "native"));
    EXPECT_FALSE(itr->matches(_ref_name, "native"));
    EXPECT_FALSE(itr->matches(_ref_cmd, "native"));

    EXPECT_TRUE(itr->matches(_ref_env, "custom", true));
    EXPECT_TRUE(itr->matches(_ref_name, "custom", true));
    EXPECT_TRUE(itr->matches(_ref_cmd, "custom", true));

    EXPECT_FALSE(itr->matches(_ref_env, "native", true));
    EXPECT_FALSE(itr->matches(_ref_name, "native", true));
    EXPECT_FALSE(itr->matches(_ref_cmd, "native", true));

    EXPECT_TRUE(itr->matches("", "custom", true));
    EXPECT_TRUE(itr->matches("", "custom", false));

    EXPECT_FALSE(itr->matches("", "native", true));
    EXPECT_FALSE(itr->matches("", "native", false));

    uint64_t _tmp = 0;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    auto _disp = itr->get_display();
    EXPECT_EQ(_disp["name"], _ref_name);
    EXPECT_EQ(_disp["count"], "1");
    EXPECT_EQ(_disp["max_count"], "-2");
    EXPECT_EQ(_disp["env_name"], _ref_env);
    EXPECT_EQ(_disp["description"], _ref_desc);
    EXPECT_EQ(_disp["command_line"], _ref_cmd);

    reference_value = 5;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_NE(_tmp, reference_value);
    EXPECT_NE(itr->get<uint64_t>().second, reference_value);
    EXPECT_NE(itr->as_string(), std::to_string(reference_value));

    EXPECT_TRUE(itr->set<uint64_t>(2));
    EXPECT_TRUE(itr->get(reference_value));
    EXPECT_EQ(reference_value, 2);

    auto citr = itr->clone();
    EXPECT_EQ(citr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(citr->as_string(), std::to_string(reference_value));

    EXPECT_TRUE(citr->set<uint64_t>(1));
    EXPECT_NE(reference_value, citr->get<uint64_t>().second);

    copied_value = citr->get<uint64_t>().second;
    EXPECT_TRUE(citr->get(_tmp));
    EXPECT_EQ(reference_value, 2);
    EXPECT_EQ(citr->get<uint64_t>().second, 1);
    EXPECT_EQ(copied_value, 1);
    EXPECT_EQ(_tmp, 1);
    EXPECT_EQ(_tmp, copied_value);
    EXPECT_EQ(citr->as_string(), std::to_string(copied_value));
}

//--------------------------------------------------------------------------------------//

TEST_F(settings_tests, metadata)
{
    tim::manager::add_metadata("lucky-number", 7);
    std::vector<std::string> _data = { "dogs", "and", "cats", "are", "good" };
    tim::manager::add_metadata("things-you-should-know", _data);
    tim::manager::instance()->set_write_metadata(1);
    tim::manager::instance()->write_metadata(tim::settings::get_global_output_prefix(),
                                             details::get_test_name().c_str());
    auto          fname = tim::settings::compose_output_filename("metadata", "json");
    std::ifstream ifs(fname);
    EXPECT_TRUE(ifs.is_open());
    std::string line;
    while(std::getline(ifs, line) && ifs)
        std::cout << line << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(settings_tests, push_pop)
{
    auto _orig_settings          = tim::settings::instance();
    auto _orig_add_second        = tim::settings::add_secondary();
    auto _orig_enabled           = tim::settings::enabled();
    auto _orig_global_components = tim::settings::global_components();
    auto _settings               = tim::settings::push<TIMEMORY_API>();

    auto _print_settings = [](const auto& _label, auto _instance) {
        std::cout << '\n' << _label << '\n';
        std::cout << "\tpointer address    : " << _instance << '\n';
        std::cout << "\tinstance<> address : " << tim::settings::instance<TIMEMORY_API>()
                  << '\n';
        std::cout << "\tadd_secondary      : " << std::boolalpha
                  << _instance->get_add_secondary() << '\n';
        std::cout << "\tenabled            : " << std::boolalpha
                  << _instance->get_enabled() << '\n';
        std::cout << "\tglobal components  : " << std::boolalpha
                  << _instance->get_global_components() << '\n';
    };

    _print_settings("ORIGINAL BEFORE COPY", _orig_settings);

    ASSERT_NE(_orig_settings, _settings.get()) << "Push did not create copy";

    _print_settings("ORIGINAL BEFORE MODIFY", _orig_settings);
    _print_settings("COPY BEFORE MODIFY", _settings);

    _settings->get_add_secondary() = !_orig_add_second;
    _settings->get_enabled()       = !_orig_enabled;
    _settings->get_global_components() =
        _orig_global_components + " wall_clock, monotonic_clock";

    _print_settings("ORIGINAL AFTER MODIFY", _orig_settings);
    _print_settings("COPY AFTER MODIFY", _settings);

    // ensure the non-template access method was NOT updated
    EXPECT_EQ(tim::settings::add_secondary(), _orig_add_second)
        << "Original settings were inappropriately updated";
    EXPECT_EQ(tim::settings::enabled(), _orig_enabled)
        << "Original settings were inappropriately updated";
    EXPECT_EQ(tim::settings::global_components(), _orig_global_components)
        << "Original settings were inappropriately updated";

    // ensure the non-template access method was NOT updated
    EXPECT_NE(_settings->get_add_secondary(), tim::settings::add_secondary())
        << "Push instance changed original settings";
    EXPECT_NE(_settings->get_enabled(), tim::settings::enabled())
        << "Push instance changed original settings";
    EXPECT_NE(_settings->get_global_components(), tim::settings::global_components())
        << "Push instance changed original settings";

    // ensure the template access method was updated
#if !defined(TIMEMORY_WINDOWS)
    EXPECT_EQ(_settings->get_add_secondary(),
              tim::settings::instance<TIMEMORY_API>()->get_add_secondary())
        << "Push instance did not propagate to static methods";
    EXPECT_EQ(_settings->get_enabled(),
              tim::settings::instance<TIMEMORY_API>()->get_enabled())
        << "Push instance did not propagate to static methods";
    EXPECT_EQ(_settings->get_global_components(),
              tim::settings::instance<TIMEMORY_API>()->get_global_components())
        << "Push instance did not propagate to static methods";
#endif

    // ensure the template access method was updated
    EXPECT_NE(_settings->get_add_secondary(), _orig_add_second)
        << "Failure to change original settings";
    EXPECT_NE(_settings->get_enabled(), _orig_enabled)
        << "Failure to change original settings";
    EXPECT_NE(_settings->get_global_components(), _orig_global_components)
        << "Failure to change original settings";

    _settings = tim::settings::pop<TIMEMORY_API>();

    EXPECT_EQ(_settings.get(), _orig_settings);

    // ensure restoration
    EXPECT_EQ(_settings->get_add_secondary(), _orig_add_second)
        << "Original settings were not restored";
    EXPECT_EQ(_settings->get_enabled(), _orig_enabled)
        << "Original settings were not restored";
    EXPECT_EQ(_settings->get_global_components(), _orig_global_components)
        << "Original settings were not restored";

    // ensure the non-template access method was NOT updated
    EXPECT_EQ(tim::settings::add_secondary(), _orig_add_second)
        << "Original settings were inappropriately updated";
    EXPECT_EQ(tim::settings::enabled(), _orig_enabled)
        << "Original settings were inappropriately updated";
    EXPECT_EQ(tim::settings::global_components(), _orig_global_components)
        << "Original settings were inappropriately updated";
}

//--------------------------------------------------------------------------------------//
#if !defined(TIMEMORY_WINDOWS)

TIMEMORY_DECLARE_API(settings_copy)

TEST_F(settings_tests, api_copy)
{
    using settings_t      = tim::settings;
    using strvec_t        = std::vector<std::string>;
    settings_t* _settings = settings_t::instance();
    std::string _ref_env  = "SETTINGS_TEST_COPY";
    std::string _ref_name = "test_copy";
    std::string _ref_cmd  = "--test-copy";
    std::string _ref_desc =
        TIMEMORY_JOIN("", "Copied value for settings_tests.", details::get_test_name());

    auto _ptr = std::make_shared<tim::tsettings<uint64_t>>(
        reference_value, _ref_name, _ref_env, _ref_desc, strvec_t({ _ref_cmd }), 1, -2);
    _settings->insert(_ptr);

    auto _api_settings = tim::settings::instance<tim::api::settings_copy>();

    ASSERT_EQ(_api_settings->ordering().size(), _settings->ordering().size());

    bool found_ref_in_api_settings = false;
    for(size_t i = 0; i < _api_settings->ordering().size(); ++i)
    {
        EXPECT_EQ(_api_settings->ordering().at(i), _settings->ordering().at(i));
        if(_api_settings->ordering().at(i) == _ref_env)
            found_ref_in_api_settings = true;
    }

    EXPECT_TRUE(found_ref_in_api_settings);

    std::stringstream api_ss;
    std::stringstream orig_ss;

    auto api_ar  = tim::policy::output_archive_t<tim::settings>::get(api_ss);
    auto orig_ar = tim::policy::output_archive_t<tim::settings>::get(orig_ss);

    settings_t::serialize_settings(*api_ar, *_api_settings);
    settings_t::serialize_settings(*orig_ar, *_settings);

    EXPECT_EQ(api_ss.str(), orig_ss.str());
}

#endif
//--------------------------------------------------------------------------------------//
