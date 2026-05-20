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

#include "timemory/components/data_tracker/components.hpp"
#include "timemory/config.hpp"
#include "timemory/data/ring_buffer.hpp"
#include "timemory/data/ring_buffer_allocator.hpp"
#include "timemory/units.hpp"

namespace comp = tim::component;

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

class ring_buffer_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    void SetUp() override
    {
        TIMEMORY_TEST_CHECK_SKIP;
        printf("[##########] Executing %s ... \n", details::get_test_name().c_str());
        if(buffer.is_initialized())
            buffer.destroy();
        bool _use_mmap = false;
        tim::set_env("TIMEMORY_USE_MMAP", "0", 1);
#if defined(TIMEMORY_LINUX)
        if(details::get_test_name().find("mmap") != std::string::npos)
        {
            _use_mmap = true;
            tim::set_env("TIMEMORY_USE_MMAP", "1", 1);
        }
#endif
        buffer.set_use_mmap(_use_mmap);
        buffer.init(1);
        std::cout << "[ SetUp    ]> " << buffer << std::endl;
        EXPECT_TRUE(buffer.is_initialized());
        EXPECT_EQ(buffer.get_use_mmap(), _use_mmap);
    }

    void TearDown() override
    {
        std::cout << "[ TearDown ]> " << buffer << '\n';
        buffer.destroy();
    }

    using comp_type   = comp::data_tracker_floating;
    using buffer_type = tim::data_storage::ring_buffer<comp_type>;

    size_t                  count      = tim::units::get_page_size() / sizeof(comp_type);
    comp_type               object     = comp_type{};
    buffer_type             buffer     = {};
    std::vector<comp_type*> buffer_vec = {};
    std::set<comp_type*>    buffer_set = {};
};

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, empty) { EXPECT_TRUE(buffer.is_empty()); }
TEST_F(ring_buffer_tests, empty_mmap) { EXPECT_TRUE(buffer.is_empty()); }

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, copy)
{
    buffer_type buffer_copy = buffer;
    for(size_t i = 0; i <= count; ++i)
    {
        if(i == count)
        {
            EXPECT_TRUE(buffer_copy.is_full()) << "[" << i << "]> " << buffer_copy;
            buffer_copy.retrieve();
        }
        EXPECT_FALSE(buffer_copy.is_full()) << "[" << i << "]> " << buffer_copy;
        comp_type tmp{};
        tmp.store(static_cast<double>(i + 1));
        auto* ptr = buffer_copy.emplace(tmp);
        EXPECT_EQ(tmp.get(), ptr->get());
        tmp.store(std::plus<double>{}, static_cast<double>(i + 1));
        EXPECT_NE(tmp.get(), ptr->get());
    }
    EXPECT_FALSE(buffer.is_full()) << buffer;
    EXPECT_TRUE(buffer_copy.is_full()) << buffer_copy;

#define COPY_EXPECT_EQ(FUNC)                                                             \
    EXPECT_EQ(lhs->FUNC(), rhs->FUNC())                                                  \
        << std::boolalpha << #FUNC << " :: lhs: " << lhs->FUNC()                         \
        << ", rhs: " << rhs->FUNC();

    std::cout << "[   copy   ]> " << buffer_copy << std::endl;
    auto buffer_temp = buffer_copy;
    std::cout << "[   temp   ]> " << buffer_temp << std::endl;
    buffer = std::move(buffer_temp);
    std::cout << "[   buff   ]> " << buffer << std::endl;
    EXPECT_TRUE(buffer.is_full()) << buffer;
    EXPECT_TRUE(buffer_temp.is_empty()) << buffer_temp;
    EXPECT_FALSE(buffer_temp.is_initialized()) << buffer_temp;
    auto n = buffer.count();
    for(size_t i = 0; i < n; ++i)
    {
        comp_type* lhs = buffer.retrieve();
        comp_type* rhs = buffer_copy.retrieve();
        ASSERT_TRUE(lhs != nullptr) << "lhs: " << lhs << ", buffer: " << buffer;
        ASSERT_TRUE(rhs != nullptr) << "rhs: " << rhs << ", buffer: " << buffer_copy;
        EXPECT_NE(lhs, rhs) << "lhs: " << lhs << ", rhs: " << rhs;
        // verify all fields
        COPY_EXPECT_EQ(get_depth_change)
        COPY_EXPECT_EQ(get_is_flat)
        COPY_EXPECT_EQ(get_is_invalid)
        COPY_EXPECT_EQ(get_is_on_stack)
        COPY_EXPECT_EQ(get_is_running)
        COPY_EXPECT_EQ(get_is_transient)
        COPY_EXPECT_EQ(get_accum)
        COPY_EXPECT_EQ(get_value)
        COPY_EXPECT_EQ(get_last)
        COPY_EXPECT_EQ(get)
        COPY_EXPECT_EQ(get_display)
        COPY_EXPECT_EQ(load)
        COPY_EXPECT_EQ(get_laps)
        COPY_EXPECT_EQ(get_iterator)
    }

    std::cout << "[   copy   ]> " << buffer_copy << std::endl;
    std::cout << "[   temp   ]> " << buffer_temp << std::endl;
    // make sure swap and move do not corrupt data
    std::swap(buffer_copy, buffer_temp);
    std::cout << "[swap//copy]> " << buffer_copy << std::endl;
    std::cout << "[swap//temp]> " << buffer_temp << std::endl;
    std::cout << "[   buff   ]> " << buffer << std::endl;
    buffer = std::move(buffer_copy);
}

//--------------------------------------------------------------------------------------//

size_t n = 4;

TEST_F(ring_buffer_tests, loop)
{
    std::ostringstream _msg{};

    auto _write = [&](size_t i) {
        _msg << "[" << std::setw(3) << i << "][" << std::setw(3) << buffer_vec.size()
             << "] " << buffer << "\n";
    };

    auto _read = [&]() {
        // this will only empty half of the buffer b/c buffer.count() will
        // decrease after each retrieve call
        buffer_vec.reserve(buffer_vec.size() + ((buffer.count() + 1) / 2));
        for(size_t i = 0; i < buffer.count(); ++i)
            buffer_vec.emplace_back(buffer.retrieve());
    };

    for(size_t i = 0; i < n * count; ++i)
    {
        object.store(static_cast<double>(i + 1));
        _write(i);
        auto* ptr = buffer.write(&object);
        _write(i);
        ASSERT_EQ(object.get(), ptr->get()) << _msg.str();
        buffer_set.insert(ptr);
        if(buffer.is_full())
            _read();
    }
    _write(n * count);
    EXPECT_EQ(buffer_set.size(), count) << _msg.str();
    buffer_set.clear();
}

TEST_F(ring_buffer_tests, loop_mmap)
{
    std::ostringstream _msg{};

    auto _write = [&](size_t i) {
        _msg << "[" << std::setw(3) << i << "][" << std::setw(3) << buffer_vec.size()
             << "] " << buffer << "\n";
    };

    auto _read = [&]() {
        buffer_vec.reserve(buffer_vec.size() + buffer.count());
        // unlike ring_buffer_test.loop above, this will fully empty the buffer
        while(!buffer.is_empty())
            buffer_vec.emplace_back(buffer.retrieve());
    };

    for(size_t i = 0; i < n * count; ++i)
    {
        object.store(static_cast<double>(i + 1));
        _write(i);
        auto* ptr = buffer.write(&object);
        _write(i);
        ASSERT_EQ(object.get(), ptr->get()) << _msg.str();
        buffer_set.insert(ptr);
        if(buffer.is_full())
            _read();
    }
    _write(n * count);
    EXPECT_EQ(buffer_set.size(), count) << _msg.str();
    buffer_set.clear();
}

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, full)
{
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_FALSE(buffer.is_full()) << "[" << i << "]> " << buffer;
        object.store(static_cast<double>(i + 1));
        auto* ptr = buffer.write(&object);
        EXPECT_EQ(object.get(), ptr->get());
        buffer_vec.emplace_back(ptr);
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

TEST_F(ring_buffer_tests, full_mmap)
{
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_FALSE(buffer.is_full()) << "[" << i << "]> " << buffer;
        object.store(static_cast<double>(i + 1));
        auto* ptr = buffer.write(&object);
        EXPECT_EQ(object.get(), ptr->get());
        buffer_vec.emplace_back(ptr);
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, destroy)
{
    buffer.destroy();
    std::cout << "[ Destroyed]> " << buffer << std::endl;
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_TRUE(buffer.is_full()) << "[" << i << "]> " << buffer;
        object.store(static_cast<double>(i + 1));
        auto* ptr = buffer.write(&object);
        EXPECT_EQ(ptr, nullptr);
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

TEST_F(ring_buffer_tests, destroy_mmap)
{
    buffer.destroy();
    std::cout << "[ Destroyed]> " << buffer << std::endl;
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_TRUE(buffer.is_full()) << "[" << i << "]> " << buffer;
        object.store(static_cast<double>(i + 1));
        auto* ptr = buffer.write(&object);
        EXPECT_EQ(ptr, nullptr);
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, emplace)
{
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_FALSE(buffer.is_full()) << "[" << i << "]> " << buffer;
        comp_type tmp{};
        tmp.store(static_cast<double>(i + 1));
        tmp.add_secondary("secondary", static_cast<double>(i + 1));
        auto* ptr = buffer.emplace();
        EXPECT_NE(tmp.get(), ptr->get());
        auto stmp = tmp.get_secondary();
        auto sptr = ptr->get_secondary();
        EXPECT_NE(stmp.size(), sptr.size());
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

TEST_F(ring_buffer_tests, emplace_mmap)
{
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_FALSE(buffer.is_full()) << "[" << i << "]> " << buffer;
        comp_type tmp{};
        tmp.store(static_cast<double>(i + 1));
        tmp.add_secondary("secondary", static_cast<double>(i + 1));
        auto* ptr = buffer.emplace();
        EXPECT_NE(tmp.get(), ptr->get());
        auto stmp = tmp.get_secondary();
        auto sptr = ptr->get_secondary();
        EXPECT_NE(stmp.size(), sptr.size());
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, emplace_copy)
{
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_FALSE(buffer.is_full()) << "[" << i << "]> " << buffer;
        comp_type tmp{};
        tmp.store(static_cast<double>(i + 1));
        auto* ptr = buffer.emplace(tmp);
        EXPECT_EQ(tmp.get(), ptr->get());
        tmp.store(std::plus<double>{}, static_cast<double>(i + 1));
        EXPECT_NE(tmp.get(), ptr->get());
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

TEST_F(ring_buffer_tests, emplace_copy_mmap)
{
    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_FALSE(buffer.is_full()) << "[" << i << "]> " << buffer;
        comp_type tmp{};
        tmp.store(static_cast<double>(i + 1));
        auto* ptr = buffer.emplace(tmp);
        EXPECT_EQ(tmp.get(), ptr->get());
        tmp.store(std::plus<double>{}, static_cast<double>(i + 1));
        EXPECT_NE(tmp.get(), ptr->get());
    }
    EXPECT_TRUE(buffer.is_full()) << buffer;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
allocator_test(size_t count)
{
    using vector_t  = Tp;
    using comp_type = typename Tp::value_type;
    vector_t _vec{};
    _vec.reserve(n * (count - 1));
    for(size_t i = 0; i < n * count; ++i)
    {
        if(i % 3 == 0)
        {
            comp_type tmp{};
            tmp.store(static_cast<double>(i + 1));
            _vec.push_back(tmp);
        }
        else if(i % 3 == 1)
        {
            _vec.emplace_back();
            _vec.back().store(static_cast<double>(i + 1));
        }
        else
        {
            comp_type tmp{};
            _vec.resize(_vec.size() + 1, tmp);
            _vec.back().store(static_cast<double>(i + 1));
        }
    }
    for(size_t i = 0; i < n * count; ++i)
    {
        EXPECT_EQ(_vec.at(i).get(), static_cast<double>(i + 1));
    }
    vector_t _cvec{};
    {
        auto _nvec = std::move(_vec);
        for(size_t i = 0; i < n * count; ++i)
        {
            EXPECT_EQ(_nvec.at(i).get(), static_cast<double>(i + 1));
        }
        _cvec = _nvec;
    }
    for(size_t i = 0; i < n * count; ++i)
    {
        EXPECT_EQ(_cvec.at(i).get(), static_cast<double>(i + 1));
    }
}

TEST_F(ring_buffer_tests, allocator)
{
    using vector_t = std::vector<comp_type, tim::data::ring_buffer_allocator<comp_type>>;
    allocator_test<vector_t>(count);
}

TEST_F(ring_buffer_tests, allocator_mmap)
{
    using vector_t =
        std::vector<comp_type, tim::data::ring_buffer_allocator<comp_type, true>>;
    allocator_test<vector_t>(count);
}

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, buffer_size)
{
    using char_type      = char[sizeof(comp_type)];
    using allocator_type = tim::data::ring_buffer_allocator<char_type>;

    allocator_type::set_buffer_count(2 * count);
    allocator_type _alloc{};
    auto           _val = _alloc.allocate(1);
    _alloc.deallocate(_val, 1);

    auto _buffer_data = _alloc.get_buffer_data();
    ASSERT_NE(_buffer_data, nullptr);
    EXPECT_EQ(_buffer_data->dangles.size(), 1);
    ASSERT_EQ(_buffer_data->buffers.size(), 1);
    EXPECT_GE(_buffer_data->buffers.front()->capacity(), 2 * count);
}

TEST_F(ring_buffer_tests, buffer_size_mmap)
{
    using char_type      = char[sizeof(comp_type)];
    using allocator_type = tim::data::ring_buffer_allocator<char_type, true>;

    allocator_type::set_buffer_count(2 * count);
    allocator_type _alloc{};
    auto           _val = _alloc.allocate(1);
    _alloc.deallocate(_val, 1);

    auto _buffer_data = _alloc.get_buffer_data();
    ASSERT_NE(_buffer_data, nullptr);
    EXPECT_EQ(_buffer_data->dangles.size(), 1);
    ASSERT_EQ(_buffer_data->buffers.size(), 1);
    EXPECT_GE(_buffer_data->buffers.front()->capacity(), 2 * count);
}

//--------------------------------------------------------------------------------------//

TEST_F(ring_buffer_tests, buffer_size_tparam)
{
    using char_type      = char[sizeof(comp_type)];
    using allocator_type = tim::data::ring_buffer_allocator<char_type, false, 256>;

    allocator_type _alloc{};
    auto           _val = _alloc.allocate(1);
    _alloc.deallocate(_val, 1);

    auto _buffer_data = _alloc.get_buffer_data();
    ASSERT_NE(_buffer_data, nullptr);
    EXPECT_EQ(_buffer_data->dangles.size(), 1);
    ASSERT_EQ(_buffer_data->buffers.size(), 1);
    EXPECT_GE(_buffer_data->buffers.front()->capacity(), 256);
}

TEST_F(ring_buffer_tests, buffer_size_tparam_mmap)
{
    using char_type      = char[sizeof(comp_type)];
    using allocator_type = tim::data::ring_buffer_allocator<char_type, true, 256>;

    allocator_type _alloc{};
    auto           _val = _alloc.allocate(1);
    _alloc.deallocate(_val, 1);

    auto _buffer_data = _alloc.get_buffer_data();
    ASSERT_NE(_buffer_data, nullptr);
    EXPECT_EQ(_buffer_data->dangles.size(), 1);
    ASSERT_EQ(_buffer_data->buffers.size(), 1);
    EXPECT_GE(_buffer_data->buffers.front()->capacity(), 256);
}

//--------------------------------------------------------------------------------------//
