//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/io/components.hpp
 * \brief Implementation of the io component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/io/backends.hpp"
#include "timemory/components/io/types.hpp"

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
/// \struct tim::component::read_char
/// \brief I/O counter for chars read. The number of bytes which this task has caused to
/// be read from storage. This is simply the sum of bytes which this process passed to
/// read() and pread(). It includes things like tty IO and it is unaffected by whether or
/// not actual physical disk IO was required (the read might have been satisfied from
/// pagecache)
struct read_char : public base<read_char, std::pair<int64_t, int64_t>>
{
    using this_type         = read_char;
    using value_type        = std::pair<int64_t, int64_t>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::pair<double, double>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string              label() { return "read_char"; }
    static std::vector<std::string> label_array();
    static std::string              description();
    static std::vector<std::string> description_array();
    static unit_type                unit();
    static unit_type                unit_array();
    static unit_type                get_unit();
    static display_unit_type        display_unit();
    static std::vector<std::string> display_unit_array();
    static display_unit_type        get_display_unit();
    static int64_t                  get_timing_unit();
    static int64_t                  get_timestamp();
    static value_type               record();

    std::string get_display() const;
    result_type get() const;

    void start();
    void stop();

    /// sample a measurement
    void sample();

    /// sample a measurement from cached data
    void sample(const cache_type& _cache);

    /// read the value from cached data
    static value_type record(const cache_type& _cache);

    /// start a measurement using the cached data
    void start(const cache_type& _cache);

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache);
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::written_char
/// \brief I/O counter for chars written. The number of bytes which this task has caused,
/// or shall cause to be written to disk. Similar caveats apply here as with \ref
/// tim::component::read_char (rchar).
struct written_char : public base<written_char, std::array<int64_t, 2>>
{
    using this_type         = written_char;
    using value_type        = std::array<int64_t, 2>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::array<double, 2>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string              label() { return "written_char"; }
    static std::vector<std::string> label_array();
    static std::string              description();
    static std::vector<std::string> description_array();
    static unit_type                unit();
    static unit_type                unit_array();
    static unit_type                get_unit();
    static display_unit_type        display_unit();
    static std::vector<std::string> display_unit_array();
    static display_unit_type        get_display_unit();
    static int64_t                  get_timing_unit();
    static int64_t                  get_timestamp();
    static value_type               record();

    std::string get_display() const;
    result_type get() const;

    void start();
    void stop();

    /// sample a measurement
    void sample();

    /// sample a measurement from cached data
    void sample(const cache_type& _cache);

    /// read the value from cached data
    static value_type record(const cache_type& _cache);

    /// start a measurement using the cached data
    void start(const cache_type& _cache);

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache);
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::read_bytes
/// \brief I/O counter for bytes read. Attempt to count the number of bytes which this
/// process really did cause to be fetched from the storage layer. Done at the
/// submit_bio() level, so it is accurate for block-backed filesystems.
struct read_bytes : public base<read_bytes, std::pair<int64_t, int64_t>>
{
    using this_type         = read_bytes;
    using value_type        = std::pair<int64_t, int64_t>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::pair<double, double>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string              label() { return "read_bytes"; }
    static std::vector<std::string> label_array();
    static std::string              description();
    static std::vector<std::string> description_array();
    static unit_type                unit();
    static unit_type                unit_array();
    static unit_type                get_unit();
    static display_unit_type        display_unit();
    static std::vector<std::string> display_unit_array();
    static display_unit_type        get_display_unit();
    static int64_t                  get_timing_unit();
    static int64_t                  get_timestamp();
    static value_type               record();

    std::string get_display() const;
    result_type get() const;

    void start();
    void stop();

    /// sample a measurement
    void sample();

    /// sample a measurement from cached data
    void sample(const cache_type& _cache);

    /// read the value from the cache
    static value_type record(const cache_type& _cache);

    /// start a measurement using the cached data
    void start(const cache_type& _cache);

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache);
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::written_bytes
/// \brief I/O counter for bytes written. Attempt to count the number of bytes which this
/// process caused to be sent to the storage layer. This is done at page-dirtying time.
struct written_bytes : public base<written_bytes, std::array<int64_t, 2>>
{
    using this_type         = written_bytes;
    using value_type        = std::array<int64_t, 2>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::array<double, 2>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string              label() { return "written_bytes"; }
    static std::vector<std::string> label_array();
    static std::string              description();
    static std::vector<std::string> description_array();
    static unit_type                unit();
    static unit_type                unit_array();
    static unit_type                get_unit();
    static display_unit_type        display_unit();
    static std::vector<std::string> display_unit_array();
    static display_unit_type        get_display_unit();
    static int64_t                  get_timing_unit();
    static int64_t                  get_timestamp();
    static value_type               record();

    std::string get_display() const;
    result_type get() const;

    void start();
    void stop();

    /// sample a measurement
    void sample();

    /// sample a measurement from cached data
    void sample(const cache_type& _cache);

    /// read the value from the cache
    static value_type record(const cache_type& _cache);

    /// start a measurement using the cached data
    void start(const cache_type& _cache);

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache);
};
}  // namespace component
}  // namespace tim

#if defined(TIMEMORY_IO_COMPONENT_HEADER_MODE) && TIMEMORY_IO_COMPONENT_HEADER_MODE > 0
#    include "timemory/components/io/components.cpp"
#endif
