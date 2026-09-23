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

#pragma once

#include "timemory/components/base.hpp"  // for component base class
#include "timemory/components/timestamp/types.hpp"
#include "timemory/data.hpp"     // for data specialization
#include "timemory/mpl/stl.hpp"  // for math specialization
#include "timemory/tpls/cereal/types.hpp"

#include <chrono>
#include <cstdint>
#include <ostream>
#include <string>

namespace tim
{
namespace component
{
//
/// \struct tim::component::timestamp
/// \brief this component stores the timestamp of when a bundle was started
/// and is specialized such that the "timeline_storage" type-trait is true.
/// This means that every entry in the call-graph for this output will be unique
/// (look in the timestamp.txt output file)
///
struct timestamp : base<timestamp, timestamp_entry_t>
{
    using value_type      = timestamp_entry_t;
    using base_type       = base<timestamp, value_type>;
    using clock_type      = std::chrono::system_clock;
    using time_point_type = typename clock_type::time_point;
    using duration_type   = std::chrono::duration<clock_type::rep, std::nano>;

    static std::string label();
    static std::string description();
    static value_type  record();
    static value_type  get_reference_ts();
    static std::string as_string(const time_point_type& _tp);
    static int64_t     count();

    template <typename ArchiveT>
    static void extra_serialization(ArchiveT&);

    void        sample();
    void        start();
    value_type  get() const;
    std::string get_display() const;

    template <typename ArchiveT>
    void load(ArchiveT& ar, const unsigned);

    template <typename ArchiveT>
    void save(ArchiveT& ar, const unsigned) const;

    timestamp& operator+=(const timestamp&);
    timestamp& operator/=(const timestamp&);
    timestamp& operator/=(int64_t);

    friend std::ostream& operator<<(std::ostream& _os, const timestamp& _ts)
    {
        _os << _ts.get_display();
        return _os;
    }
};

template <typename ArchiveT>
void
timestamp::load(ArchiveT& ar, const unsigned)
{
    time_t _val{};
    ar(cereal::make_nvp("time_since_epoch", _val));
    set_value(std::chrono::system_clock::from_time_t(_val));
}

template <typename ArchiveT>
void
timestamp::save(ArchiveT& ar, const unsigned) const
{
    auto _val = std::chrono::system_clock::to_time_t(get_value());
    ar(cereal::make_nvp("time_since_epoch", _val));
}

template <typename ArchiveT>
void
timestamp::extra_serialization(ArchiveT& ar)
{
    ar(cereal::make_nvp("reference_timestamp", get_reference_ts()));
}

}  // namespace component

//--------------------------------------------------------------------------------------//

namespace data
{
namespace base
{
using timestamp_value_t = typename component::timestamp::time_point_type;
//
template <>
void
stream_entry::construct<timestamp_value_t>(const timestamp_value_t& val);
}  // namespace base
}  // namespace data
}  // namespace tim

#if defined(TIMEMORY_TIMESTAMP_COMPONENT_HEADER_MODE) &&                                 \
    TIMEMORY_TIMESTAMP_COMPONENT_HEADER_MODE > 0
#    include "timemory/components/timestamp/timestamp.cpp"
#endif
