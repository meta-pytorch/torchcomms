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

#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"

#include <atomic>
#include <iosfwd>
#include <memory>
#include <tuple>

namespace tim
{
//
template <typename Tp>
struct statistics;
//
namespace policy
{
//======================================================================================//

template <typename CompT, typename Tp>
struct record_statistics
{
    using type            = Tp;
    using this_type       = record_statistics<CompT, type>;
    using policy_type     = this_type;
    using statistics_type = statistics<type>;

    void operator()(statistics<type>&, const CompT&, bool _accum = true);
    void operator()(type&, const CompT&) {}
};

//--------------------------------------------------------------------------------------//
//
template <typename CompT>
struct record_statistics<CompT, void>
{
    using type            = void;
    using this_type       = record_statistics<CompT, type>;
    using policy_type     = this_type;
    using statistics_type = statistics<type>;

    template <typename... ArgsT>
    void operator()(ArgsT&&...)
    {}
};

//--------------------------------------------------------------------------------------//
//
template <typename CompT>
struct record_statistics<CompT, std::tuple<>>
{
    using type            = std::tuple<>;
    using this_type       = record_statistics<CompT, type>;
    using policy_type     = this_type;
    using statistics_type = statistics<type>;

    template <typename... ArgsT>
    void operator()(ArgsT&&...)
    {}
};

//======================================================================================//

template <typename Archive, typename Api>
struct input_archive
{
    using type    = Archive;
    using pointer = std::shared_ptr<type>;

    static pointer get(std::istream& is) { return std::make_shared<type>(is); }
};

//--------------------------------------------------------------------------------------//

template <typename Api>
struct input_archive<cereal::JSONInputArchive, Api>
{
    using type    = cereal::JSONInputArchive;
    using pointer = std::shared_ptr<type>;

    static pointer get(std::istream& is) { return std::make_shared<type>(is); }
};

//--------------------------------------------------------------------------------------//

template <typename Api>
struct input_archive<cereal::PrettyJSONOutputArchive, Api>
{
    using type    = cereal::JSONInputArchive;
    using pointer = std::shared_ptr<type>;

    static pointer get(std::istream& is) { return std::make_shared<type>(is); }
};

//--------------------------------------------------------------------------------------//

template <typename Api>
struct input_archive<cereal::MinimalJSONOutputArchive, Api>
{
    using type    = cereal::JSONInputArchive;
    using pointer = std::shared_ptr<type>;

    static pointer get(std::istream& is) { return std::make_shared<type>(is); }
};

//======================================================================================//

template <typename Archive, typename Api>
struct output_archive
{
    using type    = Archive;
    using pointer = std::shared_ptr<type>;

    static pointer get(std::ostream& os) { return std::make_shared<type>(os); }
};

//--------------------------------------------------------------------------------------//

template <typename Api>
struct output_archive<cereal::PrettyJSONOutputArchive, Api>
{
    using type        = cereal::PrettyJSONOutputArchive;
    using pointer     = std::shared_ptr<type>;
    using option_type = typename type::Options;
    using indent_type = typename option_type::IndentChar;

    static unsigned int& precision()
    {
        static unsigned int value = 16;
        return value;
    }

    static unsigned int& indent_length()
    {
        static unsigned int value = 1;
        return value;
    }

    static indent_type& indent_char()
    {
        static indent_type value = indent_type::space;
        return value;
    }

    static pointer get(std::ostream& os)
    {
        //  Option args: precision, spacing, indent size
        option_type opts(precision(), indent_char(), indent_length());
        return std::make_shared<type>(os, opts);
    }
};

//--------------------------------------------------------------------------------------//
///
/// partial specialization for MinimalJSONOutputArchive
///
template <typename Api>
struct output_archive<cereal::MinimalJSONOutputArchive, Api>
{
    using type        = cereal::MinimalJSONOutputArchive;
    using pointer     = std::shared_ptr<type>;
    using option_type = typename type::Options;
    using indent_type = typename option_type::IndentChar;

    static unsigned int& precision()
    {
        static unsigned int value = 16;
        return value;
    }
    static unsigned int& indent_length()
    {
        static unsigned int value = 0;
        return value;
    }
    static indent_type& indent_char()
    {
        static indent_type value = indent_type::space;
        return value;
    }

    static pointer get(std::ostream& os)
    {
        //  Option args: precision, spacing, indent size
        //  The last two options are meaningless for the minimal writer
        option_type opts(precision(), indent_char(), indent_length());
        return std::make_shared<type>(os, opts);
    }
};

//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_XML)
///
/// partial specialization for XMLOutputArchive
///
template <typename Api>
struct output_archive<cereal::XMLOutputArchive, Api>
{
    using type        = cereal::XMLOutputArchive;
    using pointer     = std::shared_ptr<type>;
    using option_type = typename type::Options;
    using indent_type = bool;
    using output_type = bool;

    static unsigned int& precision()
    {
        static unsigned int value = 16;
        return value;
    }

    // indent the output
    static indent_type& indent()
    {
        static indent_type value =
            trait::pretty_archive<Api>::value || trait::pretty_archive<void>::value;
        return value;
    }

    // output the type as an attribute
    static output_type& type_attribute()
    {
        static output_type value = false;
        return value;
    }

    // Whether dynamically sized containers output the size=dynamic attribute
    static output_type& size_attribute()
    {
        static output_type value = false;
        return value;
    }

    static pointer get(std::ostream& os)
    {
        //  Option args: precision, spacing, indent size
        //  The last two options are meaningless for the minimal writer
        option_type opts(precision(), indent(), type_attribute(), size_attribute());
        return std::make_shared<type>(os, opts);
    }
};
#endif

//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_YAML)
///
/// partial specialization for YAMLOutputArchive
///
template <typename Api>
struct output_archive<cereal::YAMLOutputArchive, Api>
{
    using type    = cereal::YAMLOutputArchive;
    using pointer = std::shared_ptr<type>;

    static pointer get(std::ostream& os)
    {
        //  Option args: precision, spacing, indent size
        //  The last two options are meaningless for the minimal writer
        return std::make_shared<type>(os);
    }
};
#endif
//======================================================================================//

template <typename Tp>
struct instance_tracker<Tp, true>
{
public:
    using type                           = Tp;
    using int_type                       = int64_t;
    using pair_type                      = std::pair<int_type, int_type>;
    static constexpr bool thread_support = true;

    TIMEMORY_DEFAULT_OBJECT(instance_tracker)

    enum
    {
        global_count = 0,
        thread_count
    };

public:
    //----------------------------------------------------------------------------------//
    //
    static int_type get_started_count() { return get_started().load(); }

    //----------------------------------------------------------------------------------//
    //
    static int_type get_thread_started_count() { return get_thread_started(); }

protected:
    //----------------------------------------------------------------------------------//
    //
    static std::atomic<int_type>& get_started()
    {
        static std::atomic<int_type> _instance(0);
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    static int_type& get_thread_started()
    {
        static thread_local int_type _instance = 0;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    auto start()
    {
        m_tot = get_started()++;
        m_thr = get_thread_started()++;
        return pair_type{ m_tot, m_thr };
    }

    auto stop()
    {
        m_tot = --get_started();
        m_thr = --get_thread_started();
        return pair_type{ m_tot, m_thr };
    }

    //----------------------------------------------------------------------------------//
    // increment/decrement global and thread counts and return global count
    template <size_t Idx>
    enable_if_t<Idx == global_count, int_type> start()
    {
        m_tot = get_started()++;
        m_thr = get_thread_started()++;
        return m_tot;
    }

    template <size_t Idx>
    enable_if_t<Idx == global_count, int_type> stop()
    {
        m_tot = --get_started();
        m_thr = --get_thread_started();
        return m_tot;
    }

    //----------------------------------------------------------------------------------//
    // increment/decrement global and thread counts and return thread count
    template <size_t Idx>
    enable_if_t<Idx == thread_count, int_type> start()
    {
        m_tot = get_started()++;
        m_thr = get_thread_started()++;
        return m_thr;
    }

    template <size_t Idx>
    enable_if_t<Idx == thread_count, int_type> stop()
    {
        m_tot = --get_started();
        m_thr = --get_thread_started();
        return m_thr;
    }

    auto get_global_count() { return m_tot; }
    auto get_thread_count() { return m_tot; }

    auto global_tracker_start() { return (start(), m_tot); }
    auto global_tracker_stop() { return (stop(), m_tot); }
    auto thread_tracker_start() { return (start(), m_thr); }
    auto thread_tracker_stop() { return (stop(), m_thr); }

protected:
    int_type m_tot = get_started_count();
    int_type m_thr = get_thread_started_count();
};

//======================================================================================//

template <typename Tp>
struct instance_tracker<Tp, false>
{
public:
    using type                           = Tp;
    using int_type                       = int64_t;
    static constexpr bool thread_support = false;

    instance_tracker()                            = default;
    ~instance_tracker()                           = default;
    instance_tracker(const instance_tracker&)     = default;
    instance_tracker(instance_tracker&&) noexcept = default;
    instance_tracker& operator=(const instance_tracker&) = default;
    instance_tracker& operator=(instance_tracker&&) noexcept = default;

public:
    //----------------------------------------------------------------------------------//
    //
    static int_type get_started_count() { return get_started().load(); }

protected:
    //----------------------------------------------------------------------------------//
    //
    static std::atomic<int_type>& get_started()
    {
        static std::atomic<int_type> _instance(0);
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    auto start()
    {
        m_tot = get_started()++;
        return m_tot;
    }

    //----------------------------------------------------------------------------------//
    //
    auto stop()
    {
        m_tot = --get_started();
        return m_tot;
    }

    auto get_global_count() { return m_tot; }
    auto global_tracker_start() { return (start(), m_tot); }
    auto global_tracker_stop() { return (stop(), m_tot); }

protected:
    int_type m_tot = get_started_count();
};

//======================================================================================//

}  // namespace policy
}  // namespace tim
