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
 * \file timemory/components/data_tracker/components.hpp
 * \brief Implementation of the data_tracker component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/data_tracker/types.hpp"
#include "timemory/data/handler.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/demangle.hpp"

#if defined(TIMEMORY_PYBIND11_SOURCE)
#    include "pybind11/cast.h"
#    include "pybind11/pybind11.h"
#    include "pybind11/stl.h"
#endif

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

//======================================================================================//
//
namespace tim
{
namespace component
{
/// \struct tim::component::data_tracker
/// \brief This component is provided to facilitate data tracking. The first
/// template parameter is the type of data to be tracked, the second is a custom tag
/// for differentiating trackers which handle the same data types but record
/// different high-level data.
///
/// Usage:
/// \code{.cpp}
/// // declarations
///
/// struct myproject {};
///
/// using itr_tracker_type   = data_tracker<uint64_t, myproject>;
/// using err_tracker_type   = data_tracker<double, myproject>;
///
/// // add statistics capabilities
/// TIMEMORY_STATISTICS_TYPE(itr_tracker_type, int64_t)
/// TIMEMORY_STATISTICS_TYPE(err_tracker_type, double)
///
/// // set the label and descriptions
/// TIMEMORY_METADATA_SPECIALIZATION(
///     itr_tracker_type, "myproject_iterations", "short desc", "long description")
///
/// TIMEMORY_METADATA_SPECIALIZATION(
///     err_tracker_type, "myproject_convergence", "short desc", "long description")
///
/// // this is the generic bundle pairing a timer with an iteration tracker
/// // using this and not updating the iteration tracker will create entries
/// // in the call-graph with zero iterations.
/// using bundle_t           = tim::auto_tuple<wall_clock, itr_tracker_type>;
///
/// // this is a dedicated bundle for adding data-tracker entries. This style
/// // can also be used with the iteration tracker or you can bundle
/// // both trackers together. The auto_tuple will call start on construction
/// // and stop on destruction so once can construct a nameless temporary of the
/// // this bundle type and call store(...) on the nameless tmp. This will
/// // ensure that the statistics are updated for each entry
/// //
/// using err_bundle_t       = tim::auto_tuple<err_tracker_type>;
///
/// // usage in a function is implied below
///
/// double err             = std::numeric_limits<double>::max();
/// const double tolerance = 1.0e-6;
///
/// bundle_t t("iteration_time");
///
/// while(err > tolerance)
/// {
///     // store the starting error
///     double initial_err = err;
///
///     // add 1 for each iteration. Stats only updated when t is destroyed or t.stop() is
///     // called t.store(std::plus<uint64_t>{}, 1);
///
///     // ... do something ...
///
///     // construct a nameless temporary which records the change in the error and
///     // update the statistics <-- "foo" will have mean/min/max/stddev of the
///     // error
///     err_bundle_t{ "foo" }.store(err - initial_err);
///
///     // NOTE: std::plus is used with t above bc it has multiple components so std::plus
///     // helps ensure 1 doesn't get added to some other component with `store(int)`
///     // In above err_bundle_t, there is only one component so there is not concern.
/// }
/// \endcode
///
/// When creating new data trackers, it is recommended to have this in header:
///
/// \code{.cpp}
/// TIMEMORY_DECLARE_EXTERN_COMPONENT(custom_data_tracker_t, true, data_type)
/// \endcode
///
/// And this in *one* source file (preferably one that is not re-compiled often)
///
/// \code{.cpp}
/// TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(custom_data_tracker_t, true, data_type)
/// TIMEMORY_INITIALIZE_STORAGE(custom_data_tracker_t)
/// \endcode
///
/// where `custom_data_tracker_t` is the custom data tracker type (or an alias to the
/// type) and `data_type` is the data type being tracked.
///
template <typename InpT, typename Tag>
struct data_tracker : public base<data_tracker<InpT, Tag>, InpT>
{
    using value_type   = InpT;
    using this_type    = data_tracker<InpT, Tag>;
    using base_type    = base<this_type, value_type>;
    using handler_type = data::handler<InpT, Tag>;

    friend struct data::handler<InpT, Tag>;
    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    // a pair
    template <typename LhsT, typename RhsT, typename HashT = std::hash<LhsT>>
    class vector_map : public std::vector<std::pair<LhsT, RhsT>>
    {
    public:
        using value_type     = std::pair<LhsT, RhsT>;
        using base_type      = std::vector<value_type>;
        using iterator       = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using hash_type      = HashT;
        using key_type       = LhsT;
        using mapped_type    = RhsT;

        template <typename... Args>
        vector_map(Args&&... args)
        : base_type{ std::forward<Args>(args)... }
        {}

        template <typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            return { base_type::emplace(base_type::end(), std::forward<Args>(args)...),
                     true };
        }

        iterator find(const key_type& _key)
        {
            auto _hkey = hash_type{}(_key);
            for(auto itr = base_type::begin(); itr != base_type::end(); ++itr)
            {
                if(hash_type{}(itr->first) == _hkey)
                    return itr;
            }
            return base_type::end();
        }

        const_iterator find(const key_type& _key) const
        {
            auto _hkey = hash_type{}(_key);
            for(auto itr = base_type::begin(); itr != base_type::end(); ++itr)
            {
                if(hash_type{}(itr->first) == _hkey)
                    return itr;
            }
            return base_type::end();
        }

        mapped_type& operator[](const key_type& _key) { return find(_key)->second; }
    };

private:
    // private aliases
    template <typename T, typename U = int>
    using enable_if_acceptable_t =
        enable_if_t<concepts::is_acceptable_conversion<decay_t<T>, InpT>::value, U>;

    using value_ptr_t     = std::shared_ptr<value_type>;
    using secondary_map_t = vector_map<std::string, this_type>;
    using secondary_ptr_t = std::shared_ptr<secondary_map_t>;
    using start_t =
        operation::generic_operator<this_type, operation::start<this_type>, Tag>;
    using stop_t =
        operation::generic_operator<this_type, operation::stop<this_type>, Tag>;
    using compute_type = math::compute<InpT, typename trait::units<this_type>::type>;

public:
    //----------------------------------------------------------------------------------//
    //
    //  standard interface
    //
    //----------------------------------------------------------------------------------//
    /// a reference is returned here so that it can be easily updated
    static std::string& label();

    /// a reference is returned here so that it can be easily updated
    static std::string& description();

    /// override to avoid caching in default implementation
    static std::string get_label() { return label(); }

    /// override to avoid caching in default implementation
    static std::string get_description() { return description(); }

    using base_type::display_unit;
    using base_type::get_display_unit;
    using base_type::get_unit;
    using base_type::set_display_unit;
    using base_type::set_unit;
    using base_type::unit;

    // default set of ctor and assign
    TIMEMORY_DEFAULT_OBJECT(data_tracker)

    void start() {}
    void stop() {}

    /// get the data in the final form after unit conversion
    auto get() const { return handler_type::get(*this); }

    /// get the data in a form suitable for display
    auto get_display() const { return handler_type::get_display(*this); }

    /// map of the secondary entries. When TIMEMORY_ADD_SECONDARY is enabled
    /// contents of this map will be added as direct children of the current
    /// node in the call-graph.
    auto get_secondary() const
    {
        return (m_secondary) ? *m_secondary : secondary_map_t{};
    }

    using base_type::get_depth_change;
    using base_type::get_is_flat;
    using base_type::get_is_on_stack;
    using base_type::get_is_running;
    using base_type::get_is_transient;
    using base_type::get_iterator;
    using base_type::get_value;
    using base_type::laps;
    using base_type::load;

    //----------------------------------------------------------------------------------//
    //
    //  store
    //
    //----------------------------------------------------------------------------------//
    /// store some data. Uses \ref tim::data::handler for the type.
    template <typename T>
    void store(T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which takes a handler to ensure proper overload resolution
    template <typename T>
    void store(handler_type&&, T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values
    template <typename FuncT, typename T>
    auto store(FuncT&& f, T&& val, enable_if_acceptable_t<T, int> = 0)
        -> decltype(std::declval<handler_type>().store(*this, std::forward<FuncT>(f),
                                                       std::forward<T>(val)),
                    void());

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values and takes a handler to ensure proper overload
    /// resolution
    template <typename FuncT, typename T>
    auto store(handler_type&&, FuncT&& f, T&& val, enable_if_acceptable_t<T, int> = 0)
        -> decltype(std::declval<handler_type>().store(*this, std::forward<FuncT>(f),
                                                       std::forward<T>(val)),
                    void());

    //----------------------------------------------------------------------------------//
    //
    //  mark begin
    //
    //----------------------------------------------------------------------------------//
    /// The combination of `mark_begin(...)` and `mark_end(...)` can be used to
    /// store some initial data which may be needed later. When `mark_end(...)` is
    /// called, the value is updated with the difference of the value provided to
    /// `mark_end` and the temporary stored during `mark_begin`.
    template <typename T>
    void mark_begin(T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which takes a handler to ensure proper overload resolution
    template <typename T>
    void mark_begin(handler_type&&, T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values
    template <typename FuncT, typename T>
    void mark_begin(FuncT&& f, T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values and takes a handler to ensure proper
    /// overload resolution
    template <typename FuncT, typename T>
    void mark_begin(handler_type&&, FuncT&& f, T&& val,
                    enable_if_acceptable_t<T, int> = 0);

    //----------------------------------------------------------------------------------//
    //
    //  mark end
    //
    //----------------------------------------------------------------------------------//
    /// The combination of `mark_begin(...)` and `mark_end(...)` can be used to
    /// store some initial data which may be needed later. When `mark_end(...)` is
    /// called, the value is updated with the difference of the value provided to
    /// `mark_end` and the temporary stored during `mark_begin`. It may be valid
    /// to call `mark_end` without calling `mark_begin` but the result will effectively
    /// be a more expensive version of calling `store`.
    template <typename T>
    void mark_end(T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which takes a handler to ensure proper overload resolution
    template <typename T>
    void mark_end(handler_type&&, T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values
    template <typename FuncT, typename T>
    void mark_end(FuncT&& f, T&& val, enable_if_acceptable_t<T, int> = 0);

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values and takes a handler to ensure proper
    /// overload resolution
    template <typename FuncT, typename T>
    void mark_end(handler_type&&, FuncT&& f, T&& val, enable_if_acceptable_t<T, int> = 0);

    //----------------------------------------------------------------------------------//
    //
    //  add secondary
    //
    //----------------------------------------------------------------------------------//
    /// add a secondary value to the current node in the call-graph.
    /// When TIMEMORY_ADD_SECONDARY is enabled contents of this map will be added as
    /// direct children of the current node in the call-graph. This is useful
    /// for finer-grained details that might not always be desirable to display
    template <typename T>
    this_type* add_secondary(const std::string& _key, T&& val,
                             enable_if_acceptable_t<T, int> = 0);

    /// overload which takes a handler to ensure proper overload resolution
    template <typename T>
    this_type* add_secondary(const std::string& _key, handler_type&& h, T&& val,
                             enable_if_acceptable_t<T, int> = 0);

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values
    template <typename FuncT, typename T>
    this_type* add_secondary(const std::string& _key, FuncT&& f, T&& val,
                             enable_if_acceptable_t<T, int> = 0);

    /// overload which uses a lambda to bypass the default behavior of how the
    /// handler updates the values and takes a handler to ensure proper
    /// overload resolution
    template <typename FuncT, typename T>
    this_type* add_secondary(const std::string& _key, handler_type&& h, FuncT&& f,
                             T&& val, enable_if_acceptable_t<T, int> = 0);

    //----------------------------------------------------------------------------------//
    //
    //  non-standard interface
    //
    //----------------------------------------------------------------------------------//

    /// set the current value
    void set_value(const value_type& v) { value = v; }

    /// set the current value via move
    void set_value(value_type&& v) { value = std::move(v); }

    using base_type::value;

    void       set_last(value_type&& _v) { m_last_value = std::move(_v); }
    void       set_last(value_type& _v) { m_last_value = _v; }
    value_type get_last() const { return m_last_value; }

private:
    /// map of the secondary entries. When TIMEMORY_ADD_SECONDARY is enabled
    /// contents of this map will be added as direct children of the current
    /// node in the call-graph
    auto get_secondary_map()
    {
        if(!m_secondary)
            m_secondary = std::make_shared<secondary_map_t>();
        return m_secondary;
    }

    void allocate_temporary() const
    {
        if(!m_last)
            m_last = std::make_shared<value_type>();
    }

    value_type&       get_temporary() { return (allocate_temporary(), *m_last); }
    const value_type& get_temporary() const { return (allocate_temporary(), *m_last); }

private:
    mutable value_ptr_t m_last{ nullptr };
    secondary_ptr_t     m_secondary{ nullptr };
    value_type          m_last_value{};
};
//
/// \typedef typename T::handler_type tim::component::data_handler_t
/// \brief an alias for getting the handle_type of a data tracker
template <typename T>
using data_handler_t = typename T::handler_type;
//
/// \typedef tim::component::data_tracker<intmax_t, TIMEMORY_API>
/// tim::component::data_tracker_integer
///
/// \brief Specialization of \ref tim::component::data_tracker for storing signed integer
/// data
using data_tracker_integer = data_tracker<intmax_t, TIMEMORY_API>;
//
/// \typedef tim::component::data_tracker<size_t, TIMEMORY_API>
/// tim::component::data_tracker_unsigned
///
/// \brief Specialization of \ref tim::component::data_tracker for storing unsigned
/// integer data
using data_tracker_unsigned = data_tracker<size_t, TIMEMORY_API>;
//
/// \typedef tim::component::data_tracker<double, TIMEMORY_API>
/// tim::component::data_tracker_floating
///
/// \brief Specialization of \ref tim::component::data_tracker for storing floating point
/// data
using data_tracker_floating = data_tracker<double, TIMEMORY_API>;
//
//
template <typename InpT, typename Tag>
inline std::string&
data_tracker<InpT, Tag>::label()
{
    static std::string _instance = []() {
        if(metadata<this_type>::specialized())
            return metadata<this_type>::label();
        return TIMEMORY_JOIN("_", typeid(Tag).name(), typeid(InpT).name());
    }();
    return _instance;
}
//
template <typename InpT, typename Tag>
std::string&
data_tracker<InpT, Tag>::description()
{
    static std::string _instance = []() {
        if(metadata<this_type>::specialized())
        {
            auto meta_desc = metadata<this_type>::description();
            if(settings::verbose() > 0 || settings::debug())
            {
                auto meta_extra_desc = metadata<this_type>::extra_description();
                meta_desc += ". ";
                meta_desc += meta_extra_desc;
            }
            return meta_desc;
        }
        std::stringstream ss;
        ss << "Data tracker for data of type " << demangle<InpT>() << " for "
           << demangle<Tag>();
        return ss.str();
    }();
    return _instance;
}
//
template <typename InpT, typename Tag>
template <typename T>
void
data_tracker<InpT, Tag>::store(T&& val, enable_if_acceptable_t<T, int>)
{
    set_last(compute_type::divide(std::forward<T>(val), get_unit()));
    handler_type::store(*this, get_last());
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename T>
void
data_tracker<InpT, Tag>::store(handler_type&&, T&& val, enable_if_acceptable_t<T, int>)
{
    set_last(compute_type::divide(std::forward<T>(val), get_unit()));
    handler_type::store(*this, get_last());
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
auto
data_tracker<InpT, Tag>::store(FuncT&& f, T&& val, enable_if_acceptable_t<T, int>)
    -> decltype(std::declval<handler_type>().store(*this, std::forward<FuncT>(f),
                                                   std::forward<T>(val)),
                void())
{
    set_last(compute_type::divide(std::forward<T>(val), get_unit()));
    handler_type::store(*this, std::forward<FuncT>(f), get_last());
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
auto
data_tracker<InpT, Tag>::store(handler_type&&, FuncT&& f, T&& val,
                               enable_if_acceptable_t<T, int>)
    -> decltype(std::declval<handler_type>().store(*this, std::forward<FuncT>(f),
                                                   std::forward<T>(val)),
                void())
{
    set_last(compute_type::divide(std::forward<T>(val), get_unit()));
    handler_type::store(*this, std::forward<FuncT>(f), get_last());
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename T>
void
data_tracker<InpT, Tag>::mark_begin(T&& val, enable_if_acceptable_t<T, int>)
{
    handler_type::begin(get_temporary(),
                        compute_type::divide(std::forward<T>(val), get_unit()));
}
//
template <typename InpT, typename Tag>
template <typename T>
void
data_tracker<InpT, Tag>::mark_end(T&& val, enable_if_acceptable_t<T, int>)
{
    handler_type::end(*this, compute_type::divide(std::forward<T>(val), get_unit()));
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename T>
void
data_tracker<InpT, Tag>::mark_begin(handler_type&&, T&& val,
                                    enable_if_acceptable_t<T, int>)
{
    handler_type::begin(get_temporary(),
                        compute_type::divide(std::forward<T>(val), get_unit()));
}
//
template <typename InpT, typename Tag>
template <typename T>
void
data_tracker<InpT, Tag>::mark_end(handler_type&&, T&& val, enable_if_acceptable_t<T, int>)
{
    handler_type::end(*this, compute_type::divide(std::forward<T>(val), get_unit()));
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
void
data_tracker<InpT, Tag>::mark_begin(FuncT&& f, T&& val, enable_if_acceptable_t<T, int>)
{
    handler_type::begin(get_temporary(), std::forward<FuncT>(f),
                        compute_type::divide(std::forward<T>(val), get_unit()));
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
void
data_tracker<InpT, Tag>::mark_end(FuncT&& f, T&& val, enable_if_acceptable_t<T, int>)
{
    handler_type::end(*this, std::forward<FuncT>(f),
                      compute_type::divide(std::forward<T>(val), get_unit()));
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
void
data_tracker<InpT, Tag>::mark_begin(handler_type&&, FuncT&& f, T&& val,
                                    enable_if_acceptable_t<T, int>)
{
    handler_type::begin(get_temporary(), std::forward<FuncT>(f),
                        compute_type::divide(std::forward<T>(val), get_unit()));
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
void
data_tracker<InpT, Tag>::mark_end(handler_type&&, FuncT&& f, T&& val,
                                  enable_if_acceptable_t<T, int>)
{
    handler_type::end(*this, std::forward<FuncT>(f),
                      compute_type::divide(std::forward<T>(val), get_unit()));
    if(!get_is_running())
        ++laps;
}
//
template <typename InpT, typename Tag>
template <typename T>
data_tracker<InpT, Tag>*
data_tracker<InpT, Tag>::add_secondary(const std::string& _key, T&& val,
                                       enable_if_acceptable_t<T, int>)
{
    this_type _tmp{};
    if(get_is_running())
        start_t{ _tmp };
    _tmp.store(std::forward<T>(val));
    if(get_is_running())
        stop_t{ _tmp };
    auto& _map = *get_secondary_map();
    _map.emplace(_key, std::move(_tmp));
    return &(_map[_key]);
}
//
template <typename InpT, typename Tag>
template <typename T>
data_tracker<InpT, Tag>*
data_tracker<InpT, Tag>::add_secondary(const std::string& _key, handler_type&& h, T&& val,
                                       enable_if_acceptable_t<T, int>)
{
    this_type _tmp{};
    if(get_is_running())
        start_t{ _tmp };
    _tmp.store(std::forward<handler_type>(h), std::forward<T>(val));
    if(get_is_running())
        stop_t{ _tmp };
    auto& _map = *get_secondary_map();
    if(_map.empty())
        _map.reserve(20);
    _map.emplace(_key, std::move(_tmp));
    return &(_map[_key]);
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
data_tracker<InpT, Tag>*
data_tracker<InpT, Tag>::add_secondary(const std::string& _key, FuncT&& f, T&& val,
                                       enable_if_acceptable_t<T, int>)
{
    this_type _tmp{};
    if(get_is_running())
        start_t{ _tmp };
    _tmp.store(std::forward<FuncT>(f), std::forward<T>(val));
    if(get_is_running())
        stop_t{ _tmp };
    auto& _map = *get_secondary_map();
    if(_map.empty())
        _map.reserve(20);
    _map.emplace(_key, std::move(_tmp));
    return &(_map[_key]);
}
//
template <typename InpT, typename Tag>
template <typename FuncT, typename T>
data_tracker<InpT, Tag>*
data_tracker<InpT, Tag>::add_secondary(const std::string& _key, handler_type&& h,
                                       FuncT&& f, T&& val, enable_if_acceptable_t<T, int>)
{
    this_type _tmp{};
    if(get_is_running())
        start_t{ _tmp };
    _tmp.store(std::forward<handler_type>(h), std::forward<FuncT>(f),
               std::forward<T>(val));
    if(get_is_running())
        stop_t{ _tmp };
    auto& _map = *get_secondary_map();
    if(_map.empty())
        _map.reserve(20);
    _map.emplace(_key, std::move(_tmp));
    return &(_map[_key]);
}
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
