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

/** \file timemory/mpl/type_traits.hpp
 * \headerfile timemory/mpl/type_traits.hpp "timemory/mpl/type_traits.hpp"
 * These are the definitions of type-traits used by timemory and should be defined
 * separately from the class so that they can be queried without including the
 * definition of the component
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/defines.h"
#include "timemory/environment/types.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/utility/types.hpp"
//
#include <type_traits>

//======================================================================================//
//
//                                 Type Traits
//
//======================================================================================//

namespace tim
{
template <typename T>
struct statistics;

class manager;

namespace trait
{
//--------------------------------------------------------------------------------------//
//
template <typename TraitT>
inline std::string
as_string()
{
    constexpr bool _val = TraitT::value;
    return (_val) ? "true" : "false";
}

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::base_has_accum
/// \brief trait that signifies that a component has an accumulation value. In general,
/// most components implement 'value' and 'accum' data members of 'value_type'. Where
/// 'value' is generally used as intermediate storage between start/stop and after stop
/// have been called, 'value' is assigned as the difference between start/stop and added
/// to 'accum'. However, in the case where 'accum' is not a valid metric for the
/// component, this trait can be used to save memory bc it results in the 'accum' data
/// member to be implemented as a data-type of std::tuple<>, which only requires 1 byte of
/// memory.
///
template <typename T>
struct base_has_accum : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::base_has_last
/// \brief trait that signifies that a component has an "last" value which may be
/// different than the "value" value. In general, most components implement
/// 'value' and 'accum' data members of 'value_type'. Where 'value' is generally
/// used as intermediate storage between start/stop and after stop have been called,
/// 'value' is assigned as the difference between start/stop and added to 'accum'.
/// However, in the case where 'value' is valid as an individual measurement, this trait
/// can be used to store 'value' as the individual measurement and 'last' as the
/// difference or vice-versa.
///
template <typename T>
struct base_has_last : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::data
/// \brief trait to specify the value type of a component before the definition of
/// the component
///
template <typename T>
struct data
{
    // an empty type-list indicates the data type is not currently known
    // but empty tuple indicates that the type is unavailable
    using type = type_list<>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::component_apis
/// \brief trait to specify the APIs that the component is logically a part of
///
template <typename T>
struct component_apis
{
    using type = type_list<>;
};

template <typename T>
using component_apis_t = typename component_apis<T>::type;

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::supports_runtime_enabled
/// \brief trait to specify that the type supports toggling availablity at runtime.
/// Supporting runtime availability has a relatively miniscule overhead but disabling it
/// can ensure that the implementation truly maps down to the same assembly as
/// hand-implementations.
///
template <typename Tp>
struct supports_runtime_enabled : true_type
{};

//--------------------------------------------------------------------------------------//

template <>
struct runtime_enabled<void>
{
    static constexpr bool value = supports_runtime_enabled<void>::value;

    static TIMEMORY_INLINE bool get() { return get_global_value(); }
    static TIMEMORY_INLINE void set(bool val) { get_global_value() = val; }

    static TIMEMORY_INLINE bool get(scope::thread_scope) { return get_thread_value(); }
    static TIMEMORY_INLINE void set(scope::thread_scope, bool val)
    {
        get_thread_value() = val;
    }

private:
    static TIMEMORY_HOT bool& get_global_value()
    {
        static bool _v = TIMEMORY_DEFAULT_ENABLED;
        return _v;
    }

    static TIMEMORY_HOT bool& get_thread_value()
    {
        static thread_local bool _v = TIMEMORY_DEFAULT_ENABLED;
        return _v;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::default_runtime_enabled
/// \brief trait whose compile-time constant field `value` designates the default runtime
/// value of \ref tim::trait::runtime_enabled. Standard setting is true.
///
template <typename T>
struct default_runtime_enabled : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::runtime_enabled
/// \brief trait that signifies that an implementation is enabled at runtime. The
/// value returned from get() is for the specific setting for the type, the
/// global settings (type: void) and the specific settings for it's APIs
///
template <typename T>
struct runtime_enabled
{
    static constexpr bool value = supports_runtime_enabled<T>::value;

    /// type-list of APIs that are runtime configurable
    using category_type_list =
        mpl::get_true_types_t<concepts::is_runtime_configurable, component_apis_t<T>>;

    static TIMEMORY_INLINE bool get()
    {
        if constexpr(is_available<T>::value && supports_runtime_enabled<T>::value)
        {
            return (get(scope::thread_scope{}) && get_global_value() &&
                    get_category_value());
        }
        else
        {
            return is_available<T>::value && default_runtime_enabled<T>::value;
        }
    }

    static TIMEMORY_INLINE bool set(bool val)
    {
        if constexpr(is_available<T>::value && supports_runtime_enabled<T>::value)
        {
            return (get_global_value() = val);
        }
        else
        {
            return is_available<T>::value && default_runtime_enabled<T>::value;
            (void) val;
        }
    }

    //
    //
    //  thread section
    //
    //
    static TIMEMORY_INLINE bool get(scope::thread_scope)
    {
        if constexpr(is_available<T>::value && supports_runtime_enabled<T>::value)
        {
            return (get_thread_value() && get_category_value(scope::thread_scope{}));
        }
        else
        {
            return is_available<T>::value && default_runtime_enabled<T>::value;
        }
    }

    static TIMEMORY_INLINE bool set(scope::thread_scope, bool val)
    {
        if constexpr(is_available<T>::value && supports_runtime_enabled<T>::value)
        {
            return (get_thread_value() = val);
        }
        else
        {
            return is_available<T>::value && default_runtime_enabled<T>::value;
            (void) val;
        }
    }

private:
    template <typename... Args>
    static TIMEMORY_HOT bool get_category_value(Args... _args)
    {
        return runtime_enabled<void>::get(_args...) &&
               apply<runtime_enabled>{}(category_type_list{}, _args...);
    }

    static TIMEMORY_HOT bool& get_global_value()
    {
        static bool _v = is_available<T>::value && default_runtime_enabled<T>::value;
        return _v;
    }

#if defined(__GNUG__) && __GNUG__ < 8
    // using the form/style similar to above but with a thread-local value results in
    // an internal compiler error for GCC <= v7:
    //      internal compiler error: output_operand: '%&' used without
    //      any local dynamic TLS references
    static TIMEMORY_HOT bool& get_thread_value() { return thread_value; }
    static thread_local bool  thread_value;
#else
    static TIMEMORY_HOT bool& get_thread_value()
    {
        static thread_local bool _v =
            is_available<T>::value && default_runtime_enabled<T>::value;
        return _v;
    }
#endif
};

#if defined(__GNUG__) && __GNUG__ < 8
// compiler bug workaround
template <typename T>
thread_local bool            runtime_enabled<T>::thread_value =
    is_available<T>::value&& default_runtime_enabled<T>::value;

#endif

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::custom_label_printing
/// \brief trait that signifies that a component will handle printing the label(s)
///
template <typename T>
struct custom_label_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::custom_unit_printing
/// \brief trait that signifies that a component will handle printing the units(s)
///
template <typename T>
struct custom_unit_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::start_priority
/// \brief trait that designates whether there is a priority when starting the type w.r.t.
/// other types. Lower values indicate higher priority.
///
template <typename T>
struct start_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::stop_priority
/// \brief trait that designates whether there is a priority when stopping the type w.r.t.
/// other types. Lower values indicate higher priority.
///
template <typename T>
struct stop_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::fini_priority
/// \brief trait that designates whether there is a priority when finalizing the type
/// w.r.t. other types. Recommended for component which hold instances of other
/// components. Lower values indicate higher priority.
///
template <typename T>
struct fini_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_timing_category
/// \brief trait that designates the width and precision should follow formatting settings
/// related to timing measurements
///
template <typename T>
struct is_timing_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_memory_category
/// \brief trait that designates the width and precision should follow formatting settings
/// related to memory measurements
///
template <typename T>
struct is_memory_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_timing_units
/// \brief trait that designates the units should follow unit settings related to timing
/// measurements
///
template <typename T>
struct uses_timing_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_memory_units
/// \brief trait that designates the units should follow unit settings related to memory
/// measurements
///
template <typename T>
struct uses_memory_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_percent_units
/// \brief trait that designates the units are a percentage
///
template <typename T>
struct uses_percent_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_power_units
/// \brief trait that designates the value represents a power value
///
template <typename T>
struct uses_power_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_temperature_units
/// \brief trait that designates the value represents a temperature
///
template <typename T>
struct uses_temperature_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::requires_json
/// \brief trait that designates a type should always print a JSON output
///
template <typename T>
struct requires_json : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::api_components
/// \brief trait that designates components in an API (tim::api)
///
template <typename T, typename Tag>
struct api_components
{
    using type = type_list<>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::component_value_type
/// \brief trait that can be used to override the evaluation of the \ref
/// tim::trait::collects_data trait. It checks to see if \ref tim::trait::data is
/// specialized and, if not, evaluates to:
///
/// \code{.cpp}
/// typename T::value_type
/// \endcode
///
/// Unless the component has been marked as not available. If the component is not
/// available, the 'type' will always be void. When a component is available,
/// this trait will return the value type for a component regardless of whether
/// it base specified within the component definition or if it was declared via
/// a type-trait. Use the \ref tim::trait::collects_data for a constexpr boolean
/// 'value' for whether this value is a null type.
template <typename T>
struct component_value_type<T, true>
{
    using type = conditional_t<!std::is_same<type_list<>, typename data<T>::type>::value,
                               typename data<T>::type, typename T::value_type>;
};

template <typename T>
struct component_value_type<T, false>
{
    using type = void;
};

template <typename T>
using component_value_type_t =
    typename component_value_type<T, is_available<T>::value>::type;

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::collects_data
/// \brief trait that specifies or determines if a component collects any data. Default
/// behavior is to check if the component is available and extract the type and value
/// fields from \ref tim::trait::component_value_type. When a component is available,
/// the 'type' of this trait will return the 'value type' for a component regardless of
/// whether it was specified within the component definition or if it was declared via a
/// type-trait. The constexpr 'value' boolean indicates whether the 'type' is not a null
/// type.
///
template <typename T>
struct collects_data
{
    using type                  = component_value_type_t<T>;
    static constexpr bool value = (!concepts::is_null_type<type>::value);
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::supports_custom_record
/// \brief trait that designates the type supports changing the record() static function
/// per-instance
///
template <typename T>
struct supports_custom_record : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::iterable_measurement
/// \brief trait that signifies that get() returns an iterable type
///
template <typename T>
struct iterable_measurement : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::secondary_data
/// \brief trait that signifies that secondary data resembling the original data
/// exists but should be another node entry in the graph. These types
/// must provide a get_secondary() member function and that member function
/// must return a pair-wise iterable container, e.g. std::map, of types:
///     - std::string
///     - value_type
///
template <typename T>
struct secondary_data : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::thread_scope_only
/// \brief trait that signifies the component only has relevant values if it is not
/// collapsed into the master thread
///
template <typename T>
struct thread_scope_only : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::custom_serialization
/// \brief trait that signifies the component will be providing it's own split load(...)
/// and store(...) for serialization so do not provide one in the base class
///
template <typename T>
struct custom_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::record_statistics
/// \brief trait that signifies the component will calculate min/max/stddev
///
template <typename T>
struct record_statistics : default_record_statistics_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::statistics
/// \brief trait that specifies the data type of the statistics
///
template <typename T>
struct statistics
{
    using type = std::tuple<>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::permissive_statistics
/// \brief trait that will suppress compilation error in
/// `operation::add_statistics<Component>` if the data type passed is
/// implicitly convertible to the data type in `statistics<Component>::type`
/// but avoids converting integers to floating points and vice-versa.
///
template <typename T>
struct permissive_statistics : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::sampler
/// \brief trait that signifies the component supports sampling.
///
template <typename T>
struct sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::file_sampler
/// \brief trait that signifies the component samples a measurement from a file. If
/// multiple components sample from the same file, it is recommended to create a cache
/// type which performs a single read of the file and caches the values such that when
/// these components are bundled together, they can just read their data from the cache
/// structure.
///
/// See also: \ref tim::trait::cache
///
template <typename T>
struct file_sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::format_precision
/// \brief trait that specifies the default precision when printing
///
template <typename T>
struct format_precision
{
    static constexpr short value = trait::uses_percent_units<T>::value ? 1 : 3;

    auto operator()() const { return (*this)(0); }

private:
    template <typename U = T, short V = U::precision>
    auto operator()(int) const
    {
        return V;
    }
    template <typename U = T>
    auto operator()(long) const
    {
        return value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::format_width
/// \brief trait that specifies the default width when printing
///
template <typename T>
struct format_width
{
    static constexpr short value = trait::uses_percent_units<T>::value ? 6 : 8;

    auto operator()() const { return (*this)(0); }

private:
    template <typename U = T, short V = U::width>
    auto operator()(int) const
    {
        return V;
    }
    template <typename U = T>
    auto operator()(long) const
    {
        return value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::format_flags
/// \brief trait that specifies the default formatting flags when printing
///
template <typename T>
struct format_flags
{
    static constexpr std::ios_base::fmtflags value =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    auto operator()() const { return (*this)(0); }

private:
    template <typename U = T, std::ios_base::fmtflags V = U::format_flags>
    auto operator()(int) const
    {
        return V;
    }
    template <typename U = T>
    auto operator()(long) const
    {
        return value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::units
/// \brief trait that specifies the units
///
template <typename T>
struct units
{
    using type         = int64_t;
    using display_type = std::string;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::assignable_units
/// \brief trait that specifies the units should be assignable
///
template <typename T>
struct assignable_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::echo_enabled
/// \brief trait that configures echo_measurement usage
///
template <typename T>
struct echo_enabled : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::pretty_archive
/// \brief trait that configures whether output archive uses pretty formmatting. If set
/// to false_type then the JSON/XML/etc. will be compact (if supported)
///
template <typename T>
struct pretty_archive : std::false_type
{};

/// \struct tim::trait::api_input_archive
/// \brief trait that configures the default input archive type for an entire API
/// specification, e.g. TIMEMORY_API (which is `struct tim::project::timemory`). The input
/// archive format of individual components is determined from the derived \ref
/// tim::trait::input_archive
///
template <typename Api>
struct api_input_archive
{
    using type = TIMEMORY_INPUT_ARCHIVE;
};

/// \struct tim::trait::api_output_archive
/// \brief trait that configures the default output archive type for an entire API
/// specification, e.g. TIMEMORY_API (which is `struct tim::project::timemory`). The
/// output archive format of individual components is determined from the derived \ref
/// tim::trait::output_archive
///
template <typename Api>
struct api_output_archive
{
    using default_type = TIMEMORY_OUTPUT_ARCHIVE;

    static constexpr bool is_default_v = std::is_same<default_type, type_list<>>::value;
    static constexpr bool is_pretty_v  = pretty_archive<void>::value;

    using type = conditional_t<is_default_v,
                               conditional_t<is_pretty_v, cereal::PrettyJSONOutputArchive,
                                             cereal::MinimalJSONOutputArchive>,
                               default_type>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::input_archive
/// \brief trait that configures output archive type
///
template <typename T, typename Api>
struct input_archive
{
    using type = typename api_input_archive<Api>::type;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::output_archive
/// \brief trait that configures output archive type
///
template <typename T, typename Api>
struct output_archive
{
    using api_type = typename api_output_archive<Api>::type;

    using minimal_type = cereal::MinimalJSONOutputArchive;
    using pretty_type  = cereal::PrettyJSONOutputArchive;

    static constexpr bool is_pretty_v = pretty_archive<T>::value ||
                                        pretty_archive<Api>::value ||
                                        pretty_archive<void>::value;

    static constexpr bool is_json = (std::is_same<api_type, pretty_type>::value ||
                                     std::is_same<api_type, minimal_type>::value);

    using type = conditional_t<is_json, conditional_t<is_pretty_v, pretty_type, api_type>,
                               api_type>;
};

template <>
struct output_archive<manager, TIMEMORY_API>
{
    using type = cereal::PrettyJSONOutputArchive;
};

template <typename Api>
struct output_archive<manager, Api> : output_archive<manager, TIMEMORY_API>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::archive_extension
/// \brief Extension for the input or output archive types. It will throw an error if
/// used on new archive types and not specialized
template <typename T>
struct archive_extension
{
    using xml_types = type_list<cereal::XMLInputArchive, cereal::XMLOutputArchive>;
    using json_types =
        type_list<cereal::JSONInputArchive, cereal::PrettyJSONOutputArchive,
                  cereal::MinimalJSONOutputArchive>;
    using binary_types =
        type_list<cereal::BinaryInputArchive, cereal::BinaryOutputArchive,
                  cereal::PortableBinaryInputArchive,
                  cereal::PortableBinaryOutputArchive>;

    template <typename U = T>
    enable_if_t<is_one_of<U, xml_types>::value, std::string> operator()()
    {
        return ".xml";
    }

    template <typename U = T>
    enable_if_t<is_one_of<U, json_types>::value, std::string> operator()()
    {
        return ".json";
    }

    template <typename U = T>
    enable_if_t<is_one_of<U, binary_types>::value, std::string> operator()()
    {
        return ".dat";
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report
/// \brief trait that allows runtime configuration of reporting certain types of values.
/// Only applies to text output. This will allows modifying the value set by the
/// specific "report_*" type-trait.
///
template <typename T>
struct report
{
#define TIMEMORY_REPORT_GET_SET(FNAME, FIELD)                                            \
    static bool FNAME() { return get(FIELD); }                                           \
    static void FNAME(bool val) { set(FIELD, val); }

#define TIMEMORY_REPORT_ENV_QUERY(FIELD)                                                 \
    get_env<bool>("TIMEMORY_PRINT_" #FIELD, get_runtime_value().at(FIELD))

    enum field : short
    {
        COUNT = 0,
        DEPTH,
        METRIC,
        UNITS,
        SUM,
        MEAN,
        STATS,
        SELF,
        MIN,
        MAX,
        VARIANCE,
        STDDEV,
        FIELDS_END
    };

    using value_type = std::array<bool, FIELDS_END>;

    static bool get(short idx)
    {
        return get_runtime_value().at(idx % FIELDS_END) &&
               get_runtime_env().at(idx % FIELDS_END);
    }

    static void set(short idx, bool val)
    {
        get_runtime_value().at(idx % FIELDS_END) = val;
    }

    TIMEMORY_REPORT_GET_SET(count, COUNT)
    TIMEMORY_REPORT_GET_SET(depth, DEPTH)
    TIMEMORY_REPORT_GET_SET(metric, METRIC)
    TIMEMORY_REPORT_GET_SET(units, UNITS)
    TIMEMORY_REPORT_GET_SET(sum, SUM)
    TIMEMORY_REPORT_GET_SET(mean, MEAN)
    TIMEMORY_REPORT_GET_SET(stats, STATS)
    TIMEMORY_REPORT_GET_SET(self, SELF)
    TIMEMORY_REPORT_GET_SET(min, MIN)
    TIMEMORY_REPORT_GET_SET(max, MAX)
    TIMEMORY_REPORT_GET_SET(variance, VARIANCE)
    TIMEMORY_REPORT_GET_SET(stddev, STDDEV)

private:
    static value_type& get_runtime_value()
    {
        static value_type _v{
            { report_count<T>::value, (report_depth<T>::value && !flat_storage<T>::value),
              report_metric_name<T>::value, report_units<T>::value, report_sum<T>::value,
              report_mean<T>::value && !timeline_storage<T>::value,
              report_statistics<T>::value, report_self<T>::value,
              report_statistics<T>::value, report_statistics<T>::value,
              report_statistics<T>::value, report_statistics<T>::value }
        };
        return _v;
    }

    static value_type get_runtime_env()
    {
        return value_type{
            TIMEMORY_REPORT_ENV_QUERY(COUNT),    TIMEMORY_REPORT_ENV_QUERY(DEPTH),
            TIMEMORY_REPORT_ENV_QUERY(METRIC),   TIMEMORY_REPORT_ENV_QUERY(UNITS),
            TIMEMORY_REPORT_ENV_QUERY(SUM),      TIMEMORY_REPORT_ENV_QUERY(MEAN),
            TIMEMORY_REPORT_ENV_QUERY(STATS),    TIMEMORY_REPORT_ENV_QUERY(SELF),
            TIMEMORY_REPORT_ENV_QUERY(MIN),      TIMEMORY_REPORT_ENV_QUERY(MAX),
            TIMEMORY_REPORT_ENV_QUERY(VARIANCE), TIMEMORY_REPORT_ENV_QUERY(STDDEV)
        };
    }

#undef TIMEMORY_REPORT_ENV_QUERY
#undef TIMEMORY_REPORT_GET_SET
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_count
/// \brief trait that configures type to not report the number of lap count (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_count : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_count
/// \brief trait that configures type to not report the number of lap count (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_depth : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_metric_name
/// \brief trait that configures type to not report the "METRIC" column, useful if
/// redundant). Only applies to text output.
///
template <typename T>
struct report_metric_name : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_units
/// \brief trait that configures type to not report the "UNITS" column (useful if always
/// empty). Only applies to text output.
///
template <typename T>
struct report_units : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_sum
/// \brief trait that configures type to not report the accumulated value (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_sum : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_mean
/// \brief trait that configures type to not report the mean value (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_mean : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_statistics
/// \brief trait that configures type to not report the "UNITS" column (useful if always
/// empty). Only applies to text output and does NOT affect whether statistics are
/// accumulated. For disabling statistics completely, see \ref
/// tim::trait::record_statistics and \ref tim::policy::record_statistics.
///
template <typename T>
struct report_statistics : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_self
/// \brief trait that configures type to not report the % self field (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_self : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::supports_flamegraph
/// \brief trait that designates a type supports flamegraph output
///
template <typename T>
struct supports_flamegraph : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::derivation_types
/// \brief trait that designates the type supports calling assemble and derive member
/// functions with these types. Specializations MUST be structured as a
/// tim::type_list<...> of tim::type_list<...> where each inner type_list entry is
/// the list of component types required to perform a derivation.
/// \code{.cpp}
/// template <>
/// struct derivation_types<cpu_util>
/// {
///     // can derive its data when present alongside wall_clock + cpu_clock and/or
///     // wall_clock + user_clock + system_clock
///     using type = type_list<
///         type_list<wall_clock, cpu_clock>,
///         type_list<wall_clock, user_clock, system_clock>
///     >;
/// };
/// \endcode
///
template <typename T>
struct derivation_types : false_type
{
    static constexpr size_t size = 0;
    using type                   = std::tuple<type_list<>>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::python_args
/// \brief trait that designates the type supports these arguments from python.
/// Specializations MUST be structured as either one `tim::type_list<...>` or
/// a `tim::type_list<...>` of `tim::type_list<...>`.
/// The first argument is a \ref TIMEMORY_OPERATION enumerated type and for each
/// inner \ref tim::type_list, a python member function for the stand-alone component
/// will be generated with those arguments. E.g. to create a custom store member function
/// accepting integer:
/// \code{.py}
/// foo = timemory.component.CaliperLoopMarker("example")
/// foo.start()
/// for i in range(10):
///     foo.store(i)    # store member function accepting integer
///     # ...
/// foo.stop()
/// \endcode
/// The type-trait specification would look like this:
/// \code{.cpp}
/// template <>
/// struct python_args<TIMEMORY_STORE, component::caliper_loop_marker>
/// {
///     using type = type_list<size_t>;
/// };
/// \endcode
///
template <int OpT, typename T>
struct python_args
{
    using type = type_list<type_list<>>;
};

//--------------------------------------------------------------------------------------//

template <int OpT, typename T>
using python_args_t = typename python_args<OpT, T>::type;

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::cache
/// \brief trait that specifies the intermediate data type that will hold the relevant
/// data required by the component. This is useful for when multiple components
/// read different parts of the same file (e.g. /proc/<PID>/io) or an API
/// reports data in a larger data structure than the scope of the component (e.g. rusage)
/// but multiple components require access to this data structure
///
template <typename T>
struct cache
{
    using type = null_type;
};

//--------------------------------------------------------------------------------------//
//
//      determines if output is generated
//
//--------------------------------------------------------------------------------------//
/// \struct tim::trait::generates_output
/// \brief trait used to evaluate whether a component value type produces a useable value
template <typename T, typename V>
struct generates_output
{
    using type                  = V;
    static constexpr bool value = (!concepts::is_null_type<type>::value);
};

template <typename T>
struct generates_output<T, void>
{
    using type                  = void;
    static constexpr bool value = false;
};

template <typename T>
struct generates_output<T, null_type>
{
    using type                  = null_type;
    static constexpr bool value = false;
};

template <typename T>
struct generates_output<T, type_list<>>
{
    // this is default evaluation from trait::data<T>::type
    using type                  = typename T::value_type;
    static constexpr bool value = (!concepts::is_null_type<type>::value);
};

template <typename T>
struct generates_output<T, data<T>> : generates_output<T, typename data<T>::type>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_storage
/// \brief trait that designates that a component will instantiate tim::storage
///
template <typename T>
struct uses_storage : is_available<T>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::tree_storage
/// \brief trait that configures type to always use hierarchical call-stack storage
///
template <typename T>
struct tree_storage : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::flat_storage
/// \brief trait that configures type to always use flat call-stack storage
///
template <typename T>
struct flat_storage : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::timeline_storage
/// \brief trait that configures type to always use timeline call-stack storage
///
template <typename T>
struct timeline_storage : false_type
{};

//--------------------------------------------------------------------------------------//
//
//      determines if storage should be implemented
//
//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_value_storage
/// \brief This trait is used to determine whether the (expensive) instantiation of the
/// storage class happens
template <typename T, typename V, typename A>
struct uses_value_storage
{
    using value_type               = V;
    static constexpr bool avail_v  = (A::value && trait::is_available<T>::value);
    static constexpr bool output_v = trait::generates_output<T, value_type>::value;
    static constexpr bool value    = (avail_v && output_v);
};

// this specialization is from trait::data<T> when using storage
template <typename T>
struct uses_value_storage<T, type_list<>, true_type>
{
    using value_type            = typename T::value_type;
    static constexpr bool value = trait::generates_output<T, value_type>::value;
};

// this specialization is from trait::data<T> when not using storage
template <typename T>
struct uses_value_storage<T, type_list<>, false_type>
{
    using value_type            = void;
    static constexpr bool value = false;
};

template <typename T, typename A>
struct uses_value_storage<T, void, A>
{
    using value_type            = void;
    static constexpr bool value = false;
};

template <typename T, typename A>
struct uses_value_storage<T, null_type, A>
{
    using value_type            = null_type;
    static constexpr bool value = false;
};

// this specialization is from trait::data<T>
template <typename T, typename A>
struct uses_value_storage<T, type_list<>, A>
: uses_value_storage<T, type_list<>, conditional_t<A::value, true_type, false_type>>
{};

/// \struct tim::trait::perfetto_category
/// \brief Provides the static category for perfetto traces
template <typename ApiT>
struct perfetto_category;

template <>
struct perfetto_category<TIMEMORY_API>
{
    static constexpr auto value = TIMEMORY_PROJECT_NAME;
};

/// \struct tim::trait::prevent_reentry
/// \brief trait that specifies that the type should protect itself from recursion
/// or reentry if necessary globally.
template <typename Tp>
struct prevent_reentry : std::true_type
{};

/// \struct tim::trait::static_data
/// \brief trait that specifies that the type (or types used by this type)
/// should use static data retrieved via:
/// `decltype(auto) _value = policy::static_data<Tp>{}(Args...);`
/// The user must provide the policy implementation. See \ref tim::policy::static_data.
///
template <typename Tp>
struct static_data : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_component
/// \brief trait that designates the type is a timemory component
/// \deprecated{ This has been migrated to `tim::concepts` }
///
template <typename T>
struct is_component : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_gotcha
/// \brief trait that designates the type is a gotcha
/// \deprecated{ This has been migrated to `tim::concepts` }
///
template <typename T>
struct is_gotcha : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_user_bundle
/// \brief trait that designates the type is a user-bundle
/// \deprecated{ This has been migrated to `tim::concepts` }
///
template <typename T>
struct is_user_bundle : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::record_max
/// \brief trait that signifies that updating w.r.t. another instance should
/// be a max of the two instances
/// \deprecated This is no longer used. Overload the operators for +=, -=, etc. to obtain
/// previous functionality.
//
template <typename T>
struct record_max : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::array_serialization
/// \brief trait that signifies that data is an array type
/// \deprecated { This trait is no longer used as array types are determined by other
/// means }
///
template <typename T>
struct array_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::requires_prefix
/// \brief trait that signifies that a component requires the prefix to be set right after
/// construction. Types with this trait must contain a member string variable named
/// prefix
/// \deprecated { This trait is no longer used as this property is determined by other
/// means }
///
template <typename T>
struct requires_prefix : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::supports_args
/// \brief trait that designates the type supports calling a function with a certain
/// set of argument types (passed via a tuple).
/// \deprecated This is legacy code and support for calling a function with given
/// arguments is automatically determined.
///
template <typename T, typename Tuple>
struct supports_args : false_type
{};

//--------------------------------------------------------------------------------------//
}  // namespace trait
}  // namespace tim
