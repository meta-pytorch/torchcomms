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

#if defined(TIMEMORY_USE_STATISTICS)
#    undef TIMEMORY_USE_STATISTICS
#endif

#define TIMEM_DEBUG
#define TIMEMORY_DISABLE_BANNER
#define TIMEMORY_DISABLE_STORE_ENVIRONMENT
#define TIMEMORY_DISABLE_CEREAL_CLASS_VERSION
#define TIMEMORY_DISABLE_COMPONENT_STORAGE_INIT
#define TIMEMORY_DISABLE_SETTINGS_SERIALIZATION

// disables unnecessary instantiations
#define TIMEMORY_COMPILER_INSTRUMENTATION

// tweaks settings
#define TIMEMORY_SETTINGS_PREFIX "TIMEM_"
#define TIMEMORY_PROJECT_NAME "timem"

#include "timemory/api/macros.hpp"

// create an API for the compiler instrumentation whose singletons will not be shared
// with the default timemory API
// TIMEMORY_DEFINE_NS_API(project, timem)

// define the API for all instantiations before including any more timemory headers
// #define TIMEMORY_API ::tim::project::timem

#include "timemory/macros.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/macros.hpp"

#if defined(TIMEMORY_MACOS)
TIMEMORY_FORWARD_DECLARE_COMPONENT(page_rss)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::page_rss, false_type)
#endif

namespace tim
{
namespace trait
{
template <>
struct pretty_archive<void> : true_type
{};
}  // namespace trait
}  // namespace tim

#include "timemory/components/timing/child.hpp"
#include "timemory/general.hpp"
#include "timemory/operations/types/finalize/mpi_get.hpp"
#include "timemory/sampling.hpp"
#include "timemory/timemory.hpp"

// clang-format off
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::papi_array_t, true_type)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::wall_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::child_user_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::child_system_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::child_cpu_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::child_cpu_util, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::peak_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::page_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::virtual_memory, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::num_major_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::num_minor_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::priority_context_switch, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::voluntary_context_switch, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::read_char, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::read_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::written_char, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::network_stats, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::papi_array_t, false_type)
// clang-format on

#include "md5.hpp"

// C includes
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(TIMEMORY_USE_LIBEXPLAIN)
#    include <libexplain/execvp.h>
#endif

#if defined(TIMEMORY_UNIX)
#    include <unistd.h>
extern "C"
{
    extern char** environ;
}
#endif

// C++ includes
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

template <typename Tp>
using vector_t       = std::vector<Tp>;
using string_t       = std::string;
using stringstream_t = std::stringstream;
using mutex_t        = std::mutex;
using auto_lock_t    = std::unique_lock<mutex_t>;

using namespace tim::component;

template <typename Tp>
mutex_t&
type_mutex()
{
    return ::tim::type_mutex<Tp, TIMEMORY_API, 1, mutex_t>(0);
}

//--------------------------------------------------------------------------------------//
// create a custom component tuple printer
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              OPERATION SPECIALIZATION
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
struct set_print_rank
{};
//
template <typename Tp>
struct print_properties
{
    print_properties(const Tp&, set_print_rank, int32_t _rank) { rank() = _rank; }
    static int32_t& rank()
    {
        static int32_t _v = -1;
        return _v;
    }
};
//
template <typename Tp>
struct custom_print
{
    using value_type = typename Tp::value_type;
    using base_type  = component::base<Tp, value_type>;

    custom_print(std::size_t N, std::size_t /*_Ntot*/, base_type& obj, std::ostream& os)
    {
        if(!tim::trait::runtime_enabled<Tp>::get())
            return;

        stringstream_t ss;
        if(N == 0)
            ss << std::endl;
        ss << "    ";
        if(print_properties<Tp>::rank() > -1)
            ss << print_properties<Tp>::rank() << "|> ";
        ss << obj << std::endl;
        os << ss.str();
    }
};
//
/// this is a custom version of base_printer for {read,written}_{char,bytes} components
template <typename Tp>
struct custom_base_printer
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    template <typename Up                                      = value_type,
              enable_if_t<!std::is_same<Up, void>::value, int> = 0>
    explicit custom_base_printer(std::ostream& _os, const type& _obj, int32_t _rank,
                                 const std::string& _label)
    {
        stringstream_t ss;
        stringstream_t ssv;
        stringstream_t ssr;
        stringstream_t ssrank;
        auto           _prec  = base_type::get_precision();
        auto           _width = base_type::get_width();
        auto           _flags = base_type::get_format_flags();
        auto           _disp  = _obj.get_display_unit();
        auto           _val   = _obj.get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ss << ssv.str() << " " << _label;

        tim::consume_parameters(_rank);

        _os << ss.str();
    }

    template <typename Up = value_type, typename... Args,
              enable_if_t<std::is_same<Up, void>::value, int> = 0>
    explicit custom_base_printer(std::ostream&, const type&, Args&&...)
    {}
};
//
template <>
struct custom_base_printer<component::network_stats>
{
    using type       = component::network_stats;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    template <typename... Args>
    custom_base_printer(std::ostream& _os, const type& _obj, int32_t _rank, Args&&...)
    {
        auto _prec   = base_type::get_precision();
        auto _width  = base_type::get_width();
        auto _flags  = base_type::get_format_flags();
        auto _units  = type::unit_array();
        auto _disp   = type::display_unit_array();
        auto _labels = type::label_array();
        auto _val    = _obj.load();

        auto _data = _val.get_data();
        for(size_t i = 0; i < _data.size(); ++i)
            _data.at(i) /= _units.at(i);

        std::vector<size_t> _order{};
        // print tx after rx
        for(size_t i = 0; i < _data.size() / 2; ++i)
        {
            _order.emplace_back(i);
            _order.emplace_back(i + _data.size() / 2);
        }
        // account for odd size if expanded
        if(_data.size() % 2 == 1)
            _order.emplace_back(_data.size() - 1);

        for(size_t i = 0; i < _data.size(); ++i)
        {
            auto           idx = _order.at(i);
            stringstream_t ss;
            stringstream_t ssv;
            stringstream_t ssrank;
            ssv.setf(_flags);
            ssv << std::setw(_width) << std::setprecision(_prec) << _data.at(idx);
            if(!_disp.at(idx).empty())
                ssv << " " << _disp.at(idx);

            if(i > 0)
            {
                ssrank << "\n    ";
                if(_rank > -1)
                {
                    ssrank << _rank << "|> ";
                }
            }

            auto _label = _labels.at(idx);
            _label =
                str_transform(_label, "rx_", "_", [](const std::string&) -> std::string {
                    return "network_receive";
                });
            _label =
                str_transform(_label, "tx_", "_", [](const std::string&) -> std::string {
                    return "network_transmit";
                });

            ss << ssrank.str() << ssv.str() << " " << _label;
            _os << ss.str();
        }
    }
};
//
#define CUSTOM_BASE_PRINTER_SPECIALIZATION(TYPE, LABEL)                                  \
    template <>                                                                          \
    struct base_printer<TYPE> : custom_base_printer<TYPE>                                \
    {                                                                                    \
        using type = TYPE;                                                               \
        static int32_t& rank() { return print_properties<TYPE>::rank(); }                \
        explicit base_printer(std::ostream& _os, const type& _obj)                       \
        : custom_base_printer<type>(_os, _obj, rank(), LABEL)                            \
        {}                                                                               \
    };
//
CUSTOM_BASE_PRINTER_SPECIALIZATION(component::read_bytes, "bytes_read")
CUSTOM_BASE_PRINTER_SPECIALIZATION(component::read_char, "char_read")
CUSTOM_BASE_PRINTER_SPECIALIZATION(component::written_bytes, "bytes_written")
CUSTOM_BASE_PRINTER_SPECIALIZATION(component::written_char, "char_written")
CUSTOM_BASE_PRINTER_SPECIALIZATION(component::network_stats, "")
//
#if defined(TIMEMORY_USE_PAPI)
//
template <size_t N>
struct start<component::papi_array<N>>
{
    using type = papi_array<N>;

    TIMEMORY_DEFAULT_OBJECT(start)

    template <typename... Args>
    explicit start(type&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(type&, Args&&...) const
    {}
};
//
template <size_t N>
struct stop<component::papi_array<N>>
{
    using type = component::papi_array<N>;

    TIMEMORY_DEFAULT_OBJECT(stop)

    template <typename... Args>
    explicit stop(type&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(type&, Args&&...) const
    {}
};
//
#endif
//
template <>
struct stop<component::page_rss>
{
    using type = component::page_rss;

    TIMEMORY_DEFAULT_OBJECT(stop)

    template <typename... Args>
    explicit stop(type&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(type&, Args&&...) const
    {}
};
//
template <>
struct stop<component::virtual_memory>
{
    using type = component::virtual_memory;

    TIMEMORY_DEFAULT_OBJECT(stop)

    template <typename... Args>
    explicit stop(type&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(type&, Args&&...) const
    {}
};
//
template <>
struct start<component::network_stats>
{
    using type = network_stats;

    TIMEMORY_DEFAULT_OBJECT(start)

    template <typename... Args>
    explicit start(type&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(type&, Args&&...) const
    {}
};
//
template <>
struct stop<component::network_stats>
{
    using type = component::network_stats;

    TIMEMORY_DEFAULT_OBJECT(stop)

    template <typename... Args>
    explicit stop(type&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(type&, Args&&...) const
    {}
};
//
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
//                              COMPONENT SPECIALIZATION
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
//
#if defined(TIMEMORY_USE_PAPI)
//
template <>
inline std::string
papi_array_t::get_display() const
{
    auto events = common_type::get_config()->event_names;
    if(events.empty())
        return "";
    auto val          = load();
    auto _get_display = [&](std::ostream& os, size_type idx) {
        auto     _obj_value = val[idx];
        auto     _evt_type  = events[idx];
        string_t _label     = papi::get_event_info(_evt_type).short_descr;
        string_t _disp      = papi::get_event_info(_evt_type).units;
        auto     _prec      = base_type::get_precision();
        auto     _width     = base_type::get_width();
        auto     _flags     = base_type::get_format_flags();

        stringstream_t ss, ssv, ssi;
        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << _obj_value;
        if(!_disp.empty())
            ssv << " " << _disp;
        if(!_label.empty())
            ssi << " " << _label;
        if(idx > 0 && operation::print_properties<papi_array_t>::rank() > -1)
            ss << operation::print_properties<papi_array_t>::rank() << "|> ";
        ss << ssv.str() << ssi.str();
        if(idx > 0)
            os << "    ";
        os << ss.str();
    };

    stringstream_t ss;
    for(size_type i = 0; i < events.size(); ++i)
    {
        _get_display(ss, i);
        if(i + 1 < events.size())
            ss << '\n';
    }

    return ss.str();
}
//
#endif
//
}  // namespace component
//
//--------------------------------------------------------------------------------------//
//
//                         VARIADIC WRAPPER SPECIALIZATION
//
//--------------------------------------------------------------------------------------//
//
/// \class tim::timem_tuple
/// \brief A specialized variadic component wrapper which inherits from the
/// lightweight_tuple which does not automatically push and pop to storage
///
template <typename... Types>
class timem_tuple : public lightweight_tuple<Types...>
{
public:
    using base_type       = lightweight_tuple<Types...>;
    using apply_v         = tim::mpl::apply<void>;
    using data_type       = typename base_type::impl_type;
    using this_type       = timem_tuple<Types...>;
    using clock_type      = std::chrono::system_clock;
    using time_point_type = typename clock_type::time_point;
    using hist_type       = std::pair<time_point_type, data_type>;

    template <template <typename> class Op, typename Tuple = data_type>
    using custom_operation_t =
        typename base_type::template custom_operation<Op, Tuple>::type;

    template <typename... T>
    struct mpi_getter;

    template <template <typename...> class Tuple, typename... T>
    struct mpi_getter<Tuple<T...>>
    {
        using value_type = Tuple<std::vector<T>...>;
    };

public:
    TIMEMORY_DEFAULT_OBJECT(timem_tuple)

    explicit timem_tuple(string_view_cref_t key)
    : base_type{ key }
    {}

    timem_tuple(string_view_cref_t _key, data_type&& _data)
    : base_type{ _key }
    {
        m_data = std::move(_data);
    }

    using base_type::get;
    using base_type::get_labeled;
    using base_type::reset;
    using base_type::start;
    using base_type::stop;

    void set_output(std::ofstream* ofs) { m_ofs = ofs; }

    template <typename FuncT>
    void set_notify(FuncT&& _v)
    {
        tim::sampling::set_notify(m_notify, std::forward<FuncT>(_v));
    }

    void set_history(std::vector<hist_type>* _v) { m_data_hist = _v; }

    size_t get_buffer_size() const { return m_collect_size; }

    void set_buffer_size(size_t _v)
    {
        ::auto_lock_t _lk{ ::type_mutex<hist_type>(), std::defer_lock };
        if(!_lk.owns_lock())
            _lk.lock();
        m_collect_size = _v;
        m_collect_hist = (m_collect_size > 0);
        if(m_collect_hist)
        {
            m_hist_buff.reserve(std::max<size_t>(m_hist_buff.capacity(), _v));
        }
    }

    std::vector<hist_type>& swap_history(std::vector<hist_type>& _v)
    {
        ::auto_lock_t _lk{ ::type_mutex<hist_type>(), std::defer_lock };
        if(!_lk.owns_lock())
            _lk.lock();
        std::swap(m_hist_buff, _v);
        _lk.unlock();
        return _v;
    }

    template <typename... Args>
    void sample(Args&&... args)
    {
        if(base_type::m_is_active())
        {
            stop();
            base_type::sample(std::forward<Args>(args)...);
            if(m_collect_hist)
            {
                ::auto_lock_t _lk{ ::type_mutex<hist_type>(), std::defer_lock };
                if(!_lk.owns_lock())
                    _lk.lock();

                if(m_hist_buff.size() < m_collect_size)
                {
                    m_hist_buff.emplace_back(clock_type::now(), m_data);
                    if(m_hist_buff.size() + 1 >= m_collect_size)
                        m_notify(nullptr);
                }
            }
            if(m_ofs)
            {
                (*m_ofs) << get_local_datetime("[===== %r %F =====]\n") << *this
                         << std::endl;
            }
            start();
        }
    }

    auto mpi_get()
    {
        constexpr auto N      = std::tuple_size<data_type>::value;
        auto           v_data = typename mpi_getter<data_type>::value_type{};

        // merge the data
        mpi_get(v_data, m_data, make_index_sequence<N>{});

        // merge the histories
        using hist_vec_t = std::vector<hist_type>;
        auto _hist_add   = [](hist_vec_t& _lhs, const hist_vec_t& _rhs) -> hist_vec_t& {
            for(auto& itr : _rhs)
                _lhs.emplace_back(itr);
            return _lhs;
        };
        std::vector<hist_vec_t> _hist{};
        operation::finalize::mpi_get<hist_vec_t, true>{
            _hist, (m_data_hist) ? *m_data_hist : m_hist_buff, _hist_add
        };

        // return an array of this_type
        return mpi_get(v_data, _hist, make_index_sequence<N>{});
    }

    friend std::ostream& operator<<(std::ostream& os, const timem_tuple<Types...>& obj)
    {
        stringstream_t ssp;
        stringstream_t ssd;
        auto&&         _data  = obj.m_data;
        auto&&         _key   = obj.key();
        auto&&         _width = obj.output_width();

        using cprint_t = custom_operation_t<operation::custom_print, data_type>;
        mpl::apply<void>::access_with_indices<cprint_t>(_data, std::ref(ssd));

        ssp << std::setw(_width) << std::left << _key;
        os << ssp.str() << ssd.str();

        if(&os != obj.m_ofs && obj.m_ofs)
        {
            *(obj.m_ofs) << get_local_datetime("[===== %r %F =====]\n") << ssp.str()
                         << ssd.str() << std::endl;
        }

        return os;
    }

    void set_rank(int32_t _rank)
    {
        mpl::apply<void>::access<
            custom_operation_t<operation::print_properties, data_type>>(
            this->m_data, operation::set_print_rank{}, _rank);
    }

    TIMEMORY_NODISCARD bool empty() const { return m_empty; }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _timestamp_str = [](const time_point_type& _tp) {
            char _repr[64];
            std::memset(_repr, '\0', sizeof(_repr));
            std::time_t _value = std::chrono::system_clock::to_time_t(_tp);
            // alternative: "%c %Z"
            if(std::strftime(_repr, sizeof(_repr), "%a %b %d %T %Y %Z",
                             std::localtime(&_value)))
                return std::string{ _repr };
            return std::string{};
        };

        using data_tuple_type = decay_t<decltype(m_data)>;
        constexpr auto N      = std::tuple_size<data_tuple_type>::value;
        serialize_tuple(ar, m_data, make_index_sequence<N>{});

        auto* _hist = (m_data_hist) ? m_data_hist : &m_hist_buff;
        ar.setNextName("history");
        ar.startNode();
        ar.makeArray();
        for(auto& itr : *_hist)
        {
            ar.startNode();
            ar.setNextName("sample_timestamp");
            ar.startNode();
            ar(cereal::make_nvp("localtime", _timestamp_str(itr.first)));
            ar(cereal::make_nvp("time_since_epoch",
                                itr.first.time_since_epoch().count()));
            ar.finishNode();
            serialize_tuple(ar, itr.second, make_index_sequence<N>{});
            ar.finishNode();
        }
        ar.finishNode();
    }

    template <typename Up, typename Tp = decay_t<Up>>
    static auto get_metadata_label()
    {
        auto _name = metadata<Tp>::name();
        for(auto&& itr : { "::", "child_" })
        {
            auto _pos = std::string::npos;
            while((_pos = _name.find(itr)) != std::string::npos)
            {
                _name = _name.substr(_pos + std::string(itr).length());
            }
        }
        _name = _name.substr(0, _name.find('<'));
        return _name;
    }

    template <typename Archive, typename Tp>
    static auto serialize_entry(Archive& ar, Tp&& _obj)
    {
        if(!trait::runtime_enabled<decay_t<Tp>>::get())
            return;
        auto _name = get_metadata_label<Tp>();
        ar.setNextName(_name.c_str());
        ar.startNode();
        ar(cereal::make_nvp("value", _obj.get()));
        ar(cereal::make_nvp("repr", _obj.get_display()));
        ar(cereal::make_nvp("laps", _obj.get_laps()));
        ar(cereal::make_nvp("unit_value", _obj.get_unit()));
        ar(cereal::make_nvp("unit_repr", _obj.get_display_unit()));
        ar.finishNode();
        // ar(cereal::make_nvp(_name, std::forward<Tp>(_obj)));
    }

    template <typename Archive, typename... Tp, size_t... Idx>
    static void serialize_tuple(Archive& ar, std::tuple<Tp...>& _data,
                                index_sequence<Idx...>)
    {
        TIMEMORY_FOLD_EXPRESSION(serialize_entry(ar, std::get<Idx>(_data)));
    }

private:
    // this mpi_get overload merges the results from the different mpi processes
    template <typename... Tp, size_t... Idx>
    auto mpi_get(std::tuple<std::vector<Tp>...>& _data, std::tuple<Tp...>& _inp,
                 std::index_sequence<Idx...>)
    {
        tim::mpi::barrier();
        TIMEMORY_FOLD_EXPRESSION(
            operation::finalize::mpi_get<decay_t<std::tuple_element_t<Idx, data_type>>,
                                         true>(std::get<Idx>(_data),
                                               std::get<Idx>(_inp)));
        tim::mpi::barrier();
    }

    // this mpi_get overload converts the merged data into the tuples which are
    // of the same data type as the timem_tuple m_data field
    template <size_t Idx, typename... Tp>
    auto mpi_get(std::vector<std::tuple<Tp...>>& _targ,
                 std::tuple<std::vector<Tp>...>& _data)
    {
        tim::mpi::barrier();
        auto&& _entries = std::get<Idx>(_data);
        size_t n        = _entries.size();
        if(n > _targ.size())
            _targ.resize(n, std::tuple<Tp...>{});
        for(size_t i = 0; i < n; ++i)
            std::get<Idx>(_targ.at(i)) = std::move(_entries.at(i));
        tim::mpi::barrier();
    }

    // this mpi_get overload converts the data tuples into timem_tuple instances
    template <typename... Tp, size_t... Idx>
    auto mpi_get(std::tuple<std::vector<Tp>...>&      _data,
                 std::vector<std::vector<hist_type>>& _hist, std::index_sequence<Idx...>)
    {
        tim::mpi::barrier();
        // convert the tuple of vectors into a vector of tuples
        std::vector<std::tuple<Tp...>> _vec{};
        TIMEMORY_FOLD_EXPRESSION(mpi_get<Idx>(_vec, _data));
        // convert the vector of tuples into a vector of this_tupe
        std::vector<this_type> _ret{};
        _ret.reserve(_vec.size());
        for(auto&& itr : _vec)
            _ret.emplace_back(this_type(this->key(), std::move(itr)));
        for(size_t i = 0; i < _ret.size(); ++i)
            _ret.at(i).m_hist_buff = std::move(_hist.at(i));
        tim::mpi::barrier();
        return _ret;
    }

private:
    using base_type::m_data;

    bool                       m_empty        = false;
    bool                       m_collect_hist = false;
    size_t                     m_collect_size = 0;
    std::ofstream*             m_ofs          = nullptr;
    std::function<void(bool*)> m_notify       = [](bool* _v) {
        if(_v)
            *_v = true;
    };
    std::vector<hist_type>*    m_data_hist = nullptr;
    std::vector<hist_type>     m_hist_buff = {};
    std::default_random_engine m_generator{ std::random_device{}() };
};
//
template <typename... Types>
using timem_tuple_t = convert_t<mpl::available_t<type_list<Types...>>, timem_tuple<>>;
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEM_BUNDLE)
#    define TIMEM_BUNDLE                                                                 \
        tim::timem_tuple_t<wall_clock, child_user_clock, child_system_clock,             \
                           child_cpu_clock, child_cpu_util, peak_rss, page_rss,          \
                           virtual_memory, num_major_page_faults, num_minor_page_faults, \
                           priority_context_switch, voluntary_context_switch, read_char, \
                           read_bytes, written_char, written_bytes, network_stats,       \
                           papi_array_t>
#endif
//
#if !defined(TIMEM_PID_SIGNAL)
#    define TIMEM_PID_SIGNAL SIGCONT
#endif
//
#if !defined(TIMEM_PID_SIGNAL_STRING)
#    define TIMEM_PID_SIGNAL_STRING TIMEMORY_STRINGIZE(TIMEM_PID_SIGNAL)
#endif
//
using timem_bundle_t  = TIMEM_BUNDLE;
using sampler_t       = tim::sampling::sampler<TIMEM_BUNDLE, 1>;
using sampler_array_t = typename sampler_t::array_type;
static_assert(std::is_same<sampler_array_t, std::array<timem_bundle_t, 1>>::value,
              "Error! Sampler array is not std::array<timem_bundle_t, 1>");
//
//--------------------------------------------------------------------------------------//
//
inline sampler_t*&
get_sampler()
{
    static sampler_t* _instance = nullptr;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
inline timem_bundle_t*
get_measure()
{
    return get_sampler()->get_last();
}
//
//--------------------------------------------------------------------------------------//
//
using sigaction_t = struct sigaction;
//
struct signal_handler
{
    sigaction_t m_custom_sigaction;
    sigaction_t m_original_sigaction;
};
//
//--------------------------------------------------------------------------------------//
//
inline signal_handler&
get_signal_handler(int _sig)
{
    static std::map<int, signal_handler> _instance{};
    auto                                 itr = _instance.find(_sig);
    if(itr == _instance.end())
    {
        auto eitr = _instance.emplace(_sig, signal_handler{});
        if(eitr.second)
            itr = eitr.first;
    }
    if(itr == _instance.end())
        throw std::runtime_error("Error! Signal handler for " + std::to_string(_sig) +
                                 " not found.");
    return itr->second;
}
//
//--------------------------------------------------------------------------------------//
//
struct timem_config
{
    static constexpr bool papi_available = tim::trait::is_available<papi_array_t>::value;
    using hist_type                      = typename timem_bundle_t::hist_type;

    bool     use_shell        = tim::get_env("TIMEM_USE_SHELL", false);
    bool     use_mpi          = tim::get_env("TIMEM_USE_MPI", false);
    bool     use_papi         = tim::get_env("TIMEM_USE_PAPI", papi_available);
    bool     use_sample       = tim::get_env("TIMEM_SAMPLE", true);
    bool     signal_delivered = false;
    bool     completed        = false;
    bool     full_buffer      = false;
    bool     debug            = tim::get_env("TIMEM_DEBUG", false);
    int      verbose          = tim::get_env("TIMEM_VERBOSE", 0);
    string_t shell =
        tim::get_env("TIMEM_SHELL", tim::get_env<string_t>("SHELL", getusershell()));
    string_t      shell_flags    = tim::get_env<string_t>("TIMEM_SHELL_FLAGS", "");
    string_t      output_file    = tim::get_env<string_t>("TIMEM_OUTPUT", "");
    double        sample_freq    = tim::get_env<double>("TIMEM_SAMPLE_FREQ", 5.0);
    double        sample_delay   = tim::get_env<double>("TIMEM_SAMPLE_DELAY", 1.0e-6);
    pid_t         master_pid     = getpid();
    pid_t         worker_pid     = getpid();
    size_t        buffer_size    = 0;
    string_t      command        = {};
    std::set<int> signal_types   = { SIGALRM };
    std::set<int> signal_forward = { SIGINT };
    std::vector<std::string>     argvector     = {};
    std::vector<hist_type>       history       = {};
    std::unique_ptr<std::thread> buffer_thread = {};
    std::condition_variable      buffer_cv{};

    template <typename Archive>
    void serialize(Archive& ar, unsigned int);

    std::string get_output_filename(std::string inp = {}, std::string ext = {});
};
//
//--------------------------------------------------------------------------------------//
//
inline timem_config&
get_config()
{
    static timem_config _instance;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
#define TIMEM_CONFIG_FUNCTION(NAME)                                                      \
    inline auto& NAME() { return get_config().NAME; }
//
//--------------------------------------------------------------------------------------//
//
TIMEM_CONFIG_FUNCTION(use_shell)
TIMEM_CONFIG_FUNCTION(use_mpi)
TIMEM_CONFIG_FUNCTION(use_papi)
TIMEM_CONFIG_FUNCTION(use_sample)
TIMEM_CONFIG_FUNCTION(shell)
TIMEM_CONFIG_FUNCTION(shell_flags)
TIMEM_CONFIG_FUNCTION(output_file)
TIMEM_CONFIG_FUNCTION(sample_freq)
TIMEM_CONFIG_FUNCTION(sample_delay)
TIMEM_CONFIG_FUNCTION(signal_delivered)
TIMEM_CONFIG_FUNCTION(completed);
TIMEM_CONFIG_FUNCTION(full_buffer);
TIMEM_CONFIG_FUNCTION(debug)
TIMEM_CONFIG_FUNCTION(verbose)
TIMEM_CONFIG_FUNCTION(command)
TIMEM_CONFIG_FUNCTION(buffer_size)
TIMEM_CONFIG_FUNCTION(master_pid)
TIMEM_CONFIG_FUNCTION(worker_pid)
TIMEM_CONFIG_FUNCTION(signal_types)
TIMEM_CONFIG_FUNCTION(signal_forward)
TIMEM_CONFIG_FUNCTION(argvector)
TIMEM_CONFIG_FUNCTION(buffer_cv);
TIMEM_CONFIG_FUNCTION(buffer_thread);
TIMEM_CONFIG_FUNCTION(history);
//
//--------------------------------------------------------------------------------------//
//
inline void
explain(int ret, const char* pathname, char** argv)
{
    if(ret < 0)
    {
#if defined(TIMEMORY_USE_LIBEXPLAIN)
        fprintf(stderr, "%s\n", explain_execvp(pathname, argv));
#else
        fprintf(stderr, "Return code: %i : %s\n", ret, pathname);
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
#endif
    }
    else if(debug() || verbose() > 0)
    {
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
timem_config::serialize(Archive& ar, unsigned int)
{
    ar(tim::cereal::make_nvp("use_shell", use_shell),
       tim::cereal::make_nvp("use_mpi", use_mpi),
       tim::cereal::make_nvp("use_papi", use_papi),
       tim::cereal::make_nvp("use_sample", use_sample),
       tim::cereal::make_nvp("debug", debug), tim::cereal::make_nvp("verbose", verbose),
       tim::cereal::make_nvp("shell", shell),
       tim::cereal::make_nvp("shell_flags", shell_flags),
       tim::cereal::make_nvp("sample_freq", sample_freq),
       tim::cereal::make_nvp("sample_delay", sample_delay),
       tim::cereal::make_nvp("buffer_size", buffer_size));
}
//
//--------------------------------------------------------------------------------------//
//
inline std::string
timem_config::get_output_filename(std::string inp, std::string ext)
{
    if(inp.empty())
        inp = output_file;

    auto _rstrip = [](std::string& _inp, const std::string& _key) {
        auto pos = std::string::npos;
        while((pos = _inp.find(_key, _inp.length() - _key.length())) != std::string::npos)
            _inp = _inp.replace(pos, _key.length(), "");
    };

    auto _replace = [](std::string& _inp, const std::string& _key,
                       const std::string& _sub) {
        auto pos = std::string::npos;
        while((pos = _inp.find(_key)) != std::string::npos)
            _inp = _inp.replace(pos, _key.length(), _sub);
    };

    if(!ext.empty())
    {
        _rstrip(inp, ext);
        _rstrip(inp, ".json");
        _rstrip(inp, ".txt");
    }

    std::string argstring = {};
    for(size_t i = 1; i < argvector.size(); ++i)
        argstring.append(argvector.at(i));

    _replace(inp, "%m", compute_md5(argstring));

    using pair_t = std::pair<std::string, int64_t>;
    for(const auto& itr :
        { pair_t{ "%p", worker_pid }, pair_t{ "%j", tim::get_env("SLURM_JOB_ID", 0) },
          pair_t{ "%r", tim::get_env("SLURM_PROCID", tim::mpi::rank()) },
          pair_t{ "%s", tim::mpi::size() } })
    {
        _replace(inp, itr.first, std::to_string(itr.second));
    }

    if(!ext.empty())
        inp += ext;

    return inp;
}
