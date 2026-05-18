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

#include "timemory/api.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/compat/macros.h"
#include "timemory/defines.h"
#include "timemory/environment/declaration.hpp"
#include "timemory/macros.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/tsettings.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/settings/vsettings.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"

#include <ctime>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#if defined(TIMEMORY_UNIX)
//
#    include <ctime>
#    include <unistd.h>
//
extern "C"
{
    extern char** environ;
}
#endif

namespace tim
{
//
class manager;
//
//--------------------------------------------------------------------------------------//
//
//                              settings
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
/// \struct tim::operation::setting_serialization
/// \brief this operation is used to customize how the setting are serialized.
///
/// \code{.cpp}
///
/// namespace
/// {
/// struct custom_setting_serializer
/// {};
/// }  // namespace
///
/// namespace tim
/// {
/// namespace operation
/// {
/// template <typename Tp>
/// struct setting_serialization<Tp, custom_setting_serializer>
/// {
///     // discard any data, e.g. "environment" and "command_line"
///     template <typename ArchiveT>
///     void operator()(ArchiveT&, const char*, const Tp&) const
///     {}
///
///     template <typename ArchiveT>
///     void operator()(ArchiveT&, const char*, Tp&&) const
///     {}
/// };
/// //
/// template <typename Tp>
/// struct setting_serialization<tsettings<Tp>, custom_setting_serializer>
/// {
///     using value_type = tsettings<Tp>;
///
///     template <typename ArchiveT>
///     void operator()(ArchiveT& _ar, value_type& _val) const
///     {
///         // only serialize the value
///         Tp _v = _val.get();
///         _ar.setNextName(_val.get_env_name().c_str());
///         _ar.startNode();
///         _ar(cereal::make_nvp("value", _v));
///         _ar.finishNode();
///     }
/// };
/// }  // namespace operation
/// }  // namespace tim
///
/// template <typename ArchiveT, typename... Types>
/// void
/// push()
/// {
///     tim::settings::push_serialize_map_callback<ArchiveT, custom_setting_serializer>();
///     tim::settings::push_serialize_data_callback<ArchiveT, custom_setting_serializer>(
///             tim::type_list<Types...>{}));
/// }
///
/// template <typename ArchiveT, typename... Types>
/// void
/// pop()
/// {
///     tim::settings::pop_serialize_map_callback<ArchiveT, custom_setting_serializer>();
///     tim::settings::pop_serialize_data_callback<ArchiveT, custom_setting_serializer>(
///             tim::type_list<Types...>{}));
/// }
///
/// void
/// dump_config(std::string _config_file)
/// {
///     using json_t = tim::cereal::PrettyJSONOutputArchive;
///
///     // stores the original serializer behavior and
///     // replaces it with the custom one
///     push<json_t, std::vector<std::string>>();
///
///     auto _serialize = [](auto&& _ar) {
///         auto _settings = tim::settings::shared_instance();
///         _ar->setNextName(TIMEMORY_PROJECT_NAME);
///         _ar->startNode();
///         tim::settings::serialize_settings(*_ar);
///         _ar->finishNode();
///     };
///
///     std::stringstream _ss{};
///     {
///         json_t _ar{ _ss };
///         _serialize(_ar);
///     }
///     std::ofstream ofs{ _config_file };
///     ofs << _ss.str() << "n";
///
///     // restores the original serializer behavior
///     pop<json_t, std::vector<std::string>>();
/// }
/// \endcode
///
template <typename Tp, typename TagT>
struct setting_serialization
{
    using value_type = Tp;

    template <typename ArchiveT>
    void operator()(ArchiveT& _ar, const char* _name, const value_type& _data) const
    {
        _ar(cereal::make_nvp(_name, _data));
    }

    template <typename ArchiveT>
    void operator()(ArchiveT& _ar, const char* _name, value_type&& _data) const
    {
        _ar(cereal::make_nvp(_name, std::forward<Tp>(_data)));
    }
};
//
/// \brief specialization for the templated settings type
template <typename Tp, typename TagT>
struct setting_serialization<tsettings<Tp>, TagT>
{
    using value_type = tsettings<Tp>;

    template <typename ArchiveT>
    void operator()(ArchiveT& _ar, value_type& _val) const
    {
        if(!_val.get_hidden())
            _ar(cereal::make_nvp(_val.get_env_name(), _val));
    }
};
}  // namespace operation
//
struct TIMEMORY_VISIBILITY("default") settings
{
    friend void timemory_init(int, char**, const std::string&, const std::string&);
    friend void timemory_finalize(manager*, settings*, bool);
    friend class manager;

    // this is the list of the current and potentially used data types
    using data_type_list_t = setting_supported_data_types;
    using strvector_t      = std::vector<std::string>;
    using strset_t         = std::set<std::string>;
    using value_type       = std::shared_ptr<vsettings>;
    using data_type        = std::unordered_map<string_view_t, value_type>;
    using iterator         = typename data_type::iterator;
    using const_iterator   = typename data_type::const_iterator;
    using pointer_t        = std::shared_ptr<settings>;
    using strpair_t        = std::pair<std::string, std::string>;
    using compose_suffix_t = std::variant<process::id_t, std::string>;
    using update_type      = setting_update_type;

    template <typename Tp, typename Vp>
    using tsetting_pointer_t = std::shared_ptr<tsettings<Tp, Vp>>;

    template <typename Tag = TIMEMORY_API>
    static std::time_t* get_launch_time(Tag = {});
    template <typename Tag>
    static TIMEMORY_HOT const pointer_t& shared_instance();
    template <typename Tag>
    static TIMEMORY_HOT settings* instance();
    static TIMEMORY_HOT const pointer_t& shared_instance();
    static TIMEMORY_HOT settings* instance();

    settings();
    ~settings() = default;

    settings(const settings&);
    settings(settings&&) noexcept = default;

    settings& operator=(const settings&);
    settings& operator=(settings&&) noexcept = default;

    void initialize();

    /// the "tag" for settings should generally be the basename of exe
    void set_tag(std::string _v) { m_tag = std::move(_v); }

    /// the "tag" for settings should generally be the basename of exe
    std::string get_tag() const;

    /// if the tag is not explicitly set, try to compute it. Otherwise use
    /// the TIMEMORY_SETTINGS_PREFIX_
    static std::string get_fallback_tag();

    /// returns whether timemory_init has been invoked
    bool get_initialized() const { return m_initialized; }
    bool get_finalized() const { return m_finalized; }

    //==================================================================================//
    //
    //                  GENERAL SETTINGS THAT APPLY TO MULTIPLE COMPONENTS
    //
    //==================================================================================//

    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, config_file)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, suppress_parsing)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, suppress_config)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, strict_config)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, enabled)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, auto_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, cout_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, file_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, text_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, json_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, tree_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, dart_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, time_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, plot_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, diff_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, flamegraph_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, ctest_notes)
    TIMEMORY_SETTINGS_MEMBER_DECL(int, verbose)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, debug)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, banner)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, collapse_threads)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, collapse_processes)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint16_t, max_depth)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, time_format)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, precision)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, width)
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, max_width)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, scientific)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, timing_precision)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, timing_width)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, timing_units)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, timing_scientific)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, memory_precision)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, memory_width)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, memory_units)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, memory_scientific)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, output_path)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, output_prefix)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, input_path)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, input_prefix)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, input_extensions)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, dart_type)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, dart_count)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, dart_label)
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, max_thread_bookmarks)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, cpu_affinity)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, stack_clearing)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, add_secondary)
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, throttle_count)
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, throttle_value)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, global_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, tuple_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, list_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, ompt_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, mpip_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, ncclp_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, trace_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, profiler_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, kokkos_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, components)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, network_interface)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_init)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_finalize)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_thread)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, mpi_thread_type)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, upcxx_init)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, upcxx_finalize)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_threading)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_multiplexing)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_fail_on_error)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_quiet)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, papi_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_attach)
    TIMEMORY_SETTINGS_MEMBER_DECL(int, papi_overflow)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, cuda_event_batch_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, nvtx_marker_device_sync)
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, cupti_activity_level)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, cupti_activity_kinds)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, cupti_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, cupti_metrics)
    TIMEMORY_SETTINGS_MEMBER_DECL(int, cupti_device)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, roofline_mode)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, cpu_roofline_mode)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, gpu_roofline_mode)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, cpu_roofline_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, gpu_roofline_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, instruction_roofline)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_streams)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_grid_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_block_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_alignment)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, ert_skip_ops)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, craypat_categories)
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, node_count)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, destructor_report)
    TIMEMORY_SETTINGS_MEMBER_DECL(std::string, python_exe)
    // stream
    TIMEMORY_SETTINGS_MEMBER_DECL(int64_t, separator_frequency)
    // signals
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, enable_signal_handler)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, allow_signal_handler)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, enable_all_signals)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, disable_all_signals)
    // miscellaneous ref
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, flat_profile)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, timeline_profile)
    TIMEMORY_SETTINGS_REFERENCE_DECL(process::id_t, target_pid)

    static strvector_t& command_line();
    static strvector_t& environment();
    strvector_t&        get_command_line() { return m_command_line; }
    strvector_t&        get_environment() { return m_environment; }

public:
    TIMEMORY_STATIC_ACCESSOR(bool, use_output_suffix,
                             get_env<bool>(TIMEMORY_SETTINGS_PREFIX "USE_OUTPUT_SUFFIX",
                                           false))
    TIMEMORY_STATIC_ACCESSOR(compose_suffix_t, default_process_suffix, "%nid%")

    struct compose_filename_config
    {
        TIMEMORY_DEFAULT_OBJECT(compose_filename_config)

        template <typename SuffixT>
        compose_filename_config(bool, SuffixT, bool = false, std::string_view = {},
                                std::string_view = {});

        bool             use_suffix    = false;
        compose_suffix_t suffix        = "%nid%";
        bool             make_dir      = false;
        std::string      explicit_path = {};
        std::string      subdirectory  = {};
    };

    struct output_key
    {
        output_key(std::string _key, std::string _val, std::string _desc = {});

        operator std::pair<std::string, std::string>() const;

        std::string key         = {};
        std::string value       = {};
        std::string description = {};
    };

    static strvector_t get_global_environment();
    static std::string tolower(std::string str);
    static std::string toupper(std::string str);
    static std::string get_global_input_prefix(std::string subdir = {});
    static std::string get_global_output_prefix(bool        _make_dir = false,
                                                std::string subdir    = {});
    static void        store_command_line(int argc, char** argv);
    static std::string compose_output_filename(std::string _tag, std::string _ext,
                                               const compose_filename_config& = {
                                                   use_output_suffix(),
                                                   default_process_suffix(), false,
                                                   std::string{}, std::string{} });
    static std::string compose_input_filename(std::string _tag, std::string _ext,
                                              const compose_filename_config& = {
                                                  use_output_suffix(),
                                                  default_process_suffix(), false,
                                                  std::string{}, std::string{} });

    static void parse(settings* = instance<TIMEMORY_API>());

    static void parse(const std::shared_ptr<settings>&);

    static std::vector<output_key> output_keys(const std::string& _tag)
        TIMEMORY_VISIBILITY("hidden");
    static std::string format(std::string _fpath, const std::string& _tag)
        TIMEMORY_VISIBILITY("hidden");
    static std::string format(std::string _prefix, std::string _tag, std::string _suffix,
                              std::string _ext) TIMEMORY_VISIBILITY("hidden");

    template <typename... Args>
    static std::string compose_output_filename(std::string _tag, std::string _ext,
                                               bool _use_suffix, Args... args)
    {
        return compose_output_filename(std::move(_tag), std::move(_ext),
                                       compose_filename_config{ _use_suffix, args... });
    }

    template <typename... Args>
    static std::string compose_input_filename(std::string _tag, std::string _ext,
                                              bool _use_suffix, Args... args)
    {
        return compose_input_filename(std::move(_tag), std::move(_ext),
                                      compose_filename_config{ _use_suffix, args... });
    }

public:
    template <typename ArchiveT>
    void load(ArchiveT& ar, unsigned int);

    template <typename ArchiveT>
    void save(ArchiveT& ar, unsigned int) const;

    template <typename ArchiveT>
    static void serialize_settings(ArchiveT&);

    template <typename ArchiveT>
    static void serialize_settings(ArchiveT&, settings&);

    /// read a configuration file
    bool read(std::string);
    bool read(std::istream&, std::string = {});

    void init_config(bool search_default = true);

    std::vector<strpair_t> get_unknown_configs() const { return m_unknown_configs; }

public:
    template <size_t Idx = 0>
    static int64_t indent_width(int64_t _w = settings::width());

    template <typename Tp, size_t Idx = 0>
    static int64_t indent_width(int64_t _w = indent_width<Idx>());

private:
    template <typename ItrT>
    static auto get_next(ItrT _v, ItrT _e)
    {
        while(_v != _e && _v->second->get_hidden())
            ++_v;
        return _v;
    }

public:
    auto           ordering() const { return m_order; }
    iterator       begin() { return get_next(m_data.begin(), m_data.end()); }
    iterator       end() { return m_data.end(); }
    const_iterator begin() const { return get_next(m_data.cbegin(), m_data.cend()); }
    const_iterator end() const { return m_data.cend(); }
    const_iterator cbegin() const { return get_next(m_data.cbegin(), m_data.cend()); }
    const_iterator cend() const { return m_data.cend(); }

    template <typename Sp = std::string>
    auto find(Sp&& _key, bool _exact = true, const std::string& _category = {});

    template <typename Sp = std::string>
    decltype(auto) at(Sp&& _key);

    template <typename Tp, typename Sp = std::string>
    Tp get(Sp&& _key, bool _exact = true);

    template <typename Tp, typename Sp = std::string>
    bool get(Sp&& _key, Tp& _val, bool _exact);

    template <typename Tp, typename Sp = std::string>
    bool set(Sp&& _key, Tp&& _val, bool _exact = true);

    /// mark this option as enabled. returns whether option was found
    bool enable(string_view_cref_t _key, bool _exact = true);

    /// mark this option as not enabled. returns whether option was found
    bool disable(string_view_cref_t _key, bool _exact = true);

    /// mark all the options in this category as enabled. returns all options enabled
    std::set<std::string> enable_category(string_view_cref_t _key);

    /// mark all the options in this category as not enabled. returns all options disabled
    std::set<std::string> disable_category(string_view_cref_t _key);

    /// \fn bool update(const std::string& key, const std::string& val, bool exact)
    /// \param key Identifier for the setting. Either name, env-name, or command-line opt
    /// \param val Update value
    /// \param exact If true, match only options
    ///
    /// \brief Update a setting via a string. Returns whether a matching setting
    /// for the identifier was found (NOT whether the value was actually updated)
    bool update(const std::string& _key, const std::string& _val, bool _exact = false);

    /// \tparam Tp Data-type of the setting
    /// \tparam Vp Value-type of the setting (Tp or Tp&)
    /// \tparam Sp String-type
    template <typename Tp, typename Vp, typename Sp, typename... Args>
    auto insert(Sp&& _env, const std::string& _name, const std::string& _desc, Vp _init,
                Args&&... _args);

    /// \tparam Tp Data-type of the setting
    /// \tparam Vp Value-type of the setting (Tp or Tp&)
    /// \tparam Sp String-type
    template <typename Tp, typename Vp, typename Sp = std::string>
    auto insert(tsetting_pointer_t<Tp, Vp> _ptr, Sp&& _env = {});

    /// \tparam API Tagged type
    ///
    /// \brief Make a copy of the current settings and return a new instance whose values
    /// can be modified, used, and then discarded. The values modified do not change
    /// any settings accessed through static methods or the non-templated instance method.
    /// E.g. `tim::settings::enabled()` will not be affected by changes to settings
    /// instance returned by this method.
    template <typename Tag = TIMEMORY_API>
    static pointer_t push();

    /// \tparam API Tagged type
    ///
    /// \brief Restore the settings from a previous push operations.
    template <typename Tag = TIMEMORY_API>
    static pointer_t pop();

    /// \tparam Data Type
    ///
    /// \brief Try to retreive the setting by it's environment name if there is
    /// a settings instance. If not, query enironment
    template <typename Tp>
    static Tp get_with_env_fallback(const std::string& _env, Tp _default,
                                    const pointer_t& _settings = shared_instance());

protected:
    template <typename ArchiveT>
    using serialize_func_t = std::function<void(ArchiveT&, value_type)>;
    template <typename ArchiveT>
    using serialize_map_t = std::map<std::type_index, serialize_func_t<ArchiveT>>;
    template <typename ArchiveT>
    using serialize_pair_t = std::pair<std::type_index, serialize_func_t<ArchiveT>>;

    template <typename ArchiveT, typename Tp, typename TagT = void>
    static auto get_serialize_pair()
    {
        auto _func = [](ArchiveT& _ar, value_type _val) {
            using Up = tsettings<Tp>;
            if(!_val)
                _val = std::make_shared<Up>();
            operation::setting_serialization<Up, TagT>{}(_ar,
                                                         *static_cast<Up*>(_val.get()));
        };
        return serialize_pair_t<ArchiveT>{ std::type_index(typeid(Tp)), _func };
    }

    template <typename ArchiveT, typename TagT = void, typename... Tail>
    static auto get_serialize_map(tim::type_list<Tail...>)
    {
        serialize_map_t<ArchiveT> _val{};
        TIMEMORY_FOLD_EXPRESSION(_val.insert(get_serialize_pair<ArchiveT, Tail, TagT>()));
        return _val;
    }

    template <typename ArchiveT, typename TagT = void, typename ArgT = data_type_list_t>
    static auto& get_serialize_map_callback(ArgT = {})
    {
        using func_t     = serialize_map_t<ArchiveT> (*)();
        static func_t _v = []() { return get_serialize_map<ArchiveT, TagT>(ArgT{}); };
        return _v;
    }

    template <typename ArchiveT, typename TagT = void, typename ArgT>
    static auto& get_serialize_data_callback(type_list<ArgT>)
    {
        using func_t     = void (*)(ArchiveT&, const char*, const ArgT&);
        static func_t _v = [](ArchiveT& _ar, const char* _name, const ArgT& _data) {
            return operation::setting_serialization<ArgT, TagT>{}(_ar, _name, _data);
        };
        return _v;
    }

    template <typename Tp>
    struct internal_serialize_callback
    {
        struct impl
        {};
        using type = impl;
    };

public:
    template <typename ArchiveT, typename Tp, typename ArgT = data_type_list_t>
    static void push_serialize_map_callback(ArgT = {})
    {
        using internal_t = typename internal_serialize_callback<Tp>::type;
        get_serialize_map_callback<ArchiveT, internal_t>(ArgT{}) = []() {
            return get_serialize_map<ArchiveT, void>(ArgT{});
        };

        get_serialize_map_callback<ArchiveT, void>(ArgT{}) = []() {
            return get_serialize_map<ArchiveT, Tp>(ArgT{});
        };
    }

    template <typename ArchiveT, typename Tp, typename ArgT = data_type_list_t>
    static void pop_serialize_map_callback(ArgT = {})
    {
        using internal_t = typename internal_serialize_callback<Tp>::type;
        get_serialize_map_callback<ArchiveT, void>(ArgT{}) =
            get_serialize_map_callback<ArchiveT, internal_t>(ArgT{});
    }

    template <typename ArchiveT, typename Tp, typename ArgT>
    static void push_serialize_data_callback(type_list<ArgT>)
    {
        get_serialize_data_callback<ArchiveT, void>(type_list<ArgT>{}) =
            [](ArchiveT& _ar, const char* _name, const ArgT& _data) {
                return operation::setting_serialization<ArgT, Tp>{}(_ar, _name, _data);
            };
    }

    template <typename ArchiveT, typename Tp, typename ArgT>
    static void pop_serialize_data_callback(type_list<ArgT>)
    {
        get_serialize_data_callback<ArchiveT, void>(type_list<ArgT>{}) =
            [](ArchiveT& _ar, const char* _name, const ArgT& _data) {
                return operation::setting_serialization<ArgT, void>{}(_ar, _name, _data);
            };
    }

private:
    using settings_stack_t = std::stack<pointer_t>;

    template <typename Tag>
    static TIMEMORY_HOT pointer_t& private_shared_instance(
        enable_if_t<std::is_same<Tag, TIMEMORY_API>::value, int> = 0);

    template <typename Tag>
    static TIMEMORY_HOT pointer_t& private_shared_instance(
        enable_if_t<!std::is_same<Tag, TIMEMORY_API>::value, long> = 0);

    template <typename Tag>
    static settings_stack_t& get_stack()
    {
        static auto _instance = settings_stack_t{};
        return _instance;
    }

private:
    bool                   m_initialized     = false;
    bool                   m_finalized       = false;
    std::string            m_tag             = {};
    std::set<std::string>  m_read_configs    = {};
    strvector_t            m_config_stack    = {};
    strvector_t            m_order           = {};
    strvector_t            m_command_line    = {};
    strvector_t            m_environment     = get_global_environment();
    data_type              m_data            = {};
    std::vector<strpair_t> m_unknown_configs = {};

    /// This is set by timemory_init
    void set_initialized(bool _v) { m_initialized = _v; }
    void set_finalized(bool _v) { m_finalized = _v; }

private:
    void initialize_core() TIMEMORY_VISIBILITY("hidden");
    void initialize_components() TIMEMORY_VISIBILITY("hidden");
    void initialize_io() TIMEMORY_VISIBILITY("hidden");
    void initialize_format() TIMEMORY_VISIBILITY("hidden");
    void initialize_parallel() TIMEMORY_VISIBILITY("hidden");
    void initialize_tpls() TIMEMORY_VISIBILITY("hidden");
    void initialize_roofline() TIMEMORY_VISIBILITY("hidden");
    void initialize_miscellaneous() TIMEMORY_VISIBILITY("hidden");
    void initialize_ert() TIMEMORY_VISIBILITY("hidden");
    void initialize_dart() TIMEMORY_VISIBILITY("hidden");
    void initialize_disabled() TIMEMORY_VISIBILITY("hidden");

    static auto& indent_width_map()
    {
        static std::map<size_t, std::map<std::type_index, int64_t>> _instance{};
        return _instance;
    }

    static void handle_exception(std::string_view, std::string_view, std::string_view,
                                 std::string_view);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename SuffixT>
inline settings::compose_filename_config::compose_filename_config(
    bool _use, SuffixT _suffix, bool _make_dir, std::string_view _path,
    std::string_view _subdir)
: use_suffix{ _use }
, make_dir{ _make_dir }
, explicit_path{ _path }
, subdirectory{ _subdir }
{
    if constexpr(std::is_same<SuffixT, compose_suffix_t>::value)
        suffix = _suffix;
    else if constexpr(std::is_integral<SuffixT>::value)
        suffix = static_cast<process::id_t>(_suffix);
    else
        suffix = std::string{ _suffix };
}
//
inline settings::output_key::output_key(std::string _key, std::string _val,
                                        std::string _desc)
: key{ std::move(_key) }
, value{ std::move(_val) }
, description{ std::move(_desc) }
{}
//
inline settings::output_key::operator std::pair<std::string, std::string>() const
{
    return std::make_pair(key, value);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::time_t* settings::get_launch_time(Tag)
{
    // statically store the launch time, intentional memory leak
    static std::time_t* _time = new std::time_t{ std::time(nullptr) };
    return _time;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>& settings::private_shared_instance(
    enable_if_t<std::is_same<Tag, TIMEMORY_API>::value, int>)
{
    // this is the original
    static std::shared_ptr<settings> _instance = std::make_shared<settings>();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>& settings::private_shared_instance(
    enable_if_t<!std::is_same<Tag, TIMEMORY_API>::value, long>)
{
    // make a copy of the original
    static std::shared_ptr<settings> _instance =
        std::make_shared<settings>(*private_shared_instance<TIMEMORY_API>());
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>
settings::push()
{
    // ensure the non-template methods have their own static copies
    static auto* _discard_ptr  = instance();
    static auto* _discard_sptr = instance<TIMEMORY_API>();
    consume_parameters(_discard_ptr, _discard_sptr);

    auto _old = shared_instance<Tag>();
    get_stack<Tag>().push(_old);
    private_shared_instance<Tag>() = std::make_shared<settings>(*_old);
    return private_shared_instance<Tag>();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>
settings::pop()
{
    auto& _stack = get_stack<Tag>();
    if(_stack.empty())
    {
        TIMEMORY_PRINT_HERE("%s", "Ignoring settings::pop() on empty stack");
        return shared_instance<Tag>();
    }

    auto _top                      = _stack.top();
    private_shared_instance<Tag>() = _top;
    _stack.pop();
    return _top;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
const std::shared_ptr<settings>&
settings::shared_instance()
{
    static std::shared_ptr<settings>& _instance = private_shared_instance<Tag>();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
settings*
settings::instance()
{
    return shared_instance<Tag>().get();
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx>
int64_t
settings::indent_width(int64_t _w)
{
    auto        _tidx = std::type_index{ typeid(TIMEMORY_API) };
    auto_lock_t _lk{ type_mutex<settings, TIMEMORY_API>() };
    auto&       _itr = indent_width_map()[Idx][_tidx];
    return (_itr = std::max<int64_t>(_itr, _w));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, size_t Idx>
int64_t
settings::indent_width(int64_t _w)
{
    auto        _tidx = std::type_index{ typeid(Tp) };
    auto_lock_t _lk{ type_mutex<settings, TIMEMORY_API>() };
    auto&       _itr = indent_width_map()[Idx][_tidx];
    return (_itr = std::max<int64_t>(_itr, _w));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename ArchiveT>
void
settings::serialize_settings(ArchiveT& ar)
{
    if(settings::instance())
        serialize_settings<ArchiveT>(ar, *settings::instance());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename ArchiveT>
void
settings::serialize_settings(ArchiveT& ar, settings& _obj)
{
    ar(cereal::make_nvp("settings", _obj));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename ArchiveT>
void
settings::load(ArchiveT& ar, unsigned int)
{
#if !defined(TIMEMORY_DISABLE_SETTINGS_SERIALIZATION)
    using map_type = std::map<std::string, std::shared_ptr<vsettings>>;
    map_type _data;
    for(const auto& itr : m_data)
        _data.insert({ std::string{ itr.first }, itr.second->clone() });
    auto _map             = get_serialize_map_callback<ArchiveT>(data_type_list_t{})();
    bool _print_exception = (get_verbose() > 2 || get_debug());

#    define TIMEMORY_SETTINGS_TRY_LOAD(...)                                              \
        {                                                                                \
            try                                                                          \
            {                                                                            \
                __VA_ARGS__;                                                             \
            } catch(const cereal::Exception& _e)                                         \
            {                                                                            \
                if(_print_exception)                                                     \
                    TIMEMORY_PRINT_HERE("%s", _e.what());                                \
            }                                                                            \
        }

    for(const auto& itr : _data)
    {
        auto mitr = _map.find(itr.second->get_type_index());
        if(mitr != _map.end())
        {
            TIMEMORY_SETTINGS_TRY_LOAD(mitr->second(ar, itr.second));
        }
    }

    TIMEMORY_SETTINGS_TRY_LOAD(ar(cereal::make_nvp("command_line", m_command_line)));
    TIMEMORY_SETTINGS_TRY_LOAD(ar(cereal::make_nvp("environment", m_environment)));

    for(const auto& itr : _data)
    {
        auto ditr = m_data.find(itr.first);
        if(ditr != m_data.end())
        {
            ditr->second->clone(itr.second);
        }
        else
        {
            m_order.push_back(itr.first);
            m_data.insert({ m_order.back(), itr.second });
        }
    }

#    undef TIMEMORY_SETTINGS_TRY_LOAD
#else
    consume_parameters(ar);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename ArchiveT>
void
settings::save(ArchiveT& ar, unsigned int) const
{
#if !defined(TIMEMORY_DISABLE_SETTINGS_SERIALIZATION)
    using map_type = std::map<std::string, std::shared_ptr<vsettings>>;
    map_type _data;
    for(const auto& itr : m_data)
        _data.insert({ std::string{ itr.first }, itr.second->clone() });

    auto _map = get_serialize_map_callback<ArchiveT>(data_type_list_t{})();
    for(const auto& itr : _data)
    {
        auto mitr = _map.find(itr.second->get_type_index());
        if(mitr != _map.end())
            mitr->second(ar, itr.second);
    }

    static_assert(std::is_same<decltype(m_command_line), decltype(m_environment)>::value,
                  "Error! data types changed");
    using data_callback_t  = type_list<std::decay_t<decltype(m_command_line)>>;
    auto&& _data_serialize = get_serialize_data_callback<ArchiveT>(data_callback_t{});
    _data_serialize(ar, "command_line", m_command_line);
    _data_serialize(ar, "environment", m_environment);

#else
    consume_parameters(ar);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Sp>
inline auto
settings::find(Sp&& _key, bool _exact, const std::string& _category)
{
    // exact match to map key
    auto itr = m_data.find(std::forward<Sp>(_key));
    if(itr != m_data.end() && itr->second->matches(_key, _category, _exact))
        return itr;

    // match against env_name, name, command-line options
    for(auto ditr = begin(); ditr != end(); ++ditr)
    {
        if(ditr->second &&
           ditr->second->matches(std::forward<Sp>(_key), _category, _exact))
            return ditr;
    }

    // not found
    return m_data.end();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Sp>
inline decltype(auto)
settings::at(Sp&& _key)
{
    // exact match to map key
    return m_data.at(std::forward<Sp>(_key));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Sp>
Tp
settings::get(Sp&& _key, bool _exact)
{
    auto itr = find(std::forward<Sp>(_key), _exact);
    if(itr != m_data.end() && itr->second)
    {
        auto _vptr = itr->second;
        auto _tidx = std::type_index(typeid(Tp));
        auto _vidx = std::type_index(typeid(Tp&));
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _tidx)
            return static_cast<tsettings<Tp, Tp>*>(_vptr.get())->get();
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _vidx)
            return static_cast<tsettings<Tp, Tp&>*>(_vptr.get())->get();
    }
    return Tp{};
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Sp>
bool
settings::get(Sp&& _key, Tp& _val, bool _exact)
{
    auto itr = find(std::forward<Sp>(_key), _exact);
    if(itr != m_data.end() && itr->second)
    {
        auto _vptr = itr->second;
        auto _tidx = std::type_index(typeid(Tp));
        auto _vidx = std::type_index(typeid(Tp&));
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _tidx)
            return ((_val = static_cast<tsettings<Tp, Tp>*>(_vptr.get())->get()), true);
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _vidx)
            return ((_val = static_cast<tsettings<Tp, Tp&>*>(_vptr.get())->get()), true);
    }
    return false;
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp, typename Sp>
bool
settings::set(Sp&& _key, Tp&& _val, bool _exact)
{
    auto itr = find(std::forward<Sp>(_key), _exact);
    if(itr != m_data.end() && itr->second)
    {
        using Up   = decay_t<Tp>;
        auto _tidx = std::type_index(typeid(Up));
        auto _vidx = std::type_index(typeid(Up&));
        auto _tobj = dynamic_cast<tsettings<Up>*>(itr->second.get());
        auto _robj = dynamic_cast<tsettings<Up, Up&>*>(itr->second.get());
        if(itr->second->get_type_index() == _tidx &&
           itr->second->get_value_index() == _tidx && _tobj)
        {
            return (_tobj->set(std::forward<Tp>(_val)), true);
        }
        else if(itr->second->get_type_index() == _tidx &&
                itr->second->get_value_index() == _vidx && _robj)
        {
            return (_robj->set(std::forward<Tp>(_val)), true);
        }
        else
        {
            throw std::runtime_error(std::string{ "tim::settings::set(" } +
                                     std::string{ _key } + ", ...) failed");
        }
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
settings::update(const std::string& _key, const std::string& _val, bool _exact)
{
    auto itr = find(_key, _exact);
    if(itr == m_data.end())
    {
        if(get_verbose() > 0 || get_debug())
            TIMEMORY_PRINT_HERE("Key: \"%s\" did not match any known setting",
                                _key.c_str());
        return false;
    }

    itr->second->parse(_val);
    return true;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp, typename Sp, typename... Args>
auto
settings::insert(Sp&& _env, const std::string& _name, const std::string& _desc, Vp _init,
                 Args&&... _args)
{
    static_assert(is_one_of<Tp, data_type_list_t>::value,
                  "Error! Data type is not supported. See settings::data_type_list_t");
    static_assert(std::is_same<decay_t<Tp>, decay_t<Vp>>::value,
                  "Error! Initializing value is not the same as the declared type");

    auto _sid = std::string{ std::forward<Sp>(_env) };
    if(get_initialized())  // don't set env before timemory_init
    {
        if(vsettings::get_debug() >= 1)
        {
            std::ostringstream oss;
            oss << "[" << TIMEMORY_PROJECT_NAME << "][settings] set_env(\"" << _sid
                << "\", \"" << _init << "\", 0);\n";
            log::stream(std::cerr, log::color::warning()) << oss.str();
        }
        set_env(_sid, _init, 0);
    }
    else
    {
        auto _find_unknown = [&]() {
            for(auto itr = m_unknown_configs.begin(); itr != m_unknown_configs.end();
                ++itr)
            {
                if(itr->first == _sid || itr->first == _name)
                    return itr;
            }
            return m_unknown_configs.end();
        };
        auto itr = _find_unknown();
        while(itr != m_unknown_configs.end())
        {
            m_unknown_configs.erase(itr);
            itr = _find_unknown();
        }
    }
    m_order.emplace_back(_sid);
    auto& _back = m_order.back();
    return m_data.emplace(string_view_t{ _back },
                          std::make_shared<tsettings<Tp, Vp>>(
                              _init, _name, _sid, _desc, std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp, typename Sp>
auto
settings::insert(tsetting_pointer_t<Tp, Vp> _ptr, Sp&& _env)
{
    static_assert(is_one_of<Tp, data_type_list_t>::value,
                  "Error! Data type is not supported. See settings::data_type_list_t");
    if(_ptr)
    {
        auto _sid = std::string{ std::forward<Sp>(_env) };
        if(_sid.empty())
            _sid = _ptr->get_env_name();
        if(!_sid.empty())
        {
            if(get_initialized())  // don't set env before timemory_init
            {
                if(vsettings::get_debug() >= 1)
                {
                    std::ostringstream oss;
                    oss << "[" << TIMEMORY_PROJECT_NAME << "][settings] set_env(\""
                        << _sid << "\", \"" << _ptr->as_string() << "\", 0);\n";
                    log::stream(std::cerr, log::color::warning()) << oss.str();
                }
                set_env(_sid, _ptr->as_string(), 0);
            }
            else
            {
                auto _find_unknown = [&]() {
                    for(auto itr = m_unknown_configs.begin();
                        itr != m_unknown_configs.end(); ++itr)
                    {
                        if(itr->first == _ptr->get_name() ||
                           itr->first == _ptr->get_env_name())
                            return itr;
                    }
                    return m_unknown_configs.end();
                };
                auto itr = _find_unknown();
                while(itr != m_unknown_configs.end())
                {
                    m_unknown_configs.erase(itr);
                    itr = _find_unknown();
                }
            }
            m_order.emplace_back(_sid);
            auto& _back = m_order.back();
            return m_data.emplace(string_view_t{ _back }, _ptr);
        }
    }

    return std::make_pair(m_data.end(), false);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
settings::get_with_env_fallback(const std::string& _env, Tp _default,
                                const pointer_t& _settings)
{
    if(!_settings)
        return get_env<Tp>(_env, _default);
    auto itr = _settings->find(_env);
    if(itr == _settings->end())
        return get_env<Tp>(_env, _default);
    return _settings->get<Tp>(_env);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
