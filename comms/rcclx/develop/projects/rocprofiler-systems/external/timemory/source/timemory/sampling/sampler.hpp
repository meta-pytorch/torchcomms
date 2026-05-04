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

#ifndef TIMEMORY_SAMPLING_SAMPLER_HPP_
#    define TIMEMORY_SAMPLING_SAMPLER_HPP_
#endif

#include "timemory/backends/threading.hpp"
#include "timemory/components/base.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types/sample.hpp"
#include "timemory/sampling/allocator.hpp"
#include "timemory/sampling/trigger.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/macros.hpp"

// C++ includes
#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// C includes
#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(TIMEMORY_UNIX)
#    include <unistd.h>
#endif

#if !defined(TIMEMORY_SAMPLER_DEPTH_DEFAULT)
#    define TIMEMORY_SAMPLER_DEPTH_DEFAULT 64
#endif

#if !defined(TIMEMORY_SAMPLER_OFFSET_DEFAULT)
#    define TIMEMORY_SAMPLER_OFFSET_DEFAULT 3
#endif

#if !defined(TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT)
#    if defined(TIMEMORY_USE_LIBUNWIND)
#        define TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT true
#    else
#        define TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT false
#    endif
#endif

namespace tim
{
namespace sampling
{
template <typename CompT, size_t N>
struct sampler;
}
//
//--------------------------------------------------------------------------------------//
//
namespace trait
{
template <typename CompT, size_t N>
struct is_component<sampling::sampler<CompT, N>> : true_type
{};

template <typename Tp>
struct buffer_size : std::integral_constant<size_t, 0>
{};

template <typename Tp>
struct provide_backtrace : std::false_type
{};

template <typename Tp>
struct backtrace_depth : std::integral_constant<size_t, TIMEMORY_SAMPLER_DEPTH_DEFAULT>
{};

template <typename Tp>
struct backtrace_offset : std::integral_constant<size_t, TIMEMORY_SAMPLER_OFFSET_DEFAULT>
{};

template <typename Tp>
struct backtrace_use_libunwind
: std::conditional_t<TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT, std::true_type,
                     std::false_type>
{};
}  // namespace trait
//
//--------------------------------------------------------------------------------------//
//
namespace sampling
{
//
template <typename FuncT>
inline auto
set_notify(std::function<void(bool*)>& _notify, FuncT&& _v, int)
    -> decltype(std::forward<FuncT>(_v)(std::declval<bool*>()), void())
{
    _notify = std::forward<FuncT>(_v);
}

template <typename FuncT>
inline auto
set_notify(std::function<void(bool*)>& _notify, FuncT&& _v, long)
{
    _notify = [&_v](bool* _completed) {
        std::forward<FuncT>(_v)();
        if(_completed)
            *_completed = true;
    };
}
//
template <typename FuncT>
inline auto
set_notify(std::function<void(bool*)>& _notify, FuncT&& _v)
{
    set_notify(_notify, std::forward<FuncT>(_v), 0);
}
//
//--------------------------------------------------------------------------------------//
//
/// \value tim::sampling::dynamic
/// \brief A bare enumeration value implicitly convertible to zero.
enum
{
    dynamic = 0,
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t N>
struct fixed_size : std::true_type
{};
//
template <>
struct fixed_size<dynamic> : std::false_type
{};
//
template <size_t N>
using fixed_size_t = typename fixed_size<N>::type;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::sampling::sampler
/// \brief The design of the sampler struct is similar to the \ref tim::component::gotcha
/// component: the first template parameter is a specification of a bundle of components
/// which the struct internally takes measurements with and the second template parameter
/// is a size specification. The size specification is to help ensure that the allocation
/// does not grow too significantly, however, by specifying the size as either 0 (zero)
/// or \ref tim::sampling::dynamic, a std::vector is used instead of the fixed-sized
/// std::array.
/// \code{.cpp}
/// using dynamic_t        = tim::sampling::dynamic;
/// using timer_spec_t     = tim::sampling::timer;
/// // sampling components
/// using bundle_t         = tim::component_tuple<wall_clock>;
/// using sample_t         = tim::sampling::sampler<bundle_t, dynamic_t>;
///
/// // create an instance
/// sample_t _sampler("example");
///
/// // configure a real-clock timer delivering the SIGALRM signal at a frequency
/// // of 50 interrupts per second (of real time) after a 2 second delay
/// _sampler.configure(timer_spec_t{ SIGALRM, CLOCK_REALTIME, SIGEV_SIGNAL, 50.0, 2.0 });
///
/// // configure a cpu-clock timer delivering the SIGPROF signal at a frequency
/// // of 50 interrupts per second (of CPU time) after a 2 second delay. Specify
/// // the delivery should always be to the thread identified by
/// // `tim::threading::get_sys_tid()`. Note: the first thread id is just
/// // used for debugging
/// _sampler.configure(timer_spec_t{ SIGPROF, CLOCK_THREAD_CPUTIME_ID,
///                                  SIGEV_THREAD_ID, 50.0, 2.0,
///                                  tim::threading::get_id(),
///                                  tim::threading::get_sys_tid() });
///
/// _sampler.start();                       // start the sampling
///
/// // ...
/// sample_t::pause();                      // wait for one signal to be delivered
/// // ...
///
/// _sampler.stop();                        // stop sampling and wall-clock
///
/// auto _data = _sampler.get();            // get the sampled data
/// \endcode
template <template <typename...> class CompT, size_t N, typename... Types>
struct sampler<CompT<Types...>, N>
: component::base<sampler<CompT<Types...>, N>, void>
, private policy::instance_tracker<sampler<CompT<Types...>, N>, true>
{
    using this_type    = sampler<CompT<Types...>, N>;
    using base_type    = component::base<this_type, void>;
    using bundle_type  = CompT<Types...>;
    using signal_set_t = std::set<int>;
    using pid_cb_t     = std::function<bool(pid_t, int, int)>;
    using array_t      = conditional_t<fixed_size_t<N>::value, std::array<bundle_type, N>,
                                  std::vector<bundle_type>>;
    using data_type    = array_t;
    using buffer_t     = data_storage::ring_buffer<bundle_type>;
    using allocator_t =
        conditional_t<fixed_size_t<N>::value, allocator<void>, allocator<this_type>>;

    using array_type    = array_t;
    using tracker_type  = policy::instance_tracker<this_type, true>;
    using trigger_ptr_t = std::unique_ptr<trigger>;

    friend struct allocator<this_type>;

    static_assert(fixed_size<N>::value || trait::buffer_size<this_type>::value > 0,
                  "Error! Dynamic sampler has a default buffer size of zero");

    static constexpr bool is_dynamic() { return !fixed_size_t<N>::value; }

    static TIMEMORY_NOINLINE void execute(int signum);
    static TIMEMORY_NOINLINE void execute(int signum, siginfo_t*, void*);

    static auto get_latest_samples();
    static auto get_samplers()
    {
        auto_lock_t            _lk{ type_mutex<this_type>() };
        std::deque<this_type*> _v{};
        for(auto itr : get_persistent_data().m_thread_instances)
        {
            for(auto vitr : itr.second)
                _v.emplace_back(vitr);
        }
        return _v;
    }
    static auto& get_samplers(int64_t _v)
    {
        return get_persistent_data().m_thread_instances[_v];
    }

public:
    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    sampler(std::string, int64_t _tid = threading::get_id(), int _verbose = 0);

    sampler(std::shared_ptr<allocator_t>, std::string, int64_t _tid = threading::get_id(),
            int _verbose = 0);
    ~sampler();

    template <typename... Args, typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    TIMEMORY_NOINLINE void sample(Args&&...);

    template <typename... Args, typename Tp = fixed_size_t<N>,
              enable_if_t<!Tp::value> = 0>
    TIMEMORY_NOINLINE void sample(Args&&...);

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    void start();
    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    void stop();

    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    void start();
    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    void stop();

public:
    auto count() const { return m_triggers.size(); }

    bundle_type*& get_last() { return m_last; }
    bundle_type*  get_last() const { return m_last; }

    bundle_type*& get_latest() { return m_last; }
    bundle_type*  get_latest() const { return m_last; }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    const bundle_type& get(size_t idx) const;
    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    const bundle_type& get(size_t idx) const;

    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    auto get_data() const
    {
        return (m_alloc) ? m_alloc->get_data(this) : data_type{};
    }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    array_t& get_data()
    {
        return m_data;
    }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    const array_t& get_data() const
    {
        return m_data;
    }

public:
    /// \fn void configure(Tp&& _v)
    /// \param[in] _v \ref tim::sampling::timer or \ref tim::sampling::overflow
    ///
    /// \brief Set up the sampler
    template <typename Tp>
    void configure(Tp&& _overflow);

    void reset(std::vector<trigger_ptr_t>&& = {});

    /// \fn void ignore(std::set<int> _signals)
    /// \param[in] _signals Set of signals
    ///
    /// \brief Ignore the signals (applies to all threads)
    void ignore(std::set<int> _signals);

    /// \fn void clear()
    /// \brief Clear all signals. Recommended to call ignore() prior to clearing all the
    /// signals
    void clear() { m_triggers.clear(); }

    /// \fn void pause()
    /// \brief Pause until a signal is delivered
    TIMEMORY_INLINE void pause()
    {
        if(!m_triggers.empty())
            ::pause();
    }

    /// \fn int wait(pid_t _pid, int _verb, bool _debug, Func&& _cb)
    /// \param[in] _pid Process id to wait on
    /// \param[in] _verb Logging verbosity
    /// \param[in] _debug Enable debug logging
    /// \param[in] _cb Callback for checking whether to exit
    ///
    /// \brief Wait function with an optional user callback of type:
    ///
    /// \code{.cpp}
    ///     bool (*)(int a, int b)
    /// \endcode
    /// where 'a' is the status, 'b' is the error value, and returns true if waiting
    /// should continue
    template <typename Func = pid_cb_t>
    int wait(pid_t _pid, int _verbose, bool _debug, Func&& _callback = pid_callback(),
             int64_t _freq_ns = 10 * units::msec);

    template <typename Func = pid_cb_t>
    int wait(int _verbose = settings::verbose(), bool _debug = settings::debug(),
             Func&& _callback = pid_callback())
    {
        return wait(process::get_target_id(), _verbose, _debug,
                    std::forward<Func>(_callback));
    }

    template <typename Func, enable_if_t<std::is_function<Func>::value> = 0>
    int wait(pid_t _pid, Func&& _callback, int _verbose = settings::verbose(),
             bool _debug = settings::debug())
    {
        return wait(_pid, _verbose, _debug, std::forward<Func>(_callback));
    }

    template <typename Func, enable_if_t<std::is_function<Func>::value> = 0>
    int wait(Func&& _callback, int _verbose = settings::verbose(),
             bool _debug = settings::debug())
    {
        return wait(process::get_target_id(), _verbose, _debug,
                    std::forward<Func>(_callback));
    }

    /// \fn void set_flags(int)
    /// \param flags[in] the sigaction flags to use
    ///
    /// \brief Set the sigaction flags, e.g. SA_RESTART | SA_SIGINFO
    void set_flags(int _flags) { m_flags = _flags; }

    /// \fn void set_verbose(int)
    /// \brief Configure the verbosity
    void set_verbose(int _v)
    {
        m_verbose = _v;
        if(m_alloc)
            m_alloc->set_verbose(_v);
    }

    /// Pass a function to the allocator for offloading a full buffer
    template <typename FuncT>
    void set_offload(FuncT&& _v)
    {
        if(m_alloc)
            m_alloc->set_offload(std::forward<FuncT>(_v));
    }

    /// \fn void remove_trigger(int)
    /// \brief Remove from set of triggers
    void remove_trigger(int _v)
    {
        m_triggers.erase(std::remove_if(
            m_triggers.begin(), m_triggers.end(),
            [_v](const trigger_ptr_t& itr) { return itr->signal() == _v; },
            m_triggers.end()));
    }

    /// \fn size_t get_id() const
    /// \brief Get the unique global identifier
    size_t get_id() const { return m_idx; }

    /// \fn const std::vector<trigger_ptr_t>& get_triggers() const
    /// \brief Get triggers handled by the sampler
    const std::vector<trigger_ptr_t>& get_triggers() const { return m_triggers; }

    /// \fn const trigger* get_trigger(int signal) const
    /// \param signal[in] signal for desired trigger
    /// \brief Find the trigger providing this signal
    const trigger* get_trigger(int _signal) const
    {
        for(const auto& itr : m_triggers)
        {
            if(itr->signal() == _signal)
                return itr.get();
        }
        return nullptr;
    }

    /// \fn int get_verbse() const
    /// \brief Get the verbosity
    int get_verbose() const { return m_verbose; }

    size_t get_buffer_size() const { return m_buffer_size; }
    void   set_buffer_size(size_t _v) { m_buffer_size = _v; }

    auto get_allocator() const { return m_alloc; }

    template <typename FuncT>
    void set_notify(FuncT&& _v)
    {
        sampling::set_notify(m_notify, std::forward<FuncT>(_v));
    }

    template <typename FuncT>
    void set_move(FuncT&& _v)
    {
        m_move = std::forward<FuncT>(_v);
    }

    auto get_sample_count() const { return m_count; }

private:
    static void default_notify(bool* _completed)
    {
        if(_completed)
            *_completed = true;
    }

protected:
    using notify_func_t = std::function<void(bool*)>;
    using move_func_t   = std::function<void(this_type*, buffer_t&&)>;

    int                          m_verbose     = tim::settings::verbose();
    int                          m_flags       = SA_RESTART | SA_SIGINFO;
    int                          m_pid         = process::get_id();
    int64_t                      m_tid         = -1;
    sig_atomic_t                 m_sig_lock    = 0;
    size_t                       m_idx         = get_counter()++;
    size_t                       m_buffer_size = trait::buffer_size<this_type>::value;
    size_t                       m_count       = 0;
    sigaction_t                  m_custom_sigaction   = {};
    sigaction_t                  m_original_sigaction = {};
    bundle_type*                 m_last               = nullptr;
    array_t                      m_data               = {};
    notify_func_t                m_notify             = &default_notify;
    move_func_t                  m_move               = [](this_type*, buffer_t&&) {};
    buffer_t                     m_buffer             = {};
    std::shared_ptr<allocator_t> m_alloc              = {};
    std::vector<trigger_ptr_t>   m_triggers           = {};
    std::string                  m_label              = {};

private:
    struct instance_data
    {};

    struct persistent_data : instance_data
    {
        std::map<int64_t, std::deque<this_type*>> m_thread_instances = {};
    };

    static std::atomic<int64_t>& get_counter()
    {
        static std::atomic<int64_t> _ntot{ 0 };
        return _ntot;
    }

    static persistent_data& get_persistent_data()
    {
        static std::atomic<int64_t> _ntot{ 0 };
        static auto                 _main = persistent_data{};
        static thread_local auto    _n    = _ntot++;
        if(_n == 0)
            return _main;
        static thread_local auto _instance = _main;
        return _instance;
    }

    /// \fn pid_cb_t& pid_callback()
    /// \brief Default callback when configuring sampler
    static pid_cb_t pid_callback()
    {
        return [](pid_t _id, int, int) { return _id != process::get_id(); };
    }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    void _init_sampler();

    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    void _init_sampler();

public:
    static instance_data& get_default_config()
    {
        return static_cast<instance_data&>(get_persistent_data());
    }
};
}  // namespace sampling
}  // namespace tim

#if !defined(TIMEMORY_SAMPLING_SAMPLER_USE_EXTERN) &&                                    \
    !defined(TIMEMORY_SAMPLING_SAMPLER_CPP_)
#    include "timemory/sampling/sampler.cpp"
#endif
