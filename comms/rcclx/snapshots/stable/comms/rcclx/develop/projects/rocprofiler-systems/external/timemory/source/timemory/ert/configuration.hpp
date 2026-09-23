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

/** \file timemory/ert/configuration.hpp
 * \headerfile timemory/ert/configuration.hpp "timemory/ert/configuration.hpp"
 * Provides configuration for executing empirical roofline toolkit (ERT)
 *
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/defines.h"
#include "timemory/ert/aligned_allocator.hpp"
#include "timemory/ert/counter.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/ert/kernels.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/delimit.hpp"

#include <cstdint>
#include <functional>

// default vectorization width
#if !defined(TIMEMORY_VEC)
#    define TIMEMORY_VEC 256
#endif

#if !defined(TIMEMORY_USER_ERT_FLOPS)
#    define TIMEMORY_USER_ERT_FLOPS
#endif

namespace tim
{
namespace ert
{
//======================================================================================//

template <typename DeviceT, typename Tp, typename CounterT>
struct configuration
{
    using this_type       = configuration<DeviceT, Tp, CounterT>;
    using ert_data_t      = exec_data<CounterT>;
    using device_t        = DeviceT;
    using counter_t       = CounterT;
    using ert_counter_t   = counter<device_t, Tp, counter_t>;
    using ert_data_ptr_t  = std::shared_ptr<ert_data_t>;
    using executor_func_t = std::function<ert_counter_t(ert_data_ptr_t)>;
    using get_uint64_t    = std::function<uint64_t()>;
    using skip_ops_t      = std::unordered_set<size_t>;
    using get_skip_ops_t  = std::function<skip_ops_t()>;

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_num_threads()
    {
        static get_uint64_t _instance = []() {
            if(settings::ert_num_threads() > 0)
                return settings::ert_num_threads();
            // for checking if gpu
            static constexpr bool is_gpu = device::is_gpu<DeviceT>::value;
            return (is_gpu) ? settings::ert_num_threads_gpu()
                            : settings::ert_num_threads_cpu();
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_num_streams()
    {
        static get_uint64_t _instance = []() { return settings::ert_num_streams(); };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_grid_size()
    {
        static get_uint64_t _instance = []() { return settings::ert_grid_size(); };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_block_size()
    {
        static get_uint64_t _instance = []() { return settings::ert_block_size(); };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_alignment()
    {
        static get_uint64_t _instance = []() {
            return std::max<uint64_t>(settings::ert_alignment(), 8 * sizeof(Tp));
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_min_working_size()
    {
        static get_uint64_t _instance = []() {
            if(settings::ert_min_working_size() > 0)
                return settings::ert_min_working_size();
            static constexpr bool is_gpu = device::is_gpu<DeviceT>::value;
            return (is_gpu) ? settings::ert_min_working_size_gpu()
                            : settings::ert_min_working_size_cpu();
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_max_data_size()
    {
        static get_uint64_t _instance = []() -> uint64_t {
            if(settings::ert_max_data_size() > 0)
                return settings::ert_max_data_size();
            static constexpr bool is_gpu = device::is_gpu<DeviceT>::value;
            if(is_gpu)
            {
                return settings::ert_max_data_size_gpu();
            }
            {
                return 2 * ert::cache_size::get_max();
            }
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_skip_ops_t& get_skip_ops()
    {
        static get_skip_ops_t _instance = []() {
            auto       _skipstr    = settings::ert_skip_ops();
            auto       _skipstrvec = delimit(_skipstr, ",; \t");
            skip_ops_t _result;
            for(const auto& itr : _skipstrvec)
            {
                if(itr.find_first_not_of("0123456789") == std::string::npos)
                    _result.insert(atol(itr.c_str()));
            }
            return _result;
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// configure the number of threads, number of streams, block size, grid size, and
    /// alignment
    template <typename Dev                                            = DeviceT,
              enable_if_t<std::is_same<Dev, device::cpu>::value, int> = 0>
    static void configure(uint64_t nthreads, uint64_t alignment = sizeof(Tp),
                          uint64_t nstreams = 0, uint64_t block_size = 0,
                          uint64_t grid_size = 0)
    {
        get_num_threads() = [=]() -> uint64_t { return nthreads; };
        get_num_streams() = [=]() -> uint64_t { return nstreams; };
        get_grid_size()   = [=]() -> uint64_t { return grid_size; };
        get_block_size()  = [=]() -> uint64_t { return block_size; };
        get_alignment()   = [=]() -> uint64_t { return alignment; };
    }

    //----------------------------------------------------------------------------------//
    /// configure the number of threads, number of streams, block size, grid size, and
    /// alignment
    template <typename Dev                                            = DeviceT,
              enable_if_t<std::is_same<Dev, device::gpu>::value, int> = 0>
    static void configure(uint64_t nthreads, uint64_t alignment = sizeof(Tp),
                          uint64_t nstreams = 1, uint64_t block_size = 1024,
                          uint64_t grid_size = 0)
    {
        get_num_threads() = [=]() -> uint64_t { return nthreads; };
        get_num_streams() = [=]() -> uint64_t { return nstreams; };
        get_grid_size()   = [=]() -> uint64_t { return grid_size; };
        get_block_size()  = [=]() -> uint64_t { return block_size; };
        get_alignment()   = [=]() -> uint64_t { return alignment; };
    }

    //----------------------------------------------------------------------------------//

    static executor_func_t& get_executor()
    {
        static executor_func_t _instance = [](ert_data_ptr_t data) {
            using lli = long long int;
            // configuration sizes
            auto _mws_size   = get_min_working_size()();
            auto _max_size   = get_max_data_size()();
            auto _num_thread = get_num_threads()();
            auto _num_stream = get_num_streams()();
            auto _grid_size  = get_grid_size()();
            auto _block_size = get_block_size()();
            auto _align_size = get_alignment()();
            auto _skip_ops   = get_skip_ops()();

            // execution parameters
            exec_params params(_mws_size, _max_size, _num_thread, _num_stream, _grid_size,
                               _block_size);
            // operation _counter instance
            ert_counter_t _counter(params, data, _align_size);

            // set bytes per element
            _counter.bytes_per_element = sizeof(Tp);
            // set number of memory accesses per element from two functions
            _counter.memory_accesses_per_element = 2;

            for(const auto& itr : _skip_ops)
                _counter.add_skip_ops(itr);

            auto dtype = demangle(typeid(Tp).name());

            std::string _dev_name{};
            if(std::is_same<DeviceT, device::cpu>::value)
                _dev_name = "[device::cpu]";
            else if(std::is_same<DeviceT, device::gpu>::value)
                _dev_name = "[device::gpu]";

            if(settings::debug() || settings::verbose() > 0)
            {
                printf("[ert::executor]%s> "
                       "working-set = %lli, max-size = %lli, num-thread = %lli, "
                       "num-stream = "
                       "%lli, grid-size = %lli, block-size = %lli, align-size = %lli, "
                       "data-type "
                       "= %s\n",
                       _dev_name.c_str(), (lli) _mws_size, (lli) _max_size,
                       (lli) _num_thread, (lli) _num_stream, (lli) _grid_size,
                       (lli) _block_size, (lli) _align_size, dtype.c_str());
            }

            return _counter;
        };
        return _instance;
    }

public:
    bool            verbose          = false;
    get_uint64_t    num_threads      = this_type::get_num_threads();       // NOLINT
    get_uint64_t    num_streams      = this_type::get_num_streams();       // NOLINT
    get_uint64_t    min_working_size = this_type::get_min_working_size();  // NOLINT
    get_uint64_t    max_data_size    = this_type::get_max_data_size();     // NOLINT
    get_uint64_t    alignment        = this_type::get_alignment();         // NOLINT
    get_uint64_t    grid_size        = this_type::get_grid_size();         // NOLINT
    get_uint64_t    block_size       = this_type::get_block_size();        // NOLINT
    executor_func_t executor         = this_type::get_executor();          // NOLINT

public:
    TIMEMORY_DEFAULT_OBJECT(configuration)
};

//======================================================================================//

template <typename DeviceT, typename Tp, typename CounterT>
struct executor
{
    static_assert(!std::is_same<DeviceT, device::gpu>::value,
                  "Error! Device should not be gpu");

    //----------------------------------------------------------------------------------//
    // useful aliases
    //
    using device_type        = DeviceT;
    using value_type         = Tp;
    using configuration_type = configuration<device_type, value_type, CounterT>;
    using counter_type       = counter<device_type, value_type, CounterT>;
    using this_type          = executor<device_type, value_type, CounterT>;
    using callback_type      = std::function<void(counter_type&)>;
    using ert_data_t         = exec_data<CounterT>;

public:
    TIMEMORY_DEFAULT_OBJECT(executor)

    //----------------------------------------------------------------------------------//
    //  standard invocation with no callback specialization
    //
    executor(configuration_type& config, std::shared_ptr<ert_data_t> _data)
    {
        (*this)(config, _data);
    }

    //----------------------------------------------------------------------------------//
    //  specialize the counter callback
    //
    template <typename FuncT>
    executor(configuration_type& config, std::shared_ptr<ert_data_t> _data,
             FuncT&& _counter_callback)
    {
        (*this)(config, _data, std::forward<FuncT>(_counter_callback));
    }

    //----------------------------------------------------------------------------------//
    //  execute
    //
    template <typename FuncT = std::function<void(uint64_t, counter_type&)>>
    auto operator()(configuration_type& config, std::shared_ptr<ert_data_t> _data = {},
                    FuncT&& _counter_callback = FuncT{})
    {
        std::function<void(uint64_t, counter_type&)> _cb =
            std::forward<FuncT>(_counter_callback);
        try
        {
            if(!_data)
                _data.reset(new ert_data_t);
            auto _counter = config.executor(_data);
            if(_cb)
                _counter.set_callback(std::move(_cb));
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
        return _data;
    }

public:
    //----------------------------------------------------------------------------------//
    //
    callback_type callback = get_callback();  // NOLINT

public:
    //----------------------------------------------------------------------------------//
    //
    static callback_type& get_callback()
    {
        static callback_type _instance = [](counter_type& _counter) {
            this_type::execute(_counter);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    static void execute(counter_type& _counter)
    {
        // vectorization number of ops
        static constexpr const int SIZE_BITS = sizeof(Tp) * 8;
        static_assert(SIZE_BITS > 0, "Calculated bits size is not greater than zero");
        static constexpr const int VEC = TIMEMORY_VEC / SIZE_BITS;
        static_assert(VEC > 0, "Calculated vector size is zero");

        // functions
        auto store_func = [](Tp& a, const Tp& b) { a = b; };
        auto add_func   = [](Tp& a, const Tp& b, const Tp& c) { a = b + c; };
        auto fma_func   = [](Tp& a, const Tp& b, const Tp& c) { a = a * b + c; };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = "scalar_add";
        // run the kernels
        ops_main<1>(_counter, add_func, store_func);

        // set the label
        _counter.label = "vector_fma";
        // run the kernels
        if(!ops_main<TIMEMORY_USER_ERT_FLOPS>(_counter, fma_func, store_func))
            ops_main<VEC / 2, VEC, 2 * VEC, 4 * VEC>(_counter, fma_func, store_func);
    }

    //----------------------------------------------------------------------------------//
    // user specified flops as template parameters
    template <size_t... Flops, enable_if_t<(sizeof...(Flops) > 0)> = 0>
    static bool execute(counter_type&                             _counter,
                        std::array<std::string, sizeof...(Flops)> _labels)
    {
        bool _executed = false;
        auto itr       = _labels.begin();
        TIMEMORY_FOLD_EXPRESSION(_executed |= execute_impl<Flops>(_counter, *(itr++)));
        return _executed;
    }

private:
    //----------------------------------------------------------------------------------//
    //
    template <size_t Flops, enable_if_t<Flops == 1> = 0>
    static bool execute_impl(counter_type& _counter, const std::string& _label)
    {
        // functions
        auto store_func = [](Tp& a, const Tp& b) { a = b; };
        auto add_func   = [](Tp& a, const Tp& b, const Tp& c) { a = b + c; };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = _label;
        // run the kernels
        return ops_main<Flops>(_counter, add_func, store_func);
    }

    //----------------------------------------------------------------------------------//
    //
    template <size_t Flops, enable_if_t<(Flops > 1)> = 0>
    static bool execute_impl(counter_type& _counter, const std::string& _label)
    {
        // functions
        auto store_func = [](Tp& a, const Tp& b) { a = b; };
        auto fma_func   = [](Tp& a, const Tp& b, const Tp& c) { a = a * b + c; };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = _label;
        return ops_main<Flops>(_counter, fma_func, store_func);
    }
};

//======================================================================================//

template <typename Tp, typename CounterT>
struct executor<device::gpu, Tp, CounterT>
{
    using DeviceT = device::gpu;
    static_assert(std::is_same<DeviceT, device::gpu>::value,
                  "Error! Device should be gpu");

    //----------------------------------------------------------------------------------//
    // useful aliases
    //
    using device_type        = device::gpu;
    using value_type         = Tp;
    using configuration_type = configuration<device_type, value_type, CounterT>;
    using counter_type       = counter<device_type, value_type, CounterT>;
    using this_type          = executor<device_type, value_type, CounterT>;
    using callback_type      = std::function<void(counter_type&)>;
    using ert_data_t         = exec_data<CounterT>;

public:
    TIMEMORY_DEFAULT_OBJECT(executor)

    //----------------------------------------------------------------------------------//
    //  standard invocation with no callback specialization
    //
    executor(configuration_type& config, std::shared_ptr<ert_data_t> _data)
    {
        (*this)(config, _data);
    }

    //----------------------------------------------------------------------------------//
    //  specialize the counter callback
    //
    template <typename FuncT>
    executor(configuration_type& config, std::shared_ptr<ert_data_t> _data,
             FuncT&& _counter_callback)
    {
        (*this)(config, _data, std::forward<FuncT>(_counter_callback));
    }

    //----------------------------------------------------------------------------------//
    //  execute
    //
    template <typename FuncT = std::function<void(uint64_t, counter_type&)>>
    auto operator()(configuration_type& config, std::shared_ptr<ert_data_t> _data = {},
                    FuncT&& _counter_callback = FuncT{})
    {
        std::function<void(uint64_t, counter_type&)> _cb =
            std::forward<FuncT>(_counter_callback);
        try
        {
            if(!_data)
                _data.reset(new ert_data_t);
            auto _counter = config.executor(_data);
            if(_cb)
                _counter.set_callback(std::move(_cb));
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
        return _data;
    }

public:
    //----------------------------------------------------------------------------------//
    //
    callback_type callback = get_callback();  // NOLINT

public:
    //----------------------------------------------------------------------------------//
    //
    static callback_type& get_callback()
    {
        static callback_type _instance = [](counter_type& _counter) {
            this_type::execute(_counter);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    // The enclosing parent function for an extended __host__ __device__
    // lambda must allow its address to be taken
    static void execute(counter_type& _counter)
    {
        // functions
        auto store_func = [] TIMEMORY_DEVICE_LAMBDA(Tp & a, const Tp& b) { a = b; };
        auto add_func   = [] TIMEMORY_DEVICE_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
            a = b + c;
        };
        // auto mult_func = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
        //    a = b * c;
        //};
        auto fma_func = [] TIMEMORY_DEVICE_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
            a = a * b + c;
        };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = "scalar_add";
        // run the kernels
        ops_main<1>(_counter, add_func, store_func);

        // set the label
        // _counter.label = "vector_mult";
        // run the kernels
        // ops_main<4, 16, 64, 128, 256, 512>(_counter, mult_func, store_func);

        // set the label
        _counter.label = "vector_fma";
        // run the kernels
        if(!ops_main<TIMEMORY_USER_ERT_FLOPS>(_counter, fma_func, store_func))
            ops_main<4, 16, 64, 128, 256, 512>(_counter, fma_func, store_func);
    }

    //----------------------------------------------------------------------------------//
    // user specified flops as template parameters
    template <size_t... Flops, enable_if_t<(sizeof...(Flops) > 0)> = 0>
    static bool execute(counter_type&                             _counter,
                        std::array<std::string, sizeof...(Flops)> _labels)
    {
        bool _executed = false;
        auto itr       = _labels.begin();
        TIMEMORY_FOLD_EXPRESSION(_executed |= execute_impl<Flops>(_counter, *(itr++)));
        return _executed;
    }

public:
    //----------------------------------------------------------------------------------//
    //
    template <size_t Flops>
    static enable_if_t<Flops == 1, bool> execute_impl(counter_type&      _counter,
                                                      const std::string& _label)
    {
        // functions
        auto store_func = [] TIMEMORY_DEVICE_LAMBDA(Tp & a, const Tp& b) { a = b; };
        auto add_func   = [] TIMEMORY_DEVICE_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
            a = b + c;
        };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = _label;
        // run the kernels
        return ops_main<Flops>(_counter, add_func, store_func);
    }

    //----------------------------------------------------------------------------------//
    //
    template <size_t Flops>
    static enable_if_t<(Flops > 1), bool> execute_impl(counter_type&      _counter,
                                                       const std::string& _label)
    {
        // functions
        auto store_func = [] TIMEMORY_DEVICE_LAMBDA(Tp & a, const Tp& b) { a = b; };
        auto fma_func   = [] TIMEMORY_DEVICE_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
            a = a * b + c;
        };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = _label;
        return ops_main<Flops>(_counter, fma_func, store_func);
    }
};

//======================================================================================//
/// for variadic expansion to set the callback
///
template <typename ExecutorT>
struct callback
{
    template <typename FuncT>
    callback(FuncT&& f)
    {
        ExecutorT::get_callback() = f;
    }

    template <typename FuncT>
    callback(ExecutorT& _exec, FuncT&& f)
    {
        _exec.callback = f;
    }
};

//======================================================================================//

template <typename DeviceT, typename CounterT, typename Tp, typename... Types,
          typename DataType = exec_data<CounterT>,
          typename DataPtr  = std::shared_ptr<DataType>,
          typename std::enable_if<(sizeof...(Types) == 0), int>::type = 0>
std::shared_ptr<DataType>
execute(std::shared_ptr<DataType> _data = std::make_shared<DataType>())
{
    using ConfigType = configuration<DeviceT, Tp, CounterT>;
    using ExecType   = executor<DeviceT, Tp, CounterT>;

    ConfigType _config{};
    ExecType(_config, _data);

    return _data;
}

//======================================================================================//

template <typename DeviceT, typename CounterT, typename Tp, typename... Types,
          typename DataType = exec_data<CounterT>,
          typename DataPtr  = std::shared_ptr<DataType>,
          typename std::enable_if<(sizeof...(Types) > 0), int>::type = 0>
std::shared_ptr<DataType>
execute(std::shared_ptr<DataType> _data = std::make_shared<DataType>())
{
    execute<DeviceT, CounterT, Tp>(_data);
    execute<DeviceT, CounterT, Types...>(_data);
    return _data;
}

//======================================================================================//

}  // namespace ert

}  // namespace tim
