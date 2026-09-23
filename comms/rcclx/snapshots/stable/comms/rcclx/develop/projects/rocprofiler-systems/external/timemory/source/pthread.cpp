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

#if !defined(TIMEMORY_LIBRARY_SOURCE)
#    define TIMEMORY_LIBRARY_SOURCE 1
#endif

#include "timemory/backends/threading.hpp"
#include "timemory/compat/library.h"
#include "timemory/library.h"
#include "timemory/manager.hpp"
#include "timemory/operations/types/storage_initializer.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"
#include "timemory/trace.hpp"
#include "timemory/utility/signals.hpp"

#if !defined(TIMEMORY_WINDOWS)

#    include <array>
#    include <functional>
#    include <pthread.h>

using namespace tim::component;

//--------------------------------------------------------------------------------------//

extern "C"
{
    TIMEMORY_WEAK_PREFIX
    bool timemory_enable_pthread_gotcha_wrapper() TIMEMORY_WEAK_POSTFIX
        TIMEMORY_VISIBILITY("default");

    bool timemory_enable_pthread_gotcha_wrapper()
    {
        return tim::get_env("TIMEMORY_ENABLE_PTHREAD_GOTCHA_WRAPPER", false);
    }
}

//--------------------------------------------------------------------------------------//

struct pthread_gotcha : tim::component::base<pthread_gotcha, void>
{
    template <size_t... Idx>
    static auto get_storage(tim::index_sequence<Idx...>)
    {
        // array of finalization functions
        std::array<std::function<void()>, sizeof...(Idx)> _data{};
        // generate a function for finalizing
        TIMEMORY_FOLD_EXPRESSION(get_storage_impl<Idx>(_data));
        // return the array of finalization functions
        return _data;
    }

    struct wrapper
    {
        typedef void* (*routine_t)(void*);

        TIMEMORY_DEFAULT_OBJECT(wrapper)

        wrapper(routine_t _routine, void* _arg, bool _debug)
        : m_routine(_routine)
        , m_arg(_arg)
        , m_debug(_debug)
        {}

        void* operator()() const { return m_routine(m_arg); }

        bool         debug() const { return m_debug; }
        static void* wrap(void* _arg)
        {
            if(!_arg)
                return nullptr;

            // get the thread id
            auto _tid = tim::threading::get_id();
            // convert the argument
            wrapper* _wrapper = static_cast<wrapper*>(_arg);
            if(_wrapper->debug())
                TIMEMORY_PRINT_HERE("[T%li] Creating timemory manager", (long int) _tid);
            // create the manager and initialize the storage
            auto _tlm = tim::manager::instance();
#    if defined(TIMEMORY_INTERNAL_TESTING)
            if(_tlm == nullptr)
                throw std::runtime_error("nullptr to manager instance");
#    else
            (void) _tlm;
#    endif
            // initialize the storage
            auto _final =
                get_storage(tim::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
            if(_wrapper->debug())
                TIMEMORY_PRINT_HERE("[T%li] Executing original function",
                                    (long int) _tid);
            // execute the original function
            auto* _ret = (*_wrapper)();
            if(_wrapper->debug())
                TIMEMORY_PRINT_HERE("[T%li] Executing finalizing callbacks",
                                    (long int) _tid);
            // finalize
            for(auto& itr : _final)
                itr();
            if(_wrapper->debug())
                TIMEMORY_PRINT_HERE("[T%li] Returning from thread", (long int) _tid);
            // return the data
            return _ret;
        }

    private:
        routine_t m_routine = nullptr;
        void*     m_arg     = nullptr;
        bool      m_debug   = false;
    };

    // pthread_create
    int operator()(pthread_t* thread, const pthread_attr_t* attr,
                   void* (*start_routine)(void*), void*     arg)
    {
        // get the thread id
        auto  _tid      = tim::threading::get_id();
        auto* _settings = tim::settings::instance<TIMEMORY_API>();
        auto  _debug    = (_settings) ? _settings->get_debug() : true;
        if(_debug)
            TIMEMORY_PRINT_HERE("[T%li] Creating new thread", (long int) _tid);
        auto* _obj = new wrapper(start_routine, arg, _debug);
        return pthread_create(thread, attr, &wrapper::wrap, static_cast<void*>(_obj));
    }

private:
    template <size_t Idx, size_t N>
    static void get_storage_impl(std::array<std::function<void()>, N>& _data)
    {
        _data[Idx] = []() {
            using type = tim::component::enumerator_t<Idx>;
            tim::operation::fini_storage<type>{};
            // auto _instance = tim::storage<type>::instance();
            // tim::get_storage_singleton<tim::storage<type>>()->reset(_instance);
        };
    }
};

//--------------------------------------------------------------------------------------//

using empty_tuple_t    = tim::component_tuple<>;
using pthread_gotcha_t = tim::component::gotcha<2, empty_tuple_t, pthread_gotcha>;
using pthread_bundle_t = tim::component_tuple<pthread_gotcha_t, tim::quirk::auto_start>;

//--------------------------------------------------------------------------------------//

auto
setup_pthread_gotcha()
{
#    if defined(TIMEMORY_USE_GOTCHA)
    if(timemory_enable_pthread_gotcha_wrapper())
    {
        pthread_gotcha_t::get_initializer() = []() {
            TIMEMORY_C_GOTCHA(pthread_gotcha_t, 0, pthread_create);
        };
    }
    return std::make_shared<pthread_bundle_t>();
#    else
    return std::shared_ptr<pthread_bundle_t>{};
#    endif
}

//--------------------------------------------------------------------------------------//

namespace
{
auto pthread_gotcha_handle = setup_pthread_gotcha();
}

#else
namespace
{
auto pthread_gotcha_handle = false;
}
#endif
