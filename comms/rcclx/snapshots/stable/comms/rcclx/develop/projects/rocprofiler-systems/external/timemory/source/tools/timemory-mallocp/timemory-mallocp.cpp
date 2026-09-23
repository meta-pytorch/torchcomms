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

#include "timemory/components/gotcha/memory_allocations.hpp"
#include "timemory/library.h"
#include "timemory/timemory.hpp"

#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;

using malloc_toolset_t     = tim::component_tuple<memory_allocations>;
static uint64_t global_cnt = 0;
static uint64_t global_id  = std::numeric_limits<uint64_t>::max();
//
//--------------------------------------------------------------------------------------//
//
namespace
{
auto&
get_toolset()
{
    static auto _instance = std::shared_ptr<malloc_toolset_t>{};
    return _instance;
}
}  // namespace
//
//--------------------------------------------------------------------------------------//
//
extern "C"
{
    void timemory_mallocp_library_ctor() {}

    uint64_t timemory_start_mallocp()
    {
        // provide environment variable for enabling/disabling
        if(tim::get_env<bool>("TIMEMORY_ENABLE_MALLOCP", true))
        {
            auto& _handle = get_toolset();
            if(!_handle)
            {
                _handle       = std::make_shared<malloc_toolset_t>("timemory-mallocp");
                auto _cleanup = []() {
                    auto& _handle = get_toolset();
                    if(_handle)
                    {
                        _handle->stop();
                        _handle.reset();
                    }
                };

                tim::manager::instance()->add_cleanup("timemory-mallocp", _cleanup);
                global_id = global_cnt + 1;
            }

            // start the wrappers
            _handle->start();
            // if the first instance, actually call malloc and free to pre-allocate
            // storage and disable if a null pointer was returned
            if(global_cnt == 0)
            {
                size_t _size = 8 * sizeof(char);
                char*  _buff = static_cast<char*>(::malloc(_size));
                if(_buff == nullptr)
                {
                    TIMEMORY_PRINT_HERE(
                        "timemory_mallocp was started but malloc(%i) returned a "
                        "nullptr. Disabling timemory_mallocp",
                        static_cast<int>(_size));
                    _handle->stop();
                    return global_cnt;
                }
                ::free(_buff);
            }
            return ++global_cnt;
        }
        else
        {
            return 0;
        }
    }

    uint64_t timemory_stop_mallocp(uint64_t id)
    {
        if(id == global_id)
            global_cnt = 1;
        if(global_cnt == 1)
        {
            tim::manager::instance()->cleanup("timemory-mallocp");
        }
        return --global_cnt;
    }

    void timemory_register_mallocp() { global_id = timemory_start_mallocp(); }
    void timemory_deregister_mallocp() { global_id = timemory_stop_mallocp(global_id); }

    // Below are for FORTRAN codes
    void     timemory_mallocp_library_ctor_() {}
    uint64_t timemory_start_mallocp_() { return timemory_start_mallocp(); }
    uint64_t timemory_stop_mallocp_(uint64_t id) { return timemory_stop_mallocp(id); }
    void     timemory_register_mallocp_() { timemory_register_mallocp(); }
    void     timemory_deregister_mallocp_() { timemory_deregister_mallocp(); }

}  // extern "C"
