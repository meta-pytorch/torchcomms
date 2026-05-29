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

#if defined(DEBUG)
#    undef DEBUG
#endif

#include "timemory/library.h"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/insert.hpp"
#include "timemory/timemory.hpp"

#include <map>
#include <memory>

//--------------------------------------------------------------------------------------//

using namespace tim::component;

using test_list_t = tim::component_list<wall_clock, cpu_util, cpu_clock, peak_rss>;

static std::map<uint64_t, std::shared_ptr<test_list_t>> test_map;

void
custom_create_record(const char* name, uint64_t* id, int n, int* ct)
{
    uint64_t idx = timemory_get_unique_id();
    auto     tmp = std::make_shared<test_list_t>(name);
    tim::initialize(*tmp, n, ct);
    tmp->initialize<cpu_util, cpu_clock>();
    test_map[idx] = tmp;
    test_map[idx]->start();
    *id = idx;
}

void
custom_delete_record(uint64_t id)
{
    auto itr = test_map.find(id);
    if(itr != test_map.end())
    {
        itr->second->stop();
        test_map.erase(itr);
    }
}

uint64_t
get_wc_storage_size()
{
    return tim::storage<wall_clock>::instance()->size();
}

uint64_t
get_cu_storage_size()
{
    return tim::storage<cpu_util>::instance()->size();
}

uint64_t
get_cc_storage_size()
{
    return tim::storage<cpu_clock>::instance()->size();
}

uint64_t
get_pr_storage_size()
{
    return tim::storage<peak_rss>::instance()->size();
}

uint64_t
get_uc_storage_size()
{
    return tim::storage<user_clock>::instance()->size();
}

uint64_t
get_sc_storage_size()
{
    return tim::storage<system_clock>::instance()->size();
}
