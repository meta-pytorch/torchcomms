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

#if defined(TIMEMORY_USE_CUDA)
#    undef TIMEMORY_USE_CUDA
#endif

#if defined(TIMEMORY_USE_NVTX)
#    undef TIMEMORY_USE_NVTX
#endif

#include "timemory/components/timing/wall_clock.hpp"
#include "timemory/library.h"
#include "timemory/operations/definition.hpp"
#include "timemory/plotting/definition.hpp"
#include "timemory/storage/definition.hpp"
#include "timemory/variadic/definition.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define LABEL(...) TIMEMORY_LABEL(__VA_ARGS__)

using namespace tim::component;
using toolset_t     = tim::auto_tuple_t<wall_clock>;
using toolset_ptr_t = std::shared_ptr<toolset_t>;
using record_map_t  = std::unordered_map<uint64_t, toolset_ptr_t>;

void
create_record(const char* name, uint64_t* id, int, int*)
{
    auto& _records = timemory_tl_static<record_map_t>();
    *id            = timemory_get_unique_id();
    _records.insert(std::make_pair(*id, std::make_shared<toolset_t>(name)));
}

void
delete_record(uint64_t nid)
{
    auto& _records = timemory_tl_static<record_map_t>();
    // erase key from map which stops recording when object is destroyed
    _records.erase(nid);
}

long
fib(long n)
{
    return (n < 2) ? n : (fib(n - 1) + fib(n - 2));
}

int
main(int argc, char** argv)
{
    long nfib = (argc > 1) ? atol(argv[1]) : 43;

    timemory_create_function = (timemory_create_func_t) &create_record;
    timemory_delete_function = (timemory_delete_func_t) &delete_record;
    timemory_init_library(argc, argv);

    uint64_t id0 = timemory_get_begin_record(argv[0]);
    long     ans = fib(nfib);

    uint64_t id1 = timemory_get_begin_record(argv[0]);
    ans += fib(nfib + 1);

    timemory_end_record(id1);
    timemory_end_record(id0);

    timemory_end_record(20);
    printf("Answer = %li\n", ans);
    timemory_finalize_library();
    return EXIT_SUCCESS;
}
