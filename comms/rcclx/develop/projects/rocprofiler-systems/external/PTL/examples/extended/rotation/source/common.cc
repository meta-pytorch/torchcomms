// MIT License
//
// Copyright (c) 2019 Jonathan R. Madsen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#include "common.hh"
#include "macros.hh"
#include "rotate_utils.hh"

#if !defined(PTL_USE_CUDA)

//======================================================================================//

int
cuda_set_device(int)
{
    return 0;
}

//======================================================================================//

int
cuda_multi_processor_count()
{
    return 0;
}

//======================================================================================//

int
cuda_max_threads_per_block()
{
    return 0;
}

//======================================================================================//

int
cuda_warp_size()
{
    return 0;
}

//======================================================================================//

int
cuda_shared_memory_per_block()
{
    return 0;
}

//======================================================================================//

int
cuda_device_count()
{
    return 0;
}

//======================================================================================//

void
cuda_device_query()
{
    static std::atomic<int16_t> once;
    if(++once > 1)
        return;

    printf("No CUDA support enabled\n");
}

//======================================================================================//

#else

namespace
{
// add a symbol to avoid warnings about compiled file had no symbols
static int cxx_common_symbol = 0;
}  // namespace

#endif
