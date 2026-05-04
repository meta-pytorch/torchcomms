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

/** \file backends/dmp.hpp
 * \headerfile backends/dmp.hpp "timemory/backends/dmp.hpp"
 * Defines the common interface for the Distributed Memory Parallelism model
 * used (e.g. MPI, UPC++)
 *
 */

#pragma once

#include "timemory/backends/mpi.hpp"
#include "timemory/backends/upcxx.hpp"

namespace tim
{
namespace dmp
{
//--------------------------------------------------------------------------------------//

inline bool
using_mpi()
{
#if defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_MPI_HEADERS)
    return mpi::is_initialized();
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
using_upcxx()
{
#if defined(TIMEMORY_USE_UPCXX) && defined(TIMEMORY_USE_MPI)
    return (mpi::is_initialized()) ? false : upc::is_initialized();
#elif defined(TIMEMORY_USE_UPCXX)
    return upc::is_initialized();
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
is_supported()
{
#if defined(TIMEMORY_USE_UPCXX)
    return upc::is_supported();
#elif defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_MPI_HEADERS)
    return mpi::is_supported();
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
is_finalized()
{
#if defined(TIMEMORY_USE_UPCXX) && defined(TIMEMORY_USE_MPI)
    return upc::is_finalized() && mpi::is_finalized();
#elif defined(TIMEMORY_USE_UPCXX)
    return upc::is_finalized();
#elif defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_MPI_HEADERS)
    return mpi::is_finalized();
#else
    static bool _instance = true;
    return _instance;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
set_finalized(bool v)
{
#if !defined(TIMEMORY_USE_UPCXX) && !defined(TIMEMORY_USE_MPI)
    consume_parameters(v);
#else
#    if defined(TIMEMORY_USE_UPCXX)
    upc::is_finalized() = v;
#    endif
#    if defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_MPI_HEADERS)
    mpi::is_finalized() = v;
#    endif
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
is_initialized()
{
#if defined(TIMEMORY_USE_UPCXX)
    return upc::is_initialized();
#elif defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_MPI_HEADERS)
    return mpi::is_initialized();
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename... ArgsT>
inline void
initialize(ArgsT&&... _args)
{
#if defined(TIMEMORY_USE_UPCXX)
    return upc::initialize(std::forward<ArgsT>(_args)...);
#elif defined(TIMEMORY_USE_MPI)
    return mpi::initialize(std::forward<ArgsT>(_args)...);
#else
    consume_parameters(_args...);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
finalize()
{
#if defined(TIMEMORY_USE_UPCXX)
    return upc::finalize();
#elif defined(TIMEMORY_USE_MPI)
    return mpi::finalize();
#endif
}

//--------------------------------------------------------------------------------------//

template <typename... ArgsT>
inline int32_t
rank(ArgsT&&... _args)
{
#if defined(TIMEMORY_USE_UPCXX)
    return upc::rank(std::forward<ArgsT>(_args)...);
#elif defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_MPI_HEADERS)
    return mpi::rank(std::forward<ArgsT>(_args)...);
#else
    consume_parameters(_args...);
    return 0;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename... ArgsT>
inline int32_t
size(ArgsT&&... _args)
{
#if defined(TIMEMORY_USE_UPCXX)
    return upc::size(std::forward<ArgsT>(_args)...);
#elif defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_MPI_HEADERS)
    return mpi::size(std::forward<ArgsT>(_args)...);
#else
    consume_parameters(_args...);
    return 1;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
barrier()
{
#if defined(TIMEMORY_USE_UPCXX)
    upc::barrier();
#elif defined(TIMEMORY_USE_MPI)
    mpi::barrier();
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace dmp
}  // namespace tim
