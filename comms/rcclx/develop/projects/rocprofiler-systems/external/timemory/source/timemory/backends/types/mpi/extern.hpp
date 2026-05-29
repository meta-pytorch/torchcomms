//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file extern/init.hpp
 * \headerfile extern/init.hpp "timemory/extern/init.hpp"
 * Provides extern initialization
 *
 */

#pragma once

#if defined(TIMEMORY_USE_MPI)

#    if !defined(TIMEMORY_MPI_INIT)
#        define TIMEMORY_MPI_INIT 0
#    endif

#    if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_MPI_INIT_EXTERN)
#        define TIMEMORY_USE_MPI_INIT_EXTERN 1
#    endif

#    include "timemory/backends/mpi.hpp"
#    include "timemory/config.hpp"
#    include "timemory/defines.h"
#    include "timemory/manager/declaration.hpp"
#    include "timemory/settings/declaration.hpp"

#    if !defined(TIMEMORY_USE_MPI_INIT_EXTERN) && TIMEMORY_MPI_INIT > 0
#        include "timemory/components/tau_marker/backends.hpp"
#    endif

//--------------------------------------------------------------------------------------//
//
#    if defined(TIMEMORY_MPI_INIT_SOURCE)
#        define TIMEMORY_MPI_INIT_LINKAGE(...) __VA_ARGS__
#    elif defined(TIMEMORY_USE_MPI_INIT_EXTERN)
#        define TIMEMORY_MPI_INIT_LINKAGE(...) extern __VA_ARGS__
#    else
#        define TIMEMORY_MPI_INIT_LINKAGE(...) inline __VA_ARGS__
#    endif

//--------------------------------------------------------------------------------------//
//
extern "C"
{
    TIMEMORY_MPI_INIT_LINKAGE(void) timemory_MPI_Comm_set_attr(void);
    TIMEMORY_MPI_INIT_LINKAGE(void) timemory_MPI_Init(const int* argc, char*** argv);
#    if TIMEMORY_MPI_INIT > 0
    TIMEMORY_MPI_INIT_LINKAGE(int) MPI_Init(int* argc, char*** argv);
    TIMEMORY_MPI_INIT_LINKAGE(int) MPI_Init_thread(int*, char***, int, int*);
#    endif

#    if !defined(TIMEMORY_USE_MPI_INIT_EXTERN)

    //----------------------------------------------------------------------------------//
    //
    TIMEMORY_MPI_INIT_LINKAGE(void) timemory_MPI_Comm_set_attr(void)
    {
        static int comm_key = -1;
        if(comm_key < 0)
        {
            static auto _copy = [](MPI_Comm, int, void*, void*, void*, int*) {
                return MPI_SUCCESS;
            };
            static auto _fini = [](MPI_Comm, int, void*, void*) {
                if(tim::settings::debug())
                {
                    printf("[%s][%s@%s:%i]> intercepted MPI_Finalize!\n",
                           TIMEMORY_PROJECT_NAME, __FUNCTION__, __FILE__, __LINE__);
                }
                auto* manager = tim::timemory_manager_master_instance();
                if(manager)
                    manager->finalize();
                ::tim::dmp::set_finalized(true);
                if(tim::settings::debug())
                {
                    printf("[%s][%s@%s:%i]> MPI_Finalize completed!\n",
                           TIMEMORY_PROJECT_NAME, __FUNCTION__, __FILE__, __LINE__);
                }
                return MPI_SUCCESS;
            };
            PMPI_Comm_create_keyval(_copy, _fini, &comm_key, nullptr);
            PMPI_Comm_set_attr(MPI_COMM_SELF, comm_key, nullptr);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    TIMEMORY_MPI_INIT_LINKAGE(void) timemory_MPI_Init(const int* argc, char*** argv)
    {
        timemory_MPI_Comm_set_attr();
        ::tim::timemory_init(*argc, *argv);
    }

    //----------------------------------------------------------------------------------//
    //
#        if TIMEMORY_MPI_INIT > 0

    //----------------------------------------------------------------------------------//
    //
    int MPI_Init(int* argc, char*** argv)
    {
        if(tim::settings::debug())
        {
            printf("[%s][%s@%s:%i]> intercepted MPI_Init!\n", TIMEMORY_PROJECT_NAME,
                   __FUNCTION__, __FILE__, __LINE__);
        }
        TIMEMORY_TAU_INIT(argc, argv);
        auto ret = PMPI_Init(argc, argv);
        timemory_MPI_Init(argc, argv);
        return ret;
    }

    //----------------------------------------------------------------------------------//
    //
    int MPI_Init_thread(int* argc, char*** argv, int req, int* prov)
    {
        if(tim::settings::debug())
        {
            printf("[%s][%s@%s:%i]> intercepted MPI_Init_thread!\n",
                   TIMEMORY_PROJECT_NAME, __FUNCTION__, __FILE__, __LINE__);
        }
        TIMEMORY_TAU_INIT(argc, argv);
        auto ret = PMPI_Init_thread(argc, argv, req, prov);
        timemory_MPI_Init(argc, argv);
        return ret;
    }
#        endif
#    endif
}

#endif  // defined(TIMEMORY_USE_MPI)
