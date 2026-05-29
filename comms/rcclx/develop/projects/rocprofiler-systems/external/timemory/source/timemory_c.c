//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "timemory/enum.h"
#include "timemory/library.h"
#include "timemory/timemory.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__cplusplus)
extern "C"
{
#endif

#if !defined(TIMEMORY_WINDOWS)
    char* strdup(const char* s)
    {
        size_t slen   = strlen(s);
        char*  result = malloc(slen + 1);
        if(result == NULL)
        {
            return NULL;
        }

        memcpy(result, s, slen + 1);
        return result;
    }
#endif

    //==================================================================================//

    void c_timemory_init(int argc, char** argv, timemory_settings _settings)
    {
        cxx_timemory_init(argc, argv, _settings);
        timemory_init_library(argc, argv);
    }

    //==================================================================================//

    void c_timemory_finalize(void) { timemory_finalize_library(); }

    //==================================================================================//

    int c_timemory_enabled(void) { return cxx_timemory_enabled(); }

    //==================================================================================//

    void* c_timemory_create_auto_timer(const char* tag)
    {
        void* _ret = NULL;
        if(cxx_timemory_enabled())
            _ret = cxx_timemory_create_auto_timer(tag);
        return _ret;
    }

    //==================================================================================//

    void* c_timemory_create_auto_tuple(const char* tag, ...)
    {
        if(!cxx_timemory_enabled())
            return NULL;

        const int max_size       = (int) TIMEMORY_COMPONENTS_END;
        int       num_components = 0;
        int*      components     = (int*) malloc(max_size * sizeof(int));
        if(!components)
            return NULL;
        va_list args;
        va_start(args, tag);
        for(int i = 0; i < max_size; ++i)
        {
            int comp = va_arg(args, int);
            if(comp >= TIMEMORY_COMPONENTS_END)
                break;
            ++num_components;
            components[i] = comp;
        }
        va_end(args);

        void* ptr = NULL;
        if(num_components > 0)
            ptr = cxx_timemory_create_auto_tuple(tag, num_components, components);
        free(components);

        return ptr;
    }

    //==================================================================================//

    void c_timemory_delete_auto_timer(void* ctimer)
    {
        ctimer = cxx_timemory_delete_auto_timer(ctimer);
        assert(ctimer == NULL);
    }

    //==================================================================================//

    void c_timemory_delete_auto_tuple(void* ctuple)
    {
        ctuple = cxx_timemory_delete_auto_tuple(ctuple);
        assert(ctuple == NULL);
    }

    //==================================================================================//

    const char* c_timemory_blank_label(const char* _extra)
    {
        return cxx_timemory_label(0, 0, "", "", _extra);
    }

    //==================================================================================//

    const char* c_timemory_basic_label(const char* _func, const char* _extra)
    {
        return cxx_timemory_label(1, 0, _func, "", _extra);
    }

    //==================================================================================//

    const char* c_timemory_label(const char* _func, const char* _file, int _line,
                                 const char* _extra)
    {
        return cxx_timemory_label(2, _line, _func, _file, _extra);
    }

    //==================================================================================//

#if defined(__cplusplus)
}
#endif
