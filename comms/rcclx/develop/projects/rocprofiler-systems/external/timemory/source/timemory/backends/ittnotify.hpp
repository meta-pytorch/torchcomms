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

#if defined(TIMEMORY_USE_VTUNE)
#    include <ittnotify.h>
#endif

#include <string>

namespace tim
{
namespace ittnotify
{
//--------------------------------------------------------------------------------------//

template <typename... ArgsT>
void
consume_parameters(ArgsT&&...)
{}

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_VTUNE)
using id_t            = __itt_id;
using event_t         = __itt_event;
using domain_t        = __itt_domain;
using string_handle_t = __itt_string_handle;
#else
using id_t            = int;
using event_t         = int;
using domain_t        = int;
using string_handle_t = std::string;
#endif

//--------------------------------------------------------------------------------------//

inline void
pause()
{
#if defined(TIMEMORY_USE_VTUNE)
    __itt_pause();
#endif
}

//--------------------------------------------------------------------------------------//

inline void
resume()
{
#if defined(TIMEMORY_USE_VTUNE)
    __itt_resume();
#endif
}

//--------------------------------------------------------------------------------------//

inline void
detach()
{
#if defined(TIMEMORY_USE_VTUNE)
    __itt_detach();
#endif
}

//--------------------------------------------------------------------------------------//

inline domain_t*
create_domain(const std::string& _name, bool _enable = true)
{
#if defined(TIMEMORY_USE_VTUNE)
    auto _ret   = __itt_domain_create(_name.c_str());
    _ret->flags = (_enable) ? 1 : 0;  // enable domain
    return _ret;
#else
    consume_parameters(_name, _enable);
    return nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

inline string_handle_t*
get_domain_handle(const std::string& _name)
{
#if defined(TIMEMORY_USE_VTUNE)
    return __itt_string_handle_create(_name.c_str());
#else
    consume_parameters(_name);
    return nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

inline event_t
create_event(const std::string& _name)
{
#if defined(TIMEMORY_USE_VTUNE)
    return __itt_event_create(_name.c_str(), _name.length());
#else
    consume_parameters(_name);
    return 0;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
start_frame(const domain_t* _domain, id_t* _id = nullptr)
{
#if defined(TIMEMORY_USE_VTUNE)
    __itt_frame_begin_v3(_domain, _id);
#else
    consume_parameters(_domain, _id);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
end_frame(const domain_t* _domain, id_t* _id = nullptr)
{
#if defined(TIMEMORY_USE_VTUNE)
    __itt_frame_end_v3(_domain, _id);
#else
    consume_parameters(_domain, _id);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
start_event(event_t _evt)
{
#if defined(TIMEMORY_USE_VTUNE)
    __itt_event_start(_evt);
#else
    consume_parameters(_evt);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
end_event(event_t _evt)
{
#if defined(TIMEMORY_USE_VTUNE)
    __itt_event_end(_evt);
#else
    consume_parameters(_evt);
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace ittnotify
}  // namespace tim
