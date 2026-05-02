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

#include "timemory/unwind/dlinfo.hpp"

#include "timemory/macros/os.hpp"

#if defined(TIMEMORY_UNIX)
#    include <dlfcn.h>
#endif

namespace tim
{
namespace unwind
{
#if defined(TIMEMORY_UNIX)

dlinfo::dlinfo(Dl_info _info)
: location{ _info.dli_fname ? _info.dli_fname : "", _info.dli_fbase }
, symbol{ _info.dli_sname ? _info.dli_sname : "", _info.dli_saddr }
{}

dlinfo
dlinfo::construct(unw_word_t _addr)
{
    Dl_info _info{};
    dladdr(reinterpret_cast<void*>(_addr), &_info);
    return dlinfo{ _info };
}

dlinfo
dlinfo::construct(unw_word_t _addr, unw_word_t _offset)
{
    Dl_info _info{};
    dladdr(reinterpret_cast<void*>(_addr + _offset), &_info);
    return dlinfo{ _info };
}

#else

dlinfo dlinfo::construct(unw_word_t) { return dlinfo{}; }

dlinfo dlinfo::construct(unw_word_t, unw_word_t) { return dlinfo{}; }

#endif

}  // namespace unwind
}  // namespace tim
