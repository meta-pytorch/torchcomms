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

#include "timemory/utility/argparse.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/conditional.hpp"
#include "timemory/utility/console.hpp"
#include "timemory/utility/declaration.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/launch_process.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/mangler.hpp"
#include "timemory/utility/path.hpp"
#include "timemory/utility/popen.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/singleton.hpp"
#if !defined(TIMEMORY_WINDOWS) || defined(TIMEMORY_USE_WINSOCK)
#    include "timemory/utility/socket.hpp"
#endif
#include "timemory/utility/type_id.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"
