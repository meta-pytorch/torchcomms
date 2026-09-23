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

#pragma once

#include "timemory/macros/language.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

// cereal will cause some -Wclass-memaccess warnings that are quite annoying
#if defined(__GNUC__) && (__GNUC__ > 7)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

// types
#include "timemory/tpls/cereal/cereal/types/array.hpp"
#include "timemory/tpls/cereal/cereal/types/chrono.hpp"
#include "timemory/tpls/cereal/cereal/types/common.hpp"
#include "timemory/tpls/cereal/cereal/types/map.hpp"
#include "timemory/tpls/cereal/cereal/types/memory.hpp"
#include "timemory/tpls/cereal/cereal/types/set.hpp"
#include "timemory/tpls/cereal/cereal/types/string.hpp"
#include "timemory/tpls/cereal/cereal/types/tuple.hpp"
#include "timemory/tpls/cereal/cereal/types/utility.hpp"
#include "timemory/tpls/cereal/cereal/types/vector.hpp"

#if defined(CXX17)
#    include "timemory/tpls/cereal/cereal/types/variant.hpp"
#endif

#if defined(__GNUC__) && (__GNUC__ > 7)
#    pragma GCC diagnostic pop
#endif
