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

#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/tpls/cereal/types.hpp"

// cereal will cause some -Wclass-memaccess warnings that are quite annoying
#if defined(__GNUC__) && (__GNUC__ > 7)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

// forward decls
namespace tim
{
namespace cereal
{
class BinaryInputArchive;
class BinaryOutputArchive;
class PortableBinaryInputArchive;
class PortableBinaryOutputArchive;
class XMLInputArchive;
class XMLOutputArchive;
class YAMLInputArchive;
class YAMLOutputArchive;
// json are always included so no forward decl necessary
}  // namespace cereal
}  // namespace tim

// archives
#include "timemory/tpls/cereal/cereal/archives/json.hpp"
#if defined(TIMEMORY_USE_XML)
#    include "timemory/tpls/cereal/cereal/archives/xml.hpp"
#endif
#if defined(TIMEMORY_USE_YAML)
#    include "timemory/tpls/cereal/cereal/archives/yaml.hpp"
#endif

#if defined(__GNUC__) && (__GNUC__ > 7)
#    pragma GCC diagnostic pop
#endif
