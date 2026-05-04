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

#if !defined(TIMEMORY_PYBIND11_SOURCE)
#    define TIMEMORY_PYBIND11_SOURCE
#endif

#include "timemory/utility/macros.hpp"
//
#include "pybind11/cast.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "timemory/enum.h"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/insert.hpp"
#include "timemory/runtime/invoker.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/timemory.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace py::literals;
using namespace tim::component;

using manager_t = tim::manager;

namespace pytim
{
using string_t = std::string;
//
//--------------------------------------------------------------------------------------//
//
inline auto
get_ostream_handle(py::object file_handle)
{
    using pystream_t  = py::detail::pythonbuf;
    using cxxstream_t = std::ostream;

    if(!(py::hasattr(file_handle, "write") && py::hasattr(file_handle, "flush")))
    {
        throw py::type_error(
            "get_ostream_handle(file_handle): incompatible function argument:  "
            "`file` must be a file-like object, but `" +
            (std::string)(py::repr(file_handle)) + "` provided");
    }
    auto buf    = std::make_shared<pystream_t>(file_handle);
    auto stream = std::make_shared<cxxstream_t>(buf.get());
    return std::make_pair(stream, buf);
}
//
//======================================================================================//
//
//                              MANAGER
//
//======================================================================================//
//
namespace manager
{
//
//--------------------------------------------------------------------------------------//
//
string_t
write_ctest_notes(py::object man, std::string directory, bool append)
{
    py::list filenames = man.attr("text_files").cast<py::list>();

    std::stringstream ss;
    ss << std::endl;
    ss << "IF(NOT DEFINED CTEST_NOTES_FILES)" << std::endl;
    ss << "    SET(CTEST_NOTES_FILES )" << std::endl;
    ss << "ENDIF(NOT DEFINED CTEST_NOTES_FILES)" << std::endl;
    ss << std::endl;

    // loop over ASCII report filenames
    for(auto itr : filenames)
    {
        std::string fname = itr.cast<std::string>();
#if defined(_WIN32) || defined(_WIN64)
        while(fname.find("\\") != std::string::npos)
            fname = fname.replace(fname.find("\\"), 1, "/");
#endif
        ss << "LIST(APPEND CTEST_NOTES_FILES \"" << fname << "\")" << std::endl;
    }

    ss << std::endl;
    ss << "IF(NOT \"${CTEST_NOTES_FILES}\" STREQUAL \"\")" << std::endl;
    ss << "    LIST(REMOVE_DUPLICATES CTEST_NOTES_FILES)" << std::endl;
    ss << "ENDIF(NOT \"${CTEST_NOTES_FILES}\" STREQUAL \"\")" << std::endl;
    ss << std::endl;

    // create directory (easier in Python)
    auto locals = py::dict("directory"_a = directory);
    py::exec(R"(
             import os

             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)

             file_path = os.path.join(directory, "CTestNotes.cmake")
             )",
             py::globals(), locals);

    std::string   file_path = locals["file_path"].cast<std::string>();
    std::ofstream outf;
    if(append)
        outf.open(file_path.c_str(), std::ios::app);
    else
        outf.open(file_path.c_str());

    if(outf)
        outf << ss.str();
    outf.close();

    return file_path;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace manager
//
//======================================================================================//
//
//                          OPTIONS
//
//======================================================================================//
//
namespace opt
{
//
//--------------------------------------------------------------------------------------//
//
void
safe_mkdir(string_t directory)
{
    auto locals = py::dict("directory"_a = directory);
    py::exec(R"(
             import os
             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)
             )",
             py::globals(), locals);
}
//
//--------------------------------------------------------------------------------------//
//
void
ensure_directory_exists(string_t file_path)
{
    auto locals = py::dict("file_path"_a = file_path);
    py::exec(R"(
             import os

             directory = os.path.dirname(file_path)
             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)
             )",
             py::globals(), locals);
}
//
//--------------------------------------------------------------------------------------//
//
py::object
parse_args(py::object parser)
{
    auto locals = py::dict("parser"_a = parser);
    py::exec(R"(
             import timemory

             args = parser.parse_args()

             # copy over values
             timemory.settings.dart_output = args.timemory_echo_dart
             timemory.options.matplotlib_backend = args.timemory_mpl_backend
             )",
             py::globals(), locals);
    return locals["args"].cast<py::object>();
}
//
//--------------------------------------------------------------------------------------//
//
py::object
parse_known_args(py::object parser)
{
    auto locals = py::dict("parser"_a = parser);
    py::exec(R"(
             import timemory

             args, left = parser.parse_known_args()

             # replace sys.argv with unknown args only
             sys.argv = sys.argv[:1]+left

             # copy over values
             timemory.settings.dart_output = args.timemory_echo_dart
             timemory.options.matplotlib_backend = args.timemory_mpl_backend
             )",
             py::globals(), locals);
    return locals["args"].cast<py::object>();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace opt
}  // namespace pytim
