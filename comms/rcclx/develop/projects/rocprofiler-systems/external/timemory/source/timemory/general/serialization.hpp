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

#include "timemory/defines.h"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/utility/filepath.hpp"

#include <fstream>
#include <string>

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED) ||                           \
    (defined(TIMEMORY_ENABLED) && TIMEMORY_ENABLED == 0)

namespace tim
{
//
template <typename... Types, typename... Args>
void
generic_serialization(Args&&...)
{}
//
}  // namespace tim

#else

namespace tim
{
//--------------------------------------------------------------------------------------//
/// \fn void generic_serialization(std::string fname, Tp obj, std::string mname,
/// std::string dname)
/// \param[in] fname Filename
/// \param[in] obj Object to serialize
/// \param[in] mname Main label for archive (default: TIMEMORY_PROJECT_NAME)
/// \param[in] dname Label for the data (default: "data")
///
/// \brief Generic function for serializing data. Uses the \ref
/// tim::policy::output_archive to determine the output archive type and configure
/// settings such as spacing, indentation width, and precision.
///
template <typename Tp, typename FuncT>
void
generic_serialization(
    const std::string& fname, const Tp& obj,
    const std::string& _main_name = TIMEMORY_PROJECT_NAME,
    const std::string& _data_name = "data",
    FuncT&& _func = [](typename policy::output_archive_t<decay_t<Tp>>::type&) {})
{
    std::ofstream ofs{};
    if(filepath::open(ofs, fname))
    {
        // ensure json write final block during destruction before the file is closed
        using policy_type = policy::output_archive_t<decay_t<Tp>>;
        auto oa           = policy_type::get(ofs);
        oa->setNextName(_main_name.c_str());
        oa->startNode();
        _func(*oa);
        (*oa)(cereal::make_nvp(_data_name.c_str(), obj));
        oa->finishNode();
    }
    if(ofs)
        ofs << std::endl;
    ofs.close();
}
//
/// \fn void generic_serialization(std::string fname, Tp obj, std::string mname,
/// std::string dname)
/// \tparam ArchiveT Output archive type
/// \tparam ApiT API tag for \ref tim::policy::output_archive look-up
/// \param[in] fname Filename
/// \param[in] obj Object to serialize
/// \param[in] mname Main label for archive (default: TIMEMORY_PROJECT_NAME)
/// \param[in] dname Label for the data (default: "data")
///
/// \brief Generic function for serializing data. Uses the \ref
/// tim::policy::output_archive to configure settings such as spacing, indentation width,
/// and precision.
///
template <typename ArchiveT, typename ApiT = TIMEMORY_API, typename Tp, typename FuncT>
void
generic_serialization(
    const std::string& fname, const Tp& obj,
    const std::string& _main_name = TIMEMORY_PROJECT_NAME,
    const std::string& _data_name = "data",
    FuncT&& _func = [](typename policy::output_archive_t<decay_t<Tp>>::type&) {})
{
    static_assert(concepts::is_output_archive<ArchiveT>::value,
                  "Error! Not an output archive type");
    std::ofstream ofs{};
    if(filepath::open(ofs, fname))
    {
        // ensure json write final block during destruction before the file is closed
        using policy_type = policy::output_archive<ArchiveT, ApiT>;
        auto oa           = policy_type::get(ofs);
        oa->setNextName(_main_name.c_str());
        oa->startNode();
        _func(*oa);
        (*oa)(cereal::make_nvp(_data_name.c_str(), obj));
        oa->finishNode();
    }
    if(ofs)
        ofs << std::endl;
    ofs.close();
}
//
/// \fn void generic_serialization(std::string fname, Tp obj, std::string mname,
/// std::string dname)
/// \tparam ArchiveT Output archive type
/// \tparam ApiT API tag for \ref tim::policy::output_archive look-up
/// \param[in] fname Filename
/// \param[in] obj Object to serialize
/// \param[in] mname Main label for archive (default: TIMEMORY_PROJECT_NAME)
/// \param[in] dname Label for the data (default: "data")
///
/// \brief Generic function for serializing data. Uses the \ref
/// tim::policy::output_archive to configure settings such as spacing, indentation width,
/// and precision.
///
template <typename ArchiveT, typename ApiT = TIMEMORY_API, typename Tp, typename FuncT>
void
generic_serialization(
    std::ostream& ofs, const Tp& obj,
    const std::string& _main_name = TIMEMORY_PROJECT_NAME,
    const std::string& _data_name = "data",
    FuncT&& _func = [](typename policy::output_archive_t<decay_t<Tp>>::type&) {})
{
    // ensure json write final block during destruction before the file is closed
    using policy_type = policy::output_archive<ArchiveT, ApiT>;
    auto oa           = policy_type::get(ofs);

    if(!_main_name.empty())
    {
        oa->setNextName(_main_name.c_str());
        oa->startNode();
    }

    // execute the function with extra data
    _func(*oa);

    (*oa)(cereal::make_nvp(_data_name, obj));

    if(!_main_name.empty())
    {
        oa->finishNode();
    }
}
//
}  // namespace tim

#endif
