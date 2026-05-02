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

/**
 * \file timemory/operations/types/finalize_get.hpp
 * \brief Definition for various functions for finalize_get in operations
 */

#pragma once

#include "timemory/defines.h"
#include "timemory/manager/declaration.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/file_output_message.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/filepath.hpp"

namespace tim
{
namespace operation
{
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct flamegraph
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    template <typename Up = Type>
    TIMEMORY_COLD flamegraph(storage_type*, std::string,
                             enable_if_t<trait::supports_flamegraph<Up>::value, int> = 0);

    template <typename Up = Type>
    TIMEMORY_COLD flamegraph(
        storage_type*, std::string,
        enable_if_t<!trait::supports_flamegraph<Up>::value, int> = 0);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Up>
flamegraph<Type>::flamegraph(storage_type* _data, std::string _label,  // NOLINT
                             enable_if_t<trait::supports_flamegraph<Up>::value, int>)
{
    // auto node_init        = dmp::is_initialized();
    // auto node_size        = dmp::size();
    dmp::barrier();
    auto node_rank    = dmp::rank();
    auto node_results = _data->dmp_get();
    dmp::barrier();

    if(node_rank != 0 || node_results.empty())
        return;

    result_type results;
    for(auto&& itr : node_results)
    {
        for(auto&& nitr : itr)
        {
            results.emplace_back(std::move(nitr));
        }
    }

    if(results.empty())
        return;

    using Archive = cereal::MinimalJSONOutputArchive;
    // using Archive     = cereal::PrettyJSONOutputArchive;
    using policy_type = policy::output_archive<Archive, TIMEMORY_API>;

    auto outfname = settings::compose_output_filename(_label, ".flamegraph.json");

    if(outfname.length() > 0)
    {
        std::ofstream ofs{};
        if(filepath::open(ofs, outfname))
        {
            manager::instance()->add_json_output(_label, outfname);
            if(settings::verbose() >= 0)
                operation::file_output_message<Type>{}(outfname, std::string{ _label });

            // ensure write final block during destruction before the file is closed
            auto oa = policy_type::get(ofs);

            oa->setNextName("traceEvents");
            oa->startNode();
            oa->makeArray();

            using value_type   = decay_t<decltype(std::declval<const Type>().get())>;
            using offset_map_t = std::map<int64_t, value_type>;
            using useoff_map_t = std::map<int64_t, bool>;
            auto         conv  = units::usec;
            offset_map_t total_offset;
            offset_map_t last_offset;
            offset_map_t last_value;
            useoff_map_t use_last;
            int64_t      max_depth = 1;

            for(auto& itr : results)
            {
                max_depth             = std::max<int64_t>(max_depth, itr.depth() + 1);
                use_last[itr.depth()] = false;
            }

            for(auto& itr : results)
            {
                auto _prefix = itr.prefix();
                auto value   = itr.data().get() * conv;

                auto litr = last_offset.find(itr.depth());
                if(litr != last_offset.end())
                {
                    // for(int64_t i = 0; i < max_depth; ++i)
                    //    use_last[i] = false;

                    total_offset[itr.depth()] += litr->second;

                    for(int64_t i = itr.depth() + 1; i < max_depth; ++i)
                    {
                        // use_last[i] = true;
                        total_offset[i] = total_offset[itr.depth()];
                        last_value[i]   = litr->second;
                        auto ditr       = last_offset.find(i);
                        if(ditr != last_offset.end())
                            last_offset.erase(ditr);
                    }
                    last_offset.erase(litr);
                }

                value_type offset = total_offset[itr.depth()];
                if(use_last[itr.depth()])
                    offset += last_value[itr.depth()] - value;

                oa->startNode();

                // oa->setNextName("args");
                // oa->startNode();
                // (*oa)(cereal::make_nvp("detail", _prefix));
                // (*oa)(cereal::make_nvp("count", itr.data().get_laps()));
                // (*oa)(cereal::make_nvp("depth", itr.depth()));
                // (*oa)(cereal::make_nvp("units", itr.data().get_display_unit()));
                // oa->finishNode();

                string_t _ph = "X";
                if(_prefix.find(">>>") != std::string::npos)
                    _prefix = _prefix.substr(_prefix.find_first_of(">>>") + 3);
                if(_prefix.find("|_") != std::string::npos)
                    _prefix = _prefix.substr(_prefix.find_first_of("|_") + 2);

                (*oa)(cereal::make_nvp("dur", value));
                (*oa)(cereal::make_nvp("name", _prefix));
                (*oa)(cereal::make_nvp("ph", _ph));
                (*oa)(cereal::make_nvp("pid", itr.pid()));
                (*oa)(cereal::make_nvp("tid", itr.tid()));
                (*oa)(cereal::make_nvp("ts", offset));

                oa->finishNode();

                last_offset[itr.depth()] = value;
                last_value[itr.depth()]  = value;
                // total_offset[itr.depth()] += value;
            }

            /*
            oa->startNode();
            oa->setNextName("args");
            oa->startNode();
            (*oa)(cereal::make_nvp("name", _label));
            oa->finishNode();
            string_t _ph = "M";
            string_t _cat = "";
            string_t _name = "metric";
            (*oa)(cereal::make_nvp("cat", _cat));
            (*oa)(cereal::make_nvp("name", _name));
            (*oa)(cereal::make_nvp("ph", _ph));
            (*oa)(cereal::make_nvp("pid", process::get_id()));
            (*oa)(cereal::make_nvp("tid", 0));
            (*oa)(cereal::make_nvp("ts", 0));
            oa->finishNode();
            */

            oa->finishNode();
        }
        if(ofs)
            ofs << std::endl;
        ofs.close();
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Up>
flamegraph<Type>::flamegraph(storage_type*, std::string,  // NOLINT
                             enable_if_t<!trait::supports_flamegraph<Up>::value, int>)
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
