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

#include "timemory/data/stream.hpp"
#include "timemory/defines.h"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/file_output_message.hpp"
#include "timemory/operations/types/finalize/get.hpp"
#include "timemory/plotting/declaration.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/filepath.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace tim
{
namespace operation
{
namespace finalize
{
//
#if defined(TIMEMORY_OPERATIONS_SOURCE) || !defined(TIMEMORY_USE_OPERATIONS_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::print_plot(const std::string& outfname, const std::string suffix)  // NOLINT
{
    if(node_rank == 0)
    {
        auto plot_label = label;
        if(!suffix.empty())
            plot_label += std::string(" ") + suffix;

        plotting::plot(label, plot_label, m_settings->get_output_path(),
                       m_settings->get_dart_output(), outfname);
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::write(std::ostream& os, stream_type stream)  // NOLINT
{
    if(stream)
        os << *stream << std::flush;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::print_cout(stream_type stream)  // NOLINT
{
    printf("\n");
    write(std::cout, stream);  // NOLINT
    printf("\n");
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::print_text(const std::string& outfname, stream_type stream)  // NOLINT
{
    if(outfname.length() > 0 && stream)
    {
        std::ofstream fout{};
        if(filepath::open(fout, outfname))
        {
            if(m_print)
                m_print(outfname);
            write(fout, stream);
            manager::instance()->add_text_output(label, outfname);
        }
        else
        {
            fprintf(stderr, "[storage<%s>::%s @ %i]|%i> Error opening '%s'...\n",
                    label.c_str(), __FUNCTION__, __LINE__, node_rank, outfname.c_str());
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
#endif  // !defined(TIMEMORY_OPERATIONS_SOURCE)
//
template <typename Tp>
print<Tp, true>::print(const std::string& _label, storage_type* _data,
                       const settings_t& _settings)
: base_type(_label, trait::requires_json<Tp>::value, _settings)
, data(_data)
{
    m_print = [_settings, _label](const std::string& _v) {
        if(!_settings || _settings->get_verbose() >= 0)
            operation::file_output_message<Tp>{}(_v, std::string{ _label });
    };
}
//
template <typename Tp>
print<Tp, true>::print(storage_type* _data, const settings_t& _settings)
: base_type(trait::requires_json<Tp>::value, _settings)
, data(_data)
{
    m_print = [_settings](const std::string& _v) {
        if(!_settings || _settings->get_verbose() >= 0)
            operation::file_output_message<Tp>{}(_v, std::string{ Tp::label() });
    };

    dmp::barrier();
    node_init    = dmp::is_initialized();
    node_rank    = dmp::rank();
    node_size    = dmp::size();
    node_results = data->dmp_get();
    if(tree_output())
        node_tree = data->dmp_get(node_tree);
    dmp::barrier();

    settings::indent_width<Tp, 0>(Tp::get_width());
    settings::indent_width<Tp, 1>(4);
    settings::indent_width<Tp, 2>(4);

    description = Tp::get_description();
    for(auto& itr : description)
        itr = toupper(itr);

    // find the max width
    for(const auto& mitr : node_results)
    {
        for(const auto& itr : mitr)
        {
            const auto& itr_obj    = itr.data();
            const auto& itr_prefix = itr.prefix();
            const auto& itr_depth  = itr.depth();

            if(itr_depth < 0 || itr_depth > m_settings->get_max_depth() ||
               itr_depth > max_call_stack)
                continue;

            max_depth = std::max<int64_t>(max_depth, itr_depth);

            // find global max
            settings::indent_width<Tp, 0>(itr_prefix.length());
            settings::indent_width<Tp, 1>(std::log10(itr_obj.get_laps()) + 1);
            settings::indent_width<Tp, 2>(std::log10(itr_depth) + 1);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::setup()
{
    settings::indent_width<Tp, 0>(Tp::get_width());
    settings::indent_width<Tp, 1>(4);
    settings::indent_width<Tp, 2>(4);

    description = Tp::get_description();
    for(auto& itr : description)
        itr = toupper(itr);

    // find the max width
    for(const auto& mitr : node_results)
    {
        for(const auto& itr : mitr)
        {
            const auto& itr_obj    = itr.data();
            const auto& itr_prefix = itr.prefix();
            const auto& itr_depth  = itr.depth();

            if(itr_depth < 0 || itr_depth > m_settings->get_max_depth() ||
               itr_depth > max_call_stack)
                continue;

            max_depth = std::max<int64_t>(max_depth, itr_depth);

            // find global max
            settings::indent_width<Tp, 0>(itr_prefix.length());
            settings::indent_width<Tp, 1>(std::log10(itr_obj.get_laps()) + 1);
            settings::indent_width<Tp, 2>(std::log10(itr_depth) + 1);
        }
    }

    auto file_exists = [](const std::string& fname) {
        std::cout << "Checking for existing input at " << fname << "...\n";
        std::ifstream inpf(fname.c_str());
        auto          success = inpf.is_open();
        inpf.close();
        return success;
    };

    auto fext       = trait::archive_extension<trait::output_archive_t<Tp>>{}();
    auto extensions = tim::delimit(m_settings->get_input_extensions(), ",; ");

    json_outfname = settings::compose_output_filename(label, fext);
    text_outfname = settings::compose_output_filename(label, ".txt");

    if(m_settings->get_diff_output())
    {
        extensions.insert(extensions.begin(), fext);
        for(const auto& itr : extensions)
        {
            auto inpfname = settings::compose_input_filename(label, itr);
            if(file_exists(inpfname))
            {
                json_inpfname = inpfname;
                break;
            }
        }
    }

    if(!json_inpfname.empty())
    {
        auto dext     = std::string(".diff") + fext;
        json_diffname = settings::compose_output_filename(label, dext);
        text_diffname = settings::compose_output_filename(label, ".diff.txt");
        if(m_settings->get_debug())
        {
            printf("difference filenames: '%s' and '%s'\n", json_diffname.c_str(),
                   text_diffname.c_str());
        }
    }

    if(!(file_output() && text_output()) && !cout_output())
        return;

    write_stream(data_stream, node_results);
    data_stream->set_banner(description);

    if(!node_delta.empty())
    {
        write_stream(diff_stream, node_delta);
        std::stringstream ss;
        ss << description << " vs. " << json_inpfname;
        diff_stream->set_banner(ss.str());
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::write_stream(stream_type& stream, result_type& result_array)
{
    auto stream_fmt   = Tp::get_format_flags();
    auto stream_width = Tp::get_width();
    auto stream_prec  = Tp::get_precision();

    stream = std::make_shared<utility::stream>('|', '-', stream_fmt, stream_width,
                                               stream_prec);

    using get_return_type = decltype(std::declval<const Tp>().get());
    using compute_type    = math::compute<get_return_type>;

    auto_lock_t slk(type_mutex<decltype(std::cout)>(), std::defer_lock);
    if(!slk.owns_lock())
        slk.lock();

    auto result = get_flattened(result_array);
    for(auto itr = result.begin(); itr != result.end(); ++itr)
    {
        auto& itr_obj    = (*itr)->data();
        auto& itr_prefix = (*itr)->prefix();
        auto& itr_depth  = (*itr)->depth();
        auto  itr_laps   = itr_obj.get_laps();

        if(itr_depth < 0 || itr_depth > get_max_depth())
            continue;

        // counts the number of non-exclusive values
        int64_t nexclusive = 0;
        // the sum of the exclusive values
        get_return_type exclusive_values{};

        // if we are not at the bottom of the call stack (i.e. completely
        // inclusive)
        if(itr_depth < max_depth)
        {
            // get the next iteration
            auto eitr = itr;
            std::advance(eitr, 1);
            // counts the number of non-exclusive values
            nexclusive = 0;
            // the sum of the exclusive values
            exclusive_values = get_return_type{};
            // continue while not at end of graph until first sibling is
            // encountered
            if(eitr != result.end())
            {
                auto eitr_depth = (*eitr)->depth();
                while(eitr_depth != itr_depth)
                {
                    auto& eitr_obj = (*eitr)->data();

                    // if one level down, this is an exclusive value
                    if(eitr_depth == itr_depth + 1)
                    {
                        // if first exclusive value encountered: assign; else: combine
                        if(nexclusive == 0)
                        {
                            exclusive_values = eitr_obj.get();
                        }
                        else
                        {
                            compute_type::plus(exclusive_values, eitr_obj.get());
                        }
                        // increment. beyond 0 vs. 1, this value plays no role
                        ++nexclusive;
                    }
                    // increment iterator for next while check
                    ++eitr;
                    if(eitr == result.end())
                        break;
                    eitr_depth = (*eitr)->depth();
                }
            }
        }

        auto itr_self  = compute_type::percent_diff(exclusive_values, itr_obj.get());
        auto itr_stats = (*itr)->stats();

        bool _first = std::distance(result.begin(), itr) == 0;
        if(_first)
            operation::print_header<Tp>(itr_obj, *(stream.get()), itr_stats);

        operation::print<Tp>(itr_obj, *(stream.get()), itr_prefix, itr_laps, itr_depth,
                             itr_self, itr_stats);

        stream->add_row();
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::update_data()
{
    dmp::barrier();
    node_init    = dmp::is_initialized();
    node_rank    = dmp::rank();
    node_size    = dmp::size();
    node_results = data->dmp_get();
    if(tree_output())
        node_tree = data->dmp_get(node_tree);
    dmp::barrier();

    if(m_settings->get_debug())
    {
        fprintf(stderr, "[%s][%s]|%i> dmp results size: %i\n", TIMEMORY_PROJECT_NAME,
                label.c_str(), node_rank, (int) node_results.size());
    }

    setup();

    read_json();

    if(!node_input.empty() && node_rank == 0)
    {
        node_delta.resize(node_input.size());

        size_t num_ranks = std::min<size_t>(node_input.size(), node_results.size());

        for(size_t i = 0; i < num_ranks; ++i)
        {
            for(auto& iitr : node_input.at(i))
            {
                for(auto& ritr : node_results.at(i))
                {
                    if(iitr == ritr)
                    {
                        node_delta.at(i).push_back(ritr);
                        node_delta.at(i).back() -= iitr;
                        break;
                    }
                }
            }
        }
        write_stream(diff_stream, node_delta);
        std::stringstream ss;
        ss << description << " vs. " << json_inpfname;
        diff_stream->set_banner(ss.str());
    }

#if defined(DEBUG)
    if(m_settings->get_debug() && m_settings->get_verbose() > 3)
    {
        auto indiv_results = get_flattened(node_results);
        printf("\n");
        size_t w = 0;
        for(const auto& itr : indiv_results)
            w = std::max<size_t>(w, data->get_prefix(itr->hash()).length());
        for(const auto& itr : indiv_results)
        {
            std::cout << std::setw(w) << std::left << data->get_prefix(itr->hash())
                      << " : " << itr->data();
            auto _hierarchy = itr->hierarchy();
            for(size_t i = 0; i < _hierarchy.size(); ++i)
            {
                if(i == 0)
                    std::cout << " :: ";
                std::cout << data->get_prefix(_hierarchy[i]);
                if(i + 1 < _hierarchy.size())
                    std::cout << '/';
            }
            std::cout << std::endl;
        }
        printf("\n");
    }
#endif

    if(flame_output())
        operation::finalize::flamegraph<Tp>(data, label);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::print_json(const std::string& outfname, result_type& _dist,
                            result_tree& _tree)
{
    using policy_type = policy::output_archive_t<Tp>;
    if(outfname.length() > 0)
    {
        std::ofstream ofs{};
        if(filepath::open(ofs, outfname))
        {
            auto fext = outfname.substr(outfname.find_last_of('.') + 1);
            if(fext.empty())
                fext = "unknown";
            manager::instance()->add_file_output(fext, label, outfname);
            if(m_print)
                m_print(outfname);

            // ensure write final block during destruction before the file is closed
            auto oa = policy_type::get(ofs);

            oa->setNextName("timemory");
            oa->startNode();
            if(json_output() && tree_output() && !_dist.empty() && !_tree.empty())
            {
                operation::serialization<Tp>{}(*oa, _dist, _tree);
            }
            else
            {
                if(json_output() && !_dist.empty())
                    operation::serialization<Tp>{}(*oa, _dist);
                if(tree_output() && !_tree.empty())
                    operation::serialization<Tp>{}(*oa, _tree);
            }
            oa->finishNode();  // timemory
        }
        else
        {
            TIMEMORY_PRINTF_WARNING(
                stderr, "[storage<%s>::%s @ %i]|%i> Error opening '%s'...\n",
                label.c_str(), __FUNCTION__, __LINE__, node_rank, outfname.c_str());
            return;
        }
        if(ofs)
            ofs << std::endl;
        ofs.close();
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::print_dart()
{
    using strvector_t = std::vector<std::string>;

    // if only a specific type should be echoed
    if(m_settings->get_dart_type().length() > 0)
    {
        auto dtype = m_settings->get_dart_type();
        if(operation::echo_measurement<Tp>::lowercase(dtype) !=
           operation::echo_measurement<Tp>::lowercase(label))
        {
            return;
        }
    }

    uint64_t _nitr = 0;

    auto indiv_results = get_flattened(node_results);
    for(auto& itr : indiv_results)
    {
        auto& itr_depth = itr->depth();

        if(itr_depth < 0 || itr_depth > max_depth)
            continue;

        // if only a specific number of measurements should be echoed
        if(m_settings->get_dart_count() > 0 && _nitr >= m_settings->get_dart_count())
            continue;

        auto&       itr_obj       = itr->data();
        auto&       itr_hierarchy = itr->hierarchy();
        strvector_t str_hierarchy{};
        for(const auto& hitr : itr_hierarchy)
            str_hierarchy.push_back(data->get_prefix(hitr));
        operation::echo_measurement<Tp>(itr_obj, str_hierarchy);
        ++_nitr;
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::read_json()
{
    using policy_type = policy::input_archive_t<Tp>;

    if(json_inpfname.length() > 0)
    {
        std::ifstream ifs(json_inpfname.c_str());
        if(ifs)
        {
            fprintf(stderr, "[%s][%s]|%i> Reading '%s'...\n", TIMEMORY_PROJECT_NAME,
                    label.c_str(), node_rank, json_inpfname.c_str());

            // ensure write final block during destruction before the file is closed
            auto ia = policy_type::get(ifs);

            try
            {
                ia->setNextName("timemory");
                ia->startNode();
                operation::serialization<Tp>{}(*ia, node_input);
                ia->finishNode();
            } catch(tim::cereal::Exception& e)
            {
                TIMEMORY_PRINT_HERE("Exception reading %s :: %s", json_inpfname.c_str(),
                                    e.what());
#if defined(TIMEMORY_INTERNAL_TESTING)
                TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(true, 8);
                throw std::runtime_error("error reading json");
#endif
            }
        }
        else
        {
            fprintf(stderr, "[%s][%s]|%i> Failure opening '%s' for input...\n",
                    TIMEMORY_PROJECT_NAME, label.c_str(), node_rank,
                    json_inpfname.c_str());
#if defined(TIMEMORY_INTERNAL_TESTING)
            throw std::runtime_error("error opening file");
#endif
        }
        ifs.close();
    }
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
