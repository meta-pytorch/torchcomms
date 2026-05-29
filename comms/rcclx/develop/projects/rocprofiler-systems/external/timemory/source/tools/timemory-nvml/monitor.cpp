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

#include "nvml.hpp"
#include "nvml_memory_info.hpp"
#include "nvml_processes.hpp"
#include "nvml_temperature.hpp"
#include "nvml_utilization_rate.hpp"

#include <chrono>
#include <map>
#include <thread>
#include <timemory/data/ring_buffer.hpp>
#include <timemory/general/serialization.hpp>
#include <timemory/timemory.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>
#include <vector>

using bundle_t = tim::lightweight_tuple<nvml_processes, nvml_memory_info,
                                        nvml_temperature, nvml_utilization_rate>;

using ring_buffer_t       = tim::data_storage::ring_buffer<bundle_t>;
using device_bundle_map_t = std::map<int, ring_buffer_t>;

template <typename Tp, typename ArchiveT>
void
serialize_metadata(ArchiveT& ar)
{
    auto _label = Tp::label();
    ar.setNextName(_label.c_str());
    ar.startNode();
    tim::operation::serialization<Tp>{}(
        ar, typename tim::operation::serialization<Tp>::metadata{});
    ar.finishNode();
}

template <typename ArchiveT, typename... Tp>
void
serialize_metadata(ArchiveT& ar, tim::type_list<Tp...>)
{
    TIMEMORY_FOLD_EXPRESSION(serialize_metadata<Tp>(ar));
}

void
dump(const device_bundle_map_t& _data, bool report = false)
{
    using map_type = std::map<int, std::vector<bundle_t>>;
    map_type _mdata{};

    TIMEMORY_CONDITIONAL_PRINT_HERE(debug() || verbose() > 1, "Writing bundle data to %s",
                                    get_config().get_output_filename().c_str());

    for(auto& itr : _data)
    {
        std::vector<bundle_t> _vdata{};
        size_t                _n = itr.second.count();
        _vdata.resize(_n);
        TIMEMORY_CONDITIONAL_PRINT_HERE(debug() || verbose() > 1,
                                        "Device %i has %i entries", (int) itr.first,
                                        (int) _n);
        for(size_t i = 0; i < _n; ++i)
        {
            itr.second.read(&_vdata[i]);
        }
        _mdata.emplace(itr.first, _vdata);
    }
    using json_type = tim::cereal::PrettyJSONOutputArchive;
    // extra function
    auto _cmdline = [](json_type& ar) {
        ar(tim::cereal::make_nvp("config", get_config()));
        ar.setNextName("metadata");
        ar.startNode();
        serialize_metadata(ar, tim::convert_t<bundle_t, tim::type_list<>>{});
        ar.finishNode();
    };

    auto fname = get_config().get_output_filename();
    fname += ".json";
    if(debug() || verbose() > 0 || report)
        fprintf(stderr, "%s>>> Outputting '%s'...\n", (verbose() < 0) ? "" : "\n",
                fname.c_str());
    tim::generic_serialization<json_type>(fname, _mdata, "timemory", "nvml", _cmdline);
}

void
monitor(const std::vector<nvml_device_info>& _device_info)
{
    auto get_record_time = []() {
        char _buffer[128];
        auto _now = std::time_t{ std::time(nullptr) };

        if(std::strftime(_buffer, sizeof(_buffer), time_format().c_str(),
                         std::localtime(&_now)))
            return std::string{ _buffer };
        return std::string{};
    };

    device_bundle_map_t device_bundle_map{};

    // create a ring buffer for each device
    for(const auto& itr : _device_info)
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(
            debug(), "Creating ring buffer for %i entries for device %i",
            (int) buffer_count(), itr.index);
        device_bundle_map.emplace(itr.index, buffer_count());
    }

    long int _duration = 1.0e9 * sample_interval();
    size_t   ncount    = 0;
    while(true)
    {
        bool _valid = false;
        for(const auto& itr : _device_info)
        {
            bundle_t _bundle{ get_record_time() };
            _bundle.sample(itr.device);
            if(_bundle.get<nvml_processes>()->get().size() > 0)
            {
                _valid = true;
                device_bundle_map[itr.index].write(&_bundle);
            }
        }
        auto nidx = (_valid) ? ++ncount : ncount;
        if(finished())
            break;
        if(max_samples() > 0 && nidx >= max_samples())
        {
            break;
        }
        else if(nidx >= dump_interval())
        {
            ncount = 0;
            dump(device_bundle_map);
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _duration });
    }

    dump(device_bundle_map, true);
}
