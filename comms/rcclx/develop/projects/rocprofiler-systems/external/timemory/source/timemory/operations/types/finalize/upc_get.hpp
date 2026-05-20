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

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/finalize/get.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/demangle.hpp"

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
struct upc_get<Type, true>
{
    static constexpr bool value  = true;
    using storage_type           = impl::storage<Type, value>;
    using result_type            = typename storage_type::result_array_t;
    using distrib_type           = typename storage_type::dmp_result_t;
    using result_node            = typename storage_type::result_node;
    using graph_type             = typename storage_type::graph_t;
    using graph_node             = typename storage_type::graph_node;
    using hierarchy_type         = typename storage_type::uintvector_t;
    using get_type               = get<Type, value>;
    using metadata_t             = typename get_type::metadata;
    using basic_tree_type        = typename get_type::basic_tree_vector_type;
    using basic_tree_vector_type = std::vector<basic_tree_type>;

    explicit TIMEMORY_COLD upc_get(storage_type& _storage)
    : m_storage(&_storage)
    {}

    TIMEMORY_COLD distrib_type& operator()(distrib_type&);
    TIMEMORY_COLD basic_tree_vector_type& operator()(basic_tree_vector_type&);

    template <typename Archive>
    TIMEMORY_COLD enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
                  operator()(Archive&);

private:
    storage_type* m_storage = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct upc_get<Type, false>
{
    static constexpr bool value = false;
    using storage_type          = impl::storage<Type, value>;

    upc_get(storage_type&) {}

    template <typename Tp>
    Tp& operator()(Tp&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename upc_get<Type, true>::distrib_type&
upc_get<Type, true>::operator()(distrib_type& results)
{
    if(!m_storage)
        return results;

    auto& data = *m_storage;
#if !defined(TIMEMORY_USE_UPCXX)
    if(settings::debug())
        TIMEMORY_PRINT_HERE("%s", "timemory not using UPC++");

    results = distrib_type{};
    results.emplace_back(std::move(data.get()));
#else
    if(settings::debug())
        TIMEMORY_PRINT_HERE("%s", "timemory using UPC++");

    upc::barrier();

    int comm_rank = upc::rank();
    int comm_size = upc::size();

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [=](const result_type& src) {
        std::stringstream ss;
        {
            auto oa = policy::output_archive<cereal::MinimalJSONOutputArchive,
                                             TIMEMORY_API>::get(ss);
            (*oa)(cereal::make_nvp("data", src));
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [=](const std::string& src) {
        result_type       ret;
        std::stringstream ss;
        ss << src;
        {
            auto ia =
                policy::input_archive<cereal::JSONInputArchive, TIMEMORY_API>::get(ss);
            try
            {
                (*ia)(cereal::make_nvp("data", ret));
            } catch(cereal::Exception& e)
            {
                std::string _msg = ss.str();
                // truncate
                constexpr size_t max_msg_len = 60;
                if(_msg.length() > max_msg_len)
                    _msg = TIMEMORY_JOIN("...", _msg.substr(0, max_msg_len - 13),
                                         _msg.substr(_msg.length() - 10));
                TIMEMORY_PRINT_HERE("Warning! Exception in "
                                    "operation::finalize::upc_get<%s>::send_serialize: "
                                    "%s\n\t%s",
                                    demangle<Type>().c_str(), e.what(), _msg.c_str());
            }
            if(settings::debug())
                printf("[RECV: %i]> data size: %lli\n", comm_rank,
                       (long long int) ret.size());
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Function executed on remote node
    //
    auto remote_serialize = [=]() {
        return send_serialize(storage_type::master_instance()->get());
    };

    results.resize(comm_size);

    //------------------------------------------------------------------------------//
    //  Combine on master rank
    //
    if(comm_rank == 0)
    {
        for(int i = 1; i < comm_size; ++i)
        {
            upcxx::future<std::string> fut = upcxx::rpc(i, remote_serialize);
            while(!fut.ready())
                upcxx::progress();
            fut.wait();
            results[i] = recv_serialize(fut.result());
        }
        results[comm_rank] = data.get();
    }

    //------------------------------------------------------------------------------//
    //  Calculate the total number of measurement records
    //
    auto get_num_records = [&](const auto& _inp) {
        int _sz = 0;
        for(const auto& itr : _inp)
            _sz += itr.size();
        return _sz;
    };

    upcxx::barrier(upcxx::world());

    if(comm_rank != 0)
    {
        results = distrib_type{};
        results.emplace_back(std::move(data.get()));
    }

    // collapse into a single result
    if(comm_rank == 0 && settings::collapse_processes() && settings::node_count() <= 1)
    {
        auto init_size = get_num_records(results);
        if(settings::debug() || settings::verbose() > 3)
        {
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i]> collapsing %i records from %i ranks",
                demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, comm_size);
        }

        auto _collapsed = distrib_type{};
        // so we can pop off back
        std::reverse(results.begin(), results.end());
        while(!results.empty())
        {
            if(_collapsed.empty())
                _collapsed.emplace_back(std::move(results.back()));
            else
                operation::finalize::merge<Type, true>(_collapsed.front(),
                                                       results.back());
            results.pop_back();
        }

        // assign results to collapsed entry
        results = std::move(_collapsed);

        if(settings::debug() || settings::verbose() > 3)
        {
            auto fini_size = get_num_records(results);
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i]> collapsed %i records into %i records "
                "from %i ranks",
                demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, fini_size, comm_size);
        }
    }
    else if(comm_rank == 0 && settings::collapse_processes() &&
            settings::node_count() > 1)
    {
        // calculate some size parameters
        int32_t nmod  = comm_size % settings::node_count();
        int32_t bsize = comm_size / settings::node_count() + ((nmod == 0) ? 0 : 1);
        int32_t bins  = comm_size / bsize;

        if(settings::debug() || settings::verbose() > 3)
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i]> node_count = %i, comm_size = %i, bins = "
                "%i, bin size = %i",
                demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, settings::node_count(), comm_size, bins, bsize);

        // generate a map of the ranks to the node ids
        int32_t                              ncnt = 0;  // current count
        int32_t                              midx = 0;  // current bin map index
        std::map<int32_t, std::set<int32_t>> binmap;
        for(int32_t i = 0; i < comm_size; ++i)
        {
            if(settings::debug())
                TIMEMORY_PRINT_HERE("[%s][pid=%i][rank=%i]> adding rank %i to bin %i",
                                    demangle<upc_get<Type, true>>().c_str(),
                                    (int) process::get_id(), comm_rank, i, midx);

            binmap[midx].insert(i);
            // check to see if we reached the bin size
            if(++ncnt == bsize)
            {
                // set counter to zero and advance the node
                ncnt = 0;
                ++midx;
            }
        }

        auto init_size = get_num_records(results);
        if(settings::debug() || settings::verbose() > 3)
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i]> collapsing %i records from %i ranks into "
                "%i bins",
                demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, comm_size, (int) binmap.size());

        assert((int32_t) binmap.size() <= (int32_t) settings::node_count());

        // the collapsed data
        auto _collapsed = distrib_type(binmap.size());
        // loop over the node indexes
        for(const auto& itr : binmap)
        {
            // target the node index
            auto& _dst = _collapsed.at(itr.first);
            for(const auto& bitr : itr.second)
            {
                // combine the node index entry with all of the ranks in that node
                auto& _src = results.at(bitr);
                operation::finalize::merge<Type, true>(_dst, _src);
            }
        }

        // assign results to collapsed entry
        results = std::move(_collapsed);

        if(settings::debug() || settings::verbose() > 3)
        {
            auto fini_size = get_num_records(results);
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i]> collapsed %i records into %i records "
                "and %i bins",
                demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, fini_size, (int) results.size());
        }
    }

    if(settings::debug() || settings::verbose() > 1)
    {
        auto ret_size = get_num_records(results);
        TIMEMORY_PRINT_HERE("[%s][pid=%i]> %i total records on rank %i of %i",
                            demangle<upc_get<Type, true>>().c_str(),
                            (int) process::get_id(), ret_size, comm_rank, comm_size);
    }

#endif

    return results;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename upc_get<Type, true>::basic_tree_vector_type&
upc_get<Type, true>::operator()(basic_tree_vector_type& bt)
{
    if(!m_storage)
        return bt;

    auto& data = *m_storage;
#if !defined(TIMEMORY_USE_UPCXX)
    if(settings::debug())
        TIMEMORY_PRINT_HERE("%s", "timemory not using UPC++");

    auto entry = basic_tree_type{};
    bt         = basic_tree_vector_type(1, data.get(entry));
#else
    if(settings::debug())
        TIMEMORY_PRINT_HERE("%s", "timemory using UPC++");

    upc::barrier(upc::world());

    int comm_rank = upc::rank(upc::world());
    int comm_size = upc::size(upc::world());

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [&](const basic_tree_type& src) {
        std::stringstream ss;
        {
            auto oa = policy::output_archive<cereal::MinimalJSONOutputArchive,
                                             TIMEMORY_API>::get(ss);
            (*oa)(cereal::make_nvp("data", src));
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [&](const std::string& src) {
        basic_tree_type   ret;
        std::stringstream ss;
        ss << src;
        {
            auto ia =
                policy::input_archive<cereal::JSONInputArchive, TIMEMORY_API>::get(ss);
            try
            {
                (*ia)(cereal::make_nvp("data", ret));
            } catch(cereal::Exception& e)
            {
                std::string _msg = ss.str();
                // truncate
                constexpr size_t max_msg_len = 60;
                if(_msg.length() > max_msg_len)
                    _msg = TIMEMORY_JOIN("...", _msg.substr(0, max_msg_len - 13),
                                         _msg.substr(_msg.length() - 10));
                TIMEMORY_PRINT_HERE("Warning! Exception in "
                                    "operation::finalize::upc_get<%s>::recv_serialize: "
                                    "%s\n\t%s",
                                    demangle<Type>().c_str(), e.what(), _msg.c_str());
            }
            if(settings::debug())
                printf("[RECV: %i]> data size: %lli\n", comm_rank,
                       (long long int) ret.size());
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Function executed on remote node
    //
    auto remote_serialize = [=]() {
        basic_tree_type ret;
        return send_serialize(storage_type::master_instance()->get(ret));
    };

    bt       = basic_tree_vector_type(comm_size);
    auto ret = basic_tree_type{};

    if(comm_rank == 0)
    {
        //
        //  The root rank receives data from all non-root ranks and reports all data
        //
        for(int i = 1; i < comm_size; ++i)
        {
            upc::future_t<std::string> fut = upc::rpc(i, remote_serialize);
            while(!fut.ready())
                upc::progress();
            fut.wait();
            bt[i] = recv_serialize(fut.result());
        }
        bt[comm_rank] = data.get(ret);
    }

    upc::barrier(upc::world());

    if(comm_rank != 0)
        bt = basic_tree_vector_type(1, data.get(ret));

#endif
    return bt;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
upc_get<Type, true>::operator()(Archive& ar)
{
    if(!m_storage)
        return ar;

    if(!upc::is_initialized())
    {
        get_type{ m_storage }(ar);
    }
    else
    {
        auto idstr = get_type::get_identifier();
        ar.setNextName(idstr.c_str());
        ar.startNode();
        get_type{}(ar, metadata_t{});
        auto bt = basic_tree_vector_type{};
        (*this)(bt);
        ar(cereal::make_nvp("upcxx", bt));
        ar.finishNode();
    }
    return ar;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
