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
 * \file timemory/operations/types/finalize/mpi_get.hpp
 * \brief Definition for various functions for finalizing MPI data
 */

#pragma once

#include "timemory/backends/mpi.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/finalize/get.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/tpls/cereal/cereal/archives/binary.hpp"
#include "timemory/tpls/cereal/cereal/archives/json.hpp"
#include "timemory/tpls/cereal/cereal/details/helpers.hpp"
#include "timemory/utility/demangle.hpp"

#include <type_traits>

namespace tim
{
namespace concepts
{
template <typename Tp>
struct has_member_value_type
{
private:
    // clang-format off
    struct A { char a[1]; };
    struct B { char b[2]; };
    // clang-format on

    template <typename Up>
    static A query(typename Up::value_type*);

    template <typename Up>
    static B query(...);

public:
    static constexpr bool value = sizeof(query<Tp>(nullptr)) == sizeof(A);
};

template <typename Tp, bool>
struct value_type_impl;

template <typename Tp>
struct value_type_impl<Tp, true>
{
    static constexpr bool value = true;
    using type                  = typename Tp::value_type;
};

template <typename Tp>
struct value_type_impl<Tp, false>
{
    static constexpr bool value = false;
    using type                  = Tp;
};

template <typename Tp>
struct value_type
{
    static constexpr bool value = has_member_value_type<Tp>::value;
    using type                  = typename value_type_impl<Tp, value>::type;
};
}  // namespace concepts
//
namespace operation
{
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct mpi_get<Type, true>
{
    static constexpr bool value  = true;
    using this_type              = mpi_get<Type, value>;
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

    static auto& plus(Type& lhs, const Type& rhs) { return (lhs += rhs); }

    explicit TIMEMORY_COLD mpi_get(bool    _collapse   = settings::collapse_processes(),
                                   int32_t _node_count = settings::node_count())
    : m_collapse{ _collapse }
    , m_node_count{ _node_count }
    {}

    explicit TIMEMORY_COLD mpi_get(storage_type& _storage,
                                   bool    _collapse   = settings::collapse_processes(),
                                   int32_t _node_count = settings::node_count())
    : m_collapse{ _collapse }
    , m_node_count{ _node_count }
    , m_storage{ &_storage }
    {}

    // this serializes a type (src) and adds it to dst, if !collapse_processes
    // then it uses the adder to combine the data
    TIMEMORY_COLD
    mpi_get(std::vector<Type>& dst, const Type& src,
            std::function<Type&(Type& lhs, const Type& rhs)>&& adder = this_type::plus)
    {
        (*this)(dst, src, std::move(adder));
    }

    TIMEMORY_COLD distrib_type& operator()(distrib_type&);
    TIMEMORY_COLD basic_tree_vector_type& operator()(basic_tree_vector_type&);

    template <typename Archive>
    TIMEMORY_COLD enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
                  operator()(Archive&);

    TIMEMORY_COLD
    auto operator()(
        std::vector<Type>& dst, const Type& src,
        std::function<Type&(Type& lhs, const Type& rhs)>&& adder = this_type::plus) const;

    this_type& set_comm(mpi::comm_t _comm) { return (m_comm = _comm, *this); }
    this_type& set_debug(bool _v) { return (m_debug = _v, *this); }
    this_type& set_verbose(int _v) { return (m_verbose = _v, *this); }

private:
    bool          m_debug      = settings::debug();
    bool          m_collapse   = settings::collapse_processes();
    int           m_verbose    = settings::verbose();
    int32_t       m_node_count = settings::node_count();
    mpi::comm_t   m_comm       = mpi::comm_world_v;  // NOLINT
    storage_type* m_storage    = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct mpi_get<Type, false>
{
    static constexpr bool value = false;
    using this_type             = mpi_get<Type, false>;
    using storage_type          = impl::storage<Type, value>;

    explicit mpi_get(bool = false, int32_t = 0) {}
    explicit mpi_get(storage_type&, bool = false, int32_t = 0) {}

    template <typename Tp, typename... Args>
    Tp& operator()(Tp&, Args&&...)
    {}

    this_type& set_comm(mpi::comm_t) { return *this; }
    this_type& set_debug(bool) { return *this; }
    this_type& set_verbose(int) { return *this; }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename mpi_get<Type, true>::distrib_type&
mpi_get<Type, true>::operator()(distrib_type& results)
{
    if(!m_storage)
    {
#if defined(TIMEMORY_USE_MPI)
        int comm_size = mpi::size(m_comm);
        if(comm_size > 0 || (m_debug || m_verbose > 1))
        {
            int comm_rank = mpi::rank(m_comm);
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i] Warning! No storage instance on rank %i of %i",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, comm_size);
        }
#endif
        return results;
    }

    auto& data = *m_storage;
#if !defined(TIMEMORY_USE_MPI)
    if(m_debug)
        TIMEMORY_PRINT_HERE("%s", "timemory not using MPI");

    results = distrib_type{};
    results.emplace_back(std::move(data.get()));
#else
    if(m_debug)
        TIMEMORY_PRINT_HERE("%s", "timemory using MPI");

    mpi::barrier(m_comm);

    int comm_rank = mpi::rank(m_comm);
    int comm_size = mpi::size(m_comm);

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [&](const result_type& src) {
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
                constexpr size_t max_msg_len = 120;
                if(_msg.length() > max_msg_len)
                    _msg = TIMEMORY_JOIN("...", _msg.substr(0, max_msg_len - 23),
                                         _msg.substr(_msg.length() - 20));
                TIMEMORY_PRINT_HERE("Warning! Exception in "
                                    "operation::finalize::mpi_get<%s>::recv_serialize: "
                                    "%s\n\t%s",
                                    demangle<Type>().c_str(), e.what(), _msg.c_str());
            }
            if(m_debug && m_verbose >= 3)
            {
                TIMEMORY_PRINTF(stderr, "[RECV: %i] data size: %lli\n", comm_rank,
                                (long long int) ret.size());
            }
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Calculate the total number of measurement records
    //
    auto get_num_records = [&](const auto& _inp) {
        int _sz = 0;
        for(const auto& itr : _inp)
            _sz += itr.size();
        return _sz;
    };

    results = distrib_type(comm_size);

    auto ret     = data.get();
    auto str_ret = send_serialize(ret);

    if(comm_rank == 0)
    {
        //
        //  The root rank receives data from all non-root ranks and reports all data
        //
        for(int i = 1; i < comm_size; ++i)
        {
            std::string str;
            if(m_debug && m_verbose >= 3)
                TIMEMORY_PRINTF(stderr, "[RECV: %i] starting %i\n", comm_rank, i);
            mpi::recv(str, i, 0, m_comm);
            if(m_debug && m_verbose >= 3)
                TIMEMORY_PRINTF(stderr, "[RECV: %i] completed %i\n", comm_rank, i);
            results[i] = recv_serialize(str);
        }
        results[comm_rank] = std::move(ret);
    }
    else
    {
        //
        //  The non-root rank sends its data to the root rank and only reports own data
        //
        if(m_debug && m_verbose >= 3)
            TIMEMORY_PRINTF(stderr, "[SEND: %i] starting\n", comm_rank);
        mpi::send(str_ret, 0, 0, m_comm);
        if(m_debug && m_verbose >= 3)
            TIMEMORY_PRINTF(stderr, "[SEND: %i] completed\n", comm_rank);
        results = distrib_type{};
        results.emplace_back(std::move(ret));
    }

    // collapse into a single result
    if(comm_rank == 0 && m_collapse && m_node_count <= 1)
    {
        auto init_size = get_num_records(results);
        if(m_debug || m_verbose > 3)
        {
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i] collapsing %i records from %i ranks",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, comm_size);
        }

        auto _collapsed = distrib_type{};
        // so we can pop off back
        std::reverse(results.begin(), results.end());
        while(!results.empty())
        {
            if(_collapsed.empty())
            {
                _collapsed.emplace_back(std::move(results.back()));
            }
            else
            {
                operation::finalize::merge<Type, true>(_collapsed.front(),
                                                       results.back());
            }
            results.pop_back();
        }

        // assign results to collapsed entry
        results = std::move(_collapsed);

        if(m_debug || m_verbose > 3)
        {
            auto fini_size = get_num_records(results);
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i] collapsed %i records into %i records "
                "from %i ranks",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, fini_size, comm_size);
        }
    }
    else if(comm_rank == 0 && m_collapse && m_node_count > 1)
    {
        // calculate some size parameters
        int32_t nmod  = comm_size % m_node_count;
        int32_t bsize = comm_size / m_node_count + ((nmod == 0) ? 0 : 1);
        int32_t bins  = comm_size / bsize;

        if(m_debug || m_verbose > 3)
        {
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i] node_count = %i, comm_size = %i, bins = "
                "%i, bin size = %i",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, m_node_count, comm_size, bins, bsize);
        }

        // generate a map of the ranks to the node ids
        int32_t                              ncnt = 0;  // current count
        int32_t                              midx = 0;  // current bin map index
        std::map<int32_t, std::set<int32_t>> binmap;
        for(int32_t i = 0; i < comm_size; ++i)
        {
            if(m_debug)
            {
                TIMEMORY_PRINT_HERE("[%s][pid=%i][rank=%i] adding rank %i to bin %i",
                                    demangle<mpi_get<Type, true>>().c_str(),
                                    (int) process::get_id(), comm_rank, i, midx);
            }

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
        if(m_debug || m_verbose > 3)
        {
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i] collapsing %i records from %i ranks into "
                "%i bins",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, comm_size, (int) binmap.size());
        }

        assert((int32_t) binmap.size() <= (int32_t) m_node_count);

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

        if(m_debug || m_verbose > 3)
        {
            auto fini_size = get_num_records(results);
            TIMEMORY_PRINT_HERE(
                "[%s][pid=%i][rank=%i] collapsed %i records into %i records "
                "and %i bins",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, fini_size, (int) results.size());
        }
    }

    if(m_debug || m_verbose > 1)
    {
        auto ret_size = get_num_records(results);
        TIMEMORY_PRINT_HERE("[%s][pid=%i] %i total records on rank %i of %i",
                            demangle<mpi_get<Type, true>>().c_str(),
                            (int) process::get_id(), ret_size, comm_rank, comm_size);
    }

#endif

    return results;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename mpi_get<Type, true>::basic_tree_vector_type&
mpi_get<Type, true>::operator()(basic_tree_vector_type& bt)
{
    if(!m_storage)
        return bt;

    auto& data = *m_storage;

    using serialization_t = serialization<Type>;
    using mpi_data_t      = typename serialization_t::mpi_data;

    basic_tree_type _entry{};
    bt = serialization_t{}(mpi_data_t{}, mpi::comm_world_v, data.get(_entry));
    return bt;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
mpi_get<Type, true>::operator()(Archive& ar)
{
    if(!m_storage)
        return ar;

    if(!mpi::is_initialized())
    {
        get_type{ m_storage }(ar);
    }
    else
    {
        auto bt = basic_tree_vector_type{};
        (*this)(bt);
        serialization<Type>{}(ar, bt);
    }
    return ar;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
auto
mpi_get<Type, true>::operator()(
    std::vector<Type>& dst, const Type& inp,
    std::function<Type&(Type& lhs, const Type& rhs)>&& functor) const
{
#if !defined(TIMEMORY_USE_MPI)
    if(m_debug)
        TIMEMORY_PRINT_HERE("%s", "timemory not using MPI");
    consume_parameters(inp, functor);
#else
    // if Type or the underlying type is a vector of an arithmetic type, then
    // use the BinaryArchive types to avoid unnecessarily long serialization string
    using underlying_type = typename concepts::value_type<Type>::type;
    using inp_archive_t =
        std::conditional_t<concepts::is_vector<Type>::value &&
                               std::is_arithmetic<underlying_type>::value,
                           cereal::BinaryInputArchive, cereal::JSONInputArchive>;
    using out_archive_t =
        std::conditional_t<concepts::is_vector<Type>::value &&
                               std::is_arithmetic<underlying_type>::value,
                           cereal::BinaryOutputArchive, cereal::MinimalJSONOutputArchive>;

    TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "%s", "timemory using MPI");

    mpi::barrier(m_comm);

    int comm_rank = mpi::rank(m_comm);
    int comm_size = mpi::size(m_comm);

    TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "timemory using MPI [rank: %i, size: %i]",
                                    comm_rank, comm_size);

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [&](const Type& src) {
        std::stringstream ss;
        {
            auto oa = policy::output_archive<out_archive_t, TIMEMORY_API>::get(ss);
            (*oa)(cereal::make_nvp("data", src));
        }
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "sent data [rank: %i] :: %lu", comm_rank,
                                        ss.str().length());
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [&](const std::string& src) {
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "recv data [rank: %i] :: %lu", comm_rank,
                                        src.length());
        Type              ret{};
        std::stringstream ss{};
        ss << src;
        {
            auto ia = policy::input_archive<inp_archive_t, TIMEMORY_API>::get(ss);
            try
            {
                (*ia)(cereal::make_nvp("data", ret));
            } catch(cereal::Exception& e)
            {
                std::string _msg = ss.str();
                // truncate
                constexpr size_t max_msg_len = 120;
                if(_msg.length() > max_msg_len)
                    _msg = TIMEMORY_JOIN("...", _msg.substr(0, max_msg_len - 23),
                                         _msg.substr(_msg.length() - 20));
                TIMEMORY_PRINT_HERE("Warning! Exception in "
                                    "operation::finalize::mpi_get<%s>::recv_serialize: "
                                    "%s\n\t%s",
                                    demangle<Type>().c_str(), e.what(), _msg.c_str());
            }
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Calculate the total number of measurement records
    //
    auto get_num_records = [&](const auto& _inp) { return _inp.size(); };

    dst.resize(comm_size);
    auto str_ret = send_serialize(inp);

    mpi::barrier(m_comm);

    if(comm_rank == 0)
    {
        //
        //  The root rank receives data from all non-root ranks and reports all data
        //
        for(int i = 1; i < comm_size; ++i)
        {
            std::string str;
            TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "[RECV: %i] starting %i", comm_rank,
                                            i);
            mpi::recv(str, i, 0, m_comm);
            TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "[RECV: %i] completed %i", comm_rank,
                                            i);
            dst.at(i) = recv_serialize(str);
        }
        dst.at(0) = inp;
    }
    else
    {
        //
        //  The non-root rank sends its data to the root rank
        //
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "[SEND: %i] starting", comm_rank);
        mpi::send(str_ret, 0, 0, m_comm);
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_debug, "[SEND: %i] completed", comm_rank);
        dst.clear();
    }

    mpi::barrier(m_comm);

    // collapse into a single result
    if(m_collapse && comm_rank == 0)
    {
        auto init_size = get_num_records(dst);
        TIMEMORY_CONDITIONAL_PRINT_HERE(
            m_debug || m_verbose > 3,
            "[%s][pid=%i][rank=%i] collapsing %i records from %i ranks",
            demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
            (int) comm_rank, (int) init_size, (int) comm_size);

        auto _dst = std::vector<Type>{};
        for(auto& itr : dst)
        {
            if(_dst.empty())
            {
                _dst.emplace_back(std::move(itr));
            }
            else
            {
                _dst.front() = functor(_dst.front(), itr);
            }
        }

        // assign dst to collapsed entry
        dst = _dst;

        TIMEMORY_CONDITIONAL_PRINT_HERE(
            m_debug || m_verbose > 3,
            "[%s][pid=%i][rank=%i] collapsed %i records into %i records "
            "from %i ranks",
            demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
            (int) comm_rank, (int) init_size, (int) get_num_records(dst),
            (int) comm_size);
    }
    else if(m_node_count > 0 && comm_rank == 0)
    {
        // calculate some size parameters
        int32_t nmod  = comm_size % m_node_count;
        int32_t bsize = comm_size / m_node_count + ((nmod == 0) ? 0 : 1);
        int32_t bins  = comm_size / bsize;

        TIMEMORY_CONDITIONAL_PRINT_HERE(
            m_debug || m_verbose > 3,
            "[%s][pid=%i][rank=%i] node_count = %i, comm_size = %i, bins = "
            "%i, bin size = %i",
            demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(), comm_rank,
            m_node_count, comm_size, bins, bsize);

        // generate a map of the ranks to the node ids
        int32_t                              ncnt = 0;  // current count
        int32_t                              midx = 0;  // current bin map index
        std::map<int32_t, std::set<int32_t>> binmap;
        for(int32_t i = 0; i < comm_size; ++i)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(
                m_debug, "[%s][pid=%i][rank=%i] adding rank %i to bin %i",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, i, midx);

            binmap[midx].insert(i);
            // check to see if we reached the bin size
            if(++ncnt == bsize)
            {
                // set counter to zero and advance the node
                ncnt = 0;
                ++midx;
            }
        }

        auto init_size = get_num_records(dst);
        TIMEMORY_CONDITIONAL_PRINT_HERE(
            m_debug || m_verbose > 3,
            "[%s][pid=%i][rank=%i] collapsing %i records from %i ranks into %i bins",
            demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
            (int) comm_rank, (int) init_size, (int) comm_size, (int) binmap.size());

        assert((int32_t) binmap.size() <= (int32_t) m_node_count);

        // the collapsed data
        auto _dst = std::vector<Type>(binmap.size());
        // loop over the node indexes
        for(const auto& itr : binmap)
        {
            // target the node index
            auto& _targ = _dst.at(itr.first);
            for(const auto& bitr : itr.second)
            {
                // combine the node index entry with all of the ranks in that node
                auto& _src = dst.at(bitr);
                _targ      = functor(_targ, _src);
            }
        }

        // assign dst to collapsed entry
        dst = _dst;

        TIMEMORY_CONDITIONAL_PRINT_HERE(
            m_debug || m_verbose > 3,
            "[%s][pid=%i][rank=%i] collapsed %i records into %i records "
            "and %i bins",
            demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
            (int) comm_rank, (int) init_size, (int) get_num_records(dst),
            (int) dst.size());
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(
        m_debug || m_verbose > 1, "[%s][pid=%i] %i total records on rank %i of %i",
        demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
        (int) get_num_records(dst), (int) comm_rank, (int) comm_size);
#endif
    return dst;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
