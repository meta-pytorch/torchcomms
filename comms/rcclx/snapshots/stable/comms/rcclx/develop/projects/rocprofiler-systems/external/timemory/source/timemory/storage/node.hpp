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

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/data/statistics.hpp"
#include "timemory/hash.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

//--------------------------------------------------------------------------------------//
//
namespace tim
{
//
namespace operation
{
template <typename Tp>
struct dummy;
}
//
namespace node
{
#if defined(TIMEMORY_WINDOWS)
using pid_t = process::id_t;
#endif

using tid_t    = int64_t;
using tidset_t = std::set<tid_t>;
using pidset_t = std::set<pid_t>;
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::entry
/// \tparam Tp Component type
/// \tparam StatT Statistics type
///
/// \brief This data type is used in \ref tim::node::tree for inclusive and exclusive
/// values.
template <typename Tp, typename StatT>
struct entry : std::tuple<Tp, StatT>
{
    using base_type = std::tuple<Tp, StatT>;
    using this_type = entry;

    TIMEMORY_DEFAULT_OBJECT(entry)

    template <typename... Args>
    explicit entry(Args&&... args)
    : base_type(std::forward<Args>(args)...)
    {}
    explicit entry(const base_type& _obj)
    : base_type(_obj)
    {}
    explicit entry(base_type&& _obj) noexcept
    : base_type(std::forward<base_type>(_obj))
    {}

    /// component object with either inclusive or exclusive values
    Tp& data() { return std::get<0>(*this); }

    /// statistics data with either inclusive or exclusive values
    StatT& stats() { return std::get<1>(*this); }

    /// component object with either inclusive or exclusive values
    const Tp& data() const { return std::get<0>(*this); }

    /// statistics data with either inclusive or exclusive values
    const StatT& stats() const { return std::get<1>(*this); }

    this_type& operator+=(const this_type& rhs)
    {
        data() += rhs.data();
        stats() += rhs.stats();
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        data() -= rhs.data();
        stats() -= rhs.stats();
        return *this;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct data
{
    using string_t     = std::string;
    using strvector_t  = std::vector<string_t>;
    using uintvector_t = std::vector<hash_value_t>;

    using type         = typename trait::statistics<Tp>::type;
    using stats_policy = policy::record_statistics<Tp, type>;
    using stats_type   = typename stats_policy::statistics_type;
    using node_type =
        std::tuple<bool, tid_t, pid_t, hash_value_t, int64_t, Tp, stats_type>;
    using result_type = std::tuple<tid_t, pid_t, int64_t, hash_value_t, hash_value_t,
                                   string_t, uintvector_t, Tp, stats_type>;
    using entry_type  = entry<Tp, stats_type>;
    using tree_type   = std::tuple<bool, hash_value_t, int64_t, tidset_t, pidset_t,
                                 entry_type, entry_type>;
};
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::graph
/// \tparam Tp Component type
///
/// \brief This is the compact representation of a measurement in the call-graph.
template <typename Tp>
struct graph : private data<Tp>::node_type
{
    using this_type       = graph;
    using base_type       = typename data<Tp>::node_type;
    using data_value_type = typename Tp::value_type;
    using data_base_type  = typename Tp::base_type;
    using stats_type      = typename data<Tp>::stats_type;
    using string_t        = std::string;

public:
    // ctor, dtor
    graph();
    graph(hash_value_t _id, const Tp& _obj, int64_t _depth, tid_t _tid,
          pid_t _pid = process::get_id(), bool _is_dummy = false);

    ~graph()                = default;
    graph(const graph&)     = default;
    graph(graph&&) noexcept = default;

    graph& operator=(const graph&) = default;
    graph& operator=(graph&&) noexcept = default;

public:
    static Tp  get_dummy();
    bool       operator==(const graph& rhs) const;
    bool       operator!=(const graph& rhs) const;
    this_type& operator+=(const this_type& rhs)
    {
        obj() += rhs.obj();
        stats() += rhs.stats();
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        obj() -= rhs.obj();
        stats() -= rhs.stats();
        return *this;
    }

public:
    /// denotes this is a placeholder for synchronization
    bool& is_dummy() { return std::get<0>(*this); }

    /// thread identifier
    tid_t& tid() { return std::get<1>(*this); }

    /// process identifier
    pid_t& pid() { return std::get<2>(*this); }

    /// hash identifer
    hash_value_t& hash() { return std::get<3>(*this); }

    /// depth in call-graph
    int64_t& depth() { return std::get<4>(*this); }

    /// this is the instance that gets updated in call-graph
    Tp& data() { return std::get<5>(*this); }

    /// statistics data for entry in call-graph
    stats_type& stats() { return std::get<6>(*this); }

    bool              is_dummy() const { return std::get<0>(*this); }
    tid_t             tid() const { return std::get<1>(*this); }
    pid_t             pid() const { return std::get<2>(*this); }
    hash_value_t      hash() const { return std::get<3>(*this); }
    int64_t           depth() const { return std::get<4>(*this); }
    const Tp&         data() const { return std::get<5>(*this); }
    const stats_type& stats() const { return std::get<6>(*this); }

    // backwards compatibility
    auto&       id() { return this->hash(); }
    auto        id() const { return this->hash(); }
    auto&       obj() { return this->data(); }
    const auto& obj() const { return this->data(); }

    hash_value_t uniq_hash() const { return get_hash_id(id(), tid(), depth()); }

    std::string as_string() const
    {
        std::stringstream _ss{};
        _ss << std::boolalpha << "is_dummy=" << is_dummy() << ", tid=" << tid()
            << ", pid=" << pid() << ", hash=" << hash() << ", depth=" << depth()
            << ", data=" << data() << ", stats=" << stats();
        return _ss.str();
    }

    friend std::ostream& operator<<(std::ostream& _os, const this_type& _v)
    {
        return _os << _v.as_string();
    }
};
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::result
/// \tparam Tp Component type
///
/// \brief This data type is used when rendering the flat representation (i.e.
/// loop-iterable) representation of the calling-context. The prefix here will be
/// identical to the prefix in the text output.
template <typename Tp>
struct result : public data<Tp>::result_type
{
    using uintvector_t = std::vector<hash_value_t>;
    using base_type    = typename data<Tp>::result_type;
    using stats_type   = typename data<Tp>::stats_type;
    using this_type    = result<Tp>;

    result()                  = default;
    ~result()                 = default;
    result(const result&)     = default;
    result(result&&) noexcept = default;
    result& operator=(const result&) = default;
    result& operator=(result&&) noexcept = default;

    result(base_type&& _base) noexcept
    : base_type(std::forward<base_type>(_base))
    {}

    result(hash_value_t _hash, const Tp& _data, const string_t& _prefix, int64_t _depth,
           hash_value_t _rolling, const uintvector_t& _hierarchy,
           const stats_type& _stats, tid_t _tid, pid_t _pid);

    /// measurement thread. May be `std::numeric_limits<uint16_t>::max()` (i.e. 65536) if
    /// this entry is a combination of multiple threads
    tid_t& tid() { return std::get<0>(*this); }

    /// the process identifier of the reporting process, if multiple process data is
    /// combined, or the process identifier of the collecting process
    pid_t& pid() { return std::get<1>(*this); }

    /// depth of the node in the calling-context
    int64_t& depth() { return std::get<2>(*this); }

    /// hash identifer of the node
    hash_value_t& hash() { return std::get<3>(*this); }

    /// the summation of this hash and it's parent hashes
    hash_value_t& rolling_hash() { return std::get<4>(*this); }

    /// the associated string with the hash + indentation and other decoration
    string_t& prefix() { return std::get<5>(*this); }

    /// an array of the hash value + each parent hash (not serialized)
    uintvector_t& hierarchy() { return std::get<6>(*this); }

    /// reference to the component
    Tp& data() { return std::get<7>(*this); }

    /// reference to the associate statistical accumulation of the data (if any)
    stats_type& stats() { return std::get<8>(*this); }

    tid_t               tid() const { return std::get<0>(*this); }
    pid_t               pid() const { return std::get<1>(*this); }
    int64_t             depth() const { return std::get<2>(*this); }
    hash_value_t        hash() const { return std::get<3>(*this); }
    hash_value_t        rolling_hash() const { return std::get<4>(*this); }
    const string_t&     prefix() const { return std::get<5>(*this); }
    const uintvector_t& hierarchy() const { return std::get<6>(*this); }
    const Tp&           data() const { return std::get<7>(*this); }
    const stats_type&   stats() const { return std::get<8>(*this); }

    /// alias for `hash()`
    hash_value_t& id() { return this->hash(); }
    hash_value_t  id() const { return this->hash(); }

    /// alias for `data()`
    Tp&       obj() { return this->data(); }
    const Tp& obj() const { return this->data(); }

    bool operator==(const this_type& rhs) const
    {
        return (depth() == rhs.depth() && hash() == rhs.hash() &&
                rolling_hash() == rhs.rolling_hash() && prefix() == rhs.prefix());
    }

    bool operator!=(const this_type& rhs) const { return !(*this == rhs); }

    this_type& operator+=(const this_type& rhs)
    {
        data() += rhs.data();
        stats() += rhs.stats();
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        data() -= rhs.data();
        stats() -= rhs.stats();
        return *this;
    }
};
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::tree
/// \tparam Tp Generally `tim::basic_tree<ComponentT>`
///
/// \brief This data type is used when rendering the hierarchical representation (i.e.
/// requires recursion) representation of the calling-context. The prefix here has no
/// decoration.
template <typename Tp>
struct tree : private data<Tp>::tree_type
{
    using this_type       = tree;
    using base_type       = typename data<Tp>::tree_type;
    using data_value_type = typename Tp::value_type;
    using data_base_type  = typename Tp::base_type;
    using stats_type      = typename data<Tp>::stats_type;
    using entry_type      = typename data<Tp>::entry_type;
    using string_t        = std::string;

public:
    // ctor, dtor
    tree();
    explicit tree(base_type&& _base) noexcept;

    tree(const graph<Tp>&);
    tree& operator=(const graph<Tp>&);

    ~tree()               = default;
    tree(const tree&)     = default;
    tree(tree&&) noexcept = default;
    tree(bool _is_dummy, tid_t _tid, pid_t _pid, hash_value_t _hash, int64_t _depth,
         const Tp& _obj);

public:
    static Tp  get_dummy();
    bool       operator==(const tree& rhs) const;
    bool       operator!=(const tree& rhs) const;
    tree&      operator=(const tree&) = default;
    tree&      operator=(tree&&) noexcept = default;
    this_type& operator+=(const this_type& rhs)
    {
        inclusive() += rhs.inclusive();
        exclusive() += rhs.exclusive();
        for(const auto& itr : rhs.tid())
            tid().insert(itr);
        for(const auto& itr : rhs.pid())
            pid().insert(itr);
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        inclusive() -= rhs.inclusive();
        exclusive() -= rhs.exclusive();
        return *this;
    }

public:
    /// returns whether or not this node is a synchronization point and, if so, should be
    /// ignored
    bool& is_dummy() { return std::get<0>(*this); }

    /// returns the hash identifier for the associated string identifier
    hash_value_t& hash() { return std::get<1>(*this); }

    /// returns the depth of the node in the tree. NOTE: this value may be relative to
    /// dummy nodes
    int64_t& depth() { return std::get<2>(*this); }

    /// the set of thread ids this data was collected from
    tidset_t& tid() { return std::get<3>(*this); }

    /// the set of process ids this data was collected from
    pidset_t& pid() { return std::get<4>(*this); }

    /// the inclusive data + statistics
    entry_type& inclusive() { return std::get<5>(*this); }

    /// the exclusive data + statistics
    entry_type& exclusive() { return std::get<6>(*this); }

    bool              is_dummy() const { return std::get<0>(*this); }
    hash_value_t      hash() const { return std::get<1>(*this); }
    int64_t           depth() const { return std::get<2>(*this); }
    const tidset_t&   tid() const { return std::get<3>(*this); }
    const pidset_t&   pid() const { return std::get<4>(*this); }
    const entry_type& inclusive() const { return std::get<5>(*this); }
    const entry_type& exclusive() const { return std::get<6>(*this); }
};
//
//--------------------------------------------------------------------------------------//
//
//                              Definitions
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
graph<Tp>::graph()
: base_type(false, threading::get_id(), process::get_id(), 0, 0, Tp{}, stats_type{})
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
graph<Tp>::graph(hash_value_t _id, const Tp& _obj, int64_t _depth, int64_t _tid,
                 pid_t _pid, bool _is_dummy)
: base_type(_is_dummy, _tid, _pid, _id, _depth, _obj, stats_type{})
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
bool
graph<Tp>::operator==(const graph& rhs) const
{
    return (hash() == rhs.hash() && depth() == rhs.depth() && tid() == rhs.tid());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
bool
graph<Tp>::operator!=(const graph& rhs) const
{
    return !(*this == rhs);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
graph<Tp>::get_dummy()
{
    return operation::dummy<Tp>{}();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
result<Tp>::result(hash_value_t _id, const Tp& _data, const string_t& _prefix,
                   int64_t _depth, hash_value_t _rolling, const uintvector_t& _hierarchy,
                   const stats_type& _stats, int64_t _tid, pid_t _pid)
: base_type(_tid, _pid, _depth, _id, _rolling, _prefix, _hierarchy, _data, _stats)
{}
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
tree<Tp>::tree()
: base_type(false, 0, 0, tidset_t{ threading::get_id() }, pidset_t{ process::get_id() },
            entry_type{}, entry_type{})
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
tree<Tp>::tree(const graph<Tp>& rhs)
: base_type(rhs.is_dummy(), rhs.hash(), rhs.depth(), tidset_t{ rhs.tid() },
            pidset_t{ rhs.pid() }, entry_type{ rhs.data(), rhs.stats() },
            entry_type{ rhs.data(), rhs.stats() })
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
tree<Tp>&
tree<Tp>::operator=(const graph<Tp>& rhs)
{
    is_dummy()  = rhs.is_dummy();
    hash()      = rhs.hash();
    depth()     = rhs.depth();
    tid()       = { rhs.tid() };
    pid()       = { rhs.pid() };
    inclusive() = entry_type{ rhs.data(), rhs.stats() };
    exclusive() = entry_type{ rhs.data(), rhs.stats() };
    return *this;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace node
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace cereal
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const tim::node::graph<Tp>& d)
{
    auto _lbl = operation::decode<TIMEMORY_API>{}(::tim::get_hash_identifier(d.hash()));
    ar(cereal::make_nvp("hash", d.hash()), cereal::make_nvp("prefix", _lbl),
       cereal::make_nvp("entry", d.obj()), cereal::make_nvp("depth", d.depth()),
       cereal::make_nvp("stats", d.stats()), cereal::make_nvp("tid", d.tid()),
       cereal::make_nvp("pid", d.pid()), cereal::make_nvp("dummy", d.is_dummy()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, tim::node::graph<Tp>& d)
{
    std::string _prefix{};
    ar(cereal::make_nvp("hash", d.hash()), cereal::make_nvp("prefix", _prefix),
       cereal::make_nvp("entry", d.obj()), cereal::make_nvp("depth", d.depth()),
       cereal::make_nvp("stats", d.stats()), cereal::make_nvp("tid", d.tid()),
       cereal::make_nvp("pid", d.pid()), cereal::make_nvp("dummy", d.is_dummy()));
    auto _hash = tim::add_hash_id(_prefix);
    if(_hash != d.hash())
        tim::add_hash_id(_hash, d.hash());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
struct specialize<Archive, tim::node::graph<Tp>,
                  cereal::specialization::non_member_load_save>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp, typename StatT>
void
save(Archive& ar, const tim::node::entry<Tp, StatT>& e)
{
    ar(cereal::make_nvp("entry", e.data()));
    ar(cereal::make_nvp("stats", e.stats()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp, typename StatT>
void
load(Archive& ar, tim::node::entry<Tp, StatT>& e)
{
    ar(cereal::make_nvp("entry", e.data()));
    ar(cereal::make_nvp("stats", e.stats()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp, typename StatT>
struct specialize<Archive, tim::node::entry<Tp, StatT>,
                  cereal::specialization::non_member_load_save>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const tim::node::tree<Tp>& t)
{
    auto _prefix =
        operation::decode<TIMEMORY_API>{}(::tim::get_hash_identifier(t.hash()));
    ar(cereal::make_nvp("hash", t.hash()), cereal::make_nvp("prefix", _prefix),
       cereal::make_nvp("tid", t.tid()), cereal::make_nvp("pid", t.pid()),
       cereal::make_nvp("depth", t.depth()), cereal::make_nvp("is_dummy", t.is_dummy()));
    ar(cereal::make_nvp("inclusive", t.inclusive()));
    ar(cereal::make_nvp("exclusive", t.exclusive()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, tim::node::tree<Tp>& t)
{
    std::string _prefix{};
    ar(cereal::make_nvp("hash", t.hash()), cereal::make_nvp("prefix", _prefix),
       cereal::make_nvp("tid", t.tid()), cereal::make_nvp("pid", t.pid()),
       cereal::make_nvp("depth", t.depth()), cereal::make_nvp("is_dummy", t.is_dummy()));
    ar(cereal::make_nvp("inclusive", t.inclusive()));
    ar(cereal::make_nvp("exclusive", t.exclusive()));
    auto _id = tim::add_hash_id(_prefix);
    if(_id != t.hash())
        tim::add_hash_id(_id, t.hash());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
struct specialize<Archive, tim::node::tree<Tp>,
                  cereal::specialization::non_member_load_save>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const tim::node::result<Tp>& r)
{
    ar(cereal::make_nvp("hash", r.hash()), cereal::make_nvp("prefix", r.prefix()),
       cereal::make_nvp("depth", r.depth()), cereal::make_nvp("entry", r.data()),
       cereal::make_nvp("stats", r.stats()),
       cereal::make_nvp("rolling_hash", r.rolling_hash()));
    // ar(cereal::make_nvp("hierarchy", r.hierarchy()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, tim::node::result<Tp>& r)
{
    ar(cereal::make_nvp("hash", r.hash()), cereal::make_nvp("prefix", r.prefix()),
       cereal::make_nvp("depth", r.depth()), cereal::make_nvp("entry", r.data()),
       cereal::make_nvp("stats", r.stats()),
       cereal::make_nvp("rolling_hash", r.rolling_hash()));
    // ar(cereal::make_nvp("hierarchy", r.hierarchy()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const std::vector<tim::node::result<Tp>>& result_nodes)
{
    ar(cereal::make_nvp("graph_size", result_nodes.size()));
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(const auto& itr : result_nodes)
    {
        ar.startNode();
        save(ar, itr);
        ar.finishNode();
    }
    ar.finishNode();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, std::vector<tim::node::result<Tp>>& result_nodes)
{
    size_t nnodes = 0;
    ar(cereal::make_nvp("graph_size", nnodes));
    result_nodes.resize(nnodes);

    ar.setNextName("graph");
    ar.startNode();
    for(auto& itr : result_nodes)
    {
        ar.startNode();
        load(ar, itr);
        ar.finishNode();
    }
    ar.finishNode();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
struct specialize<Archive, tim::node::result<Tp>,
                  cereal::specialization::non_member_load_save>
{};
//
}  // namespace cereal
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
