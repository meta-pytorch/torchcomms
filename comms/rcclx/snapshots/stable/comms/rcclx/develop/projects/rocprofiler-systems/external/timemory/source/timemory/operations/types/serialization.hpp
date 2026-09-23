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

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/manager/manager.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/storage/node.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/utility/demangle.hpp"

#include <cctype>
#include <map>
#include <vector>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
/// \struct tim::operation::extra_serialization
/// \brief Provides a hook to add additional serialization data for the type. Examples
/// include the roofline components adding roofline data. Note that this is data for the
/// component type, not data for a component entry in the call-graph.
///
template <typename Tp>
struct extra_serialization
{
    using PrettyJson_t  = cereal::PrettyJSONOutputArchive;
    using MinimalJson_t = cereal::MinimalJSONOutputArchive;

    TIMEMORY_DEFAULT_OBJECT(extra_serialization)

    explicit extra_serialization(PrettyJson_t& ar, unsigned int ver = 0)
    {
        (*this)(ar, ver);
    }

    explicit extra_serialization(MinimalJson_t& ar, unsigned int ver = 0)
    {
        (*this)(ar, ver);
    }

    template <typename Archive>
    explicit extra_serialization(Archive& ar, unsigned int ver = 0)
    {
        (*this)(ar, ver);
    }

    auto operator()(PrettyJson_t& ar, unsigned int ver = 0) const
    {
        sfinae(ar, ver, 0, 0);
    }

    auto operator()(MinimalJson_t& ar, unsigned int ver = 0) const
    {
        sfinae(ar, ver, 0, 0);
    }

    template <typename Archive>
    auto operator()(Archive& ar, unsigned int ver = 0) const
    {
        sfinae(ar, ver, 0, 0);
    }

private:
    template <typename Archive, typename Up = Tp>
    auto sfinae(Archive& ar, unsigned int ver, int, int) const
        -> decltype(Up::extra_serialization(ar, ver), void())
    {
        Up::extra_serialization(ar, ver);
    }

    template <typename Archive, typename Up = Tp>
    auto sfinae(Archive& ar, unsigned int, int, long) const
        -> decltype(Up::extra_serialization(ar), void())
    {
        Up::extra_serialization(ar);
    }

    template <typename Archive, typename Up = Tp>
    auto sfinae(Archive&, unsigned int, long, long) const
    {}
};
//
namespace internal
{
//
TIMEMORY_INLINE std::string
                to_lower(std::string _inp)
{
    for(auto& itr : _inp)
        itr = std::tolower(itr);
    return _inp;
}
//
template <typename Tp>
struct serialization_base
{
    using type = Tp;

    struct metadata
    {};

    struct mpi_data
    {};

    struct upcxx_data
    {};

    TIMEMORY_DEFAULT_OBJECT(serialization_base)

public:
    static std::string get_identifier(const type& _obj = type{})
    {
        std::string idstr = (component::properties<type>::specialized())
                                ? to_lower(component::properties<type>::enum_string())
                                : "";
        if(idstr.empty())
            idstr = get_identifier_sfinae(_obj, 0);
        if(idstr.empty())
            idstr = demangle<type>();
        return idstr;
    }

    static auto get_label(const type& _obj = type{})
    {
        return get_label_sfinae(_obj, 0, 0);
    }

    static auto get_description(const type& _obj = type{})
    {
        return get_description_sfinae(_obj, 0, 0);
    }

    static auto get_unit(const type& _obj = type{})
    {
        return get_unit_sfinae(_obj, 0, 0);
    }

    static auto get_display_unit(const type& _obj = type{})
    {
        return get_display_unit_sfinae(_obj, 0, 0);
    }

private:
    template <typename Up>
    static auto get_identifier_sfinae(const Up& _data, int) -> decltype(_data.label())
    {
        return _data.label();
    }

    template <typename Up>
    static auto get_identifier_sfinae(const Up&, long)
    {
        return std::string{};
    }

private:
    template <typename Up>
    static auto get_label_sfinae(const Up& _data, int, int)
        -> decltype(_data.label_array())
    {
        return _data.label_array();
    }

    template <typename Up>
    static auto get_label_sfinae(const Up& _data, int, long) -> decltype(_data.label())
    {
        return _data.label();
    }

    template <typename Up>
    static auto get_label_sfinae(const Up&, long, long)
    {
        return std::string{};
    }

private:
    template <typename Up>
    static auto get_description_sfinae(const Up& _data, int, int)
        -> decltype(_data.description_array())
    {
        return _data.description_array();
    }

    template <typename Up>
    static auto get_description_sfinae(const Up& _data, int, long)
        -> decltype(_data.description())
    {
        return _data.description();
    }

    template <typename Up>
    static auto get_description_sfinae(const Up&, long, long)
    {
        return std::string{};
    }

private:
    template <typename Up>
    static auto get_unit_sfinae(const Up& _data, int, int) -> decltype(_data.unit_array())
    {
        return _data.unit_array();
    }

    template <typename Up>
    static auto get_unit_sfinae(const Up& _data, int, long) -> decltype(_data.unit())

    {
        return _data.unit();
    }

    template <typename Up>
    static auto get_unit_sfinae(const Up&, long, long) -> int64_t
    {
        return 0;
    }

private:
    template <typename Up>
    static auto get_display_unit_sfinae(const Up& _data, int, int)
        -> decltype(_data.display_unit_array())
    {
        return _data.display_unit_array();
    }

    template <typename Up>
    static auto get_display_unit_sfinae(const Up& _data, int, long)
        -> decltype(_data.display_unit())

    {
        return _data.display_unit();
    }

    template <typename Up>
    static auto get_display_unit_sfinae(const Up&, long, long)
    {
        return std::string{};
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, bool AvailV = is_enabled<Tp>::value>
struct serialization;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct serialization<Tp, false> : public serialization_base<Tp>
{
    using type      = Tp;
    using base_type = serialization_base<Tp>;
    using metadata  = typename base_type::metadata;

    TIMEMORY_DEFAULT_OBJECT(serialization)

    template <typename... Args>
    serialization(Args&&...)
    {}

    template <typename... Args>
    auto operator()(Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct serialization<Tp, true> : public serialization_base<Tp>
{
    static constexpr bool value  = true;
    using type                   = Tp;
    using base_type              = serialization_base<Tp>;
    using tree_type              = node::tree<type>;
    using graph_node             = node::graph<type>;
    using result_node            = node::result<type>;
    using result_type            = std::vector<result_node>;
    using distrib_type           = std::vector<result_type>;
    using storage_type           = impl::storage<type, value>;
    using graph_type             = graph<graph_node>;
    using hierarchy_type         = std::vector<uint64_t>;
    using basic_tree_type        = basic_tree<node::tree<type>>;
    using basic_tree_vector_type = std::vector<basic_tree_type>;
    using metadata               = typename base_type::metadata;
    using mpi_data               = typename base_type::mpi_data;
    using basic_tree_map_type =
        std::map<std::string, std::vector<basic_tree_vector_type>>;

    using base_type::get_description;
    using base_type::get_display_unit;
    using base_type::get_identifier;
    using base_type::get_label;
    using base_type::get_unit;

    TIMEMORY_DEFAULT_OBJECT(serialization)

private:
    static constexpr bool component_is_specialized()
    {
        return component::properties<Tp>::specialized();
    }

    template <typename Archive>
    static constexpr bool is_specialized()
    {
        return is_one_of<
            decay_t<Archive>,
            type_list<cereal::JSONInputArchive, cereal::MinimalJSONOutputArchive,
                      cereal::PrettyJSONOutputArchive>>::value;
    }

    template <typename DataT>
    static constexpr bool is_supported_dtype()
    {
        return is_one_of<
            decay_t<DataT>,
            type_list<basic_tree_vector_type, std::vector<basic_tree_vector_type>,
                      result_type, distrib_type, basic_tree_map_type>>::value;
    }

public:
    // template overloads -- do not get instantiated in extern template
    template <typename Archive>
    TIMEMORY_COLD serialization(const Tp& obj, Archive& ar, const unsigned int version,
                                enable_if_t<!is_specialized<Archive>(), int> = 0)
    {
        static_assert(!is_specialized<Archive>(),
                      "Error! Calling template function instead of specialized function");
        impl(obj, ar, version);
    }

    template <typename Archive>
    TIMEMORY_COLD void operator()(const Tp& obj, Archive& ar, const unsigned int version,
                                  enable_if_t<!is_specialized<Archive>(), int> = 0) const
    {
        static_assert(!is_specialized<Archive>(),
                      "Error! Calling template function instead of specialized function");
        impl(obj, ar, version);
    }

    template <typename Archive>
    TIMEMORY_COLD void operator()(Archive& ar, metadata data,
                                  enable_if_t<!is_specialized<Archive>(), int> = 0) const
    {
        static_assert(!is_specialized<Archive>(),
                      "Error! Calling template function instead of specialized function");
        impl(ar, data);
    }

    template <typename Archive, typename DataT>
    TIMEMORY_COLD void operator()(
        Archive& ar, const DataT& data,
        enable_if_t<!is_specialized<Archive>() && is_supported_dtype<DataT>(), int> =
            0) const
    {
        static_assert(!is_specialized<Archive>(),
                      "Error! Calling template function instead of specialized function");
        impl(ar, data);
    }

    template <typename Archive, typename DataT>
    TIMEMORY_COLD void operator()(
        Archive& ar, DataT& data,
        enable_if_t<!is_specialized<Archive>() && is_supported_dtype<DataT>(), int> =
            0) const
    {
        static_assert(!is_specialized<Archive>(),
                      "Error! Calling template function instead of specialized function");
        static_assert(!(std::is_same<decay_t<DataT>, basic_tree_map_type>::value &&
                        concepts::is_input_archive<decay_t<Archive>>::value),
                      "Error! basic_tree_map_type does not support input serialization");
        impl(ar, data);
    }

public:
    // MinimalJSONOutputArchive overloads -- get instantiated in extern template
    TIMEMORY_COLD serialization(const Tp& obj, cereal::MinimalJSONOutputArchive& ar,
                                const unsigned int version);

    TIMEMORY_COLD void operator()(const Tp& obj, cereal::MinimalJSONOutputArchive& ar,
                                  const unsigned int version) const;

    TIMEMORY_COLD void operator()(cereal::MinimalJSONOutputArchive& ar, metadata) const;

    TIMEMORY_COLD void operator()(cereal::MinimalJSONOutputArchive& ar,
                                  const basic_tree_vector_type&     data) const;

    TIMEMORY_COLD void operator()(cereal::MinimalJSONOutputArchive&          ar,
                                  const std::vector<basic_tree_vector_type>& data) const;

    TIMEMORY_COLD void operator()(cereal::MinimalJSONOutputArchive& ar,
                                  const basic_tree_map_type&        data) const;

    TIMEMORY_COLD void operator()(cereal::MinimalJSONOutputArchive& ar,
                                  const result_type&                data) const;

    TIMEMORY_COLD void operator()(cereal::MinimalJSONOutputArchive& ar,
                                  const distrib_type&               data) const;

    TIMEMORY_COLD void operator()(cereal::MinimalJSONOutputArchive& ar,
                                  const distrib_type&               _dist,
                                  const basic_tree_map_type&        _tree) const;

public:
    // non-MinimalJSONOutputArchive overloads -- do not get instantiated in extern
    // template
    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD serialization(const Tp& obj, ArchiveT& ar, const unsigned int version);

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(const Tp& obj, ArchiveT& ar,
                                  const unsigned int version) const;

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(ArchiveT& ar, metadata) const;

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(ArchiveT& ar, const basic_tree_vector_type& data) const;

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(ArchiveT&                                  ar,
                                  const std::vector<basic_tree_vector_type>& data) const;

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(ArchiveT& ar, const basic_tree_map_type& data) const;

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(ArchiveT& ar, const result_type& data) const;

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(ArchiveT& ar, const distrib_type& data) const;

    template <
        typename ArchiveT,
        enable_if_t<concepts::is_output_archive<ArchiveT>::value &&
                        !std::is_same<ArchiveT, cereal::MinimalJSONOutputArchive>::value,
                    int> = 0>
    TIMEMORY_COLD void operator()(ArchiveT& ar, const distrib_type& _dist,
                                  const basic_tree_map_type& _tree) const;

public:
    // JSONInputArchive overloads -- get instantiated in extern template
    TIMEMORY_COLD void operator()(cereal::JSONInputArchive& ar,
                                  basic_tree_vector_type&   data) const;

    TIMEMORY_COLD void operator()(cereal::JSONInputArchive&            ar,
                                  std::vector<basic_tree_vector_type>& data) const;

    TIMEMORY_COLD void operator()(cereal::JSONInputArchive& ar, result_type& data) const;

    TIMEMORY_COLD void operator()(cereal::JSONInputArchive& ar, distrib_type& data) const;

public:
    template <typename ValueT>
    std::vector<decay_t<ValueT>> operator()(mpi_data, mpi::comm_t comm,
                                            const ValueT& entry,
                                            int32_t       comm_target = 0) const;

private:
    template <typename Archive>
    TIMEMORY_COLD void impl(const Tp& obj, Archive& ar, const unsigned int) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(Archive& ar, metadata) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, const basic_tree_vector_type& data,
        enable_if_t<concepts::is_output_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, const std::vector<basic_tree_vector_type>& data,
        enable_if_t<concepts::is_output_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, const basic_tree_map_type& data,
        enable_if_t<concepts::is_output_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, const result_type& data,
        enable_if_t<concepts::is_output_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, const distrib_type& data,
        enable_if_t<concepts::is_output_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, const distrib_type&, const basic_tree_map_type&,
        enable_if_t<concepts::is_output_archive<Archive>::value, int> = 0) const;

    // input
    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, basic_tree_vector_type& data,
        enable_if_t<concepts::is_input_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, std::vector<basic_tree_vector_type>& data,
        enable_if_t<concepts::is_input_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, result_type& data,
        enable_if_t<concepts::is_input_archive<Archive>::value, int> = 0) const;

    template <typename Archive>
    TIMEMORY_COLD void impl(
        Archive& ar, distrib_type& data,
        enable_if_t<concepts::is_input_archive<Archive>::value, long> = 0) const;
};
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(const Tp& obj, Archive& ar, const unsigned int) const
{
    auto try_catch = [&ar](const char* key, const auto& val) {
        try
        {
            ar(cereal::make_nvp(key, val));
        } catch(cereal::Exception& e)
        {
            fprintf(stderr, "Warning! '%s' threw exception: %s\n", key, e.what());
        }
    };

    try_catch("laps", obj.get_laps());
    try_catch("value", obj.get_value());
    IF_CONSTEXPR(trait::base_has_accum<Tp>::value)
    {
        try_catch("accum", obj.get_accum());
    }
    IF_CONSTEXPR(trait::base_has_last<Tp>::value) { try_catch("last", obj.get_last()); }
    try_catch("repr_data", obj.get());
    try_catch("repr_display", obj.get_display());
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(Archive& ar, metadata) const
{
    bool _thread_scope_only = trait::thread_scope_only<type>::value;
    auto _num_thr_count     = manager::get_thread_count();
    auto _num_pid_count     = dmp::size();

    ar(cereal::make_nvp("properties", component::properties<type>{}));
    ar(cereal::make_nvp("type", get_label()));
    ar(cereal::make_nvp("description", get_description()));
    ar(cereal::make_nvp("unit_value", get_unit()));
    ar(cereal::make_nvp("unit_repr", get_display_unit()));
    ar(cereal::make_nvp("thread_scope_only", _thread_scope_only));
    ar(cereal::make_nvp("thread_count", _num_thr_count));
    ar(cereal::make_nvp("mpi_size", mpi::size()));
    ar(cereal::make_nvp("upcxx_size", upc::size()));
    ar(cereal::make_nvp("process_count", _num_pid_count));
    ar(cereal::make_nvp("num_ranks", dmp::size()));       // backwards-compat
    ar(cereal::make_nvp("concurrency", _num_thr_count));  // backwards-compat
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, const basic_tree_vector_type& data,
    enable_if_t<concepts::is_output_archive<Archive>::value, int>) const
{
    auto idstr = get_identifier();
    ar.setNextName(idstr.c_str());
    ar.startNode();
    impl(ar, metadata{});
    extra_serialization<Tp>{ ar };
    ar(cereal::make_nvp("graph", data));
    ar.finishNode();
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, const std::vector<basic_tree_vector_type>& data,
    enable_if_t<concepts::is_output_archive<Archive>::value, int>) const
{
    auto idstr = get_identifier();
    ar.setNextName(idstr.c_str());
    ar.startNode();
    impl(ar, metadata{});
    extra_serialization<Tp>{ ar };
    ar(cereal::make_nvp("graph", data));
    ar.finishNode();
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, const basic_tree_map_type& data,
    enable_if_t<concepts::is_output_archive<Archive>::value, int>) const
{
    auto idstr = get_identifier();
    ar.setNextName(idstr.c_str());
    ar.startNode();
    impl(ar, metadata{});
    extra_serialization<Tp>{ ar };
    auto pitr = data.find("process");
    if(pitr != data.end())
    {
        ar(cereal::make_nvp("graph", pitr->second));
    }
    else
    {
        for(const auto& itr : data)
            ar(cereal::make_nvp(itr.first, itr.second));
    }
    ar.finishNode();
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, const result_type& data,
    enable_if_t<concepts::is_output_archive<Archive>::value, int>) const
{
    // node
    std::string _name = get_identifier();
    ar.setNextName(_name.c_str());
    ar.startNode();
    impl(ar, metadata{});
    extra_serialization<Tp>{ ar };
    cereal::save(ar, data);
    ar.finishNode();  // ranks
    ar.finishNode();  // name
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, const distrib_type& data,
    enable_if_t<concepts::is_output_archive<Archive>::value, int>) const
{
    // node
    std::string _name = get_identifier();
    ar.setNextName(_name.c_str());
    ar.startNode();
    impl(ar, metadata{});
    extra_serialization<Tp>{ ar };
    ar.setNextName("ranks");
    ar.startNode();
    ar.makeArray();
    for(uint64_t i = 0; i < data.size(); ++i)
    {
        if(data.at(i).empty())
            continue;

        ar.startNode();

        ar(cereal::make_nvp("rank", i));
        cereal::save(ar, data.at(i));

        ar.finishNode();
    }
    ar.finishNode();  // ranks
    ar.finishNode();  // name
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, const distrib_type& _dist, const basic_tree_map_type& _tree,
    enable_if_t<concepts::is_output_archive<Archive>::value, int>) const
{
    auto _name = get_identifier();
    ar.setNextName(_name.c_str());
    ar.startNode();
    {
        impl(ar, metadata{});
        extra_serialization<Tp>{ ar };
    }
    {
        ar.setNextName("ranks");
        ar.startNode();
        ar.makeArray();
        for(uint64_t i = 0; i < _dist.size(); ++i)
        {
            if(_dist.at(i).empty())
                continue;

            ar.startNode();

            ar(cereal::make_nvp("rank", i));
            cereal::save(ar, _dist.at(i));

            ar.finishNode();
        }
        ar.finishNode();  // ranks
    }
    {
        auto pitr = _tree.find("process");
        if(pitr != _tree.end())
        {
            ar(cereal::make_nvp("graph", pitr->second));
        }
        else
        {
            for(const auto& itr : _tree)
                ar(cereal::make_nvp(itr.first, itr.second));
        }
    }
    ar.finishNode();  // name
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, basic_tree_vector_type& data,
    enable_if_t<concepts::is_input_archive<Archive>::value, int>) const
{
    auto idstr = get_identifier();
    ar.setNextName(idstr.c_str());
    ar.startNode();
    ar(cereal::make_nvp("graph", data));
    ar.finishNode();
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, std::vector<basic_tree_vector_type>& data,
    enable_if_t<concepts::is_input_archive<Archive>::value, int>) const
{
    auto idstr = get_identifier();
    ar.setNextName(idstr.c_str());
    ar.startNode();
    std::stringstream _msg;
    for(const auto* name : { "graph", "mpi", "upcxx" })
    {
        try
        {
            std::vector<basic_tree_vector_type> _data{};
            ar(tim::cereal::make_nvp(name, _data));
            data.reserve(data.size() + _data.size());
            for(auto&& itr : _data)
                data.emplace_back(itr);
        } catch(tim::cereal::Exception& e)
        {
            _msg << e.what() << '\n';
        }
    }
    ar.finishNode();
    if(data.empty())
        throw std::runtime_error(_msg.str());
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, result_type& data,
    enable_if_t<concepts::is_input_archive<Archive>::value, int>) const
{
    // node
    std::string _name = get_identifier();
    ar.setNextName(_name.c_str());
    ar.startNode();
    cereal::load(ar, data);
    ar.finishNode();  // ranks
    ar.finishNode();  // name
}
//
template <typename Tp>
template <typename Archive>
void
serialization<Tp, true>::impl(
    Archive& ar, distrib_type& data,
    enable_if_t<concepts::is_input_archive<Archive>::value, long>) const
{
    // node
    std::string _name = get_identifier();
    ar.setNextName(_name.c_str());
    ar.startNode();

    // node
    cereal::size_type _nranks = 0;
    ar.setNextName("ranks");
    ar.startNode();
    ar.loadSize(_nranks);

    data.clear();
    data.resize(_nranks);
    for(uint64_t i = 0; i < data.size(); ++i)
    {
        ar.startNode();
        ar(cereal::make_nvp("rank", i));
        try
        {
            cereal::load(ar, data.at(i));
        } catch(std::exception& e)
        {
            fprintf(stderr, "%s\n", e.what());
        }
        ar.finishNode();
    }

    ar.finishNode();  // ranks
    ar.finishNode();  // name
}
//
template <typename Tp>
template <typename ValueT>
std::vector<decay_t<ValueT>>
serialization<Tp, true>::operator()(mpi_data, mpi::comm_t comm, const ValueT& entry,
                                    int32_t comm_target) const
{
    using data_type = std::vector<decay_t<ValueT>>;

#if !defined(TIMEMORY_USE_MPI)
    consume_parameters(comm, comm_target);
    return data_type(1, entry);
#else
    using value_type = ValueT;
    // not yet implemented
    // auto comm =
    //    (settings::mpi_output_per_node()) ? mpi::get_node_comm() :
    //    mpi::comm_world_v;
    mpi::barrier(comm);

    int comm_rank = mpi::rank(comm);
    int comm_size = mpi::size(comm);

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [&](const value_type& src) {
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
        value_type        ret{};
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
                TIMEMORY_PRINT_HERE(
                    "Warning! Exception in operation::serialization<%s>::recv_serialize: "
                    "%s\n\t%s",
                    demangle<Tp>().c_str(), e.what(), _msg.c_str());
            }
        }
        return ret;
    };

    if(comm_rank == comm_target)
    {
        auto _data = data_type(comm_size);
        //
        //  The target rank receives data from all non-root ranks and reports all data
        //
        for(int i = 1; i < comm_size; ++i)
        {
            std::string str{};
            mpi::recv(str, i, 0, comm);
            _data[i] = recv_serialize(str);
        }
        _data[comm_rank] = entry;
        return _data;
    }
    else
    {
        auto str_ret = send_serialize(entry);
        //
        //  The non-target rank sends its data to the root rank and only reports own
        //  data
        //
        mpi::send(str_ret, 0, 0, comm);
        return data_type(1, entry);
    }
#endif
}
//
}  // namespace internal
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct serialization : public internal::serialization<Tp, is_enabled<Tp>::value>
{
    using type      = Tp;
    using base_type = internal::serialization<Tp, is_enabled<Tp>::value>;
    using metadata  = typename base_type::metadata;

    TIMEMORY_DEFAULT_OBJECT(serialization)

    template <typename... Args>
    serialization(Args&&... args)
    : base_type{ std::forward<Args>(args)... }
    {}

    using base_type::operator();
    using base_type::get_description;
    using base_type::get_display_unit;
    using base_type::get_identifier;
    using base_type::get_label;
    using base_type::get_unit;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim

#define TIMEMORY_OPERATIONS_TYPES_SERIALIZATION_HPP_
#include "timemory/operations/types/serialization.cpp"
#undef TIMEMORY_OPERATIONS_TYPES_SERIALIZATION_HPP_
