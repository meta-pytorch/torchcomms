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

#if !defined(TIMEMORY_PYSTORAGE_SOURCE)
#    define TIMEMORY_PYSTORAGE_SOURCE
#endif

#if !defined(TIMEMORY_USE_EXTERN)
#    define TIMEMORY_USE_EXTERN
#endif

#include "libpytimemory-components.hpp"
#include "timemory/timemory.hpp"

namespace pystorage
{
//
template <typename Tp>
struct storage_bindings
{
    static constexpr bool value =
        tim::trait::is_available<Tp>::value && tim::trait::uses_value_storage<Tp>::value;
};
//
static inline std::string
get_class_name(std::string id)
{
    static const std::set<char> delim{
        '_',
        '-',
    };

    if(id.empty())
        return std::string{};

    id = tim::settings::tolower(id);

    if(id.find("_idx") == id.length() - 4)
        id = id.substr(0, id.length() - 4);

    if(id.find("timemory_") == 0)
        id = id.substr(9);

    // capitalize after every delimiter
    for(size_t i = 0; i < id.size(); ++i)
    {
        if(i == 0)
            id.at(i) = toupper(id.at(i));
        else
        {
            if(delim.find(id.at(i)) != delim.end() && i + 1 < id.length())
            {
                id.at(i + 1) = toupper(id.at(i + 1));
                ++i;
            }
        }
    }
    // remove all delimiters
    for(auto ditr : delim)
    {
        size_t _pos = 0;
        while((_pos = id.find(ditr)) != std::string::npos)
            id = id.erase(_pos, 1);
    }

    return id;
}
//
//--------------------------------------------------------------------------------------//
//
static TIMEMORY_COLD auto
read_object(py::object _obj)
{
    std::stringstream iss;
    if(py::hasattr(_obj, "read"))
    {
        auto _data = _obj.attr("read")();
        iss << _data.cast<std::string>();
    }
    else
    {
        auto          _data = _obj.cast<std::string>();
        std::ifstream ifs{ _data };
        if(ifs)
        {
            std::string str{};
            ifs.seekg(0, std::ios::end);
            str.reserve(ifs.tellg());
            ifs.seekg(0, std::ios::beg);
            str.assign((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
            iss = std::stringstream{ str };
        }
        else
        {
            iss = std::stringstream{ _data };
        }
    }
    return iss.str();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename DataT>
TIMEMORY_COLD auto
from_json(py::object _inp)
{
    using policy_type =
        tim::policy::input_archive<tim::cereal::JSONInputArchive, tim::project::python>;

    DataT             _obj{};
    std::stringstream iss{ read_object(std::move(_inp)) };

    {
        auto ia = policy_type::get(iss);
        ia->setNextName("timemory");
        ia->startNode();
        tim::operation::serialization<Tp>{}(*ia, _obj);
        ia->finishNode();  // timemory
    }
    return _obj;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename DataT>
TIMEMORY_COLD auto
to_json(DataT&& _obj)
{
    using policy_type = tim::policy::output_archive<tim::cereal::MinimalJSONOutputArchive,
                                                    tim::project::python>;

    std::stringstream oss;
    {
        auto oa = policy_type::get(oss);
        oa->setNextName("timemory");
        oa->startNode();
        tim::operation::serialization<Tp>{}(*oa, std::forward<DataT>(_obj));
        oa->finishNode();  // timemory
    }
    auto json_str    = oss.str();
    auto json_module = py::module::import("json");
    return json_module.attr("loads")(json_str);
}

//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
TIMEMORY_COLD auto
construct(py::module& _pymod, int, tim::enable_if_t<storage_bindings<Tp>::value> = 0)
{
    using property_t = tim::component::properties<Tp>;
    // using metadata_t     = tim::component::metadata<Tp>;

    // ensure specialized
    static_assert(property_t::specialized(), "Error! Missing specialization");

    auto _base = get_class_name(property_t::enum_string());

    {
        using result_type   = tim::node::result<Tp>;
        using pyresult_type = py::class_<result_type>;

        auto _id   = TIMEMORY_JOIN("", _base, "StorageResult");
        auto _desc = TIMEMORY_JOIN(" ", "Storage result class for", _base);

        pyresult_type _pyresult(_pymod, _id.c_str(), _desc.c_str());

        auto _tid       = [](result_type* _obj) { return _obj->tid(); };
        auto _pid       = [](result_type* _obj) { return _obj->pid(); };
        auto _depth     = [](result_type* _obj) { return _obj->depth(); };
        auto _hash      = [](result_type* _obj) { return _obj->hash(); };
        auto _rhash     = [](result_type* _obj) { return _obj->rolling_hash(); };
        auto _prefix    = [](result_type* _obj) { return _obj->prefix(); };
        auto _hierarchy = [](result_type* _obj) { return _obj->hierarchy(); };
        auto _data      = [](result_type* _obj) {
            using lwtuple_t          = tim::lightweight_tuple<Tp>;
            auto* _lw                = new lwtuple_t{};
            *_lw->template get<Tp>() = _obj->data();
            return _lw;
        };
        auto _stats = [](result_type* _obj) { return _obj->stats(); };

        _pyresult.def("tid", _tid, "Thread identifier");
        _pyresult.def("pid", _pid, "Process identifier");
        _pyresult.def("depth", _depth, "Depth in call-stack");
        _pyresult.def("hash", _hash, "Hash identifier");
        _pyresult.def("rolling_hash", _rhash, "Hash identifer for hierarchy");
        _pyresult.def("prefix", _prefix,
                      "String identifier. This identifier includes the leading '>>> ' "
                      "identifer which may also be prefixed with the thread and process "
                      "identifiers, e.g. '|0|2|>>> name' for rank 0 and thread 2");
        _pyresult.def("hierarchy", _hierarchy, "Array of parent hash identifiers");
        _pyresult.def("data", _data, "Component object");
        _pyresult.def("stats", _stats, "Statistical accumulation of component values");
    }

    {
        using entry_type   = typename tim::node::tree<Tp>::entry_type;
        using pyentry_type = py::class_<entry_type>;

        auto _id   = TIMEMORY_JOIN("", _base, "StorageTreeEntryValue");
        auto _desc = TIMEMORY_JOIN(" ", "Storage tree entry class for", _base);

        pyentry_type _pyentry(_pymod, _id.c_str(), _desc.c_str());

        auto _data = [](entry_type* _obj) {
            using lwtuple_t = tim::lightweight_tuple<Tp>;
            auto     _itr   = _obj->data().get_iterator();
            uint64_t _hash  = 0;
            if(_itr)
                _hash = _itr->id();
            auto* _lw                = new lwtuple_t{ _hash };
            *_lw->template get<Tp>() = _obj->data();
            return _lw;
        };
        auto _stats = [](entry_type* _obj) { return _obj->stats(); };

        _pyentry.def("data", _data, "Component object");
        _pyentry.def("stats", _stats, "Statistical accumulation of component values");
    }

    {
        using tree_type   = tim::node::tree<Tp>;
        using pytree_type = py::class_<tree_type>;

        auto _id   = TIMEMORY_JOIN("", _base, "StorageTreeEntry");
        auto _desc = TIMEMORY_JOIN(" ", "Storage tree entry class for", _base);

        pytree_type _pytree(_pymod, _id.c_str(), _desc.c_str());

        auto _dummy = [](tree_type* _obj) {
            return (_obj->depth() == 0 || _obj->hash() == 0) ? true : _obj->is_dummy();
        };
        auto _tid    = [](tree_type* _obj) { return _obj->tid(); };
        auto _pid    = [](tree_type* _obj) { return _obj->pid(); };
        auto _depth  = [](tree_type* _obj) { return _obj->depth() - 1; };
        auto _hash   = [](tree_type* _obj) { return _obj->hash(); };
        auto _prefix = [](tree_type* _obj) {
            return tim::get_hash_identifier(_obj->hash());
        };
        auto _inclusive = [](tree_type* _obj) { return _obj->inclusive(); };
        auto _exclusive = [](tree_type* _obj) { return _obj->exclusive(); };

        _pytree.def(
            "is_dummy", _dummy,
            "Returns whether the object is placeholder for thread-synchronization");
        _pytree.def("tid", _tid, "Thread identifier");
        _pytree.def("pid", _pid, "Process identifier");
        _pytree.def("depth", _depth, "Depth in call-stack");
        _pytree.def("hash", _hash, "Hash identifier");
        _pytree.def("prefix", _prefix, "String identifier for the hash");
        _pytree.def("inclusive", _inclusive,
                    "Inclusive values for component statistics data");
        _pytree.def("exclusive", _exclusive,
                    "Exclusive values for component statistics data");
    }

    {
        using basic_node_type   = tim::node::tree<Tp>;
        using basic_tree_type   = tim::basic_tree<basic_node_type>;
        using pybasic_tree_type = py::class_<basic_tree_type>;

        auto _id   = TIMEMORY_JOIN("", _base, "StorageTree");
        auto _desc = TIMEMORY_JOIN(" ", "Storage tree class for", _base);

        pybasic_tree_type _pytree(_pymod, _id.c_str(), _desc.c_str());

        auto _value    = [](basic_tree_type* _obj) { return _obj->get_value(); };
        auto _children = [](basic_tree_type* _obj) {
            auto                         _children_ptrs = _obj->get_children();
            std::vector<basic_tree_type> _children{};
            _children.reserve(_children_ptrs.size());
            for(auto& itr : _children_ptrs)
                _children.emplace_back(*itr);
            return _children;
        };

        _pytree.def("value", _value, "Get the tree node value");
        _pytree.def("children", _children, "Get the tree node children");
    }

    {
        using storage_type           = tim::storage<Tp>;
        using basic_tree_type        = tim::basic_tree<tim::node::tree<Tp>>;
        using basic_tree_vector_type = std::vector<basic_tree_type>;
        using pystorage_type         = py::class_<storage_type>;

        auto _id   = TIMEMORY_JOIN("", _base, "Storage");
        auto _desc = TIMEMORY_JOIN(" ", "Storage class for", _base);

        static pystorage_type _pystorage(_pymod, _id.c_str(), _desc.c_str());

        auto _get     = []() { return storage_type::instance()->get(); };
        auto _dmp_get = []() { return storage_type::instance()->dmp_get(); };
        auto _mpi_get = []() { return storage_type::instance()->mpi_get(); };
        auto _upc_get = []() { return storage_type::instance()->upc_get(); };

        auto _get_tree = []() {
            basic_tree_vector_type _data;
            storage_type::instance()->get(_data);
            return _data;
        };
        auto _dmp_get_tree = []() {
            std::vector<basic_tree_vector_type> _data;
            storage_type::instance()->dmp_get(_data);
            return _data;
        };
        auto _mpi_get_tree = []() {
            std::vector<basic_tree_vector_type> _data;
            storage_type::instance()->mpi_get(_data);
            return _data;
        };
        auto _upc_get_tree = []() {
            std::vector<basic_tree_vector_type> _data;
            storage_type::instance()->upc_get(_data);
            return _data;
        };

        using basic_get_vector_type =
            tim::decay_t<decltype(storage_type::instance()->get())>;
        using basic_dmp_get_vector_type =
            tim::decay_t<decltype(storage_type::instance()->dmp_get())>;

        auto _from_json_tree = [](py::object _obj) {
            return from_json<Tp, basic_tree_vector_type>(_obj);
        };

        auto _from_json_tree_dmp = [](py::object _obj) {
            return from_json<Tp, std::vector<basic_tree_vector_type>>(_obj);
        };

        auto _from_json = [](py::object _obj) {
            return from_json<Tp, basic_get_vector_type>(_obj);
        };

        auto _from_json_dmp = [](py::object _obj) {
            return from_json<Tp, basic_dmp_get_vector_type>(_obj);
        };

        auto _to_json_tree = [](basic_tree_vector_type _obj) {
            return to_json<Tp>(_obj);
        };

        auto _to_json_tree_dmp = [](std::vector<basic_tree_vector_type> _obj) {
            return to_json<Tp>(_obj);
        };

        auto _to_json = [](basic_get_vector_type _obj) { return to_json<Tp>(_obj); };

        auto _to_json_dmp = [](basic_dmp_get_vector_type _obj) {
            return to_json<Tp>(_obj);
        };

        _pystorage.def_static(
            "get", _get,
            "Get the component results in a flat data structure. The hierarchy is "
            "represented by the indentation of prefix string and the depth field. This "
            "returns only the data within the current process");
        _pystorage.def_static(
            "dmp_get", _dmp_get,
            "Get the full component results across all the distributed "
            "memory parallelism when called from zeroth rank. If called from non-zeroth "
            "rank, this will be the results for that individual rank. This is the "
            "general form regardless of whether MPI and/or UPC++ is used as the DMP "
            "backend");
        _pystorage.def_static("mpi_get", _mpi_get,
                              "Identical to dmp_get if the distributed memory "
                              "parallelism library is MPI");
        _pystorage.def_static("upcxx_get", _upc_get,
                              "Identical to dmp_get if the distributed memory "
                              "parallelism library is UPC++");

        _pystorage.def_static("get_tree", _get_tree,
                              "Get the component results in a hierarchical data "
                              "structure. This returns "
                              "only the data within the current process");
        _pystorage.def_static(
            "dmp_get_tree", _dmp_get_tree,
            "Get the full hierarchy of component results across all the distributed "
            "memory parallelism when called from zeroth rank. If called from non-zeroth "
            "rank, this will be the results for that individual rank. This is the "
            "general function for getting results from distributed memory parallelism "
            "regardless of whether MPI and/or UPC++ is used as the DMP "
            "backend");
        _pystorage.def_static("mpi_get_tree", _mpi_get_tree,
                              "Identical to dmp_get_tree if the distributed memory "
                              "parallelism library is MPI");
        _pystorage.def_static("upcxx_get_tree", _upc_get_tree,
                              "Identical to dmp_get_tree if the distributed memory "
                              "process library is UPC++");

        _pystorage.def_static("load_json", _from_json,
                              "Load the result of get() from a JSON dictionary");
        _pystorage.def_static("load_json_dmp", _from_json_dmp,
                              "Load the result of dmp_get() from a JSON dictionary");
        _pystorage.def_static("load_json_tree", _from_json_tree,
                              "Load the result of get_tree() from a JSON dictionary");
        _pystorage.def_static("load_json_tree_dmp", _from_json_tree_dmp,
                              "Load the result of dmp_get_tree() from a JSON dictionary");

        auto _from = [_base](py::object _obj) {
            std::stringstream _msg;
            py::object        _ret = py::none{};
            std::stringstream iss{ read_object(std::move(_obj)) };

            for(const auto& itr :
                { "load_json", "load_json_dmp", "load_json_tree", "load_json_tree_dmp" })
            {
                _msg << _base << "." << itr << ":\n";
                try
                {
                    _ret = _pystorage.attr(itr)(iss.str());
                } catch(std::exception& e)
                {
                    _msg << "\n" << e.what() << std::endl;
                    continue;
                }
                break;
            }
            if(_ret.is_none())
                throw std::runtime_error(_msg.str());
            return _ret;
        };

        _pystorage.def_static(
            "load", _from,
            "Load the result of get(), {dmp,mpi,upcxx}_get(), get_tree(), or "
            "{dmp,mpi,upcxx}_get_tree() from a JSON dictionary");

        _pystorage.def_static(
            "dumps", _to_json,
            "Load the result of get(), {dmp,mpi,upcxx}_get(), get_tree(), or "
            "{dmp,mpi,upcxx}_get_tree() from a JSON dictionary");
        _pystorage.def_static(
            "dumps", _to_json_dmp,
            "Load the result of get(), {dmp,mpi,upcxx}_get(), get_tree(), or "
            "{dmp,mpi,upcxx}_get_tree() from a JSON dictionary");
        _pystorage.def_static(
            "dumps", _to_json_tree,
            "Load the result of get(), {dmp,mpi,upcxx}_get(), get_tree(), or "
            "{dmp,mpi,upcxx}_get_tree() from a JSON dictionary");
        _pystorage.def_static(
            "dumps", _to_json_tree_dmp,
            "Load the result of get(), {dmp,mpi,upcxx}_get(), get_tree(), or "
            "{dmp,mpi,upcxx}_get_tree() from a JSON dictionary");
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
construct(py::module&, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx>
constexpr auto construct(std::index_sequence<Idx...>)
{
    return tim::mpl::available_t<tim::type_list<tim::component::enumerator_t<Idx>...>>{};
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
auto
construct(py::module& _pymod, tim::type_list<Tp...>)
{
    TIMEMORY_FOLD_EXPRESSION(construct<Tp>(_pymod, 0));
}
//
//--------------------------------------------------------------------------------------//
//
py::module
generate(py::module& _pymod)
{
    py::module _pystorage = _pymod.def_submodule(
        "storage",
        "Classes which contain the accumulated data from timemory components stored via "
        "push/pop routines either implicitly (via decorators/context-managers) or "
        "explicitly (via timemory.component classes)");

    auto _types = construct(std::make_index_sequence<TIMEMORY_NATIVE_COMPONENTS_END>{});
    construct(_pystorage, _types);
    return _pystorage;
}
//
}  // namespace pystorage
