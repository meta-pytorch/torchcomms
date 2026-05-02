#!/bin/bash

: ${MPI_HEADER:=@MPI_HEADER@}
: ${OUT:=/dev/stdout}
: ${SED:=$(which ssed)}

# if no ssed, use sed
if [ -z "${SED}" ]; then SED=$(which sed); fi

if test $(echo "hello" | ${SED} -r 's/h/t/g' 2> /dev/null); then
    SED="${SED} -r"
fi

tolower()
{
    if [ "$(uname)" == "Darwin" ]; then
        echo "$@" | awk -F '\\\|\\\|\\\|' '{print tolower($1)}';
    else
        echo "$@" | awk -F \|~\| '{print tolower($1)}';
    fi
}

get_mpi_functions()
{
    local N=0
    local funcs=$(grep -E 'int[ \t].*PMPI_' ${MPI_HEADER} \
        | grep '(' \
        | sed 's/OMPI_DECLSPEC//g' \
        | ${SED} 's/[(].*//g' \
        | awk '{print $2}' \
        | grep -E -v 'typedef|OMPI_|MPI_T_|MPI_Pcontrol|MPI_Test|MPI_Type|MPI_Init|MPI_Finalize|MPI_Abort|MPI_DUP|MPI_Address|MPI_Attr|MPI_Errhandler|MPI_Keyval' | sort -u)
    for i in ${funcs}
    do
        i=$(echo $i | sed 's/^PMPI_/MPI_/g')
        if [ -z "$(echo $i | grep MPI_)" ]; then continue; fi
        if [ -n "$(echo $i | grep -E '_c2f|MPI_Fint')" ]; then continue; fi
    	echo "        TIMEMORY_C_GOTCHA(mpip_gotcha_t, ${N}, $i);"
	    N=$((${N}+1))
    done
    echo "SIZE ${N}"
}

GOTCHA_SPEC=$(get_mpi_functions | grep -v "^SIZE")
GOTCHA_SIZE=$(get_mpi_functions | grep "^SIZE" | awk '{print $NF}')

cat <<EOF>> ${OUT}

#include "timemory/library.h"
#include "timemory/timemory.hpp"
#include "timemory/components/gotcha.hpp"

#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;
using string_t      = std::string;
using stringset_t   = std::set<std::string>;
using mpi_toolset_t = tim::complete_list_t;
using mpip_gotcha_t = tim::component::gotcha<${GOTCHA_SIZE}, mpi_toolset_t>;
using mpip_tuple_t  = tim::component_tuple<mpip_gotcha_t>;
using toolset_ptr_t = std::shared_ptr<mpip_tuple_t>;
using record_map_t  = std::unordered_map<uint64_t, toolset_ptr_t>;

static toolset_ptr_t    _tool_instance;
static std::atomic<int> _tool_count;

extern "C"
{

uint64_t init_timemory_mpip_tools()
{
    static bool is_initialized = false;
    if(!is_initialized)
    {
        // initialize manager
        auto manager = tim::manager::instance();
        tim::consume_parameters(manager);

        printf("[timemory]> %s...\n", __FUNCTION__);

        // generate the gotcha wrappers
        mpip_gotcha_t::get_initializer() = []()
        {
${GOTCHA_SPEC}
        };

        // provide environment variable for suppressing wrappers
        mpip_gotcha_t::get_reject_list() = []() {
            auto reject_list =
                tim::get_env<std::string>("TIMEMORY_MPIP_REJECT_LIST", "");
            if(reject_list.length() == 0)
                return stringset_t{};
            auto        reject_list_vec = tim::delimit(reject_list);
            stringset_t reject_list_set;
            for(const auto& itr : reject_list_vec)
                reject_list_set.insert(itr);
            return reject_list_set;
        };

        // provide environment variable for selecting wrappers
        mpip_gotcha_t::get_permit_list() = []() {
            auto permit_list =
                tim::get_env<std::string>("TIMEMORY_MPIP_PERMIT_LIST", "");
            if(permit_list.length() == 0)
                return stringset_t{};
            auto        permit_list_vec = tim::delimit(permit_list);
            stringset_t permit_list_set;
            for(const auto& itr : permit_list_vec)
                permit_list_set.insert(itr);
            return permit_list_set;
        };

        is_initialized = true;
    }

    // provide environment variable for enabling/disabling
    if(tim::get_env<bool>("ENABLE_TIMEMORY_MPIP", true))
    {
        auto env_ret  = tim::get_env<string_t>("TIMEMORY_PROFILER_COMPONENTS", "");
        auto env_enum = tim::enumerate_components(tim::delimit(env_ret));
        if(env_enum.empty())
        {
            env_ret  = tim::get_env<string_t>("TIMEMORY_COMPONENT_LIST_INIT", "");
            env_enum = tim::enumerate_components(tim::delimit(env_ret));
        }

        mpi_toolset_t::get_initializer() = [env_enum](auto& cl) {
            ::tim::initialize(cl, env_enum);
        };

        if(!_tool_instance)
            _tool_instance = std::make_shared<mpip_tuple_t>("timemory_mpip");

        auto cleanup_functor = [=]() {
            if(_tool_instance)
            {
                _tool_instance->stop();
                _tool_instance.reset();
            }
        };
        auto idx = ++_tool_count;
        auto key = TIMEMORY_JOIN("-", "timemory-mpip", idx);
        tim::manager::instance()->add_cleanup(key, cleanup_functor);

        _tool_instance->start();
        return idx;
    }
    else
    {
        return 0;
    }
}

uint64_t stop_timemory_mpip_tools(uint64_t id)
{
    if(id > 0 && _tool_instance.get())
    {
        auto key = TIMEMORY_JOIN("-", "timemory-mpip", id);
        auto idx = --_tool_count;
        tim::manager::instance()->remove_cleanup(key);
        _tool_instance->stop();
        if(idx == 0)
            _tool_instance.reset();
        return idx;
    }
    return 0;
}

uint64_t init_timemory_mpip_tools_() { return init_timemory_mpip_tools(); }
uint64_t stop_timemory_mpip_tools_(uint64_t id)
{
    return stop_timemory_mpip_tools(id);
}

}  // extern "C"
EOF
