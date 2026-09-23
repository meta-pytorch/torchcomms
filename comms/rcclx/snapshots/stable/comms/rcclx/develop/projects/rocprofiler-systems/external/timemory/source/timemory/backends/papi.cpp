//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#ifndef TIMEMORY_BACKENDS_PAPI_CPP_
#define TIMEMORY_BACKENDS_PAPI_CPP_

#include "timemory/backends/defines.hpp"

#if !defined(TIMEMORY_BACKENDS_HEADER_MODE) ||                                           \
    (defined(TIMEMORY_BACKENDS_HEADER_MODE) && TIMEMORY_BACKENDS_HEADER_MODE == 0)
#    include "timemory/backends/papi.hpp"
#endif

#include "timemory/backends/hardware_counters.hpp"
#include "timemory/backends/types/papi.hpp"
#include "timemory/log/color.hpp"
#include "timemory/macros/language.hpp"

#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tim
{
namespace papi
{
namespace details
{
TIMEMORY_BACKENDS_INLINE
size_t
generate_component_info(bool _with_qualifiers, bool _force)
{
#if defined(TIMEMORY_USE_PAPI)
    static auto& _info_map     = get_component_info_map();
    static auto  _compute_size = []() {
        size_t _v = 0;
        for(auto& itr : *_info_map)
            _v += itr.second.size();
        return _v;
    };
    static auto _print = []() {
        for(auto&& itr : *_info_map)
        {
            for(auto&& vitr : itr.second)
                TIMEMORY_PRINTF(stderr, "[papi][%s][%i][%i][%i][%i] %-40s :: %s\n",
                                itr.first.name, itr.first.index, vitr.component_index,
                                vitr.event_code, vitr.event_type, vitr.symbol,
                                vitr.short_descr);
        }
    };
    auto parse_event_qualifiers = [](PAPI_event_info_t* info) {
        char* ptr = nullptr;
        // handle the PAPI component-style events which have a component:::event type
        if((ptr = strstr(info->symbol, ":::")))
            ptr += 3;
        // handle libpfm4-style events which have a pmu::event type event name
        else if((ptr = strstr(info->symbol, "::")))
            ptr += 2;
        else
            ptr = info->symbol;

        return (strchr(ptr, ':') != nullptr);
    };

    assert(_info_map != nullptr);

    std::vector<event_info_t> _preset_events{};
    for(int i = 0; i < PAPI_END_idx; ++i)
    {
        PAPI_event_info_t info{};
        if(PAPI_get_event_info((i | PAPI_PRESET_MASK), &info) == PAPI_OK)
            _preset_events.emplace_back(info);
    }

    if(!_preset_events.empty())
    {
        PAPI_component_info_t _preset_component{};
        _preset_component.CmpIdx = -1;
        memset(&_preset_component, 0, sizeof(_preset_component));
        snprintf(_preset_component.name, sizeof(_preset_component.name) - 1, "%s",
                 "preset");
        _info_map->emplace(papi::component_info{ -1, _preset_component }, _preset_events);
    }

    auto _include_qualifiers =
        (_force) ? get_env<bool>(TIMEMORY_SETTINGS_KEY("PAPI_COMPONENT_QUALIFIERS",
                                                       _with_qualifiers))
                 : _with_qualifiers;

    for(int cid = 0; cid < PAPI_num_components(); ++cid)
    {
        const PAPI_component_info_t* component = PAPI_get_component_info(cid);

        // Skip disabled components
#    if defined(PAPI_EDELAY_INIT)
        if(component->disabled != 0 && component->disabled != PAPI_EDELAY_INIT)
            continue;
#    else
        if(component->disabled != 0)
            continue;
#    endif

        std::vector<event_info_t> _events = {};
        // Always ASK FOR the first event
        // Don't just assume it'll be the first numeric value
        int _idx = 0 | PAPI_NATIVE_MASK;
        if(PAPI_enum_cmp_event(&_idx, PAPI_ENUM_FIRST, cid) == PAPI_OK)
        {
            do
            {
                PAPI_event_info_t _info;
                memset(&_info, 0, sizeof(_info));
                if(PAPI_get_event_info(_idx, &_info) == PAPI_OK)
                {
                    _events.emplace_back(_info);
                    if(_include_qualifiers)
                    {
                        auto _qidx = _idx;
                        if(PAPI_enum_cmp_event(&_qidx, PAPI_NTV_ENUM_UMASKS, cid) ==
                           PAPI_OK)
                        {
                            do
                            {
                                PAPI_event_info_t _qinfo;
                                memset(&_qinfo, 0, sizeof(_qinfo));
                                if(PAPI_get_event_info(_qidx, &_qinfo) == PAPI_OK)
                                {
                                    if(parse_event_qualifiers(&_qinfo))
                                    {
                                        _events.emplace_back(_qinfo, _info.symbol,
                                                             _info.long_descr, true);
                                    }
                                }
                            } while(PAPI_enum_cmp_event(&_qidx, PAPI_NTV_ENUM_UMASKS,
                                                        cid) == PAPI_OK);
                        }
                    }
                }
            } while(PAPI_enum_cmp_event(&_idx, PAPI_ENUM_EVENTS, cid) == PAPI_OK);
        }

        if(!_events.empty())
        {
            for(auto&& itr : _events)
            {
                auto _cmp_info = papi::component_info{ itr.component_index, *component };
                if(!_info_map->emplace(_cmp_info, std::vector<event_info_t>{ itr })
                        .second)
                {
                    auto& _existing = _info_map->at(_cmp_info);
                    bool  _found    = false;
                    for(auto& eitr : _existing)
                    {
                        if(eitr.event_code == itr.event_code)
                        {
                            _found = true;
                            break;
                        }
                    }
                    if(!_found)
                        _existing.emplace_back(itr);
                }
            }
        }
    }

    for(auto& itr : *_info_map)
    {
        for(auto& vitr : itr.second)
        {
            if(strlen(vitr.short_descr) == 0)
            {
                std::string _descr = vitr.long_descr;
                auto        _pos   = _descr.find(". ");
                if(_pos != std::string::npos)
                    _descr = _descr.substr(0, _pos);
                auto _max_len = sizeof(vitr.short_descr) - 1;

                auto _strip = [&_descr](char _c) {
                    // remove parentheses
                    const std::regex _trailing_paren{ "(.*) \\(.*\\)$",
                                                      std::regex_constants::optimize };
                    const std::regex _trailing_masks{ "(.*)(. masks:)(.*)$",
                                                      std::regex_constants::optimize };
                    if(std::regex_match(_descr, _trailing_paren))
                    {
                        _descr = std::regex_replace(_descr, _trailing_paren, "$1");
                    }
                    else if(std::regex_search(_descr, _trailing_masks))
                    {
                        _descr = std::regex_replace(_descr, _trailing_masks, "$1 ($3)");
                    }
                    else
                    {
                        auto _dpos = _descr.find_last_of(_c);
                        if(_dpos != std::string::npos)
                            _descr = _descr.substr(0, _dpos);
                    }

                    for(auto iitr : { ' ', '.', ',' })
                    {
                        while(!_descr.empty() && _descr.at(_descr.length() - 1) == iitr)
                            _descr = _descr.substr(0, _descr.length() - 1);
                    }

                    using strpair_t = std::pair<string_t, string_t>;
                    for(auto&& iitr :
                        { strpair_t{ "[\\,\\.] \\(", " (" },
                          strpair_t{ "[\\,\\.]\\)", ")" },
                          strpair_t{ "\\(monitor ", "(" }, strpair_t{ "\\(at ", "(" } })
                    {
                        const std::regex _punct_paren{ iitr.first,
                                                       std::regex_constants::optimize };
                        while(std::regex_search(_descr, _punct_paren))
                            _descr =
                                std::regex_replace(_descr, _punct_paren, iitr.second);
                    }
                };
                // remove words from end until less than max length
                while(_descr.length() > _max_len)
                {
                    _strip(' ');
                }
                if(!_descr.empty())
                {
                    for(size_t i = 0; i < _descr.length(); ++i)
                        vitr.short_descr[i] = _descr.c_str()[i];
                    vitr.short_descr[_descr.length()] = '\0';
                    vitr.modified_short_descr         = true;
                }
            }
        }
    }

    if((settings::debug() && settings::verbose() > 0))
    {
        _print();
    }

    return _compute_size();
#else
    consume_parameters(_with_qualifiers, _force);
    return 0;
#endif
}
}  // namespace details

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
int
get_event_code(string_view_cref_t event_code_str)
{
#if defined(TIMEMORY_USE_PAPI) && defined(TIMEMORY_UNIX)
    int event_code = -1;
    int retval     = PAPI_event_name_to_code(event_code_str.data(), &event_code);
    working()      = check(retval, TIMEMORY_JOIN(' ', "Warning!! Failure converting",
                                                 event_code_str, "to enum value")
                                       .c_str());
    return (retval == PAPI_OK) ? event_code : PAPI_NOT_INITED;
#else
    consume_parameters(event_code_str);
    return PAPI_NOT_INITED;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
std::string
get_event_code_name(int event_code)
{
#if defined(TIMEMORY_USE_PAPI) && defined(TIMEMORY_UNIX)
    static const uint64_t BUFFER_SIZE = 4096;
    char                  event_code_char[BUFFER_SIZE];
    int                   retval = PAPI_event_code_to_name(event_code, event_code_char);
    working() =
        check(retval, TIMEMORY_JOIN(' ', "Warning!! Failure converting event code",
                                    event_code, "to a name")
                          .c_str());
    return (retval == PAPI_OK) ? std::string(event_code_char) : "";
#else
    consume_parameters(event_code);
    return "";
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
event_info_t
get_event_info(string_view_cref_t evt_type)
{
#if defined(TIMEMORY_USE_PAPI)
    for(const auto& itr : *get_component_info_map())
    {
        for(const auto& vitr : itr.second)
        {
            if(evt_type == string_view_t{ vitr.symbol })
                return vitr;
        }
    }
#else
    consume_parameters(evt_type);
#endif
    return PAPI_event_info_t{};
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
bool
add_event(int event_set, string_view_cref_t event_name)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        auto& _info_map = get_component_info_map();
        for(auto& itr : *_info_map)
        {
            if(settings::debug() || settings::verbose() >= 2)
                TIMEMORY_PRINTF(stderr, "[papi] checking for %s in component %s...\n",
                                event_name.data(), itr.first.name);
            for(auto& eitr : itr.second)
            {
                if(strcmp(event_name.data(), eitr.symbol) == 0)
                {
                    if(settings::debug() || settings::verbose() >= 2)
                        TIMEMORY_PRINTF(stderr, "[papi] found %s in component %s...\n",
                                        event_name.data(), itr.first.name);
                    auto retval = PAPI_add_event(event_set, eitr.event_code);
                    return check(
                        retval, TIMEMORY_JOIN(" ", "Warning!! Failure to add named event",
                                              event_name, "with code", eitr.event_code,
                                              "to event set", event_set));
                }
            }
        }

        int  retval = PAPI_add_named_event(event_set, event_name.data());
        bool _working =
            check(retval, TIMEMORY_JOIN(" ", "Warning!! Failure to add named event",
                                        event_name, "to event set", event_set));
        working() = _working;
        return _working;
    }
#else
    consume_parameters(event_set, event_name);
#endif
    return false;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
std::vector<bool>
add_events(int event_set, string_t* events, int number)
{
    std::vector<bool> _success(number, false);
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        for(int i = 0; i < number; ++i)
        {
            _success.at(i) = add_event(event_set, events[i].data());
        }
    }
#else
    consume_parameters(event_set, events, number);
#endif
    return _success;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
hwcounter_info_t
available_events_info(const std::unordered_set<std::string>& excluded_components)
{
    hwcounter_info_t evts{};

#if defined(TIMEMORY_USE_PAPI)
    details::init_library();
    if(working())
    {
        // auto _force_cmp_init = [](int cid) {
        //    int nvt_code = 0 | PAPI_NATIVE_MASK;
        //    PAPI_enum_cmp_event(&nvt_code, PAPI_ENUM_FIRST, cid);
        //};

        for(int i = 0; i < PAPI_END_idx; ++i)
        {
            PAPI_event_info_t info;
            auto              ret = PAPI_get_event_info((i | PAPI_PRESET_MASK), &info);
            if(ret != PAPI_OK)
                continue;

            bool _avail = query_event((i | PAPI_PRESET_MASK));
            // string_t _sym   = get_timemory_papi_presets()[i].symbol;
            string_t _sym   = info.symbol;
            string_t _pysym = _sym;
            for(auto& itr : _pysym)
                itr = tolower(itr);
            string_t _rm = "papi_";
            auto     idx = _pysym.find(_rm);
            if(idx != string_t::npos)
                _pysym = _pysym.substr(idx + _rm.length());
            evts.emplace_back(_avail, hardware_counters::api::papi, i, PAPI_PRESET_MASK,
                              _sym, _pysym, info.short_descr, info.long_descr,
                              // get_timemory_papi_presets()[i].short_descr,
                              // get_timemory_papi_presets()[i].long_descr,
                              info.units);
        }

        auto numcmp = PAPI_num_components();

        // auto num_events    = 0;
        auto enum_modifier = PAPI_ENUM_EVENTS;

        auto parse_event_qualifiers = [](PAPI_event_info_t* info) {
            char *pmask, *ptr;

            // handle the PAPI component-style events which have a component:::event type
            if((ptr = strstr(info->symbol, ":::")))
            {
                ptr += 3;
                // handle libpfm4-style events which have a pmu::event type event name
            }
            else if((ptr = strstr(info->symbol, "::")))
            {
                ptr += 2;
            }
            else
            {
                ptr = info->symbol;
            }

            if((pmask = strchr(ptr, ':')) == nullptr)
            {
                return false;
            }
            memmove(info->symbol, pmask, (strlen(pmask) + 1) * sizeof(char));

            //  The description field contains the event description followed by a tag
            //  'masks:' and then the mask description (if there was a mask with this
            //  event).  The following code isolates the mask description part of this
            //  information.

            pmask = strstr(info->long_descr, "masks:");
            if(pmask == nullptr)
            {
                info->long_descr[0] = 0;
            }
            else
            {
                pmask += 6;  // bump pointer past 'masks:' identifier in description
                memmove(info->long_descr, pmask, (strlen(pmask) + 1) * sizeof(char));
            }
            return true;
        };

        unsigned int event_available = 0;

        auto check_event = [&event_available](PAPI_event_info_t* info) {
            int EventSet = PAPI_NULL;

            // if this event has already passed the check test, no need to try this one
            // again
            if(event_available != 0)
                return true;

            if(PAPI_create_eventset(&EventSet) == PAPI_OK)
            {
                if(PAPI_add_named_event(EventSet, info->symbol) == PAPI_OK)
                {
                    PAPI_remove_named_event(EventSet, info->symbol);
                    event_available = 1;
                }  // else printf("********** PAPI_add_named_event( %s ) failed: event
                   // could not be added \n", info->symbol);
                if(PAPI_destroy_eventset(&EventSet) != PAPI_OK)
                {
                    TIMEMORY_PRINTF(stderr,
                                    "**********  Call to destroy eventset failed when "
                                    "trying to check event '%s'  **********\n",
                                    info->symbol);
                }
            }
            return (event_available != 0);
        };

        for(int cid = 0; cid < numcmp; cid++)
        {
            const PAPI_component_info_t* component;
            component = PAPI_get_component_info(cid);

            // Skip disabled components
#    if defined(PAPI_EDELAY_INIT)
            if(component->disabled != 0 && component->disabled != PAPI_EDELAY_INIT)
                continue;
#    else
            if(component->disabled != 0)
                continue;
#    endif

            // Skip excluded components
            if(excluded_components.count(component->name) > 0)
                continue;

            // show this component has not found any events yet
            // int num_cmp_events = 0;

            // Always ASK FOR the first event
            // Don't just assume it'll be the first numeric value
            int i = 0 | PAPI_NATIVE_MASK;

            auto retval = PAPI_enum_cmp_event(&i, PAPI_ENUM_FIRST, cid);

            if(retval == PAPI_OK)
            {
                using qualifer_vec_t = hardware_counters::qualifier_vec_t;

                PAPI_event_info_t info;
                do
                {
                    bool _check      = true;
                    bool _avail      = false;
                    auto _qualifiers = qualifer_vec_t{};

                    if(strcmp(component->name, "rocm") == 0)
                    {
                        _check = false;
                        _avail = true;
                    }

                    memset(&info, 0, sizeof(info));
                    retval = PAPI_get_event_info(i, &info);

                    // This event may not exist
                    if(retval != PAPI_OK)
                        continue;

                    // Bail if event name doesn't contain include string
                    // if(flags.include && !strstr(info.symbol, flags.istr))
                    //    continue;

                    // Bail if event name does contain exclude string
                    // if(flags.xclude && strstr(info.symbol, flags.xstr))
                    //    continue;

                    // count only events that are actually processed
                    // num_events++;
                    // num_cmp_events++;

                    if(_check)
                    {
                        _avail = check_event(&info);
                    }

                    //		modifier = PAPI_NTV_ENUM_GROUPS returns event codes with a
                    //        groups id for each group in which this
                    //        native event lives, in bits 16 - 23 of event code
                    //        terminating with PAPI_ENOEVNT at the end of the list.

                    // This is an IBM Power issue
                    // bool _groups = flags.groups; // for reference
                    bool _groups = false;
                    if(_groups)
                    {
                        auto k = i;
                        if(PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_GROUPS, cid) == PAPI_OK)
                        {
                            // printf("Groups: ");
                            do
                            {
                                // printf("%4d", ((k & PAPI_NTV_GROUP_AND_MASK) >>
                                //               PAPI_NTV_GROUP_SHIFT) -
                                //                  1);
                            } while(PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_GROUPS, cid) ==
                                    PAPI_OK);
                            // printf("\n");
                        }
                    }

                    // If the user has asked us to check the events then we need to
                    // walk the list of qualifiers and try to check the event with each
                    // one. Even if the user does not want to display the qualifiers this
                    // is necessary to be able to correctly report which events can be
                    // used on this system.
                    //
                    // We also need to walk the list if the user wants to see the
                    // qualifiers.

                    // if(flags.qualifiers || _check)
                    {
                        auto k = i;
                        if(PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_UMASKS, cid) == PAPI_OK)
                        {
                            // clear event string using first mask
                            char first_event_mask_string[PAPI_HUGE_STR_LEN] = "";

                            do
                            {
                                PAPI_event_info_t _info;
                                bool              _qualifer_avail = true;
                                memset(&_info, 0, sizeof(_info));
                                retval = PAPI_get_event_info(k, &_info);
                                if(retval == PAPI_OK)
                                {
                                    // if first event mask string not set yet, set it now
                                    if(strlen(first_event_mask_string) == 0)
                                    {
                                        // strcpy(first_event_mask_string, _info.symbol);
                                    }

                                    if(_check)
                                    {
                                        _qualifer_avail = check_event(&_info);
                                    }
                                    // now test if the event qualifiers should be
                                    // displayed to the user
                                    // if(flags.qualifiers)
                                    {
                                        if(parse_event_qualifiers(&_info))
                                        {
                                            _qualifiers.emplace_back(
                                                _qualifer_avail, _info.event_code,
                                                string_t{ _info.symbol },
                                                string_t{ _info.long_descr });
                                        }
                                    }
                                }
                            } while(PAPI_enum_cmp_event(&k, PAPI_NTV_ENUM_UMASKS, cid) ==
                                    PAPI_OK);

                            // if we are validating events and the event_available flag is
                            // not set yet, try a few more combinations
                            if(false && _check && event_available == 0)
                            {
                                // try using the event with the first mask defined for the
                                // event and the cpu mask this is a kludge but many of the
                                // uncore events require an event specific mask (usually
                                // the first one defined will do) and they all require the
                                // cpu mask
                                strcpy(info.symbol, first_event_mask_string);
                                strcat(info.symbol, ":cpu=1");
                                _avail = check_event(&info);
                            }
                            if(false && _check && event_available == 0)
                            {
                                // an even bigger kludge is that there are 4 snpep_unc_pcu
                                // events which require the 'ff' and 'cpu' qualifiers to
                                // work correctly. if nothing else has worked, this code
                                // will try those two qualifiers with the current event
                                // name to see if it works
                                strcpy(info.symbol, first_event_mask_string);
                                char* wptr = strrchr(info.symbol, ':');
                                if(wptr != nullptr)
                                {
                                    *wptr = '\0';
                                    strcat(info.symbol, ":ff=64:cpu=1");
                                    _avail = check_event(&info);
                                }
                            }
                        }
                    }
                    string_t _short_descr = info.short_descr;
                    if(_short_descr.empty())
                    {
                        _short_descr = info.long_descr;
                        // truncate to first sentence if multiple sentences
                        auto _pos = _short_descr.find(". ");
                        if(_pos != std::string::npos)
                            _short_descr = _short_descr.substr(0, _pos);
                    }
                    string_t _sym       = info.symbol;
                    auto     _get_pysym = [](string_t _pysym) {
                        size_t _pos = 0;
                        while((_pos = _pysym.find(':')) != std::string::npos)
                        {
                            _pysym = _pysym.replace(_pos, 1, "_");
                        }
                        return _pysym;
                    };
                    evts.push_back(hardware_counters::info(
                        _avail, hardware_counters::api::papi, info.event_code, 0, _sym,
                        _get_pysym(_sym), _short_descr, info.long_descr, info.units,
                        _qualifiers));

                    for(auto& itr : _qualifiers)
                    {
                        _sym       = string_t{ info.symbol } + itr.symbol;
                        auto _desc = string_t{ info.symbol } + " + " + itr.description;
                        evts.push_back(hardware_counters::info(
                            itr.available, hardware_counters::api::papi, itr.event_code,
                            0, _sym, _get_pysym(_sym), _desc, _desc, info.units));
                    }
                } while(PAPI_enum_cmp_event(&i, enum_modifier, cid) == PAPI_OK);
            }
        }
    }
#endif

    if(!working() || evts.empty())
    {
        for(int i = 0; i < TIMEMORY_PAPI_PRESET_EVENTS; ++i)
        {
            if(get_timemory_papi_presets()[i].symbol == nullptr)
                continue;

            string_t _sym   = get_timemory_papi_presets()[i].symbol;
            string_t _pysym = _sym;
            for(auto& itr : _pysym)
                itr = tolower(itr);
            evts.push_back(hardware_counters::info(
                false, hardware_counters::api::papi, i, PAPI_PRESET_MASK, _sym, _pysym,
                get_timemory_papi_presets()[i].short_descr,
                get_timemory_papi_presets()[i].long_descr));
        }
    }

    std::sort(evts.begin(), evts.end());
    evts.erase(std::unique(evts.begin(), evts.end()), evts.end());

    return evts;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
tim::hardware_counters::info
get_hwcounter_info(const std::string& event_code_str)
{
#if defined(TIMEMORY_USE_PAPI)
    details::init_library();
#endif

    if(working())
    {
        auto         idx         = get_event_code(event_code_str);
        event_info_t _info       = get_event_info(idx);
        bool         _avail      = query_event(idx);
        string_t     _sym        = _info.symbol;
        string_t     _short_desc = _info.short_descr;
        string_t     _long_desc  = _info.long_descr;
        string_t     _units      = _info.units;
        string_t     _pysym      = _sym;
        for(auto& itr : _pysym)
            itr = tolower(itr);
        // int32_t _off = _info.event_code;
        int32_t _off = 0;
        if(_short_desc.empty())
            _short_desc = _sym;
        return hardware_counters::info(_avail, hardware_counters::api::papi, idx, _off,
                                       _sym, _pysym, _short_desc, _long_desc);
    }

    return hardware_counters::info(false, hardware_counters::api::papi, -1, 0,
                                   event_code_str, "", "", "");
}

}  // namespace papi
}  // namespace tim

#endif
