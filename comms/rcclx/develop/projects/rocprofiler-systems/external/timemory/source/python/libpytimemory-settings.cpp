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

#if !defined(TIMEMORY_PYSETTINGS_SOURCE)
#    define TIMEMORY_PYSETTINGS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/defines.h"
#include "timemory/mpl/policy.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/filepath.hpp"

using string_t = std::string;

#define SETTING_PROPERTY(TYPE, FUNC)                                                     \
    settings.def_property_static(                                                        \
        TIMEMORY_STRINGIZE(FUNC), [](py::object) { return tim::settings::FUNC(); },      \
        [](py::object, TYPE v) { tim::settings::FUNC() = v; },                           \
        "Binds to 'tim::settings::" TIMEMORY_STRINGIZE(FUNC) "()'")

//======================================================================================//
//
namespace pysettings
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Up, typename... Tail, tim::enable_if_t<sizeof...(Tail) == 0> = 0>
auto
add_property(py::class_<tim::settings>& _class, std::shared_ptr<tim::vsettings> _obj)
{
    using Tp   = tim::decay_t<Up>;
    auto _tidx = std::type_index(typeid(Tp));

    if(_obj && _obj->get_type_index() == _tidx)
    {
        bool _is_ref = dynamic_cast<tim::tsettings<Tp, Tp&>*>(_obj.get()) != nullptr;

        auto _env      = _obj->get_env_name();
        auto _mem_name = TIMEMORY_JOIN("_", "get", _obj->get_name());
        auto _mem_desc = TIMEMORY_JOIN("", "[Member variant of global property \"",
                                       _obj->get_name(), "\"] ", _obj->get_description());
        // member property
        _class.def_property(
            _mem_name.c_str(),
            [_env](tim::settings* _object) { return _object->get<Tp>(_env); },
            [_env](tim::settings* _object, Tp v) { return _object->set(_env, v); },
            _mem_desc.c_str());

        // static property
        if(!_is_ref)
        {
            _class.def_property_static(
                _obj->get_name().c_str(),
                [_obj](py::object) {
                    return (_obj) ? static_cast<tim::tsettings<Tp>*>(_obj.get())->get()
                                  : Tp{};
                },
                [_obj](py::object, Tp v) {
                    if(_obj)
                        static_cast<tim::tsettings<Tp>*>(_obj.get())->get() = v;
                },
                _obj->get_description().c_str());
            return true;
        }
        else
        {
            _class.def_property_static(
                _obj->get_name().c_str(),
                [_obj](py::object) {
                    return (_obj)
                               ? static_cast<tim::tsettings<Tp, Tp&>*>(_obj.get())->get()
                               : Tp{};
                },
                [_obj](py::object, Tp v) {
                    if(_obj)
                        static_cast<tim::tsettings<Tp, Tp&>*>(_obj.get())->get() = v;
                },
                _obj->get_description().c_str());
            return true;
        }
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename... Tail, tim::enable_if_t<(sizeof...(Tail) > 0), int> = 0>
auto
add_property(py::class_<tim::settings>& _class, std::shared_ptr<tim::vsettings> _obj)
{
    auto ret = add_property<Tp>(_class, _obj);
    if(!ret)
        return add_property<Tail...>(_class, _obj);
    else
        return ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tail>
auto
add_property(py::class_<tim::settings>& _class, std::shared_ptr<tim::vsettings> _obj,
             tim::type_list<Tail...>)
{
    return add_property<Tail...>(_class, _obj);
}
//
//--------------------------------------------------------------------------------------//
//
py::class_<tim::settings>
generate(py::module& _pymod)
{
    py::class_<tim::settings> settings(_pymod, "settings",
                                       "Global configuration settings for timemory");

    static auto _argparse = [](std::shared_ptr<tim::vsettings> obj, py::object argp,
                               bool _strip) {
        if(!obj || obj->get_command_line().empty())
            return;
        py::cpp_function parse = [obj](std::string val) { obj->parse(val); };
        std::string      tidx  = tim::demangle(obj->get_type_index().name());
        if(tidx.find("basic_string") != std::string::npos)
            tidx = "std::string";
        auto locals =
            py::dict("parser"_a = argp, "args"_a = obj->get_command_line(),
                     "desc"_a = obj->get_description(), "cnt"_a = obj->get_count(),
                     "max_cnt"_a = obj->get_max_count(), "name"_a = obj->get_name(),
                     "env_name"_a = obj->get_env_name(), "type_index"_a = tidx,
                     "value"_a = obj->as_string(), "parse_action"_a = parse,
                     "strip_timemory_prefix"_a = _strip);
        py::exec(R"(
             def make_action(func):
                 class customAction(argparse.Action):
                     def __call__(self, parser, args, values, option_string=None):
                         if type(values) is list:
                             func(",".join(values))
                         else:
                             func("{}".format(values))
                         setattr(args, self.dest, values)
                 return customAction

             _args = []
             _kwargs = {}
             for itr in args:
                if strip_timemory_prefix:
                    _args.append(itr.replace("--timemory-", "--"))
                else:
                    _args.append(itr)

             _kwargs["help"] = "{} [env: {}, type: {}]".format(desc, env_name.upper(),
                                                                  type_index)
             _kwargs["required"] = False

             if cnt > 1:
                _kwargs["nargs"] = cnt

             if (max_cnt == 0):
                 if value == "true":
                     _kwargs["action"] = "store_true"
                 else:
                     _kwargs["action"] = "store_false"
             else:
                 _kwargs["action"] = make_action(parse_action)
                 _kwargs["metavar"] = "VALUE"
                 _kwargs["default"] = value
                 _kwargs["type"] = str

             parser.add_argument(*_args, **_kwargs)
             )",
                 py::globals(), locals);
    };
    auto _init = []() {
        auto ret = new tim::settings{};
        *ret     = *tim::settings::instance<TIMEMORY_API>();
        return ret;
    };
    auto _parse = [](py::object _instance) {
        auto* _obj = _instance.cast<tim::settings*>();
        if(_instance.is_none() || _obj == nullptr)
            tim::settings::parse();
        else
            tim::settings::parse(_obj);
    };
    auto _read = [](py::object _instance, std::string inp) {
        auto* _obj = _instance.cast<tim::settings*>();
        if(_instance.is_none() || _obj == nullptr)
            tim::settings::instance()->read(inp);
        else
            _obj->read(inp);
    };
    static auto _args = [](py::object parser, py::object _instance, py::object subparser,
                           std::vector<std::string> fields) {
        auto pyargparse = py::module::import("argparse");

        if(parser.is_none())
            parser = pyargparse.attr("ArgumentParser")();

        using settings_t = tim::settings;
        auto* _obj       = (_instance.is_none()) ? settings_t::instance()
                                           : _instance.cast<settings_t*>();
        if(_obj == nullptr)
            return;

        py::object _parser        = parser;
        bool       _use_subparser = !subparser.is_none();

        if(_use_subparser)
        {
            try
            {
                auto _use      = subparser.cast<bool>();
                _use_subparser = _use;
                if(_use_subparser)
                {
                    // auto _parents         = py::list{};
                    auto _subcommand_args = py::kwargs{};
                    auto _subparser_args  = py::kwargs{};

                    _subcommand_args["help"] = "sub-command help";
                    auto _subparser = parser.attr("add_subparsers")(**_subcommand_args);

                    // _parents.append(parser);
                    _subparser_args["description"] = "Configure settings for timemory";
                    _subparser_args["conflict_handler"] = "resolve";
                    // _subparser_args["parents"]          = _parents;
                    _subparser_args["formatter_class"] =
                        pyargparse.attr("ArgumentDefaultsHelpFormatter");
                    _parser = _subparser.attr("add_parser")("timemory-config",
                                                            **_subparser_args);
                }
            } catch(py::cast_error&)
            {
                _parser = subparser;
            }
        }

        TIMEMORY_DEBUG_PRINT_HERE("%s", "starting loop");
        for(const auto& itr : _obj->ordering())
        {
            TIMEMORY_DEBUG_PRINT_HERE("searching for %s", itr.c_str());
            auto sitr = _obj->find(itr);
            if(sitr != _obj->end() && sitr->second)
            {
                TIMEMORY_DEBUG_PRINT_HERE("adding args for %s", sitr->first.data());
                bool _add = true;
                if(!fields.empty())
                {
                    _add = false;
                    for(auto& fitr : fields)
                    {
                        if(sitr->second->matches(fitr, false))
                            _add = true;
                    }
                }
                if(_add)
                    _argparse(sitr->second, _parser, _use_subparser);
            }
        }

        std::string cmdline = (_use_subparser) ? "--args" : "--timemory-args";
        auto        args    = std::vector<std::string>({ cmdline });

        py::cpp_function _timemory_args = [cmdline, _obj](py::list val) {
            auto vopt = std::vector<std::string>{};
            for(auto itr : val)
            {
                try
                {
                    vopt.emplace_back(itr.cast<std::string>());
                } catch(py::cast_error& e)
                {
                    std::cerr << "[timemory_argparse]> Warning! " << e.what()
                              << std::endl;
                }
            }
            for(auto& str : vopt)
            {
                // get the args
                auto vec = tim::delimit(str, " \t;:");
                for(auto itr : vec)
                {
                    TIMEMORY_DEBUG_PRINT_HERE("Processing: %s", itr.c_str());
                    auto _pos = itr.find('=');
                    auto _key = itr.substr(0, _pos);
                    auto _val = (_pos == std::string::npos) ? "" : itr.substr(_pos + 1);
                    if(!_obj->update(_key, _val, false))
                    {
                        std::cerr << "[timemory_argparse]> Warning! For " << cmdline
                                  << ", key \"" << _key
                                  << "\" is not a recognized setting. \"" << _val
                                  << "\" was not applied." << std::endl;
                    }
                }
            }
        };

        auto descript = TIMEMORY_JOIN(
            " ",
            "A generic option for any setting. Each argument MUST be passed in form: "
            "'NAME=VALUE'. E.g.",
            cmdline, "\"papi_events=PAPI_TOT_INS,PAPI_TOT_CYC\" text_output=off");

        auto locals = py::dict("parser"_a = _parser, "args"_a = args, "desc"_a = descript,
                               "parse_action"_a = _timemory_args);
        py::exec(R"(
             def make_action(func):
                 class customAction(argparse.Action):
                     def __call__(self, parser, args, values, option_string=None):
                         if type(values) is list:
                             func(values)
                         else:
                             func(["{}".format(values)])
                         setattr(args, self.dest, values)
                 return customAction

             _args = []
             _kwargs = {}
             for itr in args:
                _args.append(itr)

             _kwargs["help"] = "{} [type: str]".format(desc)
             _kwargs["required"] = False
             _kwargs["action"] = make_action(parse_action)
             _kwargs["metavar"] = "OPTION=VALUE"
             _kwargs["type"] = str
             _kwargs["nargs"] = "*"

             parser.add_argument(*_args, **_kwargs)
             )",
                 py::globals(), locals);
    };

    auto _dump = [](std::string _fname, tim::settings* _settings) {
        if(_settings == nullptr)
            _settings = tim::settings::instance();
        if(_settings == nullptr)
            throw std::runtime_error("Nullptr to settings");
        std::stringstream _ss{};
        {
            auto _ar = tim::policy::output_archive<tim::cereal::PrettyJSONOutputArchive,
                                                   TIMEMORY_API>::get(_ss);
            tim::settings::serialize_settings(*_ar, *_settings);
        }
        if(!_fname.empty())
        {
            _fname = tim::settings::compose_output_filename(_fname, ".json");
            std::ofstream _ofs{};
            if(!tim::filepath::open(_ofs, _fname))
                throw std::runtime_error("Error opening " + _fname + " for output");
            if(tim::settings::verbose() > 0)
                fprintf(stderr, "[%s][%s]|%i> Outputting '%s'...\n",
                        TIMEMORY_PROJECT_NAME, "settings", tim::dmp::rank(),
                        _fname.c_str());
            _ofs << _ss.str() << "\n";
        }
        return py::module::import("json").attr("loads")(_ss.str());
    };

    // create an instance
    settings.def(py::init(_init), "Create a copy of the global settings");
    // to parse changes in env vars
    settings.def_static("parse", _parse,
                        "Update the values of the settings from the current environment",
                        py::arg("instance") = py::none{});
    settings.def_static("read", _read, "Read the settings from JSON or text file");
    settings.def_static("add_argparse", _args, "Add command-line argument support",
                        py::arg("parser"), py::arg("instance") = py::none{},
                        py::arg("subparser") = true,
                        py::arg("selection") = std::vector<std::string>{});
    settings.def_static("add_arguments", _args, "Add command-line argument support",
                        py::arg("parser"), py::arg("instance") = py::none{},
                        py::arg("subparser") = true,
                        py::arg("selection") = std::vector<std::string>{});
    settings.def_static(
        "dump", _dump,
        "Dump settings to a file (if filename provided). Returns dictionary of settings",
        py::arg("filename") = std::string{}, py::arg("instance") = nullptr);
    settings.def_static(
        "compose_output_filename",
        [](std::string _base, std::string _ext, bool _dmp, tim::process::id_t _rank,
           bool _make_dir, std::string _explicit, std::string _subdir) {
            return tim::settings::compose_output_filename(
                _base, _ext,
                tim::settings::compose_filename_config{ _dmp, _rank, _make_dir, _explicit,
                                                        _subdir });
        },
        "Generate an output filename based on output path and output "
        "prefix settings. Use make_dir=True to ensure directory exists. "
        "Use explicit to specify an explicit output directory",
        py::arg("basename"), py::arg("extension"),
        py::arg("dmp")      = tim::settings::use_output_suffix(),
        py::arg("rank")     = tim::settings::default_process_suffix(),
        py::arg("make_dir") = false, py::arg("explicit") = std::string{},
        py::arg("subdirectory") = std::string{});
    settings.def_static(
        "compose_input_filename",
        [](std::string _base, std::string _ext, bool _dmp, tim::process::id_t _rank,
           std::string _explicit, std::string _subdir) {
            return tim::settings::compose_input_filename(
                _base, _ext,
                tim::settings::compose_filename_config{ _dmp, _rank, false, _explicit,
                                                        _subdir });
        },
        "Generate an input filename based on input path and input "
        "prefix settings. Use explicit to specify an explicit input directory",
        py::arg("basename"), py::arg("extension"),
        py::arg("dmp")      = tim::settings::use_output_suffix(),
        py::arg("rank")     = tim::settings::default_process_suffix(),
        py::arg("explicit") = std::string{}, py::arg("subdirectory") = std::string{});

    std::set<std::string> names{};
    auto                  _settings = tim::settings::instance<TIMEMORY_API>();
    if(_settings)
    {
        for(auto& itr : *_settings)
        {
            if(!add_property(settings, itr.second, tim::settings::data_type_list_t{}))
                names.insert(itr.second->get_name());
        }
        if(names.size() > 0)
        {
            std::stringstream ss, msg;
            for(auto& itr : names)
                ss << ", " << itr;
            msg << "Warning! The following settings were not added to python: "
                << ss.str().substr(2);
            std::cerr << msg.str() << std::endl;
        }

        using strvector_t = std::vector<std::string>;
        SETTING_PROPERTY(strvector_t, command_line);
        SETTING_PROPERTY(strvector_t, environment);
    }

    return settings;
}
}  // namespace pysettings
//
//======================================================================================//
