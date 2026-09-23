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

#ifndef TIMEMORY_CONFIG_CONFIG_CPP_
#define TIMEMORY_CONFIG_CONFIG_CPP_

#include "timemory/config/macros.hpp"
#include "timemory/defines.h"

#if defined(TIMEMORY_CONFIG_SOURCE) || !defined(TIMEMORY_USE_CONFIG_EXTERN)
//
#    include "timemory/backends/process.hpp"
#    include "timemory/config/config.hpp"
#    include "timemory/config/types.hpp"
#    include "timemory/log/logger.hpp"
#    include "timemory/manager/declaration.hpp"
#    include "timemory/mpl/filters.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/settings/types.hpp"
#    include "timemory/signals/signal_handlers.hpp"
#    include "timemory/utility/argparse.hpp"
#    include "timemory/utility/delimit.hpp"
#    include "timemory/utility/signals.hpp"
#    include "timemory/utility/utility.hpp"

#    include <fstream>
#    include <string>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              config
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(int argc, char** argv, const std::string& _prefix,
              const std::string& _suffix)
{
    using signal_settings = signals::signal_settings;
    using sys_signal      = signals::sys_signal;

    auto _settings = settings::shared_instance();
    auto _manager  = manager::instance();

    if(_settings)
    {
        settings::store_command_line(argc, argv);
        if(_settings->get_debug() || _settings->get_verbose() > 3)
            TIMEMORY_PRINT_HERE("%s", "");

        if(_settings->get_enable_signal_handler())
        {
            auto default_signals = signal_settings::get_default();
            for(const auto& itr : default_signals)
                signal_settings::enable(itr);
            // should return default and any modifications from environment
            auto enabled_signals = signal_settings::get_enabled();
            enable_signal_detection(enabled_signals);
        }

        _settings->init_config();
    }

    {
        std::string exe_name = (argc > 0) ? argv[0] : "";

        while(exe_name.find('\\') != std::string::npos)
            exe_name = exe_name.substr(exe_name.find_last_of('\\') + 1);

        while(exe_name.find('/') != std::string::npos)
            exe_name = exe_name.substr(exe_name.find_last_of('/') + 1);

        static const std::vector<std::string> _exe_suffixes = { ".py", ".exe" };
        for(const auto& ext : _exe_suffixes)
        {
            if(exe_name.find(ext) != std::string::npos)
                exe_name.erase(exe_name.find(ext), ext.length() + 1);
        }

        if(_settings)
            _settings->set_tag(exe_name);
    }

    if(_settings)
    {
        auto _opitr = _settings->find("output_path");
        auto _outpath =
            (_opitr != _settings->end()) ? _opitr->second : std::shared_ptr<vsettings>{};
        if(!_outpath->get_config_updated() && !_outpath->get_environ_updated())
        {
            auto _remove_double_hyphen = [](std::string _v) {
                size_t pos = std::string::npos;
                while((pos = _v.find("--")) != std::string::npos)
                    _v.erase(pos, 1);
                return _v;
            };
            _settings->set("output_path",
                           _remove_double_hyphen(_prefix + "-%tag%-" + _suffix), false);
        }

        // allow environment overrides
        settings::parse(_settings);

        auto _unknown = _settings->get_unknown_configs();
        for(const auto& itr : _unknown)
        {
            auto _v = _settings->find(itr.first);
            if(_v == _settings->end())
                set_env(itr.first, itr.second, 0);
        }

        if(_settings->get_enable_signal_handler())
        {
            auto _exit_action = [](int nsig) {
                static bool _protect = false;
                if(_protect)
                    return;
                _protect           = true;
                auto _main_manager = manager::master_instance();
                if(_main_manager)
                {
                    std::cout << "Finalizing after signal: " << nsig << " :: "
                              << signal_settings::str(static_cast<sys_signal>(nsig))
                              << std::endl;
                    _main_manager->finalize();
                }
                _protect = false;
            };
            signal_settings::set_exit_action(_exit_action);
        }
    }

    if(_settings)
        _settings->set_initialized(true);

    if(_manager)
    {
        _manager->update_metadata_prefix();
        _manager->initialize();
    }

    if(!get_shared_ptr_pair_callback())
    {
        get_shared_ptr_pair_callback() =
            new shared_ptr_pair_callback_t{ [_manager, _settings](int64_t _n) {
                if(_n == 0)
                    timemory_finalize(_manager.get(), _settings.get());
            } };
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(const std::string& exe_name, const std::string& _prefix,
              const std::string& _suffix)
{
    auto* _cstr = const_cast<char*>(exe_name.c_str());
    auto  _argc = 1;
    auto* _argv = &_cstr;
    timemory_init(_argc, _argv, _prefix, _suffix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(int* argc, char*** argv, const std::string& _prefix,
              const std::string& _suffix)
{
    if(settings::mpi_init())
    {
        if(settings::debug())
            TIMEMORY_PRINT_HERE("%s", "initializing mpi");

        mpi::initialize(argc, argv);
    }

    if(settings::upcxx_init())
    {
        if(settings::debug())
            TIMEMORY_PRINT_HERE("%s", "initializing upcxx");

        upc::initialize();
    }

    timemory_init(*argc, *argv, _prefix, _suffix);
    timemory_argparse(argc, argv);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(int* argc, char*** argv, argparse::argument_parser& _parser,
              const std::string& _prefix, const std::string& _suffix)
{
    if(settings::mpi_init())
    {
        if(settings::debug())
            TIMEMORY_PRINT_HERE("%s", "initializing mpi");

        mpi::initialize(argc, argv);
    }

    if(settings::upcxx_init())
    {
        if(settings::debug())
            TIMEMORY_PRINT_HERE("%s", "initializing upcxx");

        upc::initialize();
    }

    timemory_init(*argc, *argv, _prefix, _suffix);
    timemory_argparse(argc, argv, &_parser);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(std::vector<std::string>& args, argparse::argument_parser& _parser,
              const std::string& _prefix, const std::string& _suffix)
{
    int    argc = args.size();
    char** argv = new char*[argc];
    for(int i = 0; i < argc; ++i)
    {
        auto len = args.at(i).length();
        argv[i]  = new char[len + 1];
        strcpy(argv[i], args.at(i).c_str());
        argv[i][len] = '\0';
    }

    if(settings::mpi_init())
    {
        if(settings::debug())
            TIMEMORY_PRINT_HERE("%s", "initializing mpi");

        mpi::initialize(argc, argv);
    }

    if(settings::upcxx_init())
    {
        if(settings::debug())
            TIMEMORY_PRINT_HERE("%s", "initializing upcxx");

        upc::initialize();
    }

    timemory_init(argc, argv, _prefix, _suffix);
    timemory_argparse(args, &_parser);

    for(int i = 0; i < argc; ++i)
        delete[] argv[i];
    delete[] argv;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_argparse(int* argc, char*** argv, argparse::argument_parser* _parser,
                  tim::settings* _settings)
{
    using parser_t = argparse::argument_parser;

    // if pointers are null, return
    if(!argc || !argv)
        return;

    // if only one argument, return
    if(*argc < 2)
        return;

    // if settings instance was not provided
    auto _shared_settings = tim::settings::shared_instance();
    if(_settings == nullptr)
        _settings = _shared_settings.get();

    if(!_settings)
        return;

    // if argument parser was not provided
    parser_t _local_parser{ (*argv)[0] };
    _local_parser.enable_help();
    if(_parser == nullptr)
        _parser = &_local_parser;

    if(!_parser)
        return;

    // add the arguments in the order they were appended to the settings
    for(const auto& itr : _settings->ordering())
    {
        auto sitr = _settings->find(itr);
        if(sitr != _settings->end() && sitr->second)
        {
            sitr->second->add_argument(*_parser);
        }
    }

    _parser->add_argument()
        .names({ "--" TIMEMORY_PROJECT_NAME "-args" })
        .description("A generic option for any setting. Each argument MUST be passed in "
                     "form: 'NAME=VALUE'. E.g. --" TIMEMORY_PROJECT_NAME "-args "
                     "\"papi_events=PAPI_TOT_INS,PAPI_TOT_CYC\" text_output=off")
        .action([&](parser_t& p) {
            // get the options
            auto vopt = p.get<std::vector<std::string>>("timemory-args");
            for(auto& str : vopt)
            {
                // get the args
                auto vec = tim::delimit(str, " \t;:");
                for(const auto& itr : vec)
                {
                    TIMEMORY_DEBUG_PRINT_HERE("Processing: %s", itr.c_str());
                    auto _pos = itr.find('=');
                    auto _key = itr.substr(0, _pos);
                    auto _val = (_pos == std::string::npos) ? "" : itr.substr(_pos + 1);
                    if(!_settings->update(_key, _val, false))
                    {
                        log::stream(std::cerr, log::color::warning())
                            << "[" << TIMEMORY_PROJECT_NAME
                            << "][argparse]> Warning! For "
                               "--" TIMEMORY_PROJECT_NAME "-args, key \""
                            << _key << "\" is not a recognized setting. \"" << _val
                            << "\" was not applied.\n";
                    }
                }
            }
        });

    _parser->parse_known_args(argc, argv, "--", settings::verbose());
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_argparse(std::vector<std::string>& args, argparse::argument_parser* _parser,
                  tim::settings* _settings)
{
    int    argc = args.size();
    char** argv = new char*[argc];
    for(int i = 0; i < argc; ++i)
    {
        // intentional memory leak
        auto len = args.at(i).length();
        argv[i]  = new char[len + 1];
        strcpy(argv[i], args.at(i).c_str());
        argv[i][len] = '\0';
    }

    timemory_argparse(&argc, &argv, _parser, _settings);

    // updates args
    args.clear();
    for(int i = 0; i < argc; ++i)
        args.emplace_back(argv[i]);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)  // NOLINT
timemory_finalize(manager* _manager, settings* _settings, bool _lookup)
{
    if(_lookup)
    {
        if(!_settings)
            _settings = settings::instance();
        if(!_manager)
            _manager = manager::instance().get();
    }

    if(_settings)
        _settings->set_initialized(false);

    if(_manager)
    {
        if(_settings && _settings->get_debug())
            TIMEMORY_PRINT_HERE("%s", "finalizing manager");
        _manager->finalize();
    }

    if(_settings)
    {
        if(_settings->get_upcxx_finalize())
        {
            if(_settings->get_debug())
                TIMEMORY_PRINT_HERE("%s", "finalizing upcxx");

            upc::finalize();
        }

        if(_settings->get_mpi_finalize())
        {
            if(_settings->get_debug())
                TIMEMORY_PRINT_HERE("%s", "finalizing mpi");
            mpi::finalize();
        }

        if(_settings->get_debug())
            TIMEMORY_PRINT_HERE("%s", "timemory finalized");

        _settings->set_finalized(true);
    }
}
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#endif  // defined(TIMEMORY_CONFIG_SOURCE) || !defined(TIMEMORY_USE_CONFIG_EXTERN)

#endif
