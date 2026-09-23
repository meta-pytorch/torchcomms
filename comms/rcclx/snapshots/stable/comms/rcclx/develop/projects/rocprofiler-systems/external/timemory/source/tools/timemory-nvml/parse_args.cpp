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
#include "timemory/utility/argparse.hpp"

//--------------------------------------------------------------------------------------//

nvml_config&
get_config()
{
    static nvml_config _instance{};
    return _instance;
}

//--------------------------------------------------------------------------------------//

void
parse_args_and_configure(int& argc, char**& argv)
{
    // other settings
    tim::settings::auto_output() = false;
    tim::settings::file_output() = false;
    tim::settings::ctest_notes() = false;
    tim::settings::scientific()  = false;
    tim::settings::width()       = 16;
    tim::settings::precision()   = 6;
    tim::settings::enabled()     = true;
    // ensure manager never writes metadata
    tim::manager::instance()->set_write_metadata(-1);

    using parser_t     = tim::argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    auto help_check = [](parser_t& p, int _argc, char** _argv) {
        std::set<std::string> help_args = { "-h", "--help", "-?" };
        return (p.exists("help") ||
                (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
    };

    auto _pec        = EXIT_SUCCESS;
    auto help_action = [&_pec, argc, argv](parser_t& p) {
        if(_pec != EXIT_SUCCESS)
        {
            stringstream_t msg;
            msg << "Error in command:";
            for(int i = 0; i < argc; ++i)
                msg << " " << argv[i];
            msg << "\n\n";
            std::cerr << msg.str() << std::flush;
        }

        stringstream_t hs;
        hs << "-- <CMD> <ARGS>\n";
        p.print_help(hs.str());
        exit(_pec);
    };

    auto parser = parser_t{ argv[0] };

    parser.enable_help();
    parser.on_error([=, &_pec](parser_t& p, const parser_err_t& _err) {
        std::cerr << _err << std::endl;
        _pec = EXIT_FAILURE;
        help_action(p);
    });

    parser.add_argument()
        .names({ "--debug" })
        .description("Debug output (env: TIMEMORY_NVML_DEBUG)")
        .count(0)
        .action([](parser_t&) { debug() = true; });
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output (env: TIMEMORY_NVML_VERBOSE)")
        .max_count(1)
        .action([](parser_t& p) {
            if(p.get_count("verbose") == 0)
            {
                verbose() = 1;
            }
            else
            {
                verbose() = p.get<int>("verbose");
            }
        });
    parser.add_argument({ "-q", "--quiet" }, "Suppress as much reporting as possible")
        .count(0)
        .action([](parser_t&) {
            tim::settings::verbose() = verbose() = -1;
            tim::settings::debug() = debug() = false;
        });
    parser
        .add_argument({ "-c", "--buffer-count" },
                      "Buffer count (env: TIMEMORY_NVML_BUFFER_COUNT)")
        .count(1)
        .dtype("integer")
        .action([](parser_t& p) { buffer_count() = p.get<size_t>("buffer-count"); });
    parser
        .add_argument({ "-m", "--max-samples" },
                      "Maximum samples (env: TIMEMORY_NVML_MAX_SAMPLES)")
        .count(1)
        .dtype("integer")
        .action([](parser_t& p) { max_samples() = p.get<size_t>("max-samples"); });
    parser
        .add_argument({ "-i", "--dump-interval" },
                      "Dump interval (env: TIMEMORY_NVML_DUMP_INTERVAL)")
        .count(1)
        .dtype("integer")
        .action([](parser_t& p) { dump_interval() = p.get<size_t>("dump-interval"); });
    parser
        .add_argument({ "-s", "--sample-interval" },
                      "Sample interval (env: TIMEMORY_NVML_SAMPLE_INTERVAL)")
        .count(1)
        .dtype("double")
        .action(
            [](parser_t& p) { sample_interval() = p.get<double>("sample-interval"); });
    parser
        .add_argument({ "-f", "--time-format" },
                      "strftime format for labels (env: TIMEMORY_NVML_TIME_FORMAT)")
        .count(1)
        .dtype("string")
        .action([](parser_t& p) { time_format() = p.get<std::string>("time-format"); });
    parser
        .add_argument({ "-o", "--output" },
                      // indented 35 spaces
                      R"(Write results to JSON output file (env: TIMEMORY_NVML_OUTPUT).
%{INDENT}% Use:
%{INDENT}% - '%p' to encode the process ID
%{INDENT}% - '%j' to encode the SLURM job ID
%{INDENT}% E.g. '-o timemory-nvml-output-%p'.)")
        .dtype("string")
        .max_count(1);

    auto _args = parser.parse_known_args(argc, argv);
    auto _argc = std::get<1>(_args);
    auto _argv = std::get<2>(_args);

    if(help_check(parser, _argc, _argv))
        help_action(parser);

    if(parser.exists("output"))
    {
        auto ofname = parser.get<std::string>("output");
        if(ofname.empty())
        {
            auto _cmd = std::string{ _argv[(_argc > 1) ? 1 : 0] };
            auto _pos = _cmd.find_last_of('/');
            if(_pos != std::string::npos)
                _cmd = _cmd.substr(_pos + 1);
            ofname = TIMEMORY_JOIN("", argv[0], "-output", '/', _cmd);
            fprintf(stderr, "[%s]> No output filename provided. Using '%s'...\n", argv[0],
                    ofname.c_str());
        }
        output_file()   = ofname;
        auto output_dir = output_file().substr(0, output_file().find_last_of('/'));
        if(output_dir != output_file())
            tim::makedir(output_dir);
    }

    // parse for settings configurations
    if(argc > 1)
        tim::timemory_init(argc, argv);

    // override a some settings
    tim::settings::auto_output()   = false;
    tim::settings::output_prefix() = "";

    for(int i = 0; i < _argc; ++i)
        argvector().emplace_back(_argv[i]);

    argc = _argc;
    argv = _argv;
}

//--------------------------------------------------------------------------------------//

std::string
nvml_config::get_output_filename(std::string inp)
{
    if(inp.empty())
        inp = output_file;

    auto _replace = [](std::string& _inp, const std::string& _key,
                       const std::string& _sub) {
        auto pos = std::string::npos;
        while((pos = _inp.find(_key)) != std::string::npos)
            _inp = _inp.replace(pos, _key.length(), _sub);
    };

    using pair_t = std::pair<std::string, int64_t>;
    for(const auto& itr :
        { pair_t{ "%p", getpid() }, pair_t{ "%j", tim::get_env("SLURM_JOB_ID", 0) } })
    {
        _replace(inp, itr.first, std::to_string(itr.second));
    }
    return inp;
}

//--------------------------------------------------------------------------------------//
