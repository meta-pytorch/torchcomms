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

#include "timem.hpp"

#include "timemory/backends/process.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/log/color.hpp"
#include "timemory/utility/argparse.hpp"

#include <csignal>

//--------------------------------------------------------------------------------------//

namespace
{
std::exception_ptr global_exception_pointer = nullptr;
void
forward_signal(int);
void
handle_exception();
void
store_history(timem_bundle_t*);
void
childpid_catcher(int);
void
parent_process(pid_t pid);
void
child_process(int argc, char** argv) TIMEMORY_ATTRIBUTE(noreturn);
bool&
timem_mpi_was_finalized();
void
create_signal_handler(int sig, signal_handler& sh, void (*func)(int));
void
restore_signal_handler(int sig, signal_handler& sh);

pid_t read_pid(pid_t);
}  // namespace

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    // disable banner if not specified
    // setenv("TIMEMORY_BANNER", "OFF", 0);
    master_pid() = getpid();

    // parallel settings
    tim::settings::mpi_init()       = false;
    tim::settings::mpi_finalize()   = false;
    tim::settings::upcxx_init()     = false;
    tim::settings::upcxx_finalize() = false;
    // other settings
    tim::settings::banner()      = false;
    tim::settings::auto_output() = false;
    tim::settings::file_output() = false;
    tim::settings::ctest_notes() = false;
    tim::settings::scientific()  = false;
    tim::settings::width()       = 16;
    tim::settings::precision()   = 6;
    tim::settings::enabled()     = true;
    // ensure manager never writes metadata
    tim::manager::instance()->set_write_metadata(-1);
    // disable network stats by default
    tim::trait::apply<tim::trait::runtime_enabled>::set<network_stats>(false);

    using parser_t     = tim::argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    auto help_check = [](parser_t& p, int _argc, char** _argv) {
        std::set<std::string> help_args = { "-h", "--help", "-?" };
        return (p.exists("help") || _argc == 1 ||
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

        if(tim::dmp::rank() == 0)
        {
            stringstream_t hs;
            hs << "-- <CMD> <ARGS>\n\n";
            hs << "Examples:\n";
            hs << "    timem sleep 2\n";
            hs << "    timem -s /bin/zsh -- find /usr\n";
            hs << "    timemory-avail -H | grep PAPI | grep -i cache\n";
            hs << "    srun -N 1 -n 1 timem -e PAPI_L1_TCM PAPI_L2_TCM PAPI_L3_TCM -- "
                  "./myexe\n";
            p.print_help(hs.str());
        }
        exit(_pec);
    };

    auto parser = parser_t(argv[0]);

    parser.enable_help();
    parser.on_error([=, &_pec](parser_t& p, const parser_err_t& _err) {
        std::cerr << _err << std::endl;
        _pec = EXIT_FAILURE;
        help_action(p);
    });

    parser.add_argument()
        .names({ "--debug" })
        .description("Debug output")
        .count(0)
        .action([](parser_t&) {
            debug() = true;
            tim::signals::enable_signal_detection();
        });
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output")
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
    parser.add_argument({ "-N", "--monochrome" }, "Disable colorized output")
        .max_count(1)
        .dtype("bool")
        .action([](parser_t& p) { tim::log::monochrome() = p.get<bool>("monochrome"); });
    parser.add_argument({ "-q", "--quiet" }, "Suppress as much reporting as possible")
        .count(0)
        .action([](parser_t&) {
            tim::settings::verbose()    = -1;
            tim::settings::debug()      = false;
            tim::settings::papi_quiet() = true;
            verbose()                   = -1;
            debug()                     = false;
        });
    parser
        .add_argument({ "-d", "--sample-delay" },
                      "Set the delay before the sampler starts (seconds)")
        .count(1)
        .action([](parser_t& p) { sample_delay() = p.get<double>("sample-delay"); });
    parser
        .add_argument({ "-f", "--sample-freq" }, "Set the frequency of the sampler "
                                                 "(number of interrupts per second)")
        .count(1)
        .action([](parser_t& p) {
            sample_freq() = p.get<double>("sample-freq");
            if(sample_freq() <= 0.0)
                use_sample() = false;
        });
    parser
        .add_argument(
            { "--disable-sample", "--disable-sampling" },
            "Disable UNIX signal-based sampling. Sampling is the most common culprit for "
            "timem hanging (i.e. failing to exit after the child process exits)")
        .count(0)
        .action([](parser_t&) { use_sample() = false; });
    parser
        .add_argument({ "-b", "--buffer-size" },
                      "If set to value > 0, timem will record a history of every sample. "
                      "This requires spawning an extra thread which will periodically "
                      "wake and flush the buffer.")
        .max_count(1)
        .dtype("size_t")
        .set_default(buffer_size())
        .action([](parser_t& p) {
            if(p.get_count("buffer-size") == 0)
            {
                buffer_size() = 500;
            }
            else
            {
                buffer_size() = p.get<size_t>("buffer-size");
            }
        });
    parser.add_argument({ "-e", "--events", "--papi-events" },
                        "Set the hardware counter events to record (ref: `timemory-avail "
                        "-H | grep PAPI`)");
    parser.add_argument({ "--disable-papi" }, "Disable hardware counters")
        .count(0)
        .action([](parser_t&) {
            use_papi() = false;
            tim::trait::runtime_enabled<papi_array_t>::set(false);
        });
    parser
        .add_argument({ "-o", "--output" },
                      // indented 35 spaces
                      R"(Write results to JSON output file.
%{INDENT}%Use:
%{INDENT}%- '%m' to encode md5sum of command line
%{INDENT}%- '%p' to encode the process ID
%{INDENT}%- '%j' to encode the SLURM job ID
%{INDENT}%- '%r' to encode the MPI comm rank
%{INDENT}%- '%s' to encode the MPI comm size
%{INDENT}%E.g. '-o timem-output-%p'.
%{INDENT}%If verbosity >= 2 or debugging is enabled, will also write sampling data to log file.)")
        .max_count(1);
    parser
        .add_argument({ "-s", "--shell" }, "Enable launching command via a shell command "
                                           "(if no arguments, $SHELL is used)")
        .max_count(1)
        .action([](parser_t& p) {
            use_shell() = true;
            if(p.get_count("shell") > 0)
                shell() = p.get<std::string>("shell");
        });
    parser
        .add_argument({ "--shell-flags" },
                      "Set the shell flags to use (pass as single "
                      "string as leading dashes can confuse parser) [default: -i]")
        .count(1)
        .action([](parser_t& p) { shell_flags() = p.get<std::string>("shell-flags"); });
    parser.add_argument()
        .names({ "--network-stats" })
        .description("Enable sampling network usage statistics (Linux only)")
        .max_count(1)
        .action([&](parser_t& p) {
            tim::trait::apply<tim::trait::runtime_enabled>::set<network_stats>(
                p.get<bool>("network-stats"));
        });
    parser.add_argument()
        .names({ "-F", "--forward-signal" })
        .description(
            "If any of these signals are sent to timem, forward them to the process. "
            "E.g. '--forward-signal 2' (default behavior) will cause timem to forward "
            "SIGINT to the process if (Cntl+C) is sent by the user. Use "
            "'--forward-signal 0' to disable this behavior.")
        .dtype("int")
        .min_count(1)
        .action([&](parser_t& p) {
            signal_forward() = p.get<std::set<int>>("forward-signal");
        });
#if defined(TIMEMORY_USE_MPI)
    parser
        .add_argument({ "--mpi" }, "Launch processes via MPI_Comm_spawn_multiple "
                                   "(reduced functionality)")
        .dtype("bool")
        .max_count(1)
        .action([](parser_t& p) { use_mpi() = p.get<bool>("mpi"); });
    parser.add_argument({ "--disable-mpi" }, "Disable MPI_Finalize")
        .max_count(1)
        .action(
            [](parser_t& p) { timem_mpi_was_finalized() = p.get<bool>("disable-mpi"); });
    parser
        .add_argument({ "-i", "--indiv" },
                      "Output individual results for each process (i.e. rank) instead of "
                      "reporting the aggregation")
        .dtype("bool")
        .max_count(1)
        .action([](parser_t& p) {
            tim::settings::collapse_processes() = !p.get<bool>("indiv");
        });
#endif

    auto  _args = parser.parse_known_args(argc, argv);
    auto  _argc = std::get<1>(_args);
    auto* _argv = std::get<2>(_args);

    if(help_check(parser, _argc, _argv))
        help_action(parser);

    if(!debug() && verbose() < 1)
        tim::settings::papi_quiet() = true;

    // make sure config is instantiated
    tim::consume_parameters(get_config());

    // sample_delay() = std::max<double>(sample_delay(), 1.0e-6);
    sample_freq() = std::min<double>(sample_freq(), 5000.);

    if(parser.exists("events"))
    {
        if(!tim::trait::is_available<papi_array_t>::value)
            throw std::runtime_error("Error! timemory was not built with PAPI support");

        auto           evts = parser.get<std::vector<std::string>>("events");
        stringstream_t ss;
        for(const auto& itr : evts)
            ss << itr << ",";
        tim::settings::papi_events() = ss.str().substr(0, ss.str().length() - 1);
    }

    // parse for settings configurations
    if(argc > 1)
    {
        if(use_mpi())
            tim::mpi::initialize(&argc, &argv);

        tim::timemory_init(argc, argv);
    }

    // override a some settings
    tim::settings::suppress_parsing() = true;
    tim::settings::papi_attach()      = true;
    tim::settings::papi_threading()   = false;
    tim::settings::auto_output()      = false;
    tim::settings::output_prefix()    = "";

    auto compose_prefix = [&]() {
        stringstream_t ss;
#if defined(TIMEMORY_USE_MPI)
        ss << "[" << command().c_str() << "]";
        if(!use_mpi())
            ss << "[PID=" << tim::process::get_id() << "]";
        ss << "> Measurement totals";
#else
        ss << "[" << command().c_str() << "]> Measurement totals";
#endif
        if(use_mpi())
        {
            ss << " (# ranks = " << tim::mpi::size() << "):";
        }
        else if(tim::dmp::size() > 1)
        {
            ss << " (# ranks = " << tim::dmp::size() << "):";
        }
        else
        {
            ss << ":";
        }
        return ss.str();
    };

    if(_argc > 1)
    {
        // e.g. timem mycmd
        command() = std::string(const_cast<const char*>(_argv[1]));
    }
    else
    {
        command()              = std::string(const_cast<const char*>(_argv[0]));
        tim::get_rusage_type() = RUSAGE_CHILDREN;
        exit(EXIT_SUCCESS);
    }

    if(parser.exists("output"))
    {
        auto ofname = parser.get<string_t>("output");
        if(ofname.empty())
        {
            auto _cmd = command();
            auto _pos = _cmd.find_last_of('/');
            if(_pos != std::string::npos)
                _cmd = _cmd.substr(_pos + 1);
            ofname = TIMEMORY_JOIN("", argv[0], "-output", '/', _cmd);
            fprintf(stderr, "[%s]> No output filename provided. Using '%s'...\n",
                    command().c_str(), ofname.c_str());
        }
        output_file()   = ofname;
        auto output_dir = output_file().substr(0, output_file().find_last_of('/'));
        if(output_dir != output_file())
            tim::makedir(output_dir);
    }

    if(tim::settings::papi_events().empty())
    {
        if(use_papi())
        {
            tim::settings::papi_events() = "PAPI_TOT_CYC";
        }
        else
        {
            tim::trait::runtime_enabled<papi_array_t>::set(false);
        }
    }

    tim::get_rusage_type() = RUSAGE_CHILDREN;
    if(!use_sample())
        signal_types().clear();

    for(auto itr : signal_forward())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 0),
                                        "timem will forward signal %i to its worker "
                                        "process if it is sent to this process (PID: %i)",
                                        itr, (int) tim::process::get_id());
        if(signal_types().count(itr) > 0)
            throw std::runtime_error(
                TIMEMORY_JOIN(" ", "Error! timem sampler is using signal", itr,
                              "to handle the sampling measurements. Cannot forward it. "
                              "Re-run timem with the '--disable-sampling' option to "
                              "forward this signal"));
    }

    // set the signal handler on this process if using mpi so that we can read
    // the file providing the PID. If not, fork provides the PID so this is
    // unnecessary
    if(use_mpi())
    {
        create_signal_handler(TIMEM_PID_SIGNAL, get_signal_handler(TIMEM_PID_SIGNAL),
                              &childpid_catcher);
        // allocate the signal handlers in map
        for(auto itr : signal_forward())
            (void) get_signal_handler(itr);
    }
    else
    {
        for(auto itr : signal_forward())
            create_signal_handler(itr, get_signal_handler(itr), &forward_signal);
    }

    for(int i = 0; i < _argc; ++i)
        argvector().emplace_back(_argv[i]);

    //----------------------------------------------------------------------------------//
    //
    //          Create subprocesses
    //
    //----------------------------------------------------------------------------------//

    pid_t pid = (use_mpi()) ? master_pid() : fork();

    // SIGCHLD notifies the parent process when a child process exits, is interrupted, or
    // resumes after being interrupted
    // if(!use_mpi())
    //    signal(SIGCHLD, childpid_catcher);

    using comm_t        = tim::mpi::comm_t;
    comm_t comm_world_v = tim::mpi::comm_world_v;

    if(!use_sample() || signal_types().empty())
    {
        tim::trait::apply<tim::trait::runtime_enabled>::set<
            page_rss, virtual_memory, read_char, read_bytes, written_char, written_bytes,
            papi_array_t>(false);
    }

#if defined(TIMEMORY_USE_MPI)
    comm_t comm_child_v;
    //
    if(use_mpi())
    {
        tim::trait::apply<tim::trait::runtime_enabled>::set<
            child_user_clock, child_system_clock, child_cpu_clock, child_cpu_util,
            peak_rss, num_major_page_faults, num_minor_page_faults,
            priority_context_switch, voluntary_context_switch, papi_array_t>(false);

        using info_t      = tim::mpi::info_t;
        using argvector_t = tim::argparse::argument_vector;
        using cargs_t     = typename argvector_t::cargs_t;

        string_t pidexe = argv[0];
        // name of the executable should be some path + "timem"
        if(pidexe.substr(pidexe.find_last_of('/') + 1) == "timem")
        {
            // append "timem" + "ory-pid" to form "timemory-pid"
            pidexe += "ory-pid";
        }
        else if(pidexe.substr(pidexe.find_last_of('/') + 1) == "timem-mpi")
        {
            // remove "-mpi" -> "timem" + "ory-pid" to form "timemory-pid"
            pidexe = pidexe.substr(0, pidexe.length() - 4) + "ory-pid";
        }
        else
        {
            if(verbose() > -1)
            {
                fprintf(stderr,
                        "Warning! Executable '%s' was expected to be 'timem'. Using "
                        "'timemory-pid' instead of adding 'ory-pid' to name of "
                        "executable",
                        argv[0]);
            }
            // otherwise, assume it can find 'timemory-pid'
            pidexe = "timemory-pid";
        }

        auto comm_size  = tim::mpi::size();
        auto comm_rank  = tim::mpi::rank();
        auto pids       = std::vector<int>(comm_size, 0);
        auto procs      = std::vector<int>(comm_size, 1);
        auto errcodes   = std::vector<int>(comm_size, 0);
        auto infos      = std::vector<info_t>(comm_size, tim::mpi::info_null_v);
        auto margv      = std::vector<argvector_t>(comm_size, argvector_t(_argc, _argv));
        auto cargv      = std::vector<cargs_t>{};
        auto cargv_arg0 = std::vector<char*>{};
        auto cargv_argv = std::vector<char**>{};

        pids.at(comm_rank) = pid;
        tim::mpi::gather(&pid, 1, MPI_INT, pids.data(), 1, MPI_INT, 0,
                         tim::mpi::comm_world_v);

        if(debug() && comm_rank == 0)
        {
            std::stringstream ss;
            std::cout << "[" << command() << "]> parent pids: ";
            for(const auto& itr : pids)
                ss << ", " << itr;
            std::cout << ss.str().substr(2) << '\n';
        };

        // create the
        for(decltype(comm_size) i = 0; i < comm_size; ++i)
        {
            auto _cargv =
                margv.at(i).get_execv({ pidexe, std::to_string(pids.at(i)) }, 1);
            cargv.push_back(_cargv);
            cargv_arg0.push_back(_cargv.argv()[0]);
            cargv_argv.push_back(_cargv.argv() + 1);
            if(debug() && comm_rank == 0)
            {
                fprintf(stderr, "[%s][rank=%i]> cmd :: %s\n", command().c_str(), (int) i,
                        _cargv.args().c_str());
            }
        }

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        tim::mpi::barrier(comm_world_v);

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        tim::mpi::comm_spawn_multiple(comm_size, cargv_arg0.data(), cargv_argv.data(),
                                      procs.data(), infos.data(), 0, comm_world_v,
                                      &comm_child_v, errcodes.data());
        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");

        // clean up
        for(auto& itr : cargv)
            itr.clear();
        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
    }
#endif
    //
    //----------------------------------------------------------------------------------//
    TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");

    if(pid != 0)
    {
        if(use_mpi())
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
            if(!signal_delivered())
            {
                // wait for timemory-pid to signal
                pause();
            }
            worker_pid() = read_pid(master_pid());
        }
        else
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
            worker_pid() = pid;
        }

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        tim::process::get_target_id() = worker_pid();
        tim::settings::papi_attach()  = true;
        get_sampler()                 = new sampler_t{ compose_prefix() };
        if(use_sample() && !signal_types().empty())
        {
            get_measure()->set_buffer_size(buffer_size());
            buffer_thread() =
                std::make_unique<std::thread>(&store_history, get_measure());
        }
#if defined(TIMEMORY_USE_PAPI)
        if(use_papi())
            papi_array_t::configure();
#endif
    }

    auto failed_fork = [&]() {
        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        // pid == -1 means error occured
        bool cond = (pid == -1);
        if(pid == -1)
            puts("failure forking, error occured!");
        return cond;
    };

    auto is_child = [&]() {
        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        // pid == 0 means child process created if not using MPI
        bool cond = (pid == 0 && !use_mpi());
        return cond;
    };

    // exit code
    int ec = 0;

    if(failed_fork())
    {
        puts("Failure to fork");
        exit(EXIT_FAILURE);
    }
    else if(is_child())
    {
        if(!use_mpi())
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(
                (debug() && verbose() > 1),
                "waiting for signal to proceed (this=%i, parent=%i)", getpid(),
                master_pid());

            int      _sig = 0;
            sigset_t _wait;
            sigemptyset(&_wait);
            sigaddset(&_wait, SIGINT);
            sigwait(&_wait, &_sig);

            TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                            "proceeding (this=%i, parent=%i)", getpid(),
                                            master_pid());
        }

        child_process(_argc, _argv);
    }
    else
    {
        // output file
        auto ofs = std::unique_ptr<std::ofstream>{};

        // ensure always disabled
        tim::settings::enabled() = true;

        if(!output_file().empty() && (debug() || verbose() > 1))
        {
            auto fname = get_config().get_output_filename({}, ".txt");
            ofs        = std::make_unique<std::ofstream>(fname.c_str());
        }

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s",
                                        "configuring sampler");

        for(auto itr : signal_types())
        {
            get_sampler()->configure(tim::sampling::timer{
                itr, CLOCK_REALTIME, SIGEV_SIGNAL, sample_freq(), sample_delay() });
        }

        if(!use_mpi())
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                            "notifying worker (PID=%i) process is ready",
                                            worker_pid());
            kill(worker_pid(), SIGINT);
        }

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s",
                                        "starting sampler");
        get_sampler()->start();

        if((debug() && verbose() > 1) || verbose() > 2)
            std::cerr << "[AFTER START][" << pid << "]> " << *get_measure() << std::endl;

        if(ofs)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s",
                                            "Setting output file");
            get_measure()->set_output(ofs.get());
        }

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "target pid = %i",
                                        (int) worker_pid());
        auto status = get_sampler()->wait(worker_pid(), verbose(), debug());

        if((debug() && verbose() > 1) || verbose() > 2)
            std::cerr << "[BEFORE STOP][" << pid << "]> " << *get_measure() << std::endl;

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s",
                                        "stopping sampler");
        get_sampler()->stop();

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s",
                                        "ignoring signals");
        get_sampler()->ignore(signal_types());

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "barrier");
        tim::mpi::barrier(comm_world_v);

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "processing");
        parent_process(pid);

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "barrier");
        tim::mpi::barrier(tim::mpi::comm_world_v);

        TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "exit code = %i",
                                        status);
        ec = status;
    }

    delete get_sampler();

    TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "Completed");
    if(use_mpi() || (!timem_mpi_was_finalized() && tim::dmp::size() == 1))
        tim::mpi::finalize();

    return ec;
}

//--------------------------------------------------------------------------------------//

namespace
{
pid_t
read_pid(pid_t _master)
{
    pid_t _worker = 0;
    // get the child pid from a file
    {
        stringstream_t fname;
        fname << tim::get_env<std::string>("TMPDIR", "/tmp") << "/.timemory-pid-"
              << _master;
        std::ifstream ifs(fname.str().c_str());
        if(ifs)
        {
            ifs >> _worker;
        }
        else
        {
            fprintf(stderr, "Error opening '%s'...\n", fname.str().c_str());
        }
        ifs.close();
    }
    if(debug())
        printf("[%s]> pid :: %i -> %i\n", __FUNCTION__, _master, _worker);
    return _worker;
}

//--------------------------------------------------------------------------------------//

void
childpid_catcher(int sig)
{
    signal_delivered() = true;
    restore_signal_handler(sig, get_signal_handler(sig));
    // generate the signal handlers for signal forwarding
    for(auto itr : signal_forward())
        create_signal_handler(itr, get_signal_handler(itr), &forward_signal);
    int _worker                   = read_pid(master_pid());
    worker_pid()                  = _worker;
    tim::process::get_target_id() = _worker;
    if(debug())
    {
        printf("[%s][pid=%i]> worker_pid() = %i, worker = %i, target_id() = %i\n",
               __FUNCTION__, getpid(), worker_pid(), _worker,
               tim::process::get_target_id());
    }
}

//--------------------------------------------------------------------------------------//

void
forward_signal(int sig)
{
    kill(worker_pid(), sig);
}

//--------------------------------------------------------------------------------------//

void
handle_exception()
{
    if(global_exception_pointer)
    {
        try
        {
            std::rethrow_exception(global_exception_pointer);
        } catch(const std::exception& ex)
        {
            std::cerr << "Thread exited with exception: " << ex.what() << "\n";
            std::rethrow_exception(global_exception_pointer);
        }
    }
}

//--------------------------------------------------------------------------------------//

void
store_history(timem_bundle_t* _bundle)
{
    using hist_type = typename timem_bundle_t::hist_type;

    auto _scompleted = []() { return completed() ? "y" : "n"; };
    auto _sfullbuff  = []() { return full_buffer() ? "y" : "n"; };
    tim::consume_parameters(_scompleted, _sfullbuff);

    static thread_local bool* notify_completed = nullptr;
    try
    {
        _bundle->set_notify([](bool* _completed) {
            full_buffer()    = true;
            notify_completed = _completed;
            buffer_cv().notify_one();
        });
        while(!completed())
        {
            std::vector<hist_type> _buff{};
            _buff.reserve(_bundle->get_buffer_size());

            TIMEMORY_CONDITIONAL_PRINT_HERE(
                debug() && verbose() > 2,
                "thread entering wait. completed: %s, full buffer: "
                "%s, buffer: %zu, history: %zu",
                _scompleted(), _sfullbuff(), _buff.size(), history().size());

            auto_lock_t _lk{ type_mutex<hist_type>() };
            buffer_cv().wait(_lk, []() { return completed() || full_buffer(); });

            TIMEMORY_CONDITIONAL_PRINT_HERE(
                debug() && verbose() > 2,
                "thread swapping history. completed: %s, full buffer: "
                "%s, buffer: %zu, history: %zu",
                _scompleted(), _sfullbuff(), _buff.size(), history().size());

            if(_lk.owns_lock())
                _lk.unlock();
            _buff = _bundle->swap_history(_buff);

            TIMEMORY_CONDITIONAL_PRINT_HERE(
                debug() && verbose() > 2,
                "thread transferring buffer contents. completed: %s, "
                "full buffer: %s, buffer: %zu, history: %zu",
                _scompleted(), _sfullbuff(), _buff.size(), history().size());

            // if the sampler requested to be notified when completed
            // set the value (held by sampler) to true and set the
            // local pointer to nullptr since the variable pointer to it
            // was probably allocated on stack
            if(notify_completed)
            {
                *notify_completed = true;
                notify_completed  = nullptr;
            }
            full_buffer() = false;
            history().reserve(history().size() + _buff.size());
            for(auto& itr : _buff)
                history().emplace_back(std::move(itr));
        }
        TIMEMORY_CONDITIONAL_PRINT_HERE(
            debug(), "thread completed. completed: %s, full buffer: %s, history: %zu",
            _scompleted(), _sfullbuff(), history().size());

        _bundle->set_buffer_size(0);
        _bundle->set_notify([]() {
            completed() = true;
            buffer_cv().notify_one();
        });

        TIMEMORY_CONDITIONAL_PRINT_HERE(
            debug(),
            "thread sorting history. completed: %s, full buffer: %s, history: %zu",
            _scompleted(), _sfullbuff(), history().size());

        std::sort(history().begin(), history().end(),
                  [](const hist_type& _lhs, const hist_type& _rhs) {
                      return _lhs.first.time_since_epoch().count() <
                             _rhs.first.time_since_epoch().count();
                  });

        TIMEMORY_CONDITIONAL_PRINT_HERE(
            debug(),
            "thread setting history. completed: %s, full buffer: %s, history: %zu",
            _scompleted(), _sfullbuff(), history().size());

        _bundle->set_history(&history());
    } catch(...)
    {
        // Set the global exception pointer in case of an exception
        global_exception_pointer = std::current_exception();
    }
}

//--------------------------------------------------------------------------------------//

void
parent_process(pid_t pid)
{
    // mark as completed and wake buffer thread so it can exit
    completed() = true;
    get_measure()->set_buffer_size(0);
    get_measure()->set_notify([]() {
        completed() = true;
        buffer_cv().notify_one();
    });

    if(buffer_thread())
    {
        using hist_type = typename timem_bundle_t::hist_type;
        auto_lock_t _lk{ type_mutex<hist_type>() };
        _lk.unlock();
        for(size_t i = 0; i < 10; ++i)
        {
            buffer_cv().notify_all();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
        }
        buffer_thread()->join();
        buffer_thread().reset();
        handle_exception();
    }

    if((debug() && verbose() > 1) || verbose() > 2)
        std::cerr << "[AFTER STOP][" << pid << "]> " << *get_measure() << std::endl;

    std::vector<timem_bundle_t> _measurements{};

    if(use_mpi() || tim::mpi::size() > 1)
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(
            debug(), "%s (size: %i)", "Getting MPI measurements", (int) tim::mpi::size());
        _measurements = get_measure()->mpi_get();
    }
    else
    {
        if(get_measure())
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "%s", "Getting serial measurement");
            _measurements = { *get_measure() };
        }
        else
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "%s", "No measurements");
            _measurements = {};
        }
    }

    tim::mpi::barrier(tim::mpi::comm_world_v);

    if(_measurements.empty())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "No measurements on rank %i. Returning",
                                        tim::mpi::rank());
        return;
    }

    stringstream_t _oss;
    for(size_t i = 0; i < _measurements.size(); ++i)
    {
        auto& itr = _measurements.at(i);
        if(itr.empty())
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "%s (iteration: %zu)",
                                            "Empty measurement. Continuing", i);
            continue;
        }
        if(!_measurements.empty() && (use_mpi() || tim::mpi::size() > 1))
            itr.set_rank(i);

        TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "streaming iteration: %zu", i);
        _oss << itr << std::flush;
    }

    if(_oss.str().empty())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "%s", "Empty output. Returning");
        return;
    }

    if(output_file().empty())
    {
        std::cerr << '\n';
    }
    else
    {
        using json_type = tim::cereal::PrettyJSONOutputArchive;
        // extra function
        auto _cmdline = [](json_type& ar) {
            ar(tim::cereal::make_nvp("command_line", argvector()),
               tim::cereal::make_nvp("config", get_config()));
        };

        auto fname = get_config().get_output_filename({}, ".json");
        fprintf(stderr, "%s%s[%s]> Outputting '%s'...\n%s", ::tim::log::color::source(),
                (verbose() < 0) ? "" : "\n", command().c_str(), fname.c_str(),
                ::tim::log::color::end());
        tim::generic_serialization<json_type>(fname, _measurements, "timemory", "timem",
                                              _cmdline);
    }

    auto quiet = !output_file().empty() && verbose() < 0 && !debug();
    if(!quiet)
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "%s", "reporting");
        tim::log::stream(std::cerr, tim::log::color::info()) << _oss.str() << "\n";
    }
    else
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(debug(), "%s", "reporting skipped (quiet)");
    }

    // tim::mpi::barrier();
    // tim::mpi::finalize();
}

//--------------------------------------------------------------------------------------//

void
child_process(int argc, char** argv)
{
    if(argc < 2)
        exit(0);

    // the argv list first argument should point to filename associated
    // with file being executed the array pointer must be terminated by
    // NULL pointer

    // launches the command with the shell, this is the default because it enables aliases
    auto launch_using_shell = [&]() {
        int ret = -1;
        if(get_config().shell.length() > 0)
        {
            if(debug() || verbose() > 0)
                fprintf(stderr, "using shell: %s\n", get_config().shell.c_str());

            auto argpv = tim::argparse::argument_vector(argc, argv);
            auto argpc = argpv.get_execv(1);
            argpv.clear();
            argpv.resize(1, argpc.args());
            argpc.clear();
            std::vector<std::string> shellp = { get_config().shell, "-c" };
            for(auto&& itr : tim::delimit(get_config().shell_flags, " "))
                shellp.push_back(itr);
            auto shellc = argpv.get_execv(shellp);

            if(debug())
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                                "[%s]> command :: %s", command().c_str(),
                                                shellc.args().c_str());
            }

            explain(0, shellc.argv()[0], shellc.argv());

            ret = execvp(shellc.argv()[0], shellc.argv());
            explain(ret, shellc.argv()[0], shellc.argv());

            if(ret != 0)
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                                "return code: %i", ret);
                explain(ret, shellc.argv()[0], shellc.argv());
                ret = execv(shellc.argv()[0], shellc.argv());
            }

            if(ret != 0)
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                                "return code: %i", ret);
                explain(ret, shellc.argv()[0], shellc.argv());
                ret = execve(shellc.argv()[0], shellc.argv(), environ);
            }

            if(debug())
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                                "return code: %i", ret);
            }
        }
        else
        {
            fprintf(stderr, "getusershell failed!\n");
        }

        if(debug())
            TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");

        return ret;
    };

    // this will launch the process and inherit the environment but aliases will not
    // be available
    auto launch_without_shell = [&]() {
        auto argpv = tim::argparse::argument_vector(argc, argv);
        auto argpc = argpv.get_execv(1);
        if(debug())
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                            "[%s]> command :: '%s'", command().c_str(),
                                            argpc.args().c_str());
        }
        int ret = execvp(argpc.argv()[0], argpc.argv());
        // explain error if enabled
        explain(ret, argpc.argv()[0], argpc.argv());
        return ret;
    };

    // default return code
    int ret = -1;

    // determine if the shell should be tested first
    bool try_shell = use_shell();

    if(try_shell)
    {
        // launch the command with shell. If that fails, launch without shell
        ret = launch_using_shell();
        if(ret < 0)
        {
            if(debug())
                puts("Error launching with shell! Trying without shell...");
            ret = launch_without_shell();
        }
    }
    else
    {
        // launch the command without shell. If that fails, launch with shell
        ret = launch_without_shell();
        if(ret < 0)
        {
            if(debug())
                puts("Error launching without shell! Trying with shell...");
            ret = launch_using_shell();
        }
    }

    exit(ret);
}

//--------------------------------------------------------------------------------------//

bool&
timem_mpi_was_finalized()
{
    static bool _instance = false;
    return _instance;
}

//--------------------------------------------------------------------------------------//

void
create_signal_handler(int sig, signal_handler& sh, void (*func)(int))
{
    if(sig < 1)
        return;
    sh.m_custom_sigaction.sa_handler = func;
    sigemptyset(&sh.m_custom_sigaction.sa_mask);
    sh.m_custom_sigaction.sa_flags = SA_RESTART;
    if(sigaction(sig, &sh.m_custom_sigaction, &sh.m_original_sigaction) == -1)
    {
        std::cerr << "Failed to create signal handler for " << sig << std::endl;
    }
}

void
restore_signal_handler(int sig, signal_handler& sh)
{
    if(sig < 1)
        return;
    if(sigaction(sig, &sh.m_original_sigaction, &sh.m_custom_sigaction) == -1)
    {
        std::cerr << "Failed to restore signal handler for " << sig << std::endl;
    }
}
}  // namespace
