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

#include "timemory/environment.hpp"
#include "timemory/macros.hpp"
#include "timemory/utility/argparse.hpp"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

//--------------------------------------------------------------------------------------//

#if !defined(TIMEM_PID_SIGNAL)
#    define TIMEM_PID_SIGNAL SIGCONT
#endif

#if !defined(TIMEM_PID_SIGNAL_STRING)
#    define TIMEM_PID_SIGNAL_STRING TIMEMORY_STRINGIZE(TIMEM_PID_SIGNAL)
#endif

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    std::stringstream fname;
    fname << tim::get_env<std::string>("TMPDIR", "/tmp") << "/.timemory-pid-";

    if(argc < 3)
    {
        std::stringstream usage;
        usage << "Usage: " << argv[0] << " <PID> <CMD> [<ARG> [<ARGS...>]]\n\n";
        usage << "This application is used by timemory MPI applications to provide "
              << "a PID for an MPI process which was launched via "
              << "MPI_Comm_spawn_multiple by writing a temporary file '" << fname.str()
              << "-<PID>' and then using " << TIMEM_PID_SIGNAL_STRING
              << " to signal to the process that it has been written.\n";
        std::cerr << usage.str();
        exit(EXIT_FAILURE);
    }

    // first argument is the PID of the parent
    pid_t monitoring_pid = atoi(argv[1]);

    // open the file that the parent reads
    fname << monitoring_pid;
    std::ofstream ofs(fname.str().c_str());
    if(!ofs)
    {
        return (fprintf(stderr, "Error opening '%s'...\n", fname.str().c_str()),
                EXIT_FAILURE);
    }

    // we need to shift args by 2, e.g. ignore 'timem-pid <PID>'
    auto argpv = tim::argparse::argument_vector(argc, argv);
    auto argpc = argpv.get_execv(2);

    fprintf(stderr, "[%s]> cmd :: %s\n", argv[0], argpc.args().c_str());

    // write the child pid to a file
    if(ofs)
        ofs << getpid() << '\n' << std::flush;

    // this causes corruption for some reason
    // if(ofs.is_open())
    //    ofs.close();

    // ignore this signal so it doesn't cause this exe to exit
    signal(TIMEM_PID_SIGNAL, SIG_IGN);
    // send signal to parent
    killpg(monitoring_pid, TIMEM_PID_SIGNAL);
    // make it the default again
    signal(TIMEM_PID_SIGNAL, SIG_DFL);
    // launch process on this PID
    return execvp(argpc.argv()[0], argpc.argv());
}

//--------------------------------------------------------------------------------------//
