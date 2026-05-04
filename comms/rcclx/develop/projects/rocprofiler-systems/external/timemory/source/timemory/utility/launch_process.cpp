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

#ifndef TIMEMORY_UTILITY_LAUNCH_PROCESS_CPP_
#define TIMEMORY_UTILITY_LAUNCH_PROCESS_CPP_

#include "timemory/environment/types.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/popen.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/macros.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_UTILITY_LINKAGE(bool)
launch_process(const char* cmd, const std::string& extra, std::ostream* os,
               const std::vector<std::pair<std::string, std::string>>& _env)
{
    // save the original env variables and override
    auto _orig_env = std::vector<std::pair<std::string, std::string>>{};
    for(const auto& itr : _env)
    {
        _orig_env.emplace_back(itr.first, get_env(itr.first, std::string{}, false));
        set_env(itr.first, itr.second, 1);
    }

#if !defined(TIMEMORY_WINDOWS)
    auto delim = delimit(cmd, " \t");
    auto fp    = std::shared_ptr<tim::popen::TIMEMORY_PIPE>{};
    if(delim.size() < 2)
    {
        fp = tim::popen::popen(cmd, nullptr);
    }
    else
    {
        static std::string   _c = "-c";
        std::array<char*, 4> _args;
        _args.fill(nullptr);
        char*       _cshell = getenv("SHELL");
        char*       _ushell = getusershell();
        std::string _shell = (_cshell) ? _cshell : (_ushell) ? getusershell() : "/bin/sh";
        _args.at(0)        = (char*) _shell.c_str();
        _args.at(1)        = (char*) _c.c_str();
        _args.at(2)        = (char*) cmd;
        fp                 = tim::popen::popen(_args.at(0), _args.data());
    }

    if(fp == nullptr)
    {
        std::stringstream ss;
        ss << "[timemory]> Error launching command: '" << cmd << "'... " << extra;
        perror(ss.str().c_str());
        return false;
    }
    if(os)
    {
        popen::flush_output(*os, fp);
    }

    auto ec = tim::popen::pclose(fp);
    if(ec != 0)
    {
        std::stringstream ss;
        ss << "[timemory]> Command: '" << cmd << "' returned a non-zero exit code: " << ec
           << "... " << extra;
        perror(ss.str().c_str());
        return false;
    }
#else
    if(std::system(nullptr) != 0)
    {
        int ec = std::system(cmd);

        if(ec != 0)
        {
            fprintf(stderr,
                    "[timemory]> Command: '%s' returned a non-zero exit code: %i... %s\n",
                    cmd, ec, extra.c_str());
            return false;
        }
    }
    else
    {
        fprintf(stderr, "std::system unavailable for command: '%s'... %s\n", cmd,
                extra.c_str());
        return false;
    }
    (void) os;
#endif

    // restore original env
    for(const auto& itr : _orig_env)
        set_env(itr.first, itr.second, 1);

    return true;
}
}  // namespace tim

#endif
