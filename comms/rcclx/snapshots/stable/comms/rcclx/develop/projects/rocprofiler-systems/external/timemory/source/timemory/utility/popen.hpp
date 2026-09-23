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
//

#pragma once

#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"

#if !defined(TIMEMORY_WINDOWS)

#    include <cerrno>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <ctime>
#    include <fcntl.h>
#    include <functional>
#    include <grp.h>
#    include <limits>
#    include <memory>
#    include <ostream>
#    include <paths.h>
#    include <sstream>
#    include <string>
#    include <sys/param.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <sys/wait.h>
#    include <unistd.h>
#    include <vector>

namespace tim
{
namespace popen
{
//
using string_t = std::string;
using strvec_t = std::vector<string_t>;
//
struct TIMEMORY_PIPE
{
    FILE* read_fd  = nullptr;
    FILE* write_fd = nullptr;
    pid_t child_pid;
    int   child_status = std::numeric_limits<int>::max();
};
//
TIMEMORY_UTILITY_LINKAGE(std::shared_ptr<TIMEMORY_PIPE>)
popen(const char* path, char** argv = nullptr, char** envp = nullptr);
//
TIMEMORY_UTILITY_LINKAGE(int)
pclose(std::shared_ptr<TIMEMORY_PIPE>& p);
//
TIMEMORY_UTILITY_LINKAGE(pid_t)
fork();
//
TIMEMORY_UTILITY_LINKAGE(void)  // NOLINT
sanitize_files();
//
TIMEMORY_UTILITY_LINKAGE(int)
open_devnull(int fd);
//
TIMEMORY_UTILITY_LINKAGE(void)
drop_privileges(int permanent);
//
TIMEMORY_UTILITY_LINKAGE(void)  // NOLINT
restore_privileges();
//
TIMEMORY_UTILITY_LINKAGE(strvec_t)
read_fork(const std::shared_ptr<TIMEMORY_PIPE>& proc, std::string_view _remove_chars,
          std::string_view                             _delimiters,
          const std::function<bool(std::string_view)>& _filter, int max_counter = 50);
//
TIMEMORY_UTILITY_LINKAGE(strvec_t)
read_ldd_fork(const std::shared_ptr<TIMEMORY_PIPE>& proc, int max_counter = 50);
//
TIMEMORY_UTILITY_LINKAGE(std::ostream&)
flush_output(std::ostream& os, const std::shared_ptr<TIMEMORY_PIPE>& proc,
             int max_counter = 0);
//
//--------------------------------------------------------------------------------------//
//
}  // namespace popen
}  // namespace tim

#    if !defined(TIMEMORY_UTILITY_SOURCE) && !defined(TIMEMORY_USE_UTILITY_EXTERN)
#        include "timemory/utility/popen.cpp"
#    endif

#endif
