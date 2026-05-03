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

#include "timemory/macros/os.hpp"

#if !defined(TIMEMORY_WINDOWS)

#    include "timemory/utility/delimit.hpp"
#    include "timemory/utility/popen.hpp"

#    include <limits>

#    if !defined(OPEN_MAX)
#        define OPEN_MAX 1024
#    endif

#    if !defined(NGROUPS_MAX)
#        define NGROUPS_MAX 16
#    endif

extern "C"
{
    extern char** environ;
}

namespace tim
{
namespace popen
{
//
//--------------------------------------------------------------------------------------//
//
struct group_info
{
    int   ngroups = -1;
    gid_t group_id;
    uid_t user_id;
    gid_t groups[NGROUPS_MAX];
};
//
//--------------------------------------------------------------------------------------//
//
inline group_info&
get_group_info()
{
    static group_info _instance;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
void
drop_privileges(int permanent)
{
    gid_t newgid = getgid();
    gid_t oldgid = getegid();
    uid_t newuid = getuid();
    uid_t olduid = geteuid();

    if(permanent == 0)
    {
        // Save information about the privileges that are being dropped so that they
        // can be restored later.
        //
        get_group_info().group_id = oldgid;
        get_group_info().user_id  = olduid;
        get_group_info().ngroups  = getgroups(NGROUPS_MAX, get_group_info().groups);
    }

    // If root privileges are to be dropped, be sure to pare down the ancillary
    // groups for the process before doing anything else because the setgroups(  )
    // system call requires root privileges.  Drop ancillary groups regardless of
    // whether privileges are being dropped temporarily or permanently.
    //
    if(olduid == 0)
        setgroups(1, &newgid);

    if(newgid != oldgid)
    {
#    if !defined(TIMEMORY_LINUX)
        auto ret = setegid(newgid);
        if(ret != 0)
            abort();
        if(permanent != 0 && setgid(newgid) == -1)
            abort();
#    else
        if(setregid(((permanent != 0) ? newgid : oldgid), newgid) == -1)
            abort();
#    endif
    }

    if(newuid != olduid)
    {
#    if !defined(TIMEMORY_LINUX)
        auto ret = seteuid(newuid);
        if(ret != 0)
            abort();
        if(permanent != 0 && setuid(newuid) == -1)
            abort();
#    else
        if(setreuid(((permanent != 0) ? newuid : olduid), newuid) == -1)
            abort();
#    endif
    }

    // verify that the changes were successful
    if(permanent != 0)
    {
        if(newgid != oldgid && (setegid(oldgid) != -1 || getegid() != newgid))
            abort();
        if(newuid != olduid && (seteuid(olduid) != -1 || geteuid() != newuid))
            abort();
    }
    else
    {
        if(newgid != oldgid && getegid() != newgid)
            abort();
        if(newuid != olduid && geteuid() != newuid)
            abort();
    }
}
//
//--------------------------------------------------------------------------------------//
//
void
restore_privileges()
{
    if(geteuid() != get_group_info().user_id)
        if(seteuid(get_group_info().user_id) == -1 ||
           geteuid() != get_group_info().user_id)
            abort();
    if(getegid() != get_group_info().group_id)
        if(setegid(get_group_info().group_id) == -1 ||
           getegid() != get_group_info().group_id)
            abort();
    if(get_group_info().user_id == 0U)
        setgroups(get_group_info().ngroups, get_group_info().groups);
}
//
//--------------------------------------------------------------------------------------//
//
int
open_devnull(int fd)
{
    FILE* f = nullptr;
    switch(fd)
    {
        case 0: f = freopen("/dev/null", "rb", stdin); break;
        case 1: f = freopen("/dev/null", "wb", stdout); break;
        case 2: f = freopen("/dev/null", "wb", stderr); break;
        default: break;
    }
    return (f != nullptr && fileno(f) == fd) ? 1 : 0;
}
//
//--------------------------------------------------------------------------------------//
//
void
sanitize_files()
{
    // int         fds;
    struct stat st;

    // Make sure all open descriptors other than the standard ones are closed
    // if((fds = getdtablesize()) == -1)
    //    fds = OPEN_MAX;

    // closing these files results in the inability to read the pipe from the parent
    // for(int fd = 3; fd < fds; ++fd)
    //    close(fd);

    // Verify that the standard descriptors are open.  If they're not, attempt to
    // open them using /dev/null.  If any are unsuccessful, abort.
    for(int fd = 0; fd < 3; ++fd)
    {
        if(fstat(fd, &st) == -1 && (errno != EBADF || open_devnull(fd) == 0))
        {
            abort();
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
pid_t
fork()
{
    pid_t childpid;

    if((childpid = ::fork()) == -1)
        return -1;

    // If this is the parent process, there's nothing more to do
    if(childpid != 0)
        return childpid;

    // This is the child process
    popen::sanitize_files();    // Close all open files.
    popen::drop_privileges(1);  // Permanently drop privileges.

    return 0;
}
//
//--------------------------------------------------------------------------------------//
//
std::shared_ptr<TIMEMORY_PIPE>
popen(const char* path, char** argv, char** envp)
{
    int  stdin_pipe[2]  = { 0, 0 };
    int  stdout_pipe[2] = { 0, 0 };
    auto p              = std::shared_ptr<TIMEMORY_PIPE>{};

    static char** _argv = []() {
        static auto* _tmp = new char*[1];
        _tmp[0]           = nullptr;
        return _tmp;
    }();

    if(envp == nullptr)
        envp = environ;
    if(argv == nullptr)
        argv = _argv;

    p = std::make_shared<TIMEMORY_PIPE>();

    if(!p)
        return p;

    p->read_fd   = nullptr;
    p->write_fd  = nullptr;
    p->child_pid = -1;

    if(pipe(stdin_pipe) == -1)
    {
        p.reset();
        return p;
    }

    if(pipe(stdout_pipe) == -1)
    {
        close(stdin_pipe[1]);
        close(stdin_pipe[0]);
        p.reset();
        return p;
    }

    if(!(p->read_fd = fdopen(stdout_pipe[0], "r")))
    {
        close(stdout_pipe[1]);
        close(stdout_pipe[0]);
        close(stdin_pipe[1]);
        close(stdin_pipe[0]);
        p.reset();
        return p;
    }

    if(!(p->write_fd = fdopen(stdin_pipe[1], "w")))
    {
        fclose(p->read_fd);
        close(stdout_pipe[1]);
        close(stdin_pipe[1]);
        close(stdin_pipe[0]);
        p.reset();
        return p;
    }

    if((p->child_pid = popen::fork()) == -1)
    {
        fclose(p->write_fd);
        fclose(p->read_fd);
        close(stdout_pipe[1]);
        close(stdin_pipe[0]);
        p.reset();
        return p;
    }

    if(p->child_pid == 0)
    {
        // this is the child process
        close(stdout_pipe[0]);
        close(stdin_pipe[1]);
        if(stdin_pipe[0] != 0)
        {
            dup2(stdin_pipe[0], 0);
            close(stdin_pipe[0]);
        }
        if(stdout_pipe[1] != 1)
        {
            dup2(stdout_pipe[1], 1);
            close(stdout_pipe[1]);
        }
#    if defined(TIMEMORY_LINUX)
        execvpe(path, argv, envp);
#    else
        execve(path, argv, envp);
#    endif
        exit(127);
    }

    close(stdout_pipe[1]);
    close(stdin_pipe[0]);

    return p;
}
//
//--------------------------------------------------------------------------------------//
//
int
pclose(std::shared_ptr<TIMEMORY_PIPE>& p)
{
    int   _status    = p->child_status;
    auto  _child_pid = p->child_pid;
    pid_t _pid       = -1;

    // clean up memory
    auto _clean = [&]() {
        if(p->read_fd)
            fclose(p->read_fd);
        if(p->write_fd)
            fclose(p->write_fd);
        p.reset();
    };

    if(_status != std::numeric_limits<int>::max())
    {
        _clean();
        if(WIFEXITED(_status))
        {
            return EXIT_SUCCESS;
        }
        else if(WIFSIGNALED(_status))
        {
            TIMEMORY_PRINTF(stderr, "process %i killed by signal %d\n", _child_pid,
                            WTERMSIG(_status));
            return EXIT_FAILURE;
        }
        else if(WIFSTOPPED(_status))
        {
            TIMEMORY_PRINTF(stderr, "process %i stopped by signal %d\n", _child_pid,
                            WSTOPSIG(_status));
        }
        else if(WIFCONTINUED(_status))
        {
            TIMEMORY_PRINTF(stderr, "process %i continued\n", _child_pid);
        }
    }
    else
    {
        if(_child_pid != -1)
        {
            do
            {
                _pid = waitpid(_child_pid, &_status, 0);
            } while(_pid == -1 && errno == EINTR);
        }
    }
    _clean();
    if(_pid != -1 && WIFEXITED(_status))
        return WEXITSTATUS(_status);
    return (_pid == -1 ? -1 : 0);
}
//
//--------------------------------------------------------------------------------------//
//
strvec_t
read_fork(const std::shared_ptr<TIMEMORY_PIPE>& proc, std::string_view _remove_chars,
          std::string_view                             _delimiters,
          const std::function<bool(std::string_view)>& _filter, int max_counter)
{
    int  _counter = 0;
    auto _lines   = strvec_t{};
    auto _ss      = std::stringstream{};

    auto _update_lines = [&]() {
        auto _line = _ss.str();
        _ss        = std::stringstream{};
        if(_line.empty())
            return;
        if(!_remove_chars.empty())
        {
            auto loc = std::string::npos;
            while((loc = _line.find_first_of(_remove_chars)) != std::string::npos)
                _line.erase(loc, 1);
        }
        auto delim = delimit(_line, _delimiters);
        for(const auto& itr : delim)
        {
            if(_filter(itr))
                _lines.emplace_back(itr);
        }
    };

    while(proc)
    {
        constexpr size_t N = 4096;
        char             buffer[N];
        memset(buffer, '\0', N * sizeof(char));
        auto* ret = fgets(buffer, N, proc->read_fd);
        if(ret == nullptr || strlen(buffer) == 0)
        {
            if(max_counter == 0)
            {
                pid_t cpid = waitpid(proc->child_pid, &proc->child_status, WNOHANG);
                if(cpid == 0)
                    continue;
                else
                    break;
            }
            if(_counter++ > max_counter)
                break;
            continue;
        }
        auto _len = strnlen(buffer, N + 1);
        if(_len < N + 1)
        {
            _ss << buffer;
            if(buffer[_len - 1] == '\n')
                _update_lines();
        }
    }

    _update_lines();

    return _lines;
}
//
//--------------------------------------------------------------------------------------//
//
strvec_t
read_ldd_fork(const std::shared_ptr<TIMEMORY_PIPE>& proc, int max_counter)
{
    return read_fork(
        proc, "\n\t", " \n\t=>", [](std::string_view itr) { return itr.find('/') == 0; },
        max_counter);
}
//
//--------------------------------------------------------------------------------------//
//
std::ostream&
flush_output(std::ostream& os, const std::shared_ptr<TIMEMORY_PIPE>& proc,
             int max_counter)
{
    int _counter = 0;
    while(proc)
    {
        constexpr size_t N = 4096;
        char             buffer[N];
        memset(buffer, '\0', N);
        auto* ret = fgets(buffer, N, proc->read_fd);
        if(ret == nullptr || strlen(buffer) == 0)
        {
            if(max_counter == 0)
            {
                pid_t cpid = waitpid(proc->child_pid, &proc->child_status, WNOHANG);
                if(cpid == 0)
                    continue;
                else
                    break;
            }
            if(_counter++ > max_counter)
                break;
            continue;
        }
        if(strnlen(buffer, N + 1) < N + 1)
            os << buffer << std::flush;
    }

    return os;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace popen
}  // namespace tim

#else

namespace
{
int windows_popen = 0;
}
#endif
