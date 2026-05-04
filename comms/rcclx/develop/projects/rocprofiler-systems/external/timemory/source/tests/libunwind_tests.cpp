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

#if !defined(UNW_LOCAL_ONLY)
#    define UNW_LOCAL_ONLY
#endif

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/data/ring_buffer.hpp"
#include "timemory/timemory.hpp"

//--------------------------------------------------------------------------------------//

namespace details
{
//  Get the current tests name
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class libunwind_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

#include <errno.h>
#include <execinfo.h>
#include <libunwind.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SIG_STACK_SIZE 0x100000

int verbose;
int num_errors;

/* These variables are global because they
 * cause the signal stack to overflow */
char                                            buf[512], name[496];
unw_cursor_t                                    cursor;
unw_context_t                                   uc;
tim::data_storage::ring_buffer<const char[512]> name_buffer;

static void
do_backtrace(void)
{
    unw_word_t      ip, sp, off;
    unw_proc_info_t pi;
    int             ret;

    if(verbose != 0)
        printf("\texplicit backtrace:\n");

    unw_getcontext(&uc);
    if(unw_init_local(&cursor, &uc) < 0)
    {
        num_errors++;
        return;
    }

    do
    {
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        unw_get_reg(&cursor, UNW_REG_SP, &sp);
        buf[0] = '\0';
        if(unw_get_proc_name(&cursor, name, sizeof(name), &off) == 0)
        {
            if(off != 0)
                snprintf(buf, sizeof(buf), "<%s+0x%lx>", name, (long) off);
            else
                snprintf(buf, sizeof(buf), "<%s>", name);
            name_buffer.write(&buf);
        }
        if(verbose != 0)
        {
            printf("%016lx %-32s (sp=%016lx)\n", (long) ip, buf, (long) sp);

            if(unw_get_proc_info(&cursor, &pi) == 0)
            {
                printf("\tproc=0x%lx-0x%lx\n\thandler=0x%lx lsda=0x%lx gp=0x%lx",
                       (long) pi.start_ip, (long) pi.end_ip, (long) pi.handler,
                       (long) pi.lsda, (long) pi.gp);
            }

#if UNW_TARGET_IA64
            {
                unw_word_t bsp;

                unw_get_reg(&cursor, UNW_IA64_BSP, &bsp);
                printf(" bsp=%lx", bsp);
            }
#endif
            printf("\n");
        }

        ret = unw_step(&cursor);
        if(ret < 0)
        {
            unw_get_reg(&cursor, UNW_REG_IP, &ip);
            printf("FAILURE: unw_step() returned %d for ip=%lx\n", ret, (long) ip);
            ++num_errors;
        }
    } while(ret > 0);

    {
        void* buffer[20];
        int   i, n;

        if(verbose != 0)
            printf("\n\tvia backtrace():\n");
        n = backtrace(buffer, 20);
        if(verbose != 0)
            for(i = 0; i < n; ++i)
                printf("[%d] ip=%p\n", i, buffer[i]);
    }
}

void
foo(long)
{
    do_backtrace();
}

TIMEMORY_NOINLINE long
f(long v)
{
    return v;
}

void
bar(long v)
{
    int* arr = new int[v];

    /* This is a vain attempt to use up lots of registers to force
       the frame-chain info to be saved on the memory stack on ia64.
       It happens to work with gcc v3.3.4 and gcc v3.4.1 but perhaps
       not with any other compiler.  */
  foo (f (arr[0]) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v)
       + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + (f (v) + f (v))
       ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
       )))))))))))))))))))))))))))))))))))))))))))))))))))))));

  delete[] arr;
}

void
sighandler(int signal, siginfo_t* siginfo, void* context)
{
    tim::consume_parameters(signal, siginfo, context);

    do_backtrace();
}

TEST_F(libunwind_tests, bt)
{
    struct sigaction act;
    stack_t          stk;
    verbose = true;

    EXPECT_FALSE(name_buffer.is_initialized());
    name_buffer.init(1000);
    std::cout << "original: " << name_buffer << "\n";

    if(verbose)
        printf("Normal backtrace:\n");

    bar(1);

    memset(&act, 0, sizeof(act));
    act.sa_sigaction = &sighandler;
    act.sa_flags     = SA_SIGINFO;
    if(sigaction(SIGTERM, &act, NULL) < 0)
    {
        FAIL() << "sigaction: " << strerror(errno);
    }

    if(verbose)
        printf("\nBacktrace across signal handler:\n");
    kill(getpid(), SIGTERM);

    if(verbose)
        printf("\nBacktrace across signal handler on alternate stack:\n");
    stk.ss_sp = malloc(SIG_STACK_SIZE);
    if(!stk.ss_sp)
    {
        FAIL() << "failed to allocate " << SIG_STACK_SIZE << " bytes:" << strerror(errno);
    }
    stk.ss_size  = SIG_STACK_SIZE;
    stk.ss_flags = 0;
    if(sigaltstack(&stk, NULL) < 0)
    {
        FAIL() << "sigaltstack: " << strerror(errno);
    }

    memset(&act, 0, sizeof(act));
    act.sa_sigaction = &sighandler;
    act.sa_flags     = SA_ONSTACK | SA_SIGINFO;
    if(sigaction(SIGTERM, &act, NULL) < 0)
    {
        FAIL() << "sigaction: " << strerror(errno);
    }

    kill(getpid(), SIGTERM);

    if(num_errors > 0)
    {
        FAIL() << "number of errors: " << num_errors;
    }

    if(verbose)
        printf("SUCCESS.\n");

    signal(SIGTERM, SIG_DFL);
    stk.ss_flags = SS_DISABLE;
    sigaltstack(&stk, NULL);
    free(stk.ss_sp);

    for(size_t i = 0; i < name_buffer.count(); ++i)
    {
        char _inp[512] = {};
        memset(_inp, '\0', 512);
        name_buffer.read(&_inp);
        auto _mangled = std::string{ _inp }.substr(1).substr(0, strlen(_inp) - 2);
        auto _ret     = _mangled.find("+0x");
        if(_ret != std::string::npos)
            _mangled = _mangled.substr(0, _ret);
        printf("Name buffer value #%lu: %s (demangled: <%s>)\n", (unsigned long) i, _inp,
               tim::demangle(_mangled).c_str());
        EXPECT_TRUE(tim::demangle(_mangled).find("_Z") != 0) << tim::demangle(_mangled);
    }

    std::cout << "final: " << name_buffer << "\n";
}

//--------------------------------------------------------------------------------------//
