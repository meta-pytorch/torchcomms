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

#define HIDDEN_VISIBILITY __attribute__((visibility("hidden")))
#define PUBLIC_VISIBILITY __attribute__((visibility("default")))
#define NEVER_INSTRUMENT __attribute__((no_instrument_function))

extern "C"
{
    extern void timemory_profile_func_enter(void* this_fn, void* call_site)
        PUBLIC_VISIBILITY NEVER_INSTRUMENT;
    extern void timemory_profile_func_exit(void* this_fn, void* call_site)
        PUBLIC_VISIBILITY NEVER_INSTRUMENT;
    extern int timemory_profile_thread_init(void) PUBLIC_VISIBILITY NEVER_INSTRUMENT;
    extern int timemory_profile_thread_fini(void) PUBLIC_VISIBILITY NEVER_INSTRUMENT;

    // bool timemory_enable_pthread_gotcha_wrapper() PUBLIC_VISIBILITY NEVER_INSTRUMENT;

    void __cyg_profile_func_enter(void* this_fn,
                                  void* call_site) PUBLIC_VISIBILITY NEVER_INSTRUMENT;
    void __cyg_profile_func_exit(void* this_fn,
                                 void* call_site) PUBLIC_VISIBILITY NEVER_INSTRUMENT;
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//
namespace
{
// this struct ensures that the thread finalization function is called
struct HIDDEN_VISIBILITY thread_exit
{
    thread_exit();
    ~thread_exit();
};
//
thread_exit::thread_exit() { timemory_profile_thread_init(); }
thread_exit::~thread_exit() { timemory_profile_thread_fini(); }
//
}  // namespace

//--------------------------------------------------------------------------------------//

extern "C"
{
    // bool timemory_enable_pthread_gotcha_wrapper() { return false; };
    //
    void __cyg_profile_func_enter(void* this_fn, void* call_site)
    {
        static thread_local auto _tl = thread_exit{};
        (void) _tl;
        timemory_profile_func_enter(this_fn, call_site);
    }
    //
    void __cyg_profile_func_exit(void* this_fn, void* call_site)
    {
        timemory_profile_func_exit(this_fn, call_site);
    }
}  // extern "C"
