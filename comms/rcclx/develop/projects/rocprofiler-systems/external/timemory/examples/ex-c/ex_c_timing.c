
#include "timemory/library.h"
#include "timemory/timemory.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static long nlaps = 0;

//======================================================================================//

void*
get_timer(const char* func, int use_tuple)
{
    if(use_tuple > 0)
    {
        return TIMEMORY_MARKER(func, WALL_CLOCK, SYS_CLOCK, USER_CLOCK, CPU_CLOCK,
                               CPU_UTIL, PAGE_RSS, PEAK_RSS, PRIORITY_CONTEXT_SWITCH,
                               VOLUNTARY_CONTEXT_SWITCH, CALIPER);
    }
    else
    {
        return TIMEMORY_AUTO_TIMER(func);
    }
}

//======================================================================================//

void
free_timer(void* timer, int use_tuple)
{
    if(use_tuple > 0)
    {
        FREE_TIMEMORY_MARKER(timer);
    }
    else
    {
        FREE_TIMEMORY_AUTO_TIMER(timer);
    }
}

//======================================================================================//

void*
get_fibonacci_timer(const char* func, int use_tuple)
{
    char buffer[64];
    sprintf(buffer, "%s[using_tuple=%i]", func, use_tuple);
    if(use_tuple > 0)
    {
        return TIMEMORY_BLANK_MARKER(buffer, WALL_CLOCK, SYS_CLOCK, USER_CLOCK);
    }
    else
    {
        return TIMEMORY_BLANK_AUTO_TIMER(buffer);
    }
}

//======================================================================================//

intmax_t
_fibonacci(intmax_t n)
{
    return (n < 2) ? n : (_fibonacci(n - 1) + _fibonacci(n - 2));
}

//======================================================================================//

intmax_t
fibonacci(intmax_t n, intmax_t cutoff, int use_tuple)
{
    if(n > cutoff)
    {
        nlaps += 3;
        void*    timer = get_fibonacci_timer(__FUNCTION__, use_tuple);
        intmax_t _n    = (n < 2) ? n
                              : (fibonacci(n - 1, cutoff, use_tuple) +
                                 fibonacci(n - 2, cutoff, use_tuple));
        free_timer(timer, use_tuple);
        return _n;
    }
    else
    {
        return _fibonacci(n);
    }
}

//======================================================================================//

int
main(int argc, char** argv)
{
    // default calc: fibonacci(43)
    int nfib = 43;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 15;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    printf("'%s' : %s @ %i. Running fibonacci(%i, %i)...\n", __FILE__, __FUNCTION__,
           __LINE__, nfib, cutoff);

    // not sure why this fails on Windows
#if !defined(_WIN32) && !defined(_WIN64)
    timemory_settings settings = TIMEMORY_SETTINGS_INIT;
    settings.auto_output       = 1;
    settings.cout_output       = 1;
    settings.json_output       = 1;
    TIMEMORY_INIT(argc, argv, settings);
#endif

    void*    timer0 = get_timer("[main (untimed)]", 1);
    intmax_t n0     = _fibonacci(nfib);
    free_timer(timer0, 1);

    printf("main (untimed): fibonacci(%i, %i) = %lli\n", nfib, nfib, (long long) n0);

    void*    timer1 = get_timer("[main (timed + tuple)]", 1);
    intmax_t n1     = fibonacci(nfib, cutoff, 1);
    free_timer(timer1, 1);

    void*    timer2 = get_timer("[main (timed + timer)]", 1);
    intmax_t n2     = fibonacci(nfib, cutoff, 0);
    free_timer(timer2, 1);

    printf("main (timed): fibonacci(%i, %i) = %lli\n", nfib, cutoff, (long long) n1);
    printf("# laps = %li\n", nlaps);

    printf("\n'%s' : %s @ %i --> n = %lli, %lli, %lli\n", __FILE__, __FUNCTION__,
           __LINE__, (long long int) n0, (long long int) n1, (long long int) n2);

    TIMEMORY_FINALIZE();
    return 0;
}

//======================================================================================//
