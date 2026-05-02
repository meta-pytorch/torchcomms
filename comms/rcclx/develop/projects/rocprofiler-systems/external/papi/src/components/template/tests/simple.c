#include <stdio.h>
#include <papi.h>
#include <papi_test.h>

int quiet;

int main(int argc, char *argv[])
{
    int papi_errno;

    quiet = tests_quiet(argc, argv); 

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

#define NUM_EVENTS (4)
    const char *events[NUM_EVENTS] = {
        "templ:::TEMPLATE_ZERO:device=0",
        "templ:::TEMPLATE_CONSTANT:device=1",
        "templ:::TEMPLATE_FUNCTION:device=2:function=exp",
        "templ:::TEMPLATE_FUNCTION:device=3:function=sum",
    };

    int eventset = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        papi_errno = PAPI_add_named_event(eventset, events[i]);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }
     }

     long long counters[NUM_EVENTS] = { 0 };
     papi_errno = PAPI_start(eventset);
     if (papi_errno != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
     }

     papi_errno = PAPI_read(eventset, counters);
     if (papi_errno != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
     }

     for (int i = 0; i < NUM_EVENTS && !quiet; ++i) {
         fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
     }
      
     papi_errno = PAPI_read(eventset, counters);
     if (papi_errno != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
     }

     for (int i = 0; i < NUM_EVENTS && !quiet; ++i) {
         fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
     }

     papi_errno = PAPI_read(eventset, counters);
     if (papi_errno != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
     }

     for (int i = 0; i < NUM_EVENTS && !quiet; ++i) {
         fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
     }

     papi_errno = PAPI_stop(eventset, counters);
     if (papi_errno != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
     }

     for (int i = 0; i < NUM_EVENTS && !quiet; ++i) {
         fprintf(stdout, "%s: %lli\n", events[i], counters[i]);
     }
     
     papi_errno = PAPI_cleanup_eventset(eventset);
     if (papi_errno != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
     }

     papi_errno = PAPI_destroy_eventset(&eventset);
     if (papi_errno != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
     }

     PAPI_shutdown();
     test_pass(__FILE__);
     return 0;
}
