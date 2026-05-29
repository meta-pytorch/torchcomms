#include <stdio.h>
#include <unistd.h>
#include <papi.h>
#include <papi_test.h>

extern void launch_kernel(int device_id);

int main(int argc, char *argv[])
{
    int papi_errno;
#define NUM_EVENTS (5)
    long long counters1[NUM_EVENTS] = { 0 };
    long long counters2[NUM_EVENTS] = { 0 };
    int eventset1 = PAPI_NULL;
    int eventset2 = PAPI_NULL;
    double exp1[NUM_EVENTS] = {1, 1300000000, 55000000000, 1, 1};
    double exp2[NUM_EVENTS] = {45000000000, 1, 40000000, 1, 1300000000};
    double exp3[NUM_EVENTS] = {28000000000, 40000000, 1, 1300000000, 1};


    const char *events1[NUM_EVENTS] = {
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=1",
                  "rocp_sdk:::TCC_CYCLE:device=1",
                  "rocp_sdk:::SQ_WAVES:device=0",
                  "rocp_sdk:::SQ_WAVES:device=1"
    };

    const char *events2[NUM_EVENTS] = {
                  "rocp_sdk:::TCC_CYCLE:device=1",
                  "rocp_sdk:::SQ_INSTS:device=0",
                  "rocp_sdk:::SQ_INSTS:device=1",
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=1"
    };

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    papi_errno = PAPI_create_eventset(&eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        papi_errno = PAPI_add_named_event(eventset1, events1[i]);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }
    }

    papi_errno = PAPI_create_eventset(&eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        papi_errno = PAPI_add_named_event(eventset2, events2[i]);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }
    }

    printf("==================== FIRST EVENTSET - DEVICE 1 ====================\n");

    papi_errno = PAPI_start(eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=3; ++rep){

        launch_kernel(1);

        usleep(1000);

        papi_errno = PAPI_read(eventset1, counters1);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < NUM_EVENTS; ++i) {
            printf("%s: %lld (%.2lf)\n", events1[i], counters1[i], 1.0*counters1[i]/((1.0+rep)*exp1[i]));
        }
    }

    papi_errno = PAPI_stop(eventset1, counters1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < NUM_EVENTS; ++i) {
        printf("%s: %lld (%.2lf)\n", events1[i], counters1[i], 1.0*counters1[i]/((1.0+3)*exp1[i]));
    }

    printf("==================== SECOND EVENTSET - DEVICE 1 ====================\n");


    papi_errno = PAPI_start(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=3; ++rep){

        launch_kernel(1);

        usleep(1000);

        papi_errno = PAPI_read(eventset2, counters2);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < NUM_EVENTS; ++i) {
            printf("%s: %lld (%.2lf)\n", events2[i], counters2[i], 1.0*counters2[i]/((1.0+rep)*exp2[i]));
        }
    }

    papi_errno = PAPI_stop(eventset2, counters2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < NUM_EVENTS; ++i) {
        printf("%s: %lld (%.2lf)\n", events2[i], counters2[i], 1.0*counters2[i]/((1.0+3)*exp2[i]));
    }

    printf("==================== SECOND EVENTSET - DEVICE 0 ====================\n");

    papi_errno = PAPI_start(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=2; ++rep){

        launch_kernel(0);

        usleep(1000);

        papi_errno = PAPI_read(eventset2, counters2);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < NUM_EVENTS; ++i) {
            printf("%s: %lld (%.2lf)\n", events2[i], counters2[i], 1.0*counters2[i]/((1.0+rep)*exp3[i]));
        }
    }

    papi_errno = PAPI_stop(eventset2, counters2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < NUM_EVENTS; ++i) {
        printf("%s: %lld (%.2lf)\n", events2[i], counters2[i], 1.0*counters2[i]/((1.0+2)*exp3[i]));
    }

    /* * * Cleanup * * */

    papi_errno = PAPI_cleanup_eventset(eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    papi_errno = PAPI_cleanup_eventset(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    PAPI_shutdown();
    test_pass(__FILE__);
    return 0;
}
