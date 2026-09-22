#include <stdio.h>
#include <unistd.h>
#include <papi.h>
#include <papi_test.h>

extern int launch_kernel(int device_id);
extern void add_desired_component_events(int eventSet, int maxEventsToAdd, const char eventsToAdd[][PAPI_MAX_STR_LEN], char metricNamesAdded[][PAPI_MAX_STR_LEN], int *numEventsSuccessfullyAdded);
extern void enumerate_and_add_component_events(const char *componentName, int eventSet, int maxEventsToAdd, char metricNamesAdded[][PAPI_MAX_STR_LEN], int *numEventsSuccessfullyAdded);

#define NUM_EVENTS (5)

int main(int argc, char *argv[])
{
    int papi_errno;
    long long counters1[NUM_EVENTS] = { 0 };
    long long counters2[NUM_EVENTS] = { 0 };
    int eventset1 = PAPI_NULL;
    int eventset2 = PAPI_NULL;
    double exp1[NUM_EVENTS] = {1, 1300000000, 55000000000, 1, 1};
    double exp2[NUM_EVENTS] = {45000000000, 1, 40000000, 1, 1300000000};
    double exp3[NUM_EVENTS] = {28000000000, 40000000, 1, 1300000000, 1};


    const char desiredEvents1[NUM_EVENTS][PAPI_MAX_STR_LEN] = {
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=1",
                  "rocp_sdk:::TCC_CYCLE:device=1",
                  "rocp_sdk:::SQ_WAVES:device=0",
                  "rocp_sdk:::SQ_WAVES:device=1"
    };

    const char desiredEvents2[NUM_EVENTS][PAPI_MAX_STR_LEN] = {
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

    /* ---------- Setup for eventset1 ---------- */

    papi_errno = PAPI_create_eventset(&eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    char nativeEventNamesAdded1[NUM_EVENTS][PAPI_MAX_STR_LEN] = { 0 };
    int numNativeEventsAdded1 = 0;
    add_desired_component_events(eventset1, NUM_EVENTS, desiredEvents1, nativeEventNamesAdded1, &numNativeEventsAdded1);

    // If we are unable to add any desired events then we enumerate through the available
    // rocp_sdk native events attempting to add up to NUM_EVENTS
    if (numNativeEventsAdded1 == 0) {
        const char *componentName = "rocp_sdk";
        enumerate_and_add_component_events(componentName, eventset1, NUM_EVENTS, nativeEventNamesAdded1, &numNativeEventsAdded1);
    }

    // If we are unable to add any rocp_sdk native events whether that is the desired events
    // or events we enumerate through then skip the test
    if (numNativeEventsAdded1 == 0) {
        fprintf(stderr, "Unable to add any rocp_sdk native events for eventset2.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    /* ---------- Setup for eventset2 ---------- */

    papi_errno = PAPI_create_eventset(&eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    char nativeEventNamesAdded2[NUM_EVENTS][PAPI_MAX_STR_LEN] = { 0 };
    int numNativeEventsAdded2 = 0;
    add_desired_component_events(eventset2, NUM_EVENTS, desiredEvents2, nativeEventNamesAdded2, &numNativeEventsAdded2);

    // If we are unable to add any desired events then we enumerate through the available
    // rocp_sdk native events attempting to add up to NUM_EVENTS
    if (numNativeEventsAdded2 == 0) {
        const char *componentName = "rocp_sdk";
        enumerate_and_add_component_events(componentName, eventset2, NUM_EVENTS, nativeEventNamesAdded2, &numNativeEventsAdded2);
    }

    // If we are unable to add any rocp_sdk native events whether that is the desired events
    // or events we enumerate through then skip the test
    if (numNativeEventsAdded2 == 0) {
        fprintf(stderr, "Unable to add any rocp_sdk native events for eventset2.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    printf("==================== FIRST EVENTSET - DEVICE 1 ====================\n");

    papi_errno = PAPI_start(eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=3; ++rep){

        papi_errno = launch_kernel(1);
        if (papi_errno != 0) {
            test_fail(__FILE__, __LINE__, "launch_kernel(1)", papi_errno);
        }

        usleep(1000);

        papi_errno = PAPI_read(eventset1, counters1);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < numNativeEventsAdded1; ++i) {
            printf("%s: %lld (%.2lf)\n", nativeEventNamesAdded1[i], counters1[i], 1.0*counters1[i]/((1.0+rep)*exp1[i]));
        }
    }

    papi_errno = PAPI_stop(eventset1, counters1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < numNativeEventsAdded1; ++i) {
        printf("%s: %lld (%.2lf)\n", nativeEventNamesAdded1[i], counters1[i], 1.0*counters1[i]/((1.0+3)*exp1[i]));
    }

    printf("==================== SECOND EVENTSET - DEVICE 1 ====================\n");


    papi_errno = PAPI_start(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=3; ++rep){

        papi_errno = launch_kernel(1);
        if (papi_errno != 0) {
            test_fail(__FILE__, __LINE__, "launch_kernel(1)", papi_errno);
        }

        usleep(1000);

        papi_errno = PAPI_read(eventset2, counters2);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < numNativeEventsAdded2; ++i) {
            printf("%s: %lld (%.2lf)\n", nativeEventNamesAdded2[i], counters2[i], 1.0*counters2[i]/((1.0+rep)*exp2[i]));
        }
    }

    papi_errno = PAPI_stop(eventset2, counters2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < numNativeEventsAdded2; ++i) {
        printf("%s: %lld (%.2lf)\n", nativeEventNamesAdded2[i], counters2[i], 1.0*counters2[i]/((1.0+3)*exp2[i]));
    }

    printf("==================== SECOND EVENTSET - DEVICE 0 ====================\n");

    papi_errno = PAPI_start(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=2; ++rep){

        papi_errno = launch_kernel(0);
        if (papi_errno != 0) {
            test_fail(__FILE__, __LINE__, "launch_kernel(0)", papi_errno);
        }

        usleep(1000);

        papi_errno = PAPI_read(eventset2, counters2);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < numNativeEventsAdded2; ++i) {
            printf("%s: %lld (%.2lf)\n", nativeEventNamesAdded2[i], counters2[i], 1.0*counters2[i]/((1.0+rep)*exp3[i]));
        }
    }

    papi_errno = PAPI_stop(eventset2, counters2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < numNativeEventsAdded2; ++i) {
        printf("%s: %lld (%.2lf)\n", nativeEventNamesAdded2[i], counters2[i], 1.0*counters2[i]/((1.0+2)*exp3[i]));
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
