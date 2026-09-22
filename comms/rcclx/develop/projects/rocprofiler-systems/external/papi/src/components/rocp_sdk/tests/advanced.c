#include <stdio.h>
#include <unistd.h>
#include <papi.h>
#include <papi_test.h>
#include <hip/hip_runtime.h>

extern int launch_kernel(int device_id);
extern void add_desired_component_events(int eventSet, int maxEventsToAdd, const char eventsToAdd[][PAPI_MAX_STR_LEN], char metricNamesAdded[][PAPI_MAX_STR_LEN], int *numEventsSuccessfullyAdded);
extern void enumerate_and_add_component_events(const char *componentName, int eventSet, int maxEventsToAdd, char metricNamesAdded[][PAPI_MAX_STR_LEN], int *numEventsSuccessfullyAdded);

#define NUM_EVENTS (14)

int main(int argc, char *argv[])
{
    int dev_count=0;
    int papi_errno;
    long long counters[NUM_EVENTS] = { 0 };

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    int eventset = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    const char desiredEvents[NUM_EVENTS][PAPI_MAX_STR_LEN] = {
                  "rocp_sdk:::SQ_CYCLES:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=3:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=4:device=0",
                  "rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=5",
                  "rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0",
                  "rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=0:device=0",
                  "rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=1:device=0",
                  "rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=2:device=0",
                  "rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=3:device=0",
                  "rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=4:device=0",
                  "rocp_sdk:::SQ_WAVE_CYCLES:device=0"
    };

    char nativeEventNamesAdded[NUM_EVENTS][PAPI_MAX_STR_LEN] = { 0 };
    int numNativeEventsAdded = 0;
    add_desired_component_events(eventset, NUM_EVENTS, desiredEvents, nativeEventNamesAdded, &numNativeEventsAdded);

    // If we are unable to add any desired events then we enumerate through the available
    // rocp_sdk native events attempting to add up to NUM_EVENTS
    if (numNativeEventsAdded == 0) {
        const char *componentName = "rocp_sdk";
        enumerate_and_add_component_events(componentName, eventset, NUM_EVENTS, nativeEventNamesAdded, &numNativeEventsAdded);
    }

    // If we are unable to add any rocp_sdk native events whether that is the desired events
    // or events we enumerate through then skip the test
    if (numNativeEventsAdded == 0) {
        fprintf(stderr, "Unable to add any rocp_sdk native events.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    papi_errno = PAPI_start(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }
    for(int rep=0; rep<=4; ++rep){

        printf("---------------------  launch_kernel(0)\n");
        papi_errno = launch_kernel(0);
        if (papi_errno != 0) {
            test_fail(__FILE__, __LINE__, "launch_kernel(0)", papi_errno);
        }

        usleep(1000);

        papi_errno = PAPI_read(eventset, counters);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (int i = 0; i < numNativeEventsAdded; ++i) {
            fprintf(stdout, "%s: %.2lfM\n", nativeEventNamesAdded[i], (double)counters[i]/1e6);
        }
    }

    papi_errno = PAPI_stop(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < numNativeEventsAdded; ++i) {
            fprintf(stdout, "%s: %.2lfM\n", nativeEventNamesAdded[i], (double)counters[i]/1e6);
    }

    if (hipGetDeviceCount(&dev_count) != hipSuccess){
        test_fail(__FILE__, __LINE__, "Error while counting AMD devices:", papi_errno);
    }

    if( dev_count > 1 ){
        printf("======================================================\n");
        printf("==================== SECOND ROUND ====================\n");
        printf("======================================================\n");

        for(int rep=0; rep<=3; ++rep){
            papi_errno = PAPI_start(eventset);
            if (papi_errno != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
            }

            printf("---------------------  launch_kernel(1)\n");
            papi_errno = launch_kernel(1);
            if (papi_errno != 0) {
                test_fail(__FILE__, __LINE__, "launch_kernel(1)", papi_errno);
            }

            usleep(1000);

            papi_errno = PAPI_read(eventset, counters);
            if (papi_errno != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
            }
            printf("---------------------  PAPI_read()\n");

            for (int i = 0; i < numNativeEventsAdded; ++i) {
                fprintf(stdout, "%s: %.2lfM\n", nativeEventNamesAdded[i], (double)counters[i]/1e6);
            }

            papi_errno = PAPI_stop(eventset, counters);
            if (papi_errno != PAPI_OK) {
                test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
            }

            printf("---------------------  PAPI_stop()\n");

            for (int i = 0; i < numNativeEventsAdded; ++i) {
                fprintf(stdout, "%s: %.2lfM\n", nativeEventNamesAdded[i], (double)counters[i]/1e6);
            }
        }
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
