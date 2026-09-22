#include <papi.h>
#include <papi_test.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * For the rocp_sdk component tests, take the desired events and attempt to add them.
 */
void add_desired_component_events(int eventSet, int maxNativeEventsToAdd, const char nativeEventsToAdd[][PAPI_MAX_STR_LEN], char nativeEventNamesAdded[][PAPI_MAX_STR_LEN], int *numNativeEventsAdded)
{
    int i;
    int *numEventsAdded = numNativeEventsAdded;
    for (i = 0; i < maxNativeEventsToAdd; i++) {
        int papi_errno = PAPI_add_named_event(eventSet, nativeEventsToAdd[i]);
        if (papi_errno == PAPI_ENOEVNT) {
            continue;
        }
        else if (papi_errno != PAPI_OK){
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }

        int strLen = snprintf(nativeEventNamesAdded[(*numEventsAdded)], PAPI_MAX_STR_LEN, "%s", nativeEventsToAdd[i]);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write native event name into index: %d\n", i);
            exit(1);
        }
        (*numEventsAdded)++;
    }

    return;
}

/*
 * Enumerate through the rocp_sdk native events and attempt to add them.
 * This function should only be called if none of the desired events were
 * successfully added. 
 */
void enumerate_and_add_component_events(const char *componentName, int eventSet, int maxNativeEventsToAdd, char nativeEventNamesAdded[][PAPI_MAX_STR_LEN], int *numNativeEventsAdded)
{

    int componentIdx = PAPI_get_component_index(componentName);
    if (componentIdx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index", componentIdx);
    }  

    int eventCode = 0 | PAPI_NATIVE_MASK, modifier = PAPI_ENUM_FIRST;
    int papi_errno = PAPI_enum_cmp_event(&eventCode, modifier, componentIdx);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", papi_errno);
    }

    int *numEventsAdded = numNativeEventsAdded;
    modifier = PAPI_ENUM_EVENTS;
    do {
        char eventName[PAPI_MAX_STR_LEN];
        papi_errno = PAPI_event_code_to_name(eventCode, eventName);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", papi_errno);
        }

        papi_errno = PAPI_add_named_event(eventSet, eventName);
        if (papi_errno == PAPI_ENOEVNT) {
            continue;
        }
        else if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }

        int strLen = snprintf(nativeEventNamesAdded[(*numEventsAdded)], PAPI_MAX_STR_LEN, "%s", eventName);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write native event name into index: %d\n", (*numEventsAdded));
            exit(1);
        }
        (*numEventsAdded)++;
    } while(PAPI_enum_cmp_event(&eventCode, modifier, componentIdx) == PAPI_OK && (*numEventsAdded) < maxNativeEventsToAdd);

    return;
}
