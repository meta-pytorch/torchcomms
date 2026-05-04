/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    nvml_power_limiting_test.cu
 * CVS:     $Id$
 * @author Tony Castaldo (tonycastaldon@icl.utk.edu) removed extraneous code and fixed a bug on multiple GPU setups. (Sept 2018). 
 * @author Asim YarKhan (yarkhan@icl.utk.edu) HelloWorld altered to test power capping (October 2017) 
 * @author Heike Jagode (jagode@icl.utk.edu)
 * Mods:  <your name here> <your email address>
 *
 * @brief

 * This file tests the ability to read power limits using NVML.
 * It does not write. 

 * The papi configure and papi Makefile will take care of the
 * compilation of the component tests (if all tests are added to a
 * directory named 'tests' in the specific component dir).  See
 * components/README for more details.
 */

#include <cuda.h>
#include <stdio.h>
#include "papi.h"
#include "papi_test.h"

#include "force_init.h"

// Host function
int main( int argc, char** argv )
{

#define NUM_EVENTS 32 /* Max number of GPUs on a node this code can handle. */
    int retval, i, j, device_count;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];
    int  device_id[NUM_EVENTS];
    char *EventName[NUM_EVENTS];
    int events[NUM_EVENTS];
    int eventCount = 0;
    const PAPI_component_info_t *cmpinfo;
    char event_name[PAPI_MAX_STR_LEN];

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) {
        fprintf( stderr, "PAPI_library_init failed.\n" );
        test_fail(__FILE__, __LINE__, "PAPI_library_init() failed.\n", retval);
    }

    printf( "PAPI_VERSION : %4d %6d %7d\n",
            PAPI_VERSION_MAJOR( PAPI_VERSION ),
            PAPI_VERSION_MINOR( PAPI_VERSION ),
            PAPI_VERSION_REVISION( PAPI_VERSION ) );

    int numcmp = PAPI_num_components();

   // Search for the NVML component. 
   int cid = 0;
    for (cid=0; cid<numcmp; cid++) {
        cmpinfo = PAPI_get_component_info(cid);
        if (cmpinfo == NULL) {                                  // NULL?
            fprintf(stderr, "PAPI error: PAPI reports %d components, but PAPI_get_component_info(%d) returns NULL pointer.\n", numcmp, cid); 
            test_fail( __FILE__, __LINE__,"PAPI_get_component_info failed\n",-1 );
        } else {
            if ( strstr( cmpinfo->name, "nvml" ) ) break;       // If we found it, 
        }
    }

    if ( cid==numcmp ) {                                        // If true we looped through all without finding nvml.
        fprintf(stderr, "NVML PAPI Component was not found.\n");       
        test_skip( __FILE__, __LINE__,"Component nvml is not present\n",-1 );
    }

    force_nvml_init(cid);

    printf( "NVML found as Component %d of %d: %s: %d events\n", (1+cmpinfo->CmpIdx), numcmp, cmpinfo->name, cmpinfo->num_native_events );
    if (cmpinfo->disabled) {                                    // If disabled,
        fprintf(stderr, "NVML PAPI Component is disabled.\n");
        test_skip( __FILE__,__LINE__,"Component nvml is disabled", 0 );
    }

    cudaGetDeviceCount( &device_count );
    printf("Found %d cuda devices\n", device_count);
    int code = PAPI_NATIVE_MASK;
    int ii=0;
    int event_modifier = PAPI_ENUM_FIRST;
    for ( ii=0; ii<cmpinfo->num_native_events; ii++ ) {
        retval = PAPI_enum_cmp_event( &code, event_modifier, cid );
        event_modifier = PAPI_ENUM_EVENTS;
        if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
        retval = PAPI_event_code_to_name( code, event_name );
        char *ss; 
        // We need events that END in power_management_limit; and must 
        // exclude those that end in power_management_limit_min or _max, 
        // and correspond to an existing cuda device.
        ss = strstr(event_name, "power_management_limit");              // get position of this string.
        if (ss == NULL) continue;                                       // skip if not present.
        if (ss[22] != 0) continue;                                      // skip if there is anything after it.
        ss = strstr(event_name, "device_");                             // Look for the device id.
        if (ss == NULL) continue;                                       // Not a valid name.
        int did = atoi(ss+7);                                           // convert it.
        if (did >= device_count) continue;                              // Invalid device count.
        EventName[eventCount] = strdup(event_name);                     // Valid! Remember the name.
        device_id[eventCount] = did;                                    // Remember the device id.  
        printf("Found event '%s' for device %i.\n", event_name, did);   // Report what we found.
        eventCount++;                                                   // Add to the number of events found.
    }


    if (eventCount == 0) {                // If we found nothing, 
        fprintf(stderr, "No NVML events found. Skipping Test.\n");
        test_skip( __FILE__,__LINE__,"Component nvml does not have a power_management_limit event.", 0 );
    }

    /* convert PAPI native events to PAPI code */
    for(i=0; i < eventCount; i++) {
        retval = PAPI_event_name_to_code( ( char * )EventName[i], &events[i] );
        if( retval != PAPI_OK ) {
            for (j=0; j<eventCount; j++) free(EventName[j]);            // clean up memory.
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", EventName[i], retval, PAPI_strerror(retval));
            test_fail( __FILE__,__LINE__,"PAPI_event_name_to_code failed.", retval );
        }
    }

    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_create_eventset failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        test_fail( __FILE__,__LINE__,"PAPI_create_eventset failed.", retval );
    }

    for (i=0; i< eventCount; i++) {
        printf( "cuda_device %d is being used\n", device_id[i]);
        cudaSetDevice(device_id[i]);

        retval = PAPI_add_events( EventSet, &events[i], 1);
        if( retval != PAPI_OK ) {
            for (j=0; j<eventCount; j++) free(EventName[j]);            // clean up memory.
            PAPI_cleanup_eventset(EventSet);                            // Empty it.
            PAPI_destroy_eventset(&EventSet);                           // Release memory.
            fprintf(stderr, "PAPI_add_events failure returned %i [%s].\n", retval, PAPI_strerror(retval));
            test_fail( __FILE__,__LINE__,"PAPI_add_events failed.", retval );
        }

        retval = PAPI_start( EventSet );
        if( retval != PAPI_OK ) {
            for (j=0; j<eventCount; j++) free(EventName[j]);            // clean up memory.
            PAPI_cleanup_eventset(EventSet);                            // Empty it.
            PAPI_destroy_eventset(&EventSet);                           // Release memory.
            fprintf(stderr, "PAPI_startfailure returned %i [%s].\n", retval, PAPI_strerror(retval));
            test_fail( __FILE__,__LINE__,"PAPI_start failed.", retval );
        }

        retval = PAPI_read( EventSet, values+i );                       // Get initial value for this event.
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_read failure returned %i [%s].\n", retval, PAPI_strerror(retval));
            test_fail( __FILE__, __LINE__, "PAPI_read failed.", retval );
        }

        printf( "Read: %s = %lld\n", EventName[i], values[i]);
        PAPI_stop(EventSet, values);                                    // Stop it so we can clear it.
        PAPI_cleanup_eventset(EventSet);                                // Empty it  for the next one.
    } // end loop for all found events.

    for (j=0; j<eventCount; j++) free(EventName[j]);                    // clean up memory.
    PAPI_destroy_eventset(&EventSet);                                   // All done, don't leak memory.

    test_pass( __FILE__);
    return 0;
} // end main.


