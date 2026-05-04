/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    nvmlcap_plot.cu
 * CVS:     $Id$
 * @author Tony Castaldo (tonycastaldon@icl.utk.edu)
 * Mods:  <your name here> <your email address>
 *
 * @brief

 * This file reads power limits using NVML and writes them
 * every 50ms to nvmlcap_out.csv.
 * 
 * It takes at least one argument; the number of seconds to
 * run. 
 * 
 * If there is ONE additional argument, it is a power cap 
 * and all GPUs will be set to it. This is good if the GPUs
 * are all the same model. 
 * 
 * If there are MULTIPLE additional arguments, there must be
 * one per GPU, and they are individual power limits for the
 * GPUs. This is useful if they are not all the same model.
 * 
 * The output is written as tab-seperated-values (TSV) in 
 * PowerReadGPU.tsv.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#include "papi.h"
#include "papi_test.h"

#define dprintf if (0) printf /* debug printf; change to (1) to enable. */

int CTL_Z = 0;                         // No SIGTSTP signalled yet.
void cbSignal_SIGTSTP(int signalNumber) {
   CTL_Z = 1;                          // Indicate it was received.
} // end signal handler.

void helpText(void) {
    fprintf(stderr, "This program requires at least one argument.\n");
    fprintf(stderr, "First arg is number of seconds to run. If 0, will run   \n");
    fprintf(stderr, "until killed. A graceful exit can be made by signalling \n");
    fprintf(stderr, "SIGTSTP (Terminal Stop, like Ctrl-z). We will trap it   \n");
    fprintf(stderr, "and close files, free memory, etc. On SLURM, get job id \n");
    fprintf(stderr, "using 'squeue', then 'scancel -s SIGTSTP JOBID'         \n");
    fprintf(stderr, "2nd (optional) argument is a global power limit to set  \n");
    fprintf(stderr, "on all GPUs. If more than two arguments are given, then \n");
    fprintf(stderr, "there must be a power argument for EACH GPU we find,    \n");
    fprintf(stderr, "each is the individual power limit for that GPU (in the \n");
    fprintf(stderr, "order we report them).                                  \n");
    fprintf(stderr, "                                                        \n");
    fprintf(stderr, "We report to stderr the hardware found and current power\n");
    fprintf(stderr, "limit settings. If you change the power limit here, it  \n");
    fprintf(stderr, "does limit other programs; the original power limits are\n");
    fprintf(stderr, "automatically restored upon any exit of this program.   \n");
    fprintf(stderr, "                                                        \n");
    fprintf(stderr, "Typically, you will start this program on a node, then  \n");
    fprintf(stderr, "while it is running execute ANOTHER program on the node \n");
    fprintf(stderr, "that exercises the GPU.                                 \n");
    fprintf(stderr, "                                                        \n");
    fprintf(stderr, "After changing power settings (if specified), this code \n");
    fprintf(stderr, "READS the spot power usage every 50ms, for all GPUs on  \n");
    fprintf(stderr, "the node, and reports those (tab-separated) to the file \n");
    fprintf(stderr, "PowerReadGPUs.tsv.                                      \n");
    fprintf(stderr, "                                                        \n");
    fprintf(stderr, "It will also output PowerReadGPU.gnuplot, a gnuplot     \n");
    fprintf(stderr, "script to plot the power usage for each GPU on the node.\n");
    fprintf(stderr, "This is just an ascii file and can be edited if needed. \n");
}; 

// Host function
int main( int argc, char** argv )
{

#define NUM_EVENTS 32 /* Max number of GPUs on a node this code can handle. */
    int retval, i, j, device_count;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];                       // For reading either limit or current power.
    char *LimitEventName[NUM_EVENTS];
    char *PowerEventName[NUM_EVENTS];
    char *minEventName[NUM_EVENTS];
    char *maxEventName[NUM_EVENTS];
    int powerEvents[NUM_EVENTS];                        // PAPI codes for current power events.
    int limitEvents[NUM_EVENTS];                        // PAPI codes for power limit setting.
    int minEvents[NUM_EVENTS];
    int maxEvents[NUM_EVENTS];
    long long minSetting[NUM_EVENTS];
    long long maxSetting[NUM_EVENTS];
    long long UserLimitGiven[NUM_EVENTS];               // These are the values per GPU set by user.
    long long OrigLimitFound[NUM_EVENTS];               // original limit read from device.
    int PowerEventCount = 0, LimitEventCount = 0, minEventCount = 0, maxEventCount = 0;
    const PAPI_component_info_t *cmpinfo;
    char event_name[PAPI_MAX_STR_LEN];
    signal(SIGTSTP, cbSignal_SIGTSTP);                  // register the signal handler for CTL_Z.

    if (argc < 2) {
        helpText();
        exit(-1);
    }

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) {
        fprintf( stderr, "PAPI_library_init failed.\n" );
        helpText();
        exit(-1);
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
        exit(-1);
    }

    printf( "NVML found as Component %d of %d: %s: %d events\n", (1+cmpinfo->CmpIdx), numcmp, cmpinfo->name, cmpinfo->num_native_events );
    if (cmpinfo->disabled) {                                    // If disabled,
        fprintf(stderr, "NVML PAPI Component is disabled.\n");
        exit(-1);
    }

    cudaGetDeviceCount( &device_count );
    printf("Cuda Device Count: %d.\n", device_count);
    if (device_count < 1) {
        fprintf(stderr, "There are no GPUs to manage.\n");
        exit(-1);
    } 

    FILE *myOut = fopen("PowerReadGPU.tsv", "w");               // Open the file.
    if (myOut == NULL) {                                        // If that failed,
        fprintf(stderr, "Failed to open output file PowerReadGPU.csv.\n");
        exit(-1);
    }

    FILE *myGnuplot = fopen("PowerReadGPU.gnuplot", "w");
    if (myGnuplot == NULL) {
        fprintf(stderr, "Failed to open gnuplot output file PowerReadGPU.gnuplot.\n");
        exit(-1);
    }
 
    // Scan events to find nvml power events.
    int code = PAPI_NATIVE_MASK;
    int ii=0;
    int event_modifier = PAPI_ENUM_FIRST;
    for ( ii=0; ii<cmpinfo->num_native_events; ii++ ) {
        retval = PAPI_enum_cmp_event( &code, event_modifier, cid );
        event_modifier = PAPI_ENUM_EVENTS;
        if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
        retval = PAPI_event_code_to_name( code, event_name );
        char *ss; 

        ss = strstr(event_name, "device_");                             // Look for the device id.
        if (ss == NULL) continue;                                       // Not a valid name.
        int did = atoi(ss+7);                                           // convert it.
        if (did >= device_count) continue;                              // Invalid device count.

        // Have some event, anyway.
        ss = strstr(event_name, "power");                                       // First, see if we have power.
        if (ss != NULL && ss[5] == 0) {                                         // If found and the last thing on the line, 
            PowerEventName[did] = strdup(event_name);                           // .. remember the name, in device order.
            dprintf("Found powerEvent '%s' for device %i.\n", event_name, did);
            PowerEventCount++;                                                  // .. bump total power events.
            continue;                                                           // .. done with this event.
        }

        ss = strstr(event_name, "power_management_limit");                      // get position of this string.
        if (ss != NULL && ss[22] == 0) {                                        // If found and last thing on the line, 
            LimitEventName[did] = strdup(event_name);                           // Valid! Remember the name.
            dprintf("Found limitEvent '%s' for device %i.\n", event_name, did); // Report what we found.
            LimitEventCount++;                                                  // Add to the number of events found.
            continue;                                                           // Done with it.
        }

        ss = strstr(event_name, "power_management_limit_constraint_min");       // get position of this string.
        if (ss != NULL && ss[37] == 0) {                                        // If found and last thing on the line, 
            minEventName[did] = strdup(event_name);                             // Valid! Remember the name.
            dprintf("Found minEvent '%s' for device %i.\n", event_name, did);   // Report what we found.
            minEventCount++;                                                    // Add to the number of events found.
            continue;                                                           // Done with it.
        }

        ss = strstr(event_name, "power_management_limit_constraint_max");       // get position of this string.
        if (ss != NULL && ss[37] == 0) {                                        // If found and last thing on the line, 
            maxEventName[did] = strdup(event_name);                             // Valid! Remember the name.
            dprintf("Found maxEvent '%s' for device %i.\n", event_name, did);   // Report what we found.
            maxEventCount++;                                                    // Add to the number of events found.
            continue;                                                           // Done with it.
        }

    } // end of for each event. 


    if (PowerEventCount != device_count || 
        LimitEventCount != device_count ||
          minEventCount != device_count ||
          maxEventCount != device_count) {                              // If we did not get all the events,
        fprintf(stderr, "Too few NVML events found; %i devices, %i PowerEvents, %i LimitEvents, %i maxEvents, %i minEvents. Aborting\n",
                device_count, PowerEventCount, LimitEventCount, minEventCount, maxEventCount);
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1);
    }

    fflush(stdout);
    // Interpret command line arguments.
    int runSeconds = atoi(argv[1]);                                     // get run seconds.
    dprintf("runSeconds=%i.\n", runSeconds);
    if (runSeconds < 0) {
        fprintf(stderr, "First argument must be # seconds to run, or 0 to run until CTRL-z (=SIGTSTP). It cannot be negative.\n");
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1);
    }

    if (argc == 3) {                                                       // If a global limit is set,
        long long plv = atoll(argv[2]);                                    // .. get it.
        for (i=0; i<device_count; i++) UserLimitGiven[i]=plv;              // .. set them all.
    } else if (argc > 2) {
        if (argc != device_count+2) {
            fprintf(stderr, "You have specified %i power limits, it doesn't match with %i devices.\n", argc-2, device_count);
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }
            helpText();
            exit(-1);
        }

        for (i=0; i<device_count; i++) {
            UserLimitGiven[i] = atoll(argv[i+2]);                      // interpret the power limit.
            if (UserLimitGiven[i] < 1) {                               // This could use the actual limits, as an improvement.
                fprintf(stderr, "You have specified %i power limits, it doesn't match with %i devices.\n", argc-2, device_count);
                for (j=0; j<device_count; j++) {                                // Clean up memory.
                    free(PowerEventName[j]); 
                    free(LimitEventName[j]);
                    free(  minEventName[j]);
                    free(  maxEventName[j]);
                }
                helpText();
                exit(-1);
            }
        }                
    }

    dprintf("UserLimitGiven[0] = %llu\n", UserLimitGiven[0]);

    /* convert PAPI native events to PAPI code */
    for(i=0; i < device_count; i++) {
        retval = PAPI_event_name_to_code( ( char * )PowerEventName[i], &powerEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", PowerEventName[i], retval, PAPI_strerror(retval));
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }
            helpText();
            exit(-1); 
        }

        retval = PAPI_event_name_to_code( ( char * )LimitEventName[i], &limitEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", LimitEventName[i], retval, PAPI_strerror(retval));
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }
            helpText();
            exit(-1); 
        }

        retval = PAPI_event_name_to_code( ( char * )minEventName[i], &minEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", minEventName[i], retval, PAPI_strerror(retval));
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }
            helpText();
            exit(-1); 
        }

        retval = PAPI_event_name_to_code( ( char * )maxEventName[i], &maxEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", maxEventName[i], retval, PAPI_strerror(retval));
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }
            helpText();
            exit(-1); 
        }
    }


    for (i=0; i<device_count; i++) {
        printf("Power Event: %s Code %i\n", PowerEventName[i], powerEvents[i]);
        printf("Limit Event: %s Code %i\n", LimitEventName[i], limitEvents[i]);
        printf("  min Event: %s Code %i\n", minEventName[i],   minEvents[i]);
        printf("  max Event: %s Code %i\n", maxEventName[i],   maxEvents[i]);
    }

    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_create_eventset failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1); 
    }


    // Get the minimum values we can set each device to.
    retval = PAPI_add_events(EventSet, minEvents, device_count);  // Add the events in.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events (minEvents) failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1); 
    }

    retval = PAPI_start(EventSet);                          // Start the event set.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1); 
    }

    retval = PAPI_stop(EventSet, minSetting);               // Read it, and get values.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1); 
    }

    PAPI_cleanup_eventset(EventSet);                        // get rid of those events.
    
    // Get the maximum values we can set each device to.
    retval = PAPI_add_events(EventSet, maxEvents, device_count);  // Add the events in.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events (maxEvents) failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1); 
    }

    retval = PAPI_start(EventSet);                          // Start the event set.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1); 
    }

    retval = PAPI_stop(EventSet, maxSetting);                // Read it, and get values.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        helpText();
        exit(-1); 
    }

    // We have the min and max. 
    for (i=0; i<device_count; i++) {
        printf("Device %i: MinSetting=%llu, MaxSetting=%llu.\n", i, minSetting[i], maxSetting[i]);
    }

    // check to see if user settings are in range.
    retval = 0;                                             // count violations.
    if (argc > 2) {                                         // If we have settings to check,
        for (i=0; i<device_count; i++) {
            if (UserLimitGiven[i] < minSetting[i] ||
                UserLimitGiven[i] > maxSetting[i]) {
                fprintf(stderr, "User Power Limit of %llu is out of range for device %i.\n", UserLimitGiven[i], i);
                retval++;                                   // increase violations.
            }
        }

        if (retval > 0) {                                   // Any out of range, we get out.
            for (j=0; j<device_count; j++) {                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }

            exit(-1); 
        }
    }

    // Go ahead and read settings.
    PAPI_cleanup_eventset(EventSet);                                 // Delete existing events.
    retval = PAPI_add_events(EventSet, limitEvents, device_count);  // Add the events in.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }

        exit(-1); 
    }

    retval = PAPI_start(EventSet);                          // Start the event set.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }

        exit(-1); 
    }

    retval = PAPI_read(EventSet, OrigLimitFound);           // Read it, and get values.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_read failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }

        exit(-1); 
    }

    for (i=0; i<device_count; i++) {
        printf("Original Power Limit Read: %lli for %s.\n", OrigLimitFound[i], LimitEventName[i]);
    }

    if (argc > 2) {                                         // If power limits were given,
        retval = PAPI_write(EventSet, UserLimitGiven);      // .. Try to write user values.
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_write(User Limits) failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }

            exit(-1); 
        }
        
        retval = PAPI_stop(EventSet, values);               // Check it.
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_stop failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }

            exit(-1); 
        }
        
        for (i=0; i<device_count; i++) {
            printf("User Limit %lli set, readback new Limit: %lli for %s.\n", UserLimitGiven[i], values[i], LimitEventName[i]);
        }

        retval = 0;                                         // Use as a temp counter.
        for (i=0; i<device_count; i++) {                    // Make sure it worked.
            if (UserLimitGiven[i] != values[i]) {           // .. If this one did not,
                fprintf(stderr, "Write Failure on device %i: Attempted to write %lli, readback was different at %lli.\n",
                    i, UserLimitGiven[i], values[i]); 
                retval++;                                   // .. count the errors.
            }
        }

        if (retval > 0) {
            fprintf(stderr, "Aborting for %i write failure(s).\n", retval);
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }

            exit(-1); 
        }
    } else {                                                // end if we wanted to write new power values. If we did not,
        retval = PAPI_stop(EventSet, values);               // stop reading.
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_stop failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
            for (j=0; j<device_count; j++) {                                // Clean up memory.
                free(PowerEventName[j]); 
                free(LimitEventName[j]);
                free(  minEventName[j]);
                free(  maxEventName[j]);
            }

            exit(-1); 
        }
    } // end handling of whether we set power limits. Either way, eventset is stopped.

    fflush(stderr);
    fflush(stdout);

    PAPI_cleanup_eventset(EventSet);                            // Clean it up.
    retval = PAPI_add_events(EventSet, powerEvents, device_count);  // Add the power reading events in.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events failure (for power reading events) returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        PAPI_cleanup_eventset(EventSet);                            // Empty it.
        PAPI_destroy_eventset(&EventSet);                           // Release memory.
        exit(-1); 
    }

    retval = PAPI_start(EventSet);                          // Start the event set.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure (for power reading events) returned %i [%s].\n", retval, PAPI_strerror(retval));
        for (j=0; j<device_count; j++) {                                // Clean up memory.
            free(PowerEventName[j]); 
            free(LimitEventName[j]);
            free(  minEventName[j]);
            free(  maxEventName[j]);
        }
        PAPI_cleanup_eventset(EventSet);                            // Empty it.
        PAPI_destroy_eventset(&EventSet);                           // Release memory.
        exit(-1); 
    }


    //--------------------------------------------------------------------------
    // Main part of program, the reading loop.
    // We use tab separated to make it easy to use gnuplot.
    //--------------------------------------------------------------------------
    int runCount = 0;
    long long t1, t2;
    double elapsedSec;

    t1 = PAPI_get_real_nsec();                                  // Get the start time.
    while (CTL_Z == 0) {                                        // While I havent recieved a CTL-Z;
        usleep(50000);                                          // .. Wait 1/20 of a second.
        if (CTL_Z) break;                                       // .. CTL-Z may have interrupted usleep.
        t2 = PAPI_get_real_nsec();                              // .. Find end time.
        PAPI_read(EventSet, values);                            // .. Read instantaneous power consumption.
        elapsedSec = ((double) (t2-t1))/1.e09;                  // .. convert elapsed nanoseconds to seconds.
        fprintf(myOut, "%.6f", elapsedSec);                     // .. Time first.
        for (i=0; i<device_count; i++) {                        // .. for each device, 
            fprintf(myOut, "\t%llu", values[i]);                // .. print a value,
        }
        fprintf(myOut, "\n");                                   // .. Finish the line.
        fflush(myOut);                                          // .. Always flush (in cased canceled).

        runCount++;                                             // Count a run.
        if (runSeconds > 0 && elapsedSec >= runSeconds) break;  // Exit if time is up.
    }

    if (CTL_Z) fprintf(stderr, "Received CTL_Z signal (SIGTSTP).\n");
    else       fprintf(stderr, "Time %i seconds expired.\n", runSeconds);
    fprintf(stderr, "Total reads: %i.\n", runCount);

    //--------------------------------------------------------------------------
    // Generate a gnuplot file instructions.
    //--------------------------------------------------------------------------
    fprintf(myGnuplot, "set xlabel 'Time (sec)'\n");                // label for x axis.
    fprintf(myGnuplot, "set nokey\n");                              // no key needed.
    fprintf(myGnuplot, "set terminal png\n");                       // generate png output when plotting.
    fprintf(myGnuplot, "set title 'Spot MW Usage During Run'\n");   // Title of graph.
    fprintf(myGnuplot, "set yrange [0:300000]\n");                  // Force the y range.

    for (i=0; i<device_count; i++) {                            // For each event...
        char *Name = strdup(PowerEventName[i]);                 // Need a shorter name.
        dprintf("Name = '%s'\n", Name);                         // Show it.
        int begin=7;                                            // start of name after nvml:::.
        char *dev = strstr(Name, "device_");                    // Find location of device.
        int did = atoi(dev+7);                                  // Get device.
        int dpos = dev-Name-1;                                  // Last position to copy.
        dprintf("begin=%i, dpos=%i.\n", begin, dpos);
        for (j=begin; j<dpos; j++) Name[j-begin] = Name[j];     // Slide down the name.
        Name[dpos-begin]=0;                                     // Z-terminate.
        dprintf("Name = '%s'\n", Name);                         // Show it.

        fprintf(myGnuplot, "set ylabel '%s_%i'\n", Name, did);                      // label for y axis.
        fprintf(myGnuplot, "set output 'plot_%s_%i.png'\n", Name, did);             // Unique output file.
        fprintf(myGnuplot, "plot 'PowerReadGPU.tsv' using 1:%i with lines\n", i+2); // Always time against value. (columns are 1 relative).
        free(Name);
    }        

    fclose(myGnuplot);                                          // close file.

    // Clean up and exit.
    PAPI_stop(EventSet, values);                                // killing PAPI event set.
    PAPI_cleanup_eventset(EventSet);                            // ..
    PAPI_destroy_eventset(&EventSet);                           // ..

    for (j=0; j<device_count; j++) {                            // Clean up memory for names.
        free(PowerEventName[j]);                        
        free(LimitEventName[j]);
        free(  minEventName[j]);
        free(  maxEventName[j]);
    }

    fclose(myOut);                                              // Close the file.
    return 0;
} // end main.


