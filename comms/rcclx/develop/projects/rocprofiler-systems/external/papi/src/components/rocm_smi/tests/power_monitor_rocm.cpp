/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    power_report_rocm.cpp
 * CVS:     $Id$
 * @author Tony Castaldo (tonycastaldo@icl.utk.edu)
 * Mods:  <your name here> <your email address>
 *
 * @brief

 * This file reads power limits using ROCM_SMI and writes them
 * periodically to an output file.
 * 
 * See helpText() routine for arguments.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

#include "papi.h"
#include "papi_test.h"

#include "force_init.h"

#define dprintf if (0) printf /* debug printf; change to (1) to enable. */

// --------- GLOBALS -----------
#define NUM_EVENTS 32 /* Max number of GPUs on a node this code can handle. */
char *OutputName = NULL;
char *ScriptName = NULL;
char DefaultOutput[] = "/tmp/PowerReadGPU.tsv";
int  Interval = 100;
int  Duration = 0;
int  PowerCapped=0;                        // flag to remember if we capped power.
char Restore = 'Y';
long long UserLimitGiven[NUM_EVENTS];      // Values given, -1 = Don't Change.
double dUserLimitGiven[NUM_EVENTS];
int  DeviceCount = 0;
double ValueScale = 1000000.;               // Reports are in millionths of watts.
FILE *myGnuplot = NULL;

int CTL_Z = 0;                         // No SIGTSTP signalled yet.

void cbSignal_SIGTSTP(int signalNumber) {
   (void) signalNumber;                // No warning about unused.
   CTL_Z = 1;                          // Indicate it was received.
} // end signal handler.

void helpText(void) {
    fprintf(stderr, "This utility reads power usage registers on AMD GPU devices and writes the\n");
    fprintf(stderr, "results to a tab-separated file. Optionally, it can set the maximum power \n");
    fprintf(stderr, "used, for all devices or individually per device. It can also produce a   \n");
    fprintf(stderr, "simple gnuplot script to create a graph from the data. Both the data and  \n");
    fprintf(stderr, "script can be edited. The utiliity can run in the background while other  \n");
    fprintf(stderr, "unmodified user code exercises the GPUs. This code serves as an example   \n");
    fprintf(stderr, "for using PAPI to incorporate power control directly into user code.      \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "No arguments are required, the following are available. Default in [].    \n");
    fprintf(stderr, "--interval=#    [100]. milliseconds between reads.                        \n");
    fprintf(stderr, "--duration=#    [0].   seconds to run. If 0, runs until canceled, but will\n");
    fprintf(stderr, "                       terminate gracefully with a SIGTSTP signal. This is\n");
    fprintf(stderr, "                       sent by Ctl-Z, see below.                          \n");
    fprintf(stderr, "--globalcap=#.# []     No default; set in Watts, any double precision; but\n");
    fprintf(stderr, "                       micro-watts is maximum resolution.                 \n");
    fprintf(stderr, "                       If not specified, no powercaps will be set.        \n");
    fprintf(stderr, "                       Otherwise, the #.# will be applied to all GPUs on  \n");
    fprintf(stderr, "                       the system. May require sudo access to write       \n");
    fprintf(stderr, "                       powercaps.                                         \n");
    fprintf(stderr, "--perGPUcap=#.#,#.#,...No default; individual power caps, must be one per \n");
    fprintf(stderr, "                       GPU. if both --globalcap and --perGPUcap are       \n");
    fprintf(stderr, "                       present, the last one given is used.               \n");
    fprintf(stderr, "--output=file [/tmp/PowerReadGPU.tsv]                                     \n");
    fprintf(stderr, "                       Filename for tab-separated-file of output. This    \n");
    fprintf(stderr, "                       consists of a timestamp followed by the power read \n");
    fprintf(stderr, "                       from each GPU. One line per --interval. Report is  \n");
    fprintf(stderr, "                       given in Watts (floating point output).            \n");
    fprintf(stderr, "--script=file [NULL]   Filename for an ascii text gnuplot script. File is \n");
    fprintf(stderr, "                       replaced; if not specified or specified as \"\"    \n");
    fprintf(stderr, "                       nothing is written. The script will plot power     \n");
    fprintf(stderr, "                       usage for each GPU on the node.                    \n");
    fprintf(stderr, "--restore=N        [Y] Default is YES. If power capped, restore original  \n");
    fprintf(stderr, "                       levels before exit. Executes ONLY on graceful exit;\n");
    fprintf(stderr, "                       by --duration, CTL-Z or SIGTSTP signal. See below. \n");
    fprintf(stderr, "help | --help | -h     This text. Any argument beginning '-h'.            \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "-----------------------------Sending SIGTSTP------------------------------\n");
    fprintf(stderr, "If you have run this program in the foreground, you can press Ctl-Z.      \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "You can run this program in background by ending the command with '&', eg \n");
    fprintf(stderr, "./power_monitor_rocm &                                                    \n");
    fprintf(stderr, "To send a signal you need the PID; you can find that with the ps command: \n");
    fprintf(stderr, ">ps                                                                       \n");
    fprintf(stderr, "  PID TTY          TIME CMD                                               \n");
    fprintf(stderr, "19004 pts/2    00:00:00 power_montior_r   #<19004 is the PID we want>     \n");
    fprintf(stderr, "19005 pts/2    00:00:00 ps                                                \n");
    fprintf(stderr, "24020 pts/2    00:00:00 bash                                              \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "Then the 'kill' command sends the SIGTSTP signal as follows:              \n");
    fprintf(stderr, ">kill -SIGTSTP 19004  #<replace with PID revealed by ps command>          \n");
    fprintf(stderr, "The response will be:                                                     \n");
    fprintf(stderr, "Received CTL_Z signal (SIGTSTP).                                          \n");
    fprintf(stderr, "Total reads: ###.                                                         \n");
    fprintf(stderr, "                                                                          \n");
    fprintf(stderr, "For SLURM users: The job ID can be found using the 'squeue' command.      \n");
    fprintf(stderr, "Then 'scancel -s SIGTSTP #####' #<replace ##### with 'squeue' JOBID>      \n");
}; 

void rocmGetDeviceCount(long long *deviceCount) 
{
    int EventSet = PAPI_NULL;
    int retval, devCntEventCode;

// rocm_smi:::NUMDevices
    retval = PAPI_event_name_to_code("rocm_smi:::NUMDevices", &devCntEventCode);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_event_name_to_code failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        helpText();
        exit(-1); 
    }

    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_create_eventset failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        helpText();
        exit(-1); 
    }

    retval = PAPI_add_event(EventSet, devCntEventCode);     // Add the event in.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_event failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        helpText();
        exit(-1); 
    }

    retval = PAPI_start(EventSet);                          // Start the event set.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        helpText();
        exit(-1); 
    }

    retval = PAPI_stop(EventSet, deviceCount);               // STop and get value.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "%i: PAPI_stop failed, returned %i [%s].\n", __LINE__, retval, PAPI_strerror(retval));
        helpText();
        exit(-1); 
    }

    PAPI_cleanup_eventset(EventSet);                        // get rid of this set.
} // end Get Devices.

void parseArgs(int argc, char **argv) {
    int i, j, n;
    double dn;
    for (i=0; i<NUM_EVENTS; i++) UserLimitGiven[i]=-1; 
    if (argc < 1) exit(-1);
    for (i=1; i<argc; i++) {
        if (strncmp("--interval=", argv[i], 11) == 0) {
            n = atoi(argv[i]+11);
            if (n < 10) {
                fprintf(stderr, "--interval of %i is too short, must be at least 10.\n", n);
                exit(-1);
            }
            
            Interval = n;
            continue;
        }

       if (strncmp("--duration=", argv[i], 11) == 0) {
            n = atoi(argv[i]+11);
            if (n < 0) {
                fprintf(stderr, "Duration of %i is illegal, must be >= 0.\n", n);
                exit(-1);
            }
       
            Duration=n;
            continue;
        }

       if (strncmp("--globalcap=", argv[i], 12) == 0) {
            dn = atof(argv[i]+12);
            if (dn <= 0.) {
                fprintf(stderr, "--globalcap of %i is illegal, must be > 0.\n", n);
                exit(-1);
            }
            dUserLimitGiven[0]=dn;
            UserLimitGiven[0]=(long long) round(dn*ValueScale);
            PowerCapped = 1;

            for (j=1; j<NUM_EVENTS; j++) {
                dUserLimitGiven[j]=dn;
                UserLimitGiven[j]=round(dn*ValueScale);
            }
            continue;
        }

       if (strncmp("--perGPUcap=", argv[i], 12) == 0) {
            char *cPos = argv[i]+12;
            dn = atof(cPos);
            if (dn <= 0.) {
                fprintf(stderr, "--perGPUcap[0] of %3.6f is illegal, must be > 0.\n", dn);
                exit(-1);
            }
            dUserLimitGiven[0]=dn;
            UserLimitGiven[0]=(long long) round(dn*ValueScale);
            PowerCapped = 1;

            j = 1;
            while (1) {
                cPos =strchr(cPos, ',');    // find next comma.
                if (cPos == NULL) break;    // Get out.
                if (j > NUM_EVENTS) {
                    fprintf(stderr, "ERROR: --perGPUcap values exceed maximum # devices of %i.\n", NUM_EVENTS);
                    exit(-1);
                }
                if (j >= DeviceCount) {
                    fprintf(stderr, "ERROR: --perGPUcap values exceed number of devices=%i.\n", DeviceCount);
                    exit(-1);
                }

                cPos++;
                dn = atof(cPos);
                if (dn <= 0) {
                    fprintf(stderr, "--perGPUcap[%i] of %3.6f is illegal, must be > 0.\n", j, dn);
                    exit(-1);
                }

                dUserLimitGiven[j]=dn;
                UserLimitGiven[j]=(long long) round(dn*ValueScale);
                j++;
            }

            if (j != DeviceCount) {
                fprintf(stderr, "--perGPUcap had %i arguments for %i devices; they must be equal.\n", j, DeviceCount);
                exit(-1);
            }
                
            continue;
        }

       if (strncmp("--output=", argv[i], 9) == 0) {
            OutputName = argv[i]+9;
            continue;
        }

       if (strncmp("--script=", argv[i], 9) == 0) {
            ScriptName = argv[i]+9;
            continue;
        }

       if (strncmp("--restore=", argv[i], 10) == 0) {
            Restore = argv[i][10];
            if (Restore == 'y') Restore='Y';
            if (Restore == 'n') Restore='N';
            if (Restore != 'Y' && Restore != 'N') {
                fprintf(stderr, "--restore= must be 'Y' or 'N', '%c' is invalid.\n", argv[i][10]);
                exit(-1);
            }

            continue;
        }

       if (strncmp("--help", argv[i], 6) == 0 ||
           strncmp("-h", argv[i], 2) == 0     ||
           strncmp("help", argv[i], 4)   == 0) {
            helpText();
            exit(-1);
        }

        fprintf(stderr, "argument %i ['%s'] is not valid.\n", i, argv[i]);
        helpText();
        exit(-1);
    } // end loop.

    // set default output name.
    if (OutputName == NULL) OutputName = DefaultOutput;
} // end parseArgs.

// Host function
int main( int argc, char** argv )
{

    int retval, i, j;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];                       // For reading either limit or current power.
    char LimitEventName[NUM_EVENTS][PAPI_MAX_STR_LEN];
    char PowerEventName[NUM_EVENTS][PAPI_MAX_STR_LEN];
    char minEventName[NUM_EVENTS][PAPI_MAX_STR_LEN];
    char maxEventName[NUM_EVENTS][PAPI_MAX_STR_LEN];
    int powerEvents[NUM_EVENTS];                        // PAPI codes for current power events.
    int limitEvents[NUM_EVENTS];                        // PAPI codes for power limit setting.
    int minEvents[NUM_EVENTS];
    int maxEvents[NUM_EVENTS];
    long long minSetting[NUM_EVENTS];
    long long maxSetting[NUM_EVENTS];
    long long OrigLimitFound[NUM_EVENTS];               // original limit read from device.
    int PowerEventCount = 0, LimitEventCount = 0, minEventCount = 0, maxEventCount = 0;
    const PAPI_component_info_t *cmpinfo;
    char event_name[PAPI_MAX_STR_LEN];
    signal(SIGTSTP, cbSignal_SIGTSTP);                  // register the signal handler for CTL_Z.

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

   // Search for the rocm_smi component. 
   int cid = 0;
    for (cid=0; cid<numcmp; cid++) {
        cmpinfo = PAPI_get_component_info(cid);
        if (cmpinfo == NULL) {
            fprintf(stderr, "PAPI error: PAPI reports %d components, but PAPI_get_component_info(%d) returns NULL pointer.\n", numcmp, cid); 
            test_fail( __FILE__, __LINE__,"PAPI_get_component_info failed\n",-1 );
        } else {
            if ( strstr( cmpinfo->name, "rocm_smi" ) ) break;
        }
    }

    if ( cid==numcmp ) {
        fprintf(stderr, "ROCM_SMI PAPI Component was not found.\n");       
        exit(-1);
    }

    force_rocm_smi_init(cid);

    if (cmpinfo->disabled) {
        fprintf(stderr, "ROCM_SMI PAPI Component is disabled.\n");
        exit(-1);
    }

    long long llDC;
    rocmGetDeviceCount( &llDC);
    DeviceCount = (int) llDC;
    printf("AMD Device Count: %d.\n", DeviceCount);
    
    if (DeviceCount < 1) {
        fprintf(stderr, "There are no GPUs to manage.\n");
        exit(-1);
    } 

    if (DeviceCount > NUM_EVENTS) {
        fprintf(stderr, "There are %i GPUs found; this code cannot handle more than %i.\n", DeviceCount, NUM_EVENTS);
        exit(-1);
    } 

    parseArgs(argc, argv);

    FILE *myOut = fopen(OutputName, "w");
    if (myOut == NULL) {
        fprintf(stderr, "Failed to open output '%s'. Error: %d (%s)\n", OutputName, errno, strerror(errno));
        if (errno == 13) fprintf(stderr,"In sudo mode, you may be restricted to output only to /tmp/ directory.\n");
        exit(-1);
    }

    if (ScriptName != NULL) {
        myGnuplot = fopen("/tmp/PowerReadGPU.gnuplot", "w");
        if (myGnuplot == NULL) {
            fprintf(stderr, "Failed to open gnuplot output '%s'. Error: %d (%s)\n", ScriptName, errno, strerror(errno));
            if (errno == 13) fprintf(stderr,"In sudo mode, you may be restricted to output only to /tmp/ directory.\n");
            exit(-1);
        }
    }

    // Scan events to find rocm power events.
    int code = PAPI_NATIVE_MASK;
    int ii=0;
    int event_modifier = PAPI_ENUM_FIRST;
    for ( ii=0; ii<cmpinfo->num_native_events; ii++ ) {
        retval = PAPI_enum_cmp_event( &code, event_modifier, cid );
        event_modifier = PAPI_ENUM_EVENTS;
        if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
        retval = PAPI_event_code_to_name( code, event_name );
        char *ss; 

        ss = strstr(event_name, "device=");                             // Look for the device id.
        if (ss == NULL) continue;                                       // Not a valid name.
        int did = atoi(ss+7);                                           // convert it.
        if (did >= DeviceCount) continue;                               // Invalid device count.

        // Have an event name to examine.
        ss = strstr(event_name, "power_average:");
        if (ss != NULL) {
            strncpy(PowerEventName[did], event_name, PAPI_MAX_STR_LEN);
            PowerEventName[did][PAPI_MAX_STR_LEN-1]=0;
            PowerEventCount++;
            continue;
        }

        ss = strstr(event_name, "power_cap:");
        if (ss != NULL) {
            strncpy(LimitEventName[did], event_name, PAPI_MAX_STR_LEN);
            LimitEventName[did][PAPI_MAX_STR_LEN-1]=0;
            LimitEventCount++;
            continue;
        }

        ss = strstr(event_name, "power_cap_range_min:");
        if (ss != NULL) {
            strncpy(minEventName[did], event_name, PAPI_MAX_STR_LEN);
            minEventName[did][PAPI_MAX_STR_LEN-1]=0;
            minEventCount++;
            continue;
        }

        ss = strstr(event_name, "power_cap_range_max");
        if (ss != NULL) {
            strncpy(maxEventName[did], event_name, PAPI_MAX_STR_LEN);
            maxEventName[did][PAPI_MAX_STR_LEN-1]=0;
            maxEventCount++;
            continue;
        }
    } // end of for each event. 


    if (PowerEventCount != DeviceCount || 
        LimitEventCount != DeviceCount ||
          minEventCount != DeviceCount ||
          maxEventCount != DeviceCount) {
        fprintf(stderr, "Too few ROCM_SMI events found; %d devices, %i PowerEvents, %i LimitEvents, %i maxEvents, %i minEvents. Aborting\n",
                DeviceCount, PowerEventCount, LimitEventCount, minEventCount, maxEventCount);
        helpText();
        exit(-1);
    }

    /* convert PAPI native events to PAPI code */
    for(i=0; i < DeviceCount; i++) {
        retval = PAPI_event_name_to_code( ( char * )PowerEventName[i], &powerEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", PowerEventName[i], retval, PAPI_strerror(retval));
            helpText();
            exit(-1); 
        }

        retval = PAPI_event_name_to_code( ( char * )LimitEventName[i], &limitEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", LimitEventName[i], retval, PAPI_strerror(retval));
            helpText();
            exit(-1); 
        }

        retval = PAPI_event_name_to_code( ( char * )minEventName[i], &minEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", minEventName[i], retval, PAPI_strerror(retval));
            helpText();
            exit(-1); 
        }

        retval = PAPI_event_name_to_code( ( char * )maxEventName[i], &maxEvents[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failure for event [%s] returned %i [%s].\n", maxEventName[i], retval, PAPI_strerror(retval));
            helpText();
            exit(-1); 
        }
    }

    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_create_eventset failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }


    // Get the minimum values we can set each device to.
    retval = PAPI_add_events(EventSet, minEvents, DeviceCount);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events (minEvents) failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    retval = PAPI_start(EventSet);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    retval = PAPI_stop(EventSet, minSetting);               // Read it, and get values.
    if( retval != PAPI_OK ) {
        fprintf(stderr, "%i: PAPI_stop failed, returned %i [%s].\n", __LINE__, retval, PAPI_strerror(retval));
        helpText();
        exit(-1); 
    }

    // get rid of those events.
    PAPI_cleanup_eventset(EventSet);
    
    // Get the maximum values we can set each device to.
    retval = PAPI_add_events(EventSet, maxEvents, DeviceCount);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events (maxEvents) failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    retval = PAPI_start(EventSet);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    retval = PAPI_stop(EventSet, maxSetting);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "%i: PAPI_stop failed, returned %i [%s].\n", __LINE__, retval, PAPI_strerror(retval));
        exit(-1); 
    }

    // We have the min and max. 
    for (i=0; i<DeviceCount; i++) {
        printf("Device %i: MinSetting=%.6f, MaxSetting=%.6f.\n", i, (minSetting[i]/ValueScale), (maxSetting[i])/ValueScale);
    }

    // check to see if user settings are in range.
    retval = 0;                                             // count violations.

    for (i=0; i<DeviceCount; i++) {
        if (UserLimitGiven[i] < 0) continue;                // ignore if never set.
        if (UserLimitGiven[i] < minSetting[i] ||
            UserLimitGiven[i] > maxSetting[i]) {
            fprintf(stderr, "User Power Limit of %.6f is out of range for device %i; min=%.6f, max=%.6f.\n", 
                dUserLimitGiven[i], i, (minSetting[i]/ValueScale), (maxSetting[i]/ValueScale));
            retval++;                                       // increase violations.
        }
    }

    // exit if any violations.
    if (retval > 0) {
        exit(-1); 
    }

    // Go ahead and read settings, all at once.
    PAPI_cleanup_eventset(EventSet);
    retval = PAPI_add_events(EventSet, limitEvents, DeviceCount);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    retval = PAPI_start(EventSet);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    // Read values.
    retval = PAPI_stop(EventSet, OrigLimitFound);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_read failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    PAPI_cleanup_eventset(EventSet);

    // For each device, set limit if provided.
    for (i=0; i<DeviceCount; i++) {
        printf("Original Power Limit Read: %.6f for %s.\n", (OrigLimitFound[i]/ValueScale), LimitEventName[i]);
        if (UserLimitGiven[i] >= 0) {
            printf("Attempting to set Power Limit for device %i to %.6f.\n", i, (UserLimitGiven[i]/ValueScale));
            retval = PAPI_add_event(EventSet, limitEvents[i]);
            if( retval != PAPI_OK ) {
                fprintf(stderr, "PAPI_add_event failure returned %i [%s].\n", retval, PAPI_strerror(retval));
                exit(-1); 
            }

            retval = PAPI_start(EventSet);
            if( retval != PAPI_OK ) {
                fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
                exit(-1); 
            }

            // Try to write user value.
            retval = PAPI_write(EventSet, &UserLimitGiven[i]);
            if( retval != PAPI_OK ) {
                fprintf(stderr, "PAPI_write(User Limit) device %i failed, returned %i [%s]. May require sudo status.\n", i, retval, PAPI_strerror(retval));
                exit(-1); 
            }

            // Check result.
            retval = PAPI_stop(EventSet, values);
            if( retval != PAPI_OK ) {
                fprintf(stderr, "%i: PAPI_stop failed, returned %i [%s].\n", __LINE__, retval, PAPI_strerror(retval));
                exit(-1); 
            }
        
            printf("User Limit %.6f set, readback new Limit: %.6f for %s.\n", (UserLimitGiven[i]/ValueScale), (values[0]/ValueScale), LimitEventName[i]);
            PAPI_cleanup_eventset(EventSet);
        } // end check if user limit given for this device.

    } // end handling setting of user Limits.

    // Eventset is cleaned up. Add all power reading events.
    retval = PAPI_add_events(EventSet, powerEvents, DeviceCount);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events failure (for power reading events) returned %i [%s].\n", retval, PAPI_strerror(retval));
        PAPI_cleanup_eventset(EventSet);
        PAPI_destroy_eventset(&EventSet);
        exit(-1); 
    }

    retval = PAPI_start(EventSet);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failure (for power reading events) returned %i [%s].\n", retval, PAPI_strerror(retval));
        PAPI_cleanup_eventset(EventSet);
        PAPI_destroy_eventset(&EventSet);
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
    while (CTL_Z == 0) {                                        // While I havent received a CTL-Z;
        usleep(Interval*1000);                                  // .. Wait (Interval given in mS, function arg is uS).
        if (CTL_Z) break;                                       // .. CTL-Z may have interrupted usleep.
        t2 = PAPI_get_real_nsec();                              // .. Find end time.
        PAPI_read(EventSet, values);                            // .. Read instantaneous power consumption.
        elapsedSec = ((double) (t2-t1))/1.e09;                  // .. convert elapsed nanoseconds to seconds.
        fprintf(myOut, "%.6f", elapsedSec);                     // .. Time first.
        for (i=0; i<DeviceCount; i++) {                         // .. for each device, 
            fprintf(myOut, "\t%.6f", (values[i]/ValueScale)); // .. print a value,
        }
        fprintf(myOut, "\n");                                   // .. Finish the line.
        fflush(myOut);                                          // .. Always flush (in cased canceled).

        runCount++;                                             // Count a run.
        if (Duration > 0 && elapsedSec >= Duration) break;  // Exit if time is up.
    }

    if (CTL_Z) fprintf(stderr, "Received CTL_Z signal (SIGTSTP).\n");
    else       fprintf(stderr, "Time %i seconds expired.\n", Duration);

    retval = PAPI_stop(EventSet, values);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop failed, returned %i [%s].\n", retval, PAPI_strerror(retval));
        exit(-1); 
    }

    PAPI_cleanup_eventset(EventSet);

    // Reset power limits to originals if capped.
    if (Restore == 'Y' && PowerCapped) {
        for (i=0; i<DeviceCount; i++) {
            printf("Original Power Limit Read: %.6f for %s.\n", (OrigLimitFound[i]/ValueScale), LimitEventName[i]);
            if (UserLimitGiven[i] >= 0) {
                printf("Attempting to reset Power Limit for device %i to %.6f.\n", i, (OrigLimitFound[i]/ValueScale));
                retval = PAPI_add_event(EventSet, limitEvents[i]);
                if( retval != PAPI_OK ) {
                    fprintf(stderr, "PAPI_add_event failure returned %i [%s].\n", retval, PAPI_strerror(retval));
                    exit(-1); 
                }

                retval = PAPI_start(EventSet);
                if( retval != PAPI_OK ) {
                    fprintf(stderr, "PAPI_start failure returned %i [%s].\n", retval, PAPI_strerror(retval));
                    exit(-1); 
                }

                // Try to write user value.
                retval = PAPI_write(EventSet, &OrigLimitFound[i]);
                if( retval != PAPI_OK ) {
                    fprintf(stderr, "PAPI_write(Original Limit) device %i failed, returned %i [%s].\n", i, retval, PAPI_strerror(retval));
                    exit(-1); 
                }

                // Check result.
                retval = PAPI_stop(EventSet, values);
                if( retval != PAPI_OK ) {
                    fprintf(stderr, "%i: PAPI_stop failed, returned %i [%s].\n", __LINE__, retval, PAPI_strerror(retval));
                    exit(-1); 
                }
            
                printf("User Limit %.6f set, readback new Limit: %.6f for %s.\n", (OrigLimitFound[i]/ValueScale), (values[0]/ValueScale), LimitEventName[i]);
                PAPI_cleanup_eventset(EventSet);
            } // end check if user limit given for this device.
        } // end resetting to original limits found on entry.
    } // END if restoring limits.

    fprintf(stderr, "Total reads: %i.\n", runCount);

    //--------------------------------------------------------------------------
    // Generate a gnuplot file instructions.
    //--------------------------------------------------------------------------
    if (ScriptName != NULL) {
        fprintf(myGnuplot, "set xlabel 'Time (sec)'\n");                // label for x axis.
        fprintf(myGnuplot, "set nokey\n");                              // no key needed.
        fprintf(myGnuplot, "set terminal png\n");                       // generate png output when plotting.
        fprintf(myGnuplot, "set title 'Spot Wattage Usage During Run'\n");   // Title of graph.
        fprintf(myGnuplot, "set yrange [0:250]\n");                     // Force the y range.

        for (i=0; i<DeviceCount; i++) {                                 // For each event...
            char *Name = strdup(PowerEventName[i]);                     // Need a shorter name.
            dprintf("Name = '%s'\n", Name);                             // Show it.
            int begin=7;                                                // start of name after nvml:::.
            char *dev = strstr(Name, "device=");                        // Find location of device.
            int did = atoi(dev+7);                                      // Get device.
            int dpos = dev-Name-1;                                      // Last position to copy.
            dprintf("begin=%i, dpos=%i.\n", begin, dpos);
            for (j=begin; j<dpos; j++) Name[j-begin] = Name[j];         // Slide down the name.
            Name[dpos-begin]=0;                                         // Z-terminate.
            dprintf("Name = '%s'\n", Name);                             // Show it.

            fprintf(myGnuplot, "set ylabel '%s_%i'\n", Name, did);                      // label for y axis.
            fprintf(myGnuplot, "set output 'plot_%s_%i.png'\n", Name, did);             // Unique output file.
            fprintf(myGnuplot, "plot 'PowerReadGPU.tsv' using 1:%i with lines\n", i+2); // Always time against value. (columns are 1 relative).
            free(Name);
        }        

        fclose(myGnuplot);                                          // close file.
    } // end if user wanted a script.

    // Clean up and exit.
    PAPI_stop(EventSet, values);                                // killing PAPI event set.
    PAPI_cleanup_eventset(EventSet);                            // ..
    PAPI_destroy_eventset(&EventSet);                           // ..
    fclose(myOut);                                              // Close the file.
    return 0;
} // end main.


