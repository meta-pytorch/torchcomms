//-----------------------------------------------------------------------------
// This program must be compiled using a special makefile:
// make -f ROCM_SMI_Makefile rocm_smi_all.out 
//-----------------------------------------------------------------------------
#define __HIP_PLATFORM_HCC__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include <hip/hip_runtime.h>
#include <unistd.h>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

// THIS MACRO EXITS if the papi call does not return PAPI_OK. Do not use for routines that
// return anything else; e.g. PAPI_num_components, PAPI_get_component_info, PAPI_library_init.
#define CALL_PAPI_OK(papi_routine)                                                        \
    do {                                                                                  \
        int _papiret = papi_routine;                                                      \
        if (_papiret != PAPI_OK) {                                                        \
            fprintf(stderr, "%s:%d macro: PAPI Error: function " #papi_routine " failed with ret=%d [%s].\n", \
                    __FILE__, __LINE__, _papiret, PAPI_strerror(_papiret));               \
            exit(-1);                                                                     \
        }                                                                                 \
    } while (0);


#define MEMORY_ALLOCATION_CALL(var)                                     \
    do {                                                                \
        if (var == NULL) {                                              \
            fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",\
                    __FILE__, __LINE__);                                \
            exit(-1);                                                   \
        }                                                               \
    } while (0);  


#define MAX_DEVICES    (32)
#define BLOCK_SIZE     (1024)
#define GRID_SIZE      (512)
#define BUF_SIZE       (32 * 1024)
#define ALIGN_SIZE     (8)
#define SUCCESS        (0)
#define NUM_METRIC     (18)
#define NUM_EVENTS     (2)
#define MAX_SIZE       (64*1024*1024)   // 64 MB

typedef union
{
    long long ll;
    unsigned long long ull;
    double    d;
    void *vp;
    unsigned char ch[8];
} convert_64_t;

typedef struct {
    char name[128];
    long long value;
    int flagged;
} eventStore_t;

int eventsFoundCount = 0;               // occupants of the array.
int eventsFoundMax;                     // Size of the array.
int eventsFoundAdd = 32;                // Blocksize for increasing the array.
int deviceCount=0;                      // Total devices seen.
int deviceEvents[32] = {0};             // Number of events for each device=??.
int globalEvents = 0;                   // events without a "device=".
eventStore_t *eventsFound = NULL;       // The array.

//-----------------------------------------------------------------------------
// HIP routine: Square each element in the array A and write to array C.
//-----------------------------------------------------------------------------
template <typename T>
__global__ void
vector_square(T *C_d, T *A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

//-----------------------------------------------------------------------------
// FreeGlobals: Frees globally allocated memories.
//-----------------------------------------------------------------------------
void FreeGlobals(void) 
{
   return;
} // end routine.


// Show help.
//-----------------------------------------------------------------------------
static void printUsage()
{
    printf("Demonstrate use of ROCM APIs\n");
    printf("This program has no options, it will attempt to try every combination of rocm PAPI events\n");
    printf("and report the results.                                                                  \n");
} // end routine.


//-----------------------------------------------------------------------------
// Interpret command line flags.
//-----------------------------------------------------------------------------
void parseCommandLineArgs(int argc, char *argv[])
{
    if(argc < 2) return;

    if((strcmp(argv[1], "--help") == 0) || 
       (strcmp(argv[1], "-help") == 0)  || 
       (strcmp(argv[1], "-h") == 0)) {
        printUsage();
        exit(0);
    }
} // end routine.

//-----------------------------------------------------------------------------
// Add an entry to the eventsFound array. On entry we always have room.
//-----------------------------------------------------------------------------
void addEventsFound(char *eventName, long long value) {
    strncpy(eventsFound[eventsFoundCount].name, eventName, 127);    // Copy up to 127 chars.
    eventsFound[eventsFoundCount].value = value;                    // Copy the value.
    eventsFound[eventsFoundCount].flagged = 0;                      // Not flagged yet.


    if (++eventsFoundCount >= eventsFoundMax) {                     // bump count, if too much, make room.
        eventsFoundMax += eventsFoundAdd;                           // Add.
        eventsFound = (eventStore_t*) realloc(eventsFound, eventsFoundMax*sizeof(eventStore_t));    // Make new room.
        memset(eventsFound+(eventsFoundMax-eventsFoundAdd), 0, eventsFoundAdd*sizeof(eventStore_t));    // zero it.
    }
} // end routine.

//-----------------------------------------------------------------------------
// conduct a test using HIP. Derived from AMD sample code 'square.cpp'.
// coming in, EventSet is already populated, we just run the test and read.
// Note values must point at an array large enough to store the events in
// Eventset.
//-----------------------------------------------------------------------------
void conductTest(int EventSet, int device, long long *values) {
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);
    int ret, thisDev, verbose=0;

	ret = PAPI_start( EventSet );
	if (ret != PAPI_OK ) {
	    fprintf(stderr,"Error! PAPI_start\n");
	    exit( ret );
	}

    hipDeviceProp_t props;                        
    if (verbose) fprintf(stderr, "args: EventSet=%i, device=%i, values=%p.\n", EventSet, device, values);
 
    CHECK(hipSetDevice(device));                      // Set device requested.
    CHECK(hipGetDevice(&thisDev));                    // Double check.
    CHECK(hipGetDeviceProperties(&props, thisDev));   // Get properties (for name).
    if (verbose) fprintf (stderr, "info: Requested Device=%i, running on device %i=%s\n", device, thisDev, props.name);

    if (verbose) fprintf (stderr, "info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);                     // standard malloc for host.
    CHECK(A_h == NULL ? hipErrorMemoryAllocation : hipSuccess );
    C_h = (float*)malloc(Nbytes);                     // standard malloc for host.
    CHECK(C_h == NULL ? hipErrorMemoryAllocation : hipSuccess );

    // Fill with Phi + i
    for (size_t i=0; i<N; i++) 
    {
        A_h[i] = 1.618f + i; 
    }

    if (verbose) fprintf (stderr, "info: allocate device mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));                   // HIP malloc for device.
    CHECK(hipMalloc(&C_d, Nbytes));                   // ...


    if (verbose) fprintf (stderr, "info: copy Host2Device\n");
    CHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));  // Copy (*dest, *source, Type).

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    if (verbose) fprintf (stderr, "info: launch 'vector_square' kernel\n");
    hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

    if (verbose) fprintf (stderr, "info: copy Device2Host\n");
    CHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));  // copy (*dest, *source, Type).

    if (verbose) fprintf (stderr, "info: check result\n");
    for (size_t i=0; i<N; i++)  {
        if (C_h[i] != A_h[i] * A_h[i]) {              // If value received is not square of value sent,
            CHECK(hipErrorUnknown);                   // ... We have a problem!
        }
    }

    // We passed. Now we need to read the event.
    if (verbose) fprintf(stderr, "Passed. info: About to read event with PAPI_stop.\n");
    ret = PAPI_stop( EventSet, values );
    if (ret != PAPI_OK ) {
        fprintf(stderr,"Error! PAPI_stop failed.\n");
        if (verbose) fprintf(stderr, "PAPI_stop failed.\n");
        exit(ret);
    }
    
    if (verbose) fprintf (stderr, "PAPI_stop succeeded.\n");

    // Memory cleanup, on device and host.
    CHECK(hipFree(A_d));                   // HIP free for device.
    CHECK(hipFree(C_d));                   // ...
    free(A_h);
    free(C_h);
} // end conductTest.

//-----------------------------------------------------------------------------
// Main program.
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int device, i = 0;
    size_t freeMemory = 0, totalMemory = 0;
    char str[64];
    (void) freeMemory;
    (void) totalMemory;
    (void) str;

    eventsFoundMax = eventsFoundAdd;                            // space allocated.
    eventsFound = (eventStore_t*) calloc(eventsFoundMax, sizeof(eventStore_t)); // make some space.

    // Parse command line arguments
    parseCommandLineArgs(argc, argv);

    // fprintf(stderr, "Setup PAPI counters internally (PAPI)\n");
    int EventSet = PAPI_NULL;
    int eventCount;
    int ret;
    int k, m, cid=-1;

    /* PAPI Initialization */
    ret = PAPI_library_init(PAPI_VER_CURRENT);
    if(ret != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed, ret=%i [%s]\n", 
            ret, PAPI_strerror(ret));
        FreeGlobals();
        exit(-1);
    }

    printf("PAPI version: %d.%d.%d\n", 
        PAPI_VERSION_MAJOR(PAPI_VERSION), 
        PAPI_VERSION_MINOR(PAPI_VERSION), 
        PAPI_VERSION_REVISION(PAPI_VERSION));
    fflush(stdout);

    // Find rocm_smi component index.
    k = PAPI_num_components();                                          // get number of components.
    for (i=0; i<k && cid<0; i++) {                                      // while not found,
        PAPI_component_info_t *aComponent = 
            (PAPI_component_info_t*) PAPI_get_component_info(i);        // get the component info.     
        if (aComponent == NULL) {                                       // if we failed,
            fprintf(stderr,  "PAPI_get_component_info(%i) failed, "
                "returned NULL. %i components reported.\n", i,k);
            FreeGlobals();
            exit(-1);    
        }

       if (strcmp("rocm_smi", aComponent->name) == 0) cid=i;            // If we found our match, record it.
    } // end search components.

    if (cid < 0) {                                                      // if no PCP component found,
        fprintf(stderr, "Failed to find rocm_smi component among %i "
            "reported components.\n", k);
        FreeGlobals();
        PAPI_shutdown();
        exit(-1); 
    }

    printf("Found ROCM_SMI Component at id %d\n", cid);

    // Add events at a GPU specific level ... eg rocm:::device=0:Whatever
    eventCount = 0;
    int eventsRead=0;

   // Begin enumeration of all events.

   printf("Events with numeric values were read; if they are zero, they may not  \n"
          "be operational, or the exercises performed by this code do not affect \n"
          "them. We report all 'rocm' events presented by the rocm component.    \n"
          "\n"
          "------------------------Event Name Found------------------------:---Value---\n");

    PAPI_event_info_t info;                                             // To get event enumeration info.
    m=PAPI_NATIVE_MASK;                                                 // Get the PAPI NATIVE mask.
    CALL_PAPI_OK(PAPI_enum_cmp_event(&m,PAPI_ENUM_FIRST,cid));          // Begin enumeration of ALL papi counters.
    do {                                                                // Enumerate all events.
        memset(&info,0,sizeof(PAPI_event_info_t));                      // Clear event info.
        k=m;                                                            // Make a copy of current code.

        // enumerate sub-events, with masks. For this test, we do not
        // have any! But we do this to test our enumeration works as
        // expected. First time through is guaranteed, of course.

        do {                                                            // enumerate masked events. 
            CALL_PAPI_OK(PAPI_get_event_info(k,&info));                 // get name of k symbol.
            char *devstr = strstr(info.symbol, "device=");              // look for device enumerator.
            if (devstr != NULL) {                                       // If device specific,
                device=atoi(devstr+7);                                      // Get the device id, for info.
//              fprintf(stderr, "Found rocm symbol '%s', device=%i.\n", info.symbol , device);
                if (device < 0 || device >= 32) continue;                   // skip any not in range.
            } else {                                                    // A few are system wide.
//              fprintf(stderr, "Found rocm symbol '%s'.\n", info.symbol);
                globalEvents++;                                         // Add to global events.
                device=0;                                               // Any device will do.
            }

            // Filter for strings being returned.
            int isString = 0;

            if (strstr(info.symbol, "device_brand:")            != NULL) isString=1;
            if (strstr(info.symbol, "device_name:")             != NULL) isString=1;
            if (strstr(info.symbol, "device_serial_number:")    != NULL) isString=1;
            if (strstr(info.symbol, "device_subsystem_name:")   != NULL) isString=1;
            if (strstr(info.symbol, "vbios_version:")           != NULL) isString=1;
            if (strstr(info.symbol, "vendor_name:")             != NULL) isString=1;
            if (strstr(info.symbol, "driver_version_str:")      != NULL) isString=1;

            // Filter out crashers.
            if (strstr(info.symbol, "temp_current:device=0:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_critical:device=0:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_critical_hyst:device=0:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_emergency:device=0:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_emergency:device=0:sensor=3") != NULL) continue;  

            if (strstr(info.symbol, "temp_current:device=1:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_critical:device=1:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_critical_hyst:device=1:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_emergency:device=1:sensor=3") != NULL) continue;  
            if (strstr(info.symbol, "temp_emergency:device=1:sensor=3") != NULL) continue;  

            CALL_PAPI_OK(PAPI_create_eventset(&EventSet)); 
            CALL_PAPI_OK(PAPI_assign_eventset_component(EventSet, cid)); 

            ret = PAPI_add_named_event(EventSet, info.symbol);          // Don't want to fail program if name not found...
            if(ret == PAPI_OK) {
                eventCount++;                                           // Bump number of events we could test.
                if (deviceEvents[device] == 0) deviceCount++;           // Increase count of devices if first for this device.
                deviceEvents[device]++;                                 // Add to count of events on this device.
            } else {
                fprintf(stderr, "FAILED to add event '%s', ret=%i='%s'.\n", info.symbol, ret, PAPI_strerror(ret));
                CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));          // Delete all events in set.
                CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));         // destroy the event set.
                continue; 
            }

            long long value=0;                                          // The only value we read.
            
            // Prep stuff.
    
            fprintf(stderr, "conductTest on single event: %s.\n", info.symbol);
            conductTest(EventSet, device, &value);                      // Conduct a test, on device given. 
            addEventsFound(info.symbol, value);                         // Add to events we were able to read.
            
            CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));              // Delete all events in set.
            CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));             // destroy the event set.

            // report each event counted.
            eventsRead++;                                               // .. count and report.
            if (value == 0) {
                printf("%-64s: %lli (perhaps not exercised by current test code.)\n", info.symbol, value);
            } else {
                if (isString) printf("%-64s: %-64s\n", info.symbol, ((char*) value));
                else         printf("%-64s: %lli\n", info.symbol, value);
            }
        } while(PAPI_enum_cmp_event(&k,PAPI_NTV_ENUM_UMASKS,cid)==PAPI_OK); // Get next umask entry (bits different) (should return PAPI_NOEVNT).
    } while(PAPI_enum_cmp_event(&m,PAPI_ENUM_EVENTS,cid)==PAPI_OK);         // Get next event code.

//  fprintf(stderr, "%s:%i Finished Event Loops.\n", __FILE__, __LINE__);

    if (eventCount < 1) {                                                   // If we failed on all of them,
        fprintf(stderr, "Unable to add any ROCM events; they are not present in the component.\n");
        fprintf(stderr, "Unable to proceed with this test.\n");
        FreeGlobals();
        PAPI_shutdown();                                                    // Returns no value.
        exit(-1);                                                           // exit no matter what.
    }
        
    if (eventsRead < 1) {                                                   // If failed to read any,
        fprintf(stderr, "\nFailed to read any ROCM events.\n");             // report a failure.
        fprintf(stderr, "Unable to proceed with pair testing.\n");
        FreeGlobals();
        PAPI_shutdown();                                                    // Returns no value.
        exit(-1);                                                           // exit no matter what.
    }

    printf("\nTotal ROCM events identified: %i.\n\n", eventsFoundCount);

    // EARLY SHUT DOWN.
//  PAPI_shutdown();
//  return(0);

    // Next section is pair testing information. 
    if (eventsFoundCount < 2) {                                             // If failed to get counts on any,
        printf("Insufficient events are exercised by the current test code to perform pair testing.\n"); // report a failure.
        FreeGlobals();
        PAPI_shutdown();                                                    // Returns no value.
        exit(0);                                                            // exit no matter what.
    }


    for (i=0; i<32; i++) {
        if (deviceEvents[i] == 0) continue;                             // skip if none found.
        if (i==0 && globalEvents >0) {
            printf("Device %i assigned %i events (%i of which are not device specific). %i potential pairings for this device.\n", i, deviceEvents[i], globalEvents, deviceEvents[i]*(deviceEvents[i]-1)/2);
        } else {
            printf("Device %i assigned %i events. %i potential pairings for this device.\n", i, deviceEvents[i], deviceEvents[i]*(deviceEvents[i]-1)/2);
        }
    }

    // Begin pair testing. We consider every possible pairing of events
    // that, tested alone, returned a value greater than zero.
//  fprintf(stderr, "Begin Pair Testing.\n");

    int mainEvent, pairEvent, mainDevice, pairDevice;
    long long readValues[2];
    int  goodOnSame=0, failOnDiff=0, badSameCombo=0, pairProblems=0;        // Some counters.
    int type;                                                               // 0 succeed on same device, 1 = fail across devices.
    for (type=0; type<2; type++) {
        if (type == 0) {
            printf("List of Pairings on SAME device:\n");
            printf("* means value changed by more than 10%% when paired (vs measured singly, above).\n");
            printf("^ means a pair was rejected as an invalid combo.\n");
        } else {
            printf("List of Pairings causing an error when on DIFFERENT devices:\n");
        }

        for (mainEvent = 0; mainEvent<eventsFoundCount-1; mainEvent++) {                // Through all but one events.
             char *devstr = strstr(eventsFound[mainEvent].name, "device=");             // look for device enumerator.
             if (devstr != NULL) mainDevice=atoi(devstr+7);                             // Get the device id.
             else mainDevice=0;                                                         // Use zero if not found.
            
            for (pairEvent = mainEvent+1; pairEvent<eventsFoundCount; pairEvent++) {    // Through all possible pairs,
                devstr = strstr(eventsFound[pairEvent].name, "device=");                // look for device enumerator.
                if (devstr != NULL) pairDevice=atoi(devstr+7);                          // Get the device id.
                else pairDevice=0;                                                      // Use zero if not found.

                if (type == 0 && mainDevice != pairDevice) continue;                    // Skip if we need same device.
                if (type == 1 && mainDevice == pairDevice) continue;                    // Skip if we need different devices.

                CALL_PAPI_OK(PAPI_create_eventset(&EventSet)); 
                CALL_PAPI_OK(PAPI_assign_eventset_component(EventSet, cid)); 
                CALL_PAPI_OK(PAPI_add_named_event(EventSet, eventsFound[mainEvent].name));
                // Here we must examine the return code.
                int ret = PAPI_add_named_event(EventSet, eventsFound[pairEvent].name);
                if (type == 0 && ret == PAPI_ECOMBO) {                                  // A bad combination when looking for valid combos.
                    printf("%c %64s + %-64s [Invalid Combo]\n", '^',                    // report it.
                        eventsFound[mainEvent].name, eventsFound[pairEvent].name);
                    badSameCombo++;                                                     // .. count an explicit rejection.
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. done with event set.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                if (type == 1 && ret == PAPI_ECOMBO) {                                  // A bad  combination when we are looking for that.
                    printf("%64s + %-64s BAD COMBINATION ACROSS DEVICES.\n", 
                        eventsFound[mainEvent].name, eventsFound[pairEvent].name);      // report it.
                    failOnDiff++;                                                       // count the bad combos.
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. don't need to go further.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                if (ret != PAPI_OK) {                                                   // If it failed for some other reason,
                    fprintf(stderr, "%s:%d Attempt to add event '%s' to set "
                            "with event '%s' produced an unexpected error: "
                            "[%s]. Ignoring this pair.\n", 
                        __FILE__, __LINE__, eventsFound[pairEvent].name, 
                        eventsFound[mainEvent].name, PAPI_strerror(ret));
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. didn't work.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                // We were able to add the pair. In type 1, we just skip it,
                // because we presume a single event on a device isn't changed
                // by any event on another device.
                if (type == 1) {
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. worked fine; don't measure it.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                // We were able to add the pair, in type 0, get a measurement. 
                readValues[0]= -1; readValues[1] = -1;
//              fprintf(stderr, "conductTest on paired events: '%s' v. '%s'.\n", eventsFound[mainEvent].name, eventsFound[pairEvent].name);
                conductTest(EventSet, device, readValues);                              // Conduct a test, on device given. 
//              fprintf(stderr, "readValues[0]=%lli readValues[1]=%lli.\n", readValues[0], readValues[1]);
                goodOnSame++;                                                           // Was accepted by cuda as a valid pairing.

                // For the checks, we add 2 (so -1 becomes +1) to avoid any
                // divide by zeros. It won't make a significant difference 
                // in the ratios. (none if readings are the same). 
                double mainSingle = (2.0 + eventsFound[mainEvent].value);               // Get value when read alone.
                double pairSingle = (2.0 + eventsFound[pairEvent].value);               // ..
                double mainCheck  = mainSingle/(2.0 + readValues[0]);                   // Get ratio when paired.
                double pairCheck  = pairSingle/(2.0 + readValues[1]);                   // ..

                char flag=' ', flag1=' ', flag2=' ';                                    // Presume all okay.
                if (mainCheck < 0.90 || mainCheck > 1.10) {                             // Flag as significantly different for main.
                    flag1='*';
                    eventsFound[mainEvent].flagged = 1;                                 // .. remember this event is suspect.
                }

                if (pairCheck < 0.90 || pairCheck > 1.10) {                             // Flag as significantly different for pair.
                    flag2='*';                    
                    eventsFound[pairEvent].flagged = 1;                                 // .. remember this event is suspect.
                }

                if (flag1 == '*' || flag2 == '*') {
                    pairProblems++;                                                     // Remember number of problems.
                    flag = '*';                                                         // set global flag.
                }

                printf("%c %64s + %-64s [", flag, eventsFound[mainEvent].name, eventsFound[pairEvent].name);
                if (flag1 == '*') printf("%c%lli (vs %lli),", flag1, readValues[0], eventsFound[mainEvent].value);
                else              printf("%c%lli,", flag1, readValues[0]);

                if (flag2 == '*') printf("%c%lli (vs %lli)]\n", flag2, readValues[1], eventsFound[pairEvent].value);
                else              printf("%c%lli]\n", flag2, readValues[1]);

                CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                          // Delete all events in set.
                CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                         // destroy the event set.
            } // end for each possible pairing event.
        } // end loop for each possible primary event.

        if (type == 0) {                                                                // For good pairings on same devices,
            if (goodOnSame == 0) {
                printf("NO valid pairings of above events if both on the SAME device.\n");
            } else {
                printf("%i valid pairings of above events if both on the SAME device.\n", goodOnSame);
            }

            printf("%i unique pairings on SAME device were rejected as bad combinations.\n", badSameCombo);
            
            if (pairProblems > 0) {
                printf("%i pairings resulted in a change of one or both event values > 10%%.\n", pairProblems);
                printf("The following events were changed by pairing:\n");
                for (mainEvent = 0; mainEvent<eventsFoundCount; mainEvent++) {          // Through all events.
                    if (eventsFound[mainEvent].flagged) {                               // ..If it was flagged,
                        printf("%s\n", eventsFound[mainEvent].name);                    // .... report it.
                    } 
                } // end for each event.
            } else {
                printf("No significant change in event values read for any pairings.\n");
            }
        } else {                                                                        // Must be reporting bad pairings across devies.
            if (failOnDiff == 0) printf("NO pairings of above events caused an error if each were on DIFFERENT devices.\n");
            else printf("%i pairings of above events caused an error with each on a DIFFERENT device.\n", failOnDiff);
        }
    } // end loop on type.

    PAPI_shutdown();                                                                    // Returns no value.
    return(0);                                                                          // exit OK.
} // end MAIN.
