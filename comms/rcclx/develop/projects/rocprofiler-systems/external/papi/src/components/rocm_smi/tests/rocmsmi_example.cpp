//-----------------------------------------------------------------------------
// rocmsmi_example.cpp is a minimal example of using PAPI with rocm_smi. SMI
// is for "System Management Interface", it provide information on hardware 
// sensors such as fan speed, temperature, and power consumption.
//
// Unfortunately, the power consumption is a "spot" reading, and most users
// want a reading *during* the execution of a function, not just before or
// after a call to the GPU kernel returns; when the GPU will likely be at or
// near its idling power.
//
// In this example, we will show how to call a function from rocblas (the sgemm
// routine (single precision general matrix multiply), while using a separate
// pthread to sample and record time & power while it runs.

// This is intended as a simple example upon which programmers can expand; for
// a more comprehensive approach see power_monitor_rocm.cpp, that can deal with
// multiple GPUs and allows power-capping and other output control. It is in
// this same directory. power_monitor_rocm is a standalone code, run in the
// background to monitor another application (two processes). (On some clusters
// you must ensure the GPU *can* be shared by two processes simultaneously.)

// A separate process has the advantage of being able to monitor library code
// and other application code that has no PAPI instrumentation code in it. The
// pthread approach coded here has the advantage of being a single executable
// and more flexible, for example you can incorporate other elements into your
// output, such as PAPI event values into the timed outputs, or output labels
// to indicate processing landmarks, etc. For example, with multiple papi event
// sets, we could also read device temperature with every sample, or memory
// usage or cache statistics, or even I/O bandwidth statistics on devices other
// than the GPU, and output those stats for the same time steps as our power
// consumption.

// rocblas is generally part of the installed package from AMD, in
// $PAPI_ROCM_ROOT/rocblas/, with subdirectories /lib and /include.  We don't
// use HipBlas (also included by AMD), HipBlas is a higher level "switch" that
// calls either cuBlas or rocBlas. This would be an unnecessary complication
// for this example.
//
// The corresponding Makefile is also instructional. 

// An advantage of rocblas is it is also a "switch", automatically detecting
// the hardware and using the appropriate tuned code for it.

// > make rocmsmi_example 

// This code is intentionally heavily commented to be instructional.  We use
// the library code to exercise the GPU with a realistic workload instead of a
// toy kernel. For examples of how to include your own kernels, see the
// distribution directory $PAPI_ROCM_ROOT/hip/samples/, which contains
// sub-directories with working applications.

// To Compile, the environment variable PAPI_ROCM_ROOT must be defined to point
// at a rocm directory. No other environment variables are necessary, but if
// you wish to use ROCM events or events from other components, check the
// appropriate README.md files for instructions. These will be found in
// papi/src/components/COMPONENT_NAME/README.md files. 

// Because this program uses AMD HIP functions to manage memory on the GPU, it
// must be compiled with the hipcc compiler. Typically this is found in:
// $PAPI_ROCM_ROOT/bin/hipcc 

// hipcc is a c++ compiler, so c++ conventions for strings must be followed.
// (PAPI does not require c++; it is simple C; but PAPI++ will require c++).

// Note for Clusters: Many clusters have "head nodes" (aka "login nodes") that
// do not contain any gpus; the head node is used for compiling but the code is
// actually run on a batch node (e.g. using SLURM and srun). Ensure when
// running code that must access a GPU, including our utilities like
// papi_component_avail and papi_native_avail, and this example code, that the
// code is run on a batch node, not the head node. 
//-----------------------------------------------------------------------------

// Necessary to specify a platform for the hipcc compiler.
#define __HIP_PLATFORM_HCC__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#include "papi.h"
#include "rocblas.h"
#include <hip/hip_runtime.h>

// This is a structure to hold variables for our work example to exercise the
// GPU. The work is an SGEMM, using AMD's rocblas library. PrepWork() fills in
// this structure.

typedef struct rb_sgemm_parms {
    rocblas_handle    handle;
    hipStream_t       hipStream;
    rocblas_status    rb_ret;
    rocblas_operation transA;
    rocblas_operation transB;
    rocblas_int       m;
    rocblas_int       n;
    rocblas_int       k;
    float*            alpha;
    float*            A;        // Host side.
    float*            A_d;      // Device side.
    rocblas_int       lda;
    float*            B;        // Host side.
    float*            B_d;      // Device side.
    rocblas_int       ldb;
    float*            beta;
    float*            C;        // Host side.
    float*            C_d;      // Device side.
    rocblas_int       ldc;
} rb_sgemm_parms_t;

enum {
    samplingCommand_wait = -1,
    samplingCommand_record = 0,
    samplingCommand_exit = 1
};

typedef struct samplingOrders {
    // To avoid any multi-thread race conditions:
    // The following variables are only written by Main, not by the sampling Thread.
    int               EventSet;     // The events to read.
    volatile int      command;      // -1=wait, 0=sample and record, 1=exit.
    int               maxSamples;   // extent of array.

    // The following variables are only written by the Thread, not by Main.
    volatile int      sampleIdx;    // next sample to store.
    long long         *samples;     // array of [2*maxSamples] allocated by caller. Only first event is stored.
                                    // Note each sample takes two slots; for nanosecond time stamp and event value.
    long long         *eventsRead;  // array long enough to read all samples in EventSet, allocated by caller.
} samplingOrders_t;

// prototypes for doing some work to exercise the GPU.
rb_sgemm_parms_t* PrepWork(int MNK);
void DoWork(rb_sgemm_parms_t *myParms);
void FinishWork(rb_sgemm_parms_t *myParms);

// Used to initialize arrays.
#define SRAND48_SEED 1001

// Select GPU to test [0,1,...] 
#define GPU_TO_TEST 0
#define MS_SAMPLE_INTERVAL 5
#define VERBOSE 0 /* 1 prints action-by-action report for debugging. */


//-----------------------------------------------------------------------------
// This is the timer pthread; note that time intervals of 2ms or less may be
// a problem for some operating systems. The operation here is simple, we 
// sample, store the value read in an array, increment our index, and sleep.
// but we also check our passed in structure to look for an exit signal. If 
// we run out of room in the provided array, we stop sampling. This is an 
// example, programmers can get more sophisticated if they like.
// The PAPI EventSet must already be initialized, built, and started; all this
// thread does is PAPI_read().
//-----------------------------------------------------------------------------
long long countInterrupts=0;
void* sampler(void* vMyOrders) {
    int ret;
    struct timespec req, rem, cont;
    if (VERBOSE) fprintf(stderr, "entered sampler.\n");
    samplingOrders_t *myOrders = (samplingOrders_t*) vMyOrders; // recast for use.

    // Initializations first time through.
    // These are the only two elements in timespec.
    cont.tv_sec=0;                              // always.
    req.tv_sec=0;
    req.tv_nsec = MS_SAMPLE_INTERVAL*(1000000);
    myOrders->sampleIdx = 0;
    myOrders->eventsRead[0]=0;
    
    // Sleep for time given in req. If interrupted by a signal, time remaining in rem.
    while (myOrders->command != samplingCommand_exit) {  // run until instructed to exit. 
        ret = nanosleep(&req, &rem);        // sleep.
        while (ret != 0) {                  // If interrupted by a signal (almost never happens),
            countInterrupts++; 
            cont = rem;                     // Set up continuation.
            ret = nanosleep(&cont, &rem);   // try again.  
        }
        
        // We have completed a sleep cycle. If we need to sample, do that.
        if (myOrders->command == samplingCommand_record && 
            myOrders->sampleIdx < myOrders->maxSamples) {   
            long long nsElapsed=0;
            int x2 = myOrders->sampleIdx<<1;            // compute double the sample index.
            long long tstamp = PAPI_get_real_nsec();    // PAPI function returns 
            ret = PAPI_read(myOrders->EventSet, myOrders->eventsRead);
            if (x2 > 0) nsElapsed=tstamp - myOrders->samples[x2-2];
            if (VERBOSE) fprintf(stderr, "reading sample %d elapsed=%llu.\n", myOrders->sampleIdx, nsElapsed);
            if (ret == PAPI_OK) {
                myOrders->samples[x2]=tstamp;
                myOrders->samples[x2+1]=myOrders->eventsRead[0];
                myOrders->sampleIdx++;
            } else {
                if (VERBOSE) fprintf(stderr, "Failed PAPI_read(EventSet, values), retcode=%d ->'%s'\n", ret, PAPI_strerror(ret));
                exit(ret);
            }
        }
    } // end while.
                
    // We are exiting the thread. The data is contained in the passed in structure.
    if (VERBOSE) fprintf(stderr, "exiting sampler.\n");
    pthread_exit(NULL);
} // end pthread sampler.

//-----------------------------------------------------------------------------
// Begin main code.
//-----------------------------------------------------------------------------
int main( int argc, char **argv )
{
    (void) argc;
    (void) argv;

    int i, x2, retval;
    int EventSet = PAPI_NULL;

    // Step 1: Initialize the PAPI library. The library returns its version,
    // and we compare that to the include file version to ensure compatibility.
    // if for some reason the PAPI_libary_init() fails, it will not return 
    // its version but an error code.
    // Note that PAPI_strerror(retcode) will return a pointer to a string that
    // describes the error. 

    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if (retval != PAPI_VER_CURRENT ) {
        fprintf(stderr,"Error! PAPI_library_init failed, retcode=%d -> '%s'\n",
            retval, PAPI_strerror(retval));
        exit(retval);
    }

    // Step 2: Create an event set. A PAPI event set is a collection of 
    // events we wish to read. We can add new events, delete events, and
    // so on. Upon creation the event set is empty.
    // Note that the EventSet is just an integer, an index into an internal 
    // array of event sets. We pass the address so PAPI can populate the
    // integer with the correct index.

    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK ) {
        fprintf(stderr,"Error! PAPI_create_eventset failed, retcode=%d -> '%s'\n",
            retval, PAPI_strerror(retval));
        exit(retval);
    }

    // When we read an eventset it returns an array of values, one per event
    // contained in the eventset. PAPI always works in 'long long' 64 bit
    // values.  For some events the value returned should be recast to some
    // other type, e.g. unsigned long long, or even floating point.  ROCM has
    // several events that are percentages in the range [0,100].  In general it
    // is up to the programmers to know how to use the event data, based on the
    // event description, and whether it needs to be recast.  Event
    // descriptions can be obtained using the PAPI utility PAPI_native_avail,
    // found in the papi/src/utils directory. The following is an excerpt of a
    // few such such events reported by PAPI_native_avail (out of hundreds).
    // Note: Be sure to use the events for the device you are testing; defined
    //       above as 'GPU_TO_TEST'. 

    //--------------------------------------------------------------------------------
    //| rocm_smi:::power_average:device=0:sensor=0                                   |
    //|            Current Average Power consumption in microwatts. Requires root pri|
    //|            vilege.                                                           |
    //--------------------------------------------------------------------------------

    // We declare an array of long long here to receive counter values from
    // PAPI. It could also be allocated if the number of events are not known
    // at compile time.

    long long values[1]={0};  // declare a single event value.

    // We define a string for the event we choose to monitor. 
    char eventname[]="rocm_smi:::power_average:device=0:sensor=0";

    // Now we add the named event to the event set. It is also possible to add
    // names by their numerical code; but most applications use the named
    // events. Be aware that the numeric codes for the same event can change
    // from run to run; so using the name is the safer approach. Also note that
    // what we have done thus far is all setup, we have not tried to
    // communicate with the GPU yet.

    retval = PAPI_add_named_event(EventSet, eventname); // Note we pass pointer to string. 
    if (retval != PAPI_OK) {
        fprintf(stderr, "Failed PAPI_add_named_event(EventSet, %s), retcode=%d ->'%s'\n", eventname, retval, PAPI_strerror(retval));
        exit(retval);
    }

    // Now we ask PAPI to interface with the GPU driver software to start
    // keeping track of this event on the GPU. At this point, if the GPU has
    // some issue with the events in the event set, we may see an error in
    // starting the event counting. One example is if we attempt to combine
    // events that the GPU cannot collect together, due to lack of resources.
    // The GPU may use the same hardware counter that it switches to count
    // event A *or* event B, and thus cannot count both event A *and* event B
    // simultaneously.

    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Failed PAPI_start(EventSet), retcode=%d ->'%s'\n", retval, PAPI_strerror(retval));
        exit(retval);
    }

    // Set up the sampler thread.
    pthread_t samplingThread;
    samplingOrders_t myOrders;
    myOrders.EventSet = EventSet;
    myOrders.command = samplingCommand_wait;            // Wait for me to say start.
    myOrders.maxSamples = 20*1000/MS_SAMPLE_INTERVAL;   // 20 seconds worth.
    myOrders.sampleIdx = 0;
    myOrders.samples = (long long*) calloc(myOrders.maxSamples*2, sizeof(long long));   // allocate space.
    myOrders.eventsRead = values;
    retval = pthread_create(&samplingThread, NULL, sampler, &myOrders);
    if (retval != 0) {
        fprintf(stderr, "pthread_create() failed, retcode=%d. Aborting.\n", retval);
        exit(-1);
    }
    
    if (VERBOSE) fprintf(stderr, "Launched Sampler.\n");
    // Do Some Work: This is a subroutine do just make the GPU do something to
    // run up counters so we have something to report. We do this in three 
    // parts. Information for the run is contained in 'myParms'.
    rb_sgemm_parms_t *myParms = PrepWork(16384); // set up, param is M,N,K.

    // If something went wrong it was reported by PrepWork; so just exit.
    if (myParms->rb_ret != rocblas_status_success) exit(myParms->rb_ret);

    // Start sampling.
    myOrders.command = samplingCommand_record;
    while (myOrders.sampleIdx < 1); 

    // Call rocblas and do an SGEMM.
    long long timeDoWork = PAPI_get_real_nsec();
    DoWork(myParms);
    timeDoWork = PAPI_get_real_nsec() - timeDoWork;
    if (VERBOSE) fprintf(stderr, "DoWork consumed %.3f ms.\n", (timeDoWork+0.)/1000000.0);
    // If something went wrong it was reported by DoWork; so just exit.
    if (myParms->rb_ret != rocblas_status_success) exit(myParms->rb_ret);

    // stop the samplings.
    int checkTime = myOrders.sampleIdx;
    // Wait for some trailing samples, if possible.
    while (myOrders.sampleIdx <= checkTime+5 && myOrders.sampleIdx < myOrders.maxSamples);
    if (VERBOSE) fprintf(stderr, "Stopping Sampler, joining thread.\n");
    myOrders.command = samplingCommand_exit;       // Tell the sampling thread to exit.

    // Wait for the sampler thread to finish.
    retval = pthread_join(samplingThread, NULL); 
    if (retval != 0) {
        fprintf(stderr, "Failed to join the sampling thread, ret=%d.\n", retval);
        exit(retval);
    }

    if (VERBOSE) fprintf(stderr, "Joined Thread.\n");
    // Success. Report what we read for the value.
    if (VERBOSE) printf("Read %d samples, =%.3f seconds. %llu nanosleepInterrupts.\n", myOrders.sampleIdx, (myOrders.sampleIdx*MS_SAMPLE_INTERVAL)/1000., countInterrupts);
    printf("ns timeStamp, ns Diff, microWatts, Joules (Watts*Seconds)\n");

    float duration, avgWatts=0.0, totJoules=0.0;
    long long minWatts=myOrders.samples[1];
    long long maxWatts=minWatts;
    for (i=0; i<myOrders.sampleIdx; i++) {
        x2 = i<<1;
        printf("%llu,", myOrders.samples[x2]);
        if (i==0) printf("0,");
        else      printf("%llu,", myOrders.samples[x2]-myOrders.samples[x2-2]);
        if (myOrders.samples[x2+1] < minWatts) minWatts = myOrders.samples[x2+1];
        if (myOrders.samples[x2+1] > maxWatts) maxWatts = myOrders.samples[x2+1];
        avgWatts += (myOrders.samples[x2+1]+0.0)*1.e-6;
        printf("%llu,", myOrders.samples[x2+1]);
        if (i==0) printf("0.0\n");
        else {
            float w= myOrders.samples[x2+1]*1.e-6;
            float s= (myOrders.samples[x2]-myOrders.samples[x2-2])*1.e-9;
            totJoules += (w*s);
            printf("%.6f\n", (w*s));
        }
    }
   
    x2 = (myOrders.sampleIdx-1)<<1; // Final index.
    duration = (float) (myOrders.samples[x2]-myOrders.samples[0]);
    duration *= 1.e-6;  // compute milliseconds from nano seconds.
    avgWatts /= (myOrders.sampleIdx-1.0);   // one less, first reading is zero.
    printf("ms Duration=%.3f\n", duration);
    printf("ms AvgInterval=%.3f\n", duration/(myOrders.sampleIdx-1));
    printf("avg Watts=%.3f, minWatts=%.3f, maxWatts=%.3f\n", avgWatts, (minWatts*1.e-6), (maxWatts*1.e-6));
    printf("total Joules=%.3f\n", totJoules); 

    // Now we clean up. First, we stop the event set. This will re-do the read;
    // we could prevent that by passing a NULL pointer for the 'values'.

    retval = PAPI_stop( EventSet, values );       // ROCM added stop and test.
    if (retval != PAPI_OK) {
        fprintf(stderr, "Failed PAPI_stop(EventSet, values), retcode=%d ->'%s'\n", retval, PAPI_strerror(retval));
        exit( retval );
    }

    // This is an example of modifying an EventSet. This will happen
    // automatically if we destroyed the event set, but is shown here as an
    // example. We can remove events, add other events, do more work and read
    // those. Of course you should allow room in the 'values[]' array for the
    // maximum number of events you might read.
    
    retval = PAPI_remove_named_event(EventSet, eventname); // remove the event we added.
    if (retval != PAPI_OK) {
        fprintf(stderr, "Failed PAPI_remove_named_event(EventSet, eventname), retcode=%d ->'%s'\n", retval, PAPI_strerror(retval));
        exit(retval);
    }

    // Being good memory citizens, we want to destroy the event set PAPI created for us.
    // Once again, we pass the address of an integer.
    retval=PAPI_destroy_eventset(&EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Failed PAPI_destroy_event(&EventSet), retcode=%d ->'%s'\n", retval, PAPI_strerror(retval));
        exit(retval);
    }

    // Cleanup the work portion. It will delete any allocated memories.
    FinishWork(myParms);

    // And finally, we tell PAPI to do a clean shut down, release all its allocations.
    PAPI_shutdown();                                    // Shut it down.
    return 0;
} // END MAIN.



//-----------------------------------------------------------------------------
// The following are dedicated to rocblas; the PAPI examples are above. 
// rocblas_handle is a structure holding the rocblas library context.
// It must be created using rocblas_create_handle(), passed to all function
// calls, and destroyed using rocblas_destroy_handle().
//-----------------------------------------------------------------------------
static const char* rocblas_return_strings[13]={ // from comments in rocblas_types.h.
    "Success",
    "Handle not initialized, invalid or null",
    "Function is not implemented",
    "Invalid pointer argument",
    "Invalid size argument",
    "Failed internal memory allocation, copy or dealloc",
    "Other internal library failure",
    "Performance degraded due to low device memory",
    "Unmatched start/stop size query",
    "Queried device memory size increased",
    "Queried device memory size unchanged",
    "Passed argument not valid",
    "Nothing preventing function to proceed",
    };


//-----------------------------------------------------------------------------
// helper. Report any info about the context of the error first; eg.
// fprintf(stderr, "calloc(1, %lu) for rb_sgemm_parts_t ", sizeof(rb_sgemm_parms_t));
//-----------------------------------------------------------------------------
void rb_report_error(int ret) {
        fprintf(stderr, "failed, ret=%d -> ", ret);
        if (ret >=0 && ret<=12) {
            fprintf(stderr, "%s.\n", rocblas_return_strings[ret]);
        } else {
            fprintf(stderr, "Meaning Unknown.\n"); 
        }
} // end rb_report_error.

//-----------------------------------------------------------------------------
// helper. Report any info about the context of the error first; eg.
// fprintf(stderr, "calloc(1, %lu) for rb_sgemm_parts_t ", sizeof(rb_sgemm_parms_t));
//-----------------------------------------------------------------------------
void hip_report_error(hipError_t ret) {
        fprintf(stderr, "failed, ret=%d -> %s.\n", ret, hipGetErrorString(ret));
} // end rb_report_error.

rb_sgemm_parms_t* PrepWork(int MNK) {
    // We use calloc to ensure all pointers are NULL. 
    rb_sgemm_parms_t *myParms = (rb_sgemm_parms_t*) calloc(1, sizeof(rb_sgemm_parms_t));

    // Check that we successfully allocated memory.
    if (myParms == NULL) {
        fprintf(stderr, "calloc(1, %lu) for rb_sgemm_parms_t failed.", sizeof(rb_sgemm_parms_t));
        exit(-1);
    }

    hipError_t hipret;
   
    // GPU_TO_TEST is defined at top of file; [0,1,...]
    hipret = hipSetDevice(GPU_TO_TEST);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipSetDevice(%d) ", __FILE__, __func__, __LINE__, GPU_TO_TEST);
        hip_report_error(hipret);
        exit(-1);
    }        
 
    // initialize rocblas.
    rocblas_initialize();

    // rocblas requires the creation of a 'handle' to call any functions.
    myParms->rb_ret = rocblas_create_handle(&myParms->handle);

    // Check that we were able to create a handle.
    if (myParms->rb_ret != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle ");
        rb_report_error(myParms->rb_ret);
        exit(-1);
    }

    // Set constants.
    myParms->m = (rocblas_int) MNK;
    myParms->n = (rocblas_int) MNK;
    myParms->k = (rocblas_int) MNK;
    myParms->lda = (rocblas_int) MNK;
    myParms->ldb = (rocblas_int) MNK;
    myParms->ldc = (rocblas_int) MNK;
    myParms->transA = rocblas_operation_none;
    myParms->transB = rocblas_operation_none;

    // Allocate memory; check each time.
    myParms->alpha = (float*) calloc(1, sizeof(float));
    if (myParms->alpha == NULL) {
        fprintf(stderr, "calloc(1, %lu) failed for myParms->alpha.\n", sizeof(float));
        exit(-1);
    }

    myParms->beta = (float*) calloc(1, sizeof(float));
    if (myParms->beta == NULL) {
        fprintf(stderr, "calloc(1, %lu) failed for myParms->beta.\n", sizeof(float));
        exit(-1);
    }

    myParms->A = (float*) calloc(1, sizeof(float)*MNK*MNK);
    if (myParms->A == NULL) {
        fprintf(stderr, "calloc(1, %lu) failed for myParms->A.\n", sizeof(float)*MNK*MNK);
        exit(-1);
    }

    myParms->B = (float*) calloc(1, sizeof(float)*MNK*MNK);
    if (myParms->B == NULL) {
        fprintf(stderr, "calloc(1, %lu) failed for myParms->B.\n", sizeof(float)*MNK*MNK);
        exit(-1);
    }

    myParms->C = (float*) calloc(1, sizeof(float)*MNK*MNK);
    if (myParms->C == NULL) {
        fprintf(stderr, "calloc(1, %lu) failed for myParms->A.\n", sizeof(float)*MNK*MNK);
        exit(-1);
    }

    // Set up allocated areas.
    myParms->alpha[0]= 1.0;
    myParms->beta[0] = 1.0;

    srand48(SRAND48_SEED);
    
    // Init square arrays, uniform distribution; values [0.0,1.0).
    int i;
    for (i=0; i<(MNK*MNK); i++) {
        myParms->A[i] = (float) drand48();
        myParms->B[i] = (float) drand48();
        myParms->C[i] = (float) drand48();
    } 

    int thisDevice;
    hipret = hipGetDevice(&thisDevice);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipGetDevice(&thisDevice) ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }        

    if (thisDevice != 0) {
        fprintf(stderr, "%s:%s:%i Unexpected result, thisDevice = %d.\n", __FILE__, __func__, __LINE__, thisDevice);
    }        

    // Not used here, but useful for debug.
    hipDeviceProp_t devProps;
    hipret = hipGetDeviceProperties(&devProps, thisDevice); 
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipGetDeviceProperties(&devProps) ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }        
    
    if (0) {
        fprintf(stderr, "info: device=%i name=%s.\n", thisDevice, devProps.name); 
    }

    // Allocate memory on the GPU for three arrays.
    hipret = hipMalloc(&myParms->A_d, sizeof(float)*MNK*MNK);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipMalloc((&myParms->A_d, %lu) ", __FILE__, __func__, __LINE__, sizeof(float)*MNK*MNK);
        hip_report_error(hipret);
        exit(-1);
    }        

    hipret = hipMalloc(&myParms->B_d, sizeof(float)*MNK*MNK);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipMalloc((&myParms->B_d, %lu) ", __FILE__, __func__, __LINE__, sizeof(float)*MNK*MNK);
        hip_report_error(hipret);
        exit(-1);
    }        

    hipret = hipMalloc(&myParms->C_d, sizeof(float)*MNK*MNK);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipMalloc((&myParms->C_d, %lu) ", __FILE__, __func__, __LINE__, sizeof(float)*MNK*MNK);
        hip_report_error(hipret);
        exit(-1);
    }        

    // Copy each array from Host to Device. Note args for
    // hipMemcpy(*dest, *source, count, type of copy)
    hipret = hipMemcpy(myParms->A_d, myParms->A, sizeof(float)*MNK*MNK, hipMemcpyHostToDevice);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipMemcpy(A) HostToDevice) ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }        

    hipret = hipMemcpy(myParms->B_d, myParms->B, sizeof(float)*MNK*MNK, hipMemcpyHostToDevice);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipMemcpy(B) HostToDevice) ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }        

    hipret = hipMemcpy(myParms->C_d, myParms->C, sizeof(float)*MNK*MNK, hipMemcpyHostToDevice);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipMemcpy(C) HostToDevice) ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }        

    return (myParms);
} // end PrepWork.

void DoWork(rb_sgemm_parms_t *myParms) {
    long long elapsed;
    hipError_t hipret;
    (void) elapsed;         // No warnings if not used.

    // Execute the SGEMM we set up.
    myParms->rb_ret = rocblas_sgemm(
        myParms->handle,
        myParms->transA,
        myParms->transB,
        myParms->m,
        myParms->n,
        myParms->k,
        myParms->alpha,
        myParms->A_d,
        myParms->lda,
        myParms->B_d,
        myParms->ldb,
        myParms->beta,
        myParms->C_d,
        myParms->ldc);

    if (myParms->rb_ret != rocblas_status_success) {
        fprintf(stderr, "rocblas_sgemm ");
        rb_report_error(myParms->rb_ret);
        exit(-1);
    }

    long unsigned mtxSize = sizeof(float)*(myParms->m)*(myParms->n);

    // Copy the result (matrix C) back to host space.
    hipret = hipMemcpy(myParms->C, myParms->C_d, mtxSize, hipMemcpyHostToDevice);
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipMemcpy(C) HostToDevice) ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }        

    // AMD GPU "Streams" are command queues for the device.
    // hipDeviceSynchronize() blocks until all streams are empty. Failing to
    // Synchronize has resulted in incorrect reading of performance counters.
    // In timings, this takes about 3 uS if all commands ARE complete. But
    // immediately after the rocblas_call above, it has taken up to 239 ms, and
    // without it, we have read zeros for the performance event. (Typically the
    // memory copy will take long enough that we don't see this; reading zeros
    // for the event occurred when we had a PAPI_read() immediately after the
    // rocblas_sgemm() call.)

    // Example of timing:
    // elapsed = PAPI_get_real_nsec();
    // ... do something ... 
    // elapsed = PAPI_get_real_nsec() - elapsed;
    // fprintf(stderr, "Elapsed time %llu ns.\n", elapsed);

    hipret = hipDeviceSynchronize();
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipDeviceSynchronize() ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }
        
    if (VERBOSE) fprintf(stderr, "Successful rocblas_sgemm with M,N,K=%d,%d,%d.\n", myParms->m, myParms->n, myParms->k);
    return;
} // end DoWork. 

void FinishWork(rb_sgemm_parms_t *myParms) {

    hipError_t hipret;

    // Clean up host memory.
    if (myParms->alpha != NULL) {free(myParms->alpha); myParms->alpha = NULL;}
    if (myParms->beta  != NULL) {free(myParms->beta ); myParms->beta  = NULL;}
    if (myParms->A     != NULL) {free(myParms->A    ); myParms->A     = NULL;}
    if (myParms->B     != NULL) {free(myParms->B    ); myParms->B     = NULL;}
    if (myParms->C     != NULL) {free(myParms->C    ); myParms->C     = NULL;}

    // Clean up device memory.
    if (myParms->A_d != NULL) {
        hipret = hipFree(myParms->A_d);
        if (hipret != hipSuccess) {
            fprintf(stderr, "%s:%s:%i hipFree(myParms->A_d) ", __FILE__, __func__, __LINE__);
            hip_report_error(hipret);
            exit(-1);
        }        
        myParms->A_d = NULL;
    }

    if (myParms->B_d != NULL) {
        hipret = hipFree(myParms->B_d);
        if (hipret != hipSuccess) {
            fprintf(stderr, "%s:%s:%i hipFree(myParms->B_d) ", __FILE__, __func__, __LINE__);
            hip_report_error(hipret);
            exit(-1);
        }        
        myParms->B_d = NULL;
    }

    if (myParms->C_d != NULL) {
        hipret = hipFree(myParms->C_d);
        if (hipret != hipSuccess) {
            fprintf(stderr, "%s:%s:%i hipFree(myParms->C_d) ", __FILE__, __func__, __LINE__);
            hip_report_error(hipret);
            exit(-1);
        }        
        myParms->C_d = NULL;
    }

    // Make sure the device is done with everything.
    hipret = hipDeviceSynchronize();
    if (hipret != hipSuccess) {
        fprintf(stderr, "%s:%s:%i hipStreamSynchronize() ", __FILE__, __func__, __LINE__);
        hip_report_error(hipret);
        exit(-1);
    }
        
    // Tell rocblas to clean up the handle. 
    myParms->rb_ret = rocblas_destroy_handle(myParms->handle);
    if (myParms->rb_ret != rocblas_status_success) {
        fprintf(stderr, "rocblas_destroy_handle ");
        rb_report_error(myParms->rb_ret);
    }

    // free our parameter structure.    
    if (myParms != NULL) {free(myParms); myParms = NULL;}
    return;
} // end FinishWork.

