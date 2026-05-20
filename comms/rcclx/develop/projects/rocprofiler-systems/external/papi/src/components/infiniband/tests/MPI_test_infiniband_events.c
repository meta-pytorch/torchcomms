/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    MPI_test_infiniband_events.c
 * 
 * @author  Rizwan Ashraf
 *          rizwan@icl.utk.edu
 *       
 * MPI-based test case for the infiniband component.
 *      
 * @brief
 *       The test code uses the message passing interface (MPI) to test all interconnect 
 *       related events available in the infiniband component. It is designed to generate 
 *       network traffic using MPI routines with the goal to trigger some network counters. 
 *       The code automatically checks if the infiniband component is enabled and 
 *       correspondingly adds all available PAPI events in the event set, one at a time.
 *       In each invocation, different data sizes are communicated over the network. 
 *       The event values are recorded in each case, and listed at the completion of the 
 *       test. Mostly, the event values need to be checked manually for correctness. 
 *       The code automatically tests expected behavior of the code for transmit (TX)/ 
 *       receive (RX) event types. 
 *
 *       In this test, the master process distributes workload to all other processes 
 *       (NumProcs-1) and then receives the results of the corresponding sub-computations. 
 *       As far as message transfers is concerned, the expected behavior of this code
 *       is as follows:
 *       1. Master TX event ~= Sum of all RX events across all workers (NumProcs-1),
 *       2. Master RX event ~= Sum of all TX events across all workers (NumProcs-1).
 *       Usage: mpirun -n <NumProcs> ./MPI_test_infiniband_events
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* headers required by PAPI */
#include "papi.h"
#include "papi_test.h"

/* constants */
// NSIZE_MIN/MAX: min/max no. of double floating point
//                values allocated at each node
#define NSIZE_MIN 10000
#define NSIZE_MAX 100000
// No. of different data sizes to be 
// tested b/w NSIZE_MIN and NSIZE_MAX
#define NSTEPS 9
// The max no. of infiniband events expected
#define MAX_IB_EVENTS 150
// Threshold value to use when comparing TX/RX event values,
// i.e., error will be recorded when any difference greater 
// than the threshold occurs
#define EVENT_VAL_DIFF_THRESHOLD 100
// PASS_THRESHOLD: percentage of values out of all possibilities
//                 which need to be correct for the test to PASS
// WARN_THRESHOLD: If PASS_THRESHOLD is not met, then this threshold 
//                 is used to check if the test can be declared 
//                 PASS WITH WARNING. Otherwise, the test is declared 
//                 as FAILED.
// NSIZE_* : No. of Data Sizes out of all possible NSTEPS data sizes
//           where TX/RX event value comparison will be performed to check 
//           expected behavior.                  
#define NSIZE_PASS_THRESHOLD 90
#define NSIZE_WARN_THRESHOLD 50
// EVENT_* : No. of events out of all possible events as reported by
//           component_info which need to be added successfully to the
//           event set.
#define EVENT_PASS_THRESHOLD 90
#define EVENT_WARN_THRESHOLD 50

int main (int argc, char **argv) {

    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

    /*************************  SETUP PAPI ENV *************************************
     *******************************************************************************/                              
    int retVal, r, code;
    int ComponentID, NumComponents, IB_ID = -1;
    int EventSet = PAPI_NULL;
    int eventCount = 0;  // total events as reported by component info
    int eventNum = 0;    // number of events successfully tested

    /* error reporting */
    int addEventFailCount = 0, codeConvertFailCount = 0, eventInfoFailCount = 0;  
    int PAPIstartFailCount = 0, PAPIstopFailCount = 0;
    int failedEventCodes[MAX_IB_EVENTS];
    int failedEventIndex = 0;

    /* Note: these are fixed length arrays */ 
    char eventNames[MAX_IB_EVENTS][PAPI_MAX_STR_LEN];
    char description[MAX_IB_EVENTS][PAPI_MAX_STR_LEN];
    long long values[NSTEPS][MAX_IB_EVENTS];

    /* these record certain event values for event value testing */
    long long rxCount[NSTEPS], txCount[NSTEPS];

    const PAPI_component_info_t *cmpInfo = NULL;
    PAPI_event_info_t eventInfo;

    /* for timing the test */
    long long startTime, endTime;
    double elapsedTime;

    /* PAPI Initialization */
    retVal = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retVal != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__,"PAPI_library_init failed. The test has been terminated.\n",retVal);
    }

    /* Get total number of components detected by PAPI */ 
    NumComponents = PAPI_num_components();

    /* Check if infiniband component exists */
    for ( ComponentID = 0; ComponentID < NumComponents; ComponentID++ ) {

        if ( (cmpInfo = PAPI_get_component_info(ComponentID)) == NULL ) {
            fprintf(stderr, "WARNING: PAPI_get_component_info failed on one of the components.\n"
                    "\t The test will continue for now, but it will be skipped later on\n"
                    "\t if this error was for a component under test.\n");
            continue;
        }

        if (strcmp(cmpInfo->name, "infiniband") != 0) {
            continue;
        }

        // if we are here, Infiniband component is found 
        if (!TESTS_QUIET) {
            printf("INFO: Component %d (%d) - %d events - %s\n",
                    ComponentID, cmpInfo->CmpIdx,
                    cmpInfo->num_native_events, cmpInfo->name);
        }

        if (cmpInfo->disabled) {
            test_skip(__FILE__,__LINE__,"Infiniband Component is disabled. The test has been terminated.\n", 0);
            break;
        }

        eventCount = cmpInfo->num_native_events;
        IB_ID = ComponentID;
        break;
    }

    /* if we did not find any valid events, just skip the test. */
    if (eventCount==0) {
        fprintf(stderr, "FATAL: No events found for the Infiniband component, even though it is enabled.\n"
                "       The test will be skipped.\n"); 
        test_skip(__FILE__,__LINE__,"No events found for the Infiniband component.\n", 0);
    }

    /*************************  SETUP MPI ENV **************************************
     *******************************************************************************/
    int NumProcs, Rank;

    /* Initialize MPI environment */
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &NumProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &Rank);

    if ((!TESTS_QUIET) && (Rank == 0)) {
        printf("INFO: This test should trigger some network events.\n");
    }

    /* data sizes assigned here */ 
    int Nmax_per_Proc = NSIZE_MAX;   
    int Nmin_per_Proc = NSIZE_MIN;
    // fix data size if not appropriately set  
    while (Nmax_per_Proc <= Nmin_per_Proc) 
        Nmax_per_Proc = Nmin_per_Proc*10;
    int Nmax = Nmax_per_Proc * NumProcs;
    int NstepSize = (Nmax_per_Proc - Nmin_per_Proc)/NSTEPS; 

    int i, j, k;  // loop variables
    int memoryAllocateFailure = 0, ALLmemoryAllocateFailure = 0; // error flags

    /* data arrays */
    double *X, *Y, *Out;
    double *Xp, *Yp, *Outp;

    /* Master will initialize data arrays */
    if (Rank == 0) {
        X = (double *) malloc (sizeof(double) * Nmax);
        Y = (double *) malloc (sizeof(double) * Nmax);
        Out = (double *) malloc (sizeof(double) * Nmax);

        // check if memory was successfully allocated.
        // Do NOT quit from here. Need to quit safely.
        if ( (X == NULL) || (Y == NULL) || (Out == NULL) ) {
            fprintf(stderr, "FATAL: Failed to allocate memory on Master Node.\n");
            memoryAllocateFailure = 1;
        }      

        if (memoryAllocateFailure == 0) {

            if (!TESTS_QUIET)
                printf("INFO: Master is initializing data.\n");

            for ( i = 0; i < Nmax; i++ ) {
                X[i] = i*0.25;
                Y[i] = i*0.75;
            }

            if (!TESTS_QUIET) 
                printf("INFO: Master has successfully initialized arrays.\n");

        }
    }

    // communicate to workers if master was able to successfully allocate memory
    MPI_Bcast (&memoryAllocateFailure, 1, MPI_INT, 0, MPI_COMM_WORLD);   
    if (memoryAllocateFailure == 1) 
        test_fail(__FILE__,__LINE__,"Could not allocate memory during the test. This is fatal and the test has been terminated.\n", 0);

    memoryAllocateFailure = 0; // re-use flag

    /* allocate memory for all nodes */
    Xp = (double *) malloc (sizeof(double) * Nmax_per_Proc);
    Yp = (double *) malloc (sizeof(double) * Nmax_per_Proc);
    Outp = (double *) malloc (sizeof(double) * Nmax_per_Proc); 

    // handle error cases for memory allocation failure for all nodes.
    if ( (Xp == NULL) || (Yp == NULL) || (Outp == NULL) ) {
        fprintf(stderr, "FATAL: Failed to allocate %zu bytes on Rank %d.\n", sizeof(double)*Nmax_per_Proc, Rank);
        memoryAllocateFailure = 1;
    }
    MPI_Allreduce (&memoryAllocateFailure, &ALLmemoryAllocateFailure, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (ALLmemoryAllocateFailure > 0) 
        test_fail(__FILE__,__LINE__,"Could not allocate memory during the test. This is fatal and the test has been terminated.\n", 0);

    /* calculate data size for each compute step */
    int Nstep_per_Proc;
    int DataSizes[NSTEPS];
    for (i = 0; i < NSTEPS; i++) {
        Nstep_per_Proc = Nmin_per_Proc + (i * NstepSize);
        //last iteration or when max size is exceeded
        if ((i == (NSTEPS - 1)) || (Nstep_per_Proc > Nmax_per_Proc))
            Nstep_per_Proc = Nmax_per_Proc;
        DataSizes[i] = Nstep_per_Proc;
    }

    /*************************  MAIN TEST CODE *************************************
     *******************************************************************************/
    startTime = PAPI_get_real_nsec();

    /* create an eventSet */
    retVal = PAPI_create_eventset ( &EventSet );
    if (retVal != PAPI_OK) {
        // handle error cases for PAPI_create_eventset() 
        // Two outcomes are possible here:
        // 1. PAPI_EINVAL: invalid argument. This should not occur.
        // 2. PAPI_ENOMEM: insufficient memory. If this is the case, then we need to quit the test.
        fprintf(stderr, "FATAL: Could not create an eventSet on MPI Rank %d due to: %s.\n"
                "       Test will not proceed.\n", Rank, PAPI_strerror(retVal));
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed. This is fatal and the test has been terminated.\n", retVal);
    } // end -- handle error cases for PAPI_create_eventset()

    /* find the code for first event in component */
    code = PAPI_NATIVE_MASK;
    r = PAPI_enum_cmp_event ( &code, PAPI_ENUM_FIRST, IB_ID );

    /* add each event individually in the eventSet and measure event values. */
    /* for each event, repeat work with different data sizes. */
    while ( r == PAPI_OK ) {

        // attempt to add event to event set 
        retVal = PAPI_add_event (EventSet, code);
        if (retVal != PAPI_OK ) {
            // handle error cases for PAPI_add_event()
            if (retVal == PAPI_ENOMEM) {  
                fprintf(stderr, "FATAL: Could not add an event to eventSet on MPI Rank %d due to insufficient memory.\n"
                        "       Test will not proceed.\n", Rank);
                test_fail(__FILE__, __LINE__, "PAPI_add_event failed due to fatal error and the test has been terminated.\n", retVal);
            }

            if (retVal == PAPI_ENOEVST) {
                fprintf(stderr, "WARNING: Could not add an event to eventSet on MPI Rank %d since eventSet does not exist.\n"
                        "\t Test will proceed attempting to create a new eventSet\n", Rank);
                EventSet = PAPI_NULL;  
                retVal = PAPI_create_eventset ( &EventSet );
                if (retVal != PAPI_OK)
                    test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed while handling failure of PAPI_add_event." 
                            " This is fatal and the test has been terminated.\n", retVal);
                continue;  
            }

            if (retVal == PAPI_EISRUN) {
                long long tempValue;
                fprintf(stderr, "WARNING: Could not add an event to eventSet on MPI Rank %d since eventSet is already counting.\n"
                        "\t Test will proceed attempting to stop counting and re-attempting to add current event.\n", Rank);
                retVal = PAPI_stop (EventSet, &tempValue);
                if (retVal != PAPI_OK) 
                    test_fail(__FILE__,__LINE__,"PAPI_stop failed while handling failure of PAPI_add_event." 
                            " This is fatal and the test has been terminated.\n", retVal);  
                retVal = PAPI_cleanup_eventset( EventSet );
                if (retVal != PAPI_OK)
                    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset failed while handling failure of PAPI_add_event."
                            " This is fatal and the test has been terminated.\n", retVal);
                continue;              
            }

            // for all other errors, skip an event
            addEventFailCount++; // error reporting  
            failedEventCodes[failedEventIndex] = code;
            failedEventIndex++;        
            fprintf(stderr, "WARNING: Could not add an event to eventSet on MPI Rank %d due to: %s.\n" 
                    "\t Test will proceed attempting to add other events.\n", Rank, PAPI_strerror(retVal));

            r = PAPI_enum_cmp_event (&code, PAPI_ENUM_EVENTS, IB_ID);

            if (addEventFailCount >= eventCount) // if no event was added successfully
                break;  

            continue; 
        }  // end -- handle error cases for PAPI_add_event()

        /* get event name of added event */
        retVal = PAPI_event_code_to_name (code, eventNames[eventNum]);
        if (retVal != PAPI_OK ) {
            // handle error cases for PAPI_event_code_to_name().
            codeConvertFailCount++; // error reporting
            fprintf(stderr, "WARNING: PAPI_event_code_to_name failed due to: %s.\n"
                    "\t Test will proceed but an event name will not be available.\n", PAPI_strerror(retVal));
            strncpy(eventNames[eventNum], "ERROR:NOT_AVAILABLE", sizeof(eventNames[0])-1);
            eventNames[eventNum][sizeof(eventNames[0])-1] = '\0'; 
        } // end -- handle error cases for PAPI_event_code_to_name()

        /* get long description of added event */
        retVal = PAPI_get_event_info (code, &eventInfo);
        if (retVal != PAPI_OK ) {
            // handle error cases for PAPI_get_event_info()
            eventInfoFailCount++;  // error reporting
            fprintf(stderr, "WARNING: PAPI_get_event_info failed due to: %s.\n"
                    "\t Test will proceed but an event description will not be available.\n", PAPI_strerror(retVal));
            strncpy(description[eventNum], "ERROR:NOT_AVAILABLE", sizeof(description[0])-1);
            description[eventNum][sizeof(description[0])-1] = '\0';
        } else {
            strncpy(description[eventNum], eventInfo.long_descr, sizeof(description[0])-1);
            description[eventNum][sizeof(description[0])-1] = '\0';
        }

        /****************** PERFORM WORK (W/ DIFFERENT DATA SIZES) *********************
         *******************************************************************************/
        for (i = 0; i < NSTEPS; i++) { 

            /* start recording event value */
            retVal = PAPI_start (EventSet);
            if (retVal != PAPI_OK ) {
                // handle error cases for PAPI_start()
                // we need to skip the current event being counted for all errors,
                // in all cases, errors will be handled later on.

                PAPIstartFailCount++;  // error reporting
                failedEventCodes[failedEventIndex] = code;
                failedEventIndex++;
                fprintf(stderr, "WARNING: PAPI_start failed on Event Number %d (%s) due to: %s.\n"
                        "\t Test will proceed with other events if available.\n",
                        eventNum, eventNames[eventNum], PAPI_strerror(retVal));

                for (k = i; k < NSTEPS; k++)  // fill invalid event values.
                    values[k][eventNum] = (unsigned long long) - 1; 

                break;  // try next event
            } // end -- handle error cases for PAPI_start()

            if ((!TESTS_QUIET) && (Rank == 0))
                printf("INFO: Doing MPI communication for %s: min. %ld bytes transferred by each process.\n", 
                        eventNames[eventNum], DataSizes[i]*sizeof(double));

            MPI_Scatter (X, DataSizes[i], MPI_DOUBLE, Xp, DataSizes[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter (Y, DataSizes[i], MPI_DOUBLE, Yp, DataSizes[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);

            /* perform calculation. */
            /* Note: there is redundant computation here. */ 
            for (j = 0; j < DataSizes[i]; j++)
                Outp [j] = Xp [j] + Yp [j];

            MPI_Gather (Outp, DataSizes[i], MPI_DOUBLE, Out, DataSizes[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);

            /* stop recording and collect event value */ 
            retVal = PAPI_stop (EventSet, &values[i][eventNum]);
            if (retVal != PAPI_OK ) {
                // handle error cases for PAPI_stop()
                // we need to skip the current event for all errors 
                // except one case, as below.
                PAPIstopFailCount++;  // error reporting
                if (retVal == PAPI_ENOTRUN) { 
                    fprintf(stderr, "WARNING: PAPI_stop failed on Event Number %d (%s) since eventSet is not running.\n"
                            "\t Test will attempt to restart counting on this eventSet.\n",
                            eventNum, eventNames[eventNum]);
                    if (PAPIstopFailCount < NSTEPS) { 
                        i = i - 1; // re-attempt this data size
                        continue;
                    }
                }
                // for all other errors, try next event.
                failedEventCodes[failedEventIndex] = code;
                failedEventIndex++;
                fprintf(stderr, "WARNING: PAPI_stop failed on Event Number %d (%s) due to: %s.\n"
                        "\t Test will proceed with other events if available.\n",
                        eventNum, eventNames[eventNum], PAPI_strerror(retVal));

                for (k = i; k < NSTEPS; k++)   // fill invalid event values
                    values[k][eventNum] = (unsigned long long) - 1;

                break;  
            }  // end -- handle error cases for PAPI_stop()

            /* record number of bytes received */
            if (strstr(eventNames[eventNum], ":port_rcv_data")) {
                rxCount[i] = values[i][eventNum] * 4;  // counter value needs to be multiplied by 4 to get total number of bytes
            } 
            /* record number of bytes transmitted */
            if (strstr(eventNames[eventNum], ":port_xmit_data")) {
                txCount[i] = values[i][eventNum] * 4;
            }

        }   // end -- work loop

        /* Done, clean up eventSet for next iteration */
        retVal = PAPI_cleanup_eventset( EventSet );
        if (retVal != PAPI_OK) {
            // handle failure cases for PAPI_cleanup_eventset()
            if (retVal == PAPI_ENOEVST) {
                fprintf(stderr, "WARNING: Could not clean up eventSet on MPI Rank %d since eventSet does not exist.\n"
                        "\t Test will proceed attempting to create a new eventSet\n", Rank);
                EventSet = PAPI_NULL;
                retVal = PAPI_create_eventset ( &EventSet );
                if (retVal != PAPI_OK)
                    test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed while handling failure of PAPI_cleanup_eventset.\n"
                            "This is fatal and the test has been terminated.\n", retVal);                                                    
            } else if (retVal == PAPI_EISRUN) {
                long long tempValue;
                fprintf(stderr, "WARNING: Could not clean up eventSet on MPI Rank %d since eventSet is already counting.\n"
                        "\t Test will proceed attempting to stop counting and re-attempting to clean up.\n", Rank);
                retVal = PAPI_stop (EventSet, &tempValue);
                if (retVal != PAPI_OK)
                    test_fail(__FILE__,__LINE__,"PAPI_stop failed while handling failure of PAPI_cleanup_eventset."
                            "This is fatal and the test has been terminated.\n", retVal);
                retVal = PAPI_cleanup_eventset( EventSet );
                if (retVal != PAPI_OK)
                    test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset failed once again while handling failure of PAPI_cleanup_eventset."
                            "This is fatal and the test has been terminated.\n", retVal);
            } else {
                test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset failed:", retVal);
            }
        } // end -- handle failure cases for PAPI_cleanup_eventset()

        /* get next event */
        eventNum++;
        r = PAPI_enum_cmp_event (&code, PAPI_ENUM_EVENTS, IB_ID);

    } // end -- event loop

    // free memory at all nodes
    free (Xp); free (Yp); free (Outp);

    /* Done, destroy eventSet */
    retVal = PAPI_destroy_eventset( &EventSet );
    if (retVal != PAPI_OK) {
        // handle error cases for PAPI_destroy_eventset() 
        if (retVal == PAPI_ENOEVST || retVal == PAPI_EINVAL) {
            fprintf(stderr, "WARNING: Could not destroy eventSet on MPI Rank %d since eventSet does not exist or has invalid value.\n"
                    "\t Test will proceed with other operations.\n", Rank);
        } else if (retVal == PAPI_EISRUN) {
            long long tempValue;
            fprintf(stderr, "WARNING: Could not destroy eventSet on MPI Rank %d since eventSet is already counting.\n"
                    "\t Test will proceed attempting to stop counting and re-attempting to clean up.\n", Rank);
            retVal = PAPI_stop (EventSet, &tempValue);
            if (retVal != PAPI_OK)
                test_fail(__FILE__,__LINE__,"PAPI_stop failed while handling failure of PAPI_destroy_eventset."
                        "This is fatal and the test has been terminated.\n", retVal);
            retVal = PAPI_cleanup_eventset( EventSet );
            if (retVal != PAPI_OK)
                test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset failed while handling failure of PAPI_destroy_eventset."
                        "This is fatal and the test has been terminated.\n", retVal); 
            retVal = PAPI_destroy_eventset(&EventSet);
            if (retVal != PAPI_OK)
                test_fail(__FILE__,__LINE__,"PAPI_destroy_eventset failed once again while handling failure of PAPI_destroy_eventset."
                        " This is fatal and the test has been terminated.\n", retVal);
        } else {
            fprintf(stderr, "WARNING: Could not destroy eventSet on MPI Rank %d since there is an internal bug in PAPI.\n"
                    "\t Please report this to the developers. Test will proceed and operation may be unexpected.\n", Rank);
        }
    }  // end -- handle failure cases for PAPI_destroy_eventset() 

    /*************************** SUMMARIZE RESULTS ********************************
     ******************************************************************************/
    endTime = PAPI_get_real_nsec();
    elapsedTime = ((double) (endTime-startTime))/1.0e9;

    /* print results: event values and descriptions */
    if (!TESTS_QUIET) {
        int eventX;
        // print event values at each process/rank
        printf("POST WORK EVENT VALUES (Rank, Event Name, List of Event Values w/ Different Data Sizes)>>>\n");
        for (eventX = 0; eventX < eventNum; eventX++) {
            printf("\tRank %d> %s --> \t\t", Rank, eventNames[eventX]);
            for (i = 0; i < NSTEPS; i++) {
                if (i < NSTEPS-1) 
                    printf("%lld, ", values[i][eventX]);
                else
                    printf("%lld.", values[i][eventX]);
            }
            printf("\n");
        }

        // print description of each event 
        if (Rank == 0) {
            printf("\n\nTHE DESCRIPTION OF EVENTS IS AS FOLLOWS>>>\n"); 
            for (eventX = 0; eventX < eventNum; eventX++) {
                printf("\t%s \t\t--> %s \n", eventNames[eventX], description[eventX]);
            }
        }
    }

    /* test summary: 1) sanity check on floating point computation */
    int computeTestPass = 0, computeTestPassCount = 0;
    if (Rank == 0) {
        // check results of computation 
        for (i = 0; i < Nmax; i++) {
            if ( fabs(Out[i] - (X[i] + Y[i])) < 0.00001 )
                computeTestPassCount++;
        }
        // summarize results of computation
        if (computeTestPassCount == Nmax)
            computeTestPass = 1;

        // free memory
        free (X); free (Y); free (Out);
    }
    // communicate test results to everyone
    MPI_Bcast (&computeTestPass, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* test summary: 2) check TX and RX event values, if available */       
    long long rxCountSumWorkers[NSTEPS], txCountSumWorkers[NSTEPS];
    long long *allProcessRxEvents, *allProcessTxEvents;
    int txFailedIndex = 0, rxFailedIndex = 0;
    int txFailedDataSizes[NSTEPS], rxFailedDataSizes[NSTEPS];     
    int eventValueTestPass = 0;   // for test summary 
    if ((txCount[0] > 0) && (rxCount[0] > 0)) {
        if (Rank == 0) {
            allProcessRxEvents = (long long*) malloc(sizeof(long long) * NumProcs * NSTEPS);
            allProcessTxEvents = (long long*) malloc(sizeof(long long) * NumProcs * NSTEPS);
        }
        // get all rxCount/txCount at master. Used to check if rx/tx counts match up. 
        MPI_Gather (&rxCount, NSTEPS, MPI_LONG_LONG, allProcessRxEvents, NSTEPS, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather (&txCount, NSTEPS, MPI_LONG_LONG, allProcessTxEvents, NSTEPS, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        // perform event count check at master 
        if (Rank == 0) {
            memset (rxCountSumWorkers, 0, sizeof(long long) * NSTEPS);
            memset (txCountSumWorkers, 0, sizeof(long long) * NSTEPS);
            for (i = 0; i < NSTEPS; i++) { 
                for (j = 1; j < NumProcs; j++) {   
                    rxCountSumWorkers[i] += allProcessRxEvents[j*NSTEPS+i];
                    txCountSumWorkers[i] += allProcessTxEvents[j*NSTEPS+i];
                }
            }

            if (!TESTS_QUIET) printf("\n\n");
            for (i = 0; i < NSTEPS; i++) {
                // check: Master TX event ~= Sum of all RX events across all workers (NumProcs-1) 
                // difference threshold may need to be adjusted based on observed values 
                if ((llabs(rxCountSumWorkers[i] - txCount[i]) > EVENT_VAL_DIFF_THRESHOLD)) {
                    txFailedDataSizes[txFailedIndex] = DataSizes[i];
                    txFailedIndex++;
                    if (!TESTS_QUIET) 
                        printf("WARNING: The transmit event count at Master Node (%lld) is not equal"
                                " to receive event counts at Worker Nodes (%lld) when using %ld bytes!\n"
                                "\t A difference of %lld was recorded.\n", txCount[i], rxCountSumWorkers[i], 
                                DataSizes[i]*sizeof(double), llabs(rxCountSumWorkers[i] - txCount[i]));
                } else {
                    if (!TESTS_QUIET) 
                        printf("PASSED: The transmit event count at Master Node (%lld) is almost equal"
                                " to receive event counts at Worker Nodes (%lld) when using %ld bytes.\n", 
                                txCount[i], rxCountSumWorkers[i], DataSizes[i]*sizeof(double));
                }

                // check: Master RX event ~= Sum of all TX events across all workers (NumProcs-1)
                if ((llabs(txCountSumWorkers[i] - rxCount[i]) > EVENT_VAL_DIFF_THRESHOLD)) { 
                    rxFailedDataSizes[rxFailedIndex] = DataSizes[i];
                    rxFailedIndex++;
                    if (!TESTS_QUIET) 
                        printf("WARNING: The receive event count at Master Node (%lld) is not equal" 
                                " to transmit event counts at Worker Nodes (%lld) when using %ld bytes!\n"
                                " A difference of %lld was recorded.\n", rxCount[i], txCountSumWorkers[i], 
                                DataSizes[i]*sizeof(double), llabs(txCountSumWorkers[i] - rxCount[i])); 
                } else {
                    if (!TESTS_QUIET) 
                        printf("PASSED: The receive event count at Master Node (%lld) is almost equal"
                                " to transmit event counts at Worker Nodes (%lld) when using %ld bytes.\n", 
                                rxCount[i], txCountSumWorkers[i], DataSizes[i]*sizeof(double));
                }
            }

            // test evaluation criteria
            if ( (((float) txFailedIndex / NSTEPS) <= (1.0 - (float) NSIZE_PASS_THRESHOLD/100)) &&
                    (((float) rxFailedIndex / NSTEPS) <= (1.0 - (float) NSIZE_PASS_THRESHOLD/100)) )
                eventValueTestPass = 1;  // pass            
            else if ( (((float) txFailedIndex / NSTEPS) <= (1.0 - (float) NSIZE_WARN_THRESHOLD/100)) &&
                    (((float) rxFailedIndex / NSTEPS) <= (1.0 - (float) NSIZE_WARN_THRESHOLD/100)) )
                eventValueTestPass = -1; // warning
            else
                eventValueTestPass = 0;  // fail

        }  // end -- check RX/TX counts for all data sizes at Master node.

        // communicate test results to everyone, since only master knows the result
        MPI_Bcast (&eventValueTestPass, 1, MPI_INT, 0, MPI_COMM_WORLD);

    } else {
        eventValueTestPass = -2; // not available
    } // end -- event value test

    /* test summary: 3) number of events added and counted successfully */
    // Note: under some rare circumstances, the number of failed events at each node may be different.
    int eventNumTestPass = 0;
    // test evaluation criteria
    if (((float) failedEventIndex / eventCount) <= (1.0 - (float) EVENT_PASS_THRESHOLD/100) )
        eventNumTestPass = 1;
    else if (((float) failedEventIndex / eventCount) <= (1.0 - (float) EVENT_WARN_THRESHOLD/100) ) 
        eventNumTestPass = -1; 
    else
        eventNumTestPass = 0;


    /* print test summary */
    if ((!TESTS_QUIET) && (Rank == 0)) {

        printf("\n\n************************ TEST SUMMARY (EVENTS) ******************************\n"
                "No. of Events NOT tested successfully: %d (%.1f%%)\n"
                "Note: the above failed event count is for Master node.\n"
                "Total No. of Events reported by component info: %d\n", 
                failedEventIndex, ((float) failedEventIndex/eventCount)*100.0, eventCount);

        if (failedEventIndex > 0) {
            printf("\tNames of Events NOT tested: ");
            char failedEventName[PAPI_MAX_STR_LEN];
            for (i = 0; i < failedEventIndex; i++) {
                retVal = PAPI_event_code_to_name (failedEventCodes[i], failedEventName);
                if (retVal != PAPI_OK) {
                    strncpy(failedEventName, "ERROR:NOT_AVAILABLE", sizeof(failedEventName)-1);
                    failedEventName[sizeof(failedEventName)-1] = '\0';
                }
                printf("%s ", failedEventName);
                if ((i > 0) && (i % 2 == 1)) printf("\n   \t\t\t\t");
            }
            printf("\n");

            printf("\tThe error counts for different PAPI routines are as follows:\n"
                    "\t\t\tNo. of PAPI add event errors (major) --> %d\n"
                    "\t\t\tNo. of PAPI code convert errors (minor) --> %d\n"
                    "\t\t\tNo. of PAPI event info errors (minor) --> %d\n" 
                    "\t\t\tNo. of PAPI start errors (major) --> %d\n"
                    "\t\t\tNo. of PAPI stop errors (major) --> %d\n", 
                    addEventFailCount, codeConvertFailCount, eventInfoFailCount, PAPIstartFailCount, PAPIstopFailCount);
        }
        printf("The PAPI event test has ");
        if (eventNumTestPass == 1) printf("PASSED\n");
        else if (eventNumTestPass == -1) printf("PASSED WITH WARNING\n");
        else printf("FAILED\n");

        // event values
        printf("************************ TEST SUMMARY (EVENT VALUES) ************************\n");
        if ((txCount[0] > 0) && (rxCount[0] > 0)) {
            printf("No. of times transmit event at Master node did NOT match up receive events at worker nodes: %d (%.1f%%)\n"
                    "No. of times receive event at Master node did NOT match up transmit events at worker nodes: %d (%.1f%%)\n"
                    "Total No. of data sizes tested: %d\n"
                    "\tList of Data Sizes tested in bytes:\n\t\t\t", 
                    txFailedIndex, ((float) txFailedIndex/NSTEPS)*100.0, rxFailedIndex, ((float) rxFailedIndex/NSTEPS)*100.0, NSTEPS);
            for (i = 0; i < NSTEPS; i++)
                printf("%ld ",DataSizes[i]*sizeof(double));
            printf("\n");
            if (txFailedIndex > 0 || rxFailedIndex > 0) {
                printf("\tList of Data Sizes where transmit count at Master was not equal to sum of all worker receive counts:\n"
                        "\t\t\t");
                for (i = 0; i < txFailedIndex; i++) 
                    printf("%ld ", txFailedDataSizes[i]*sizeof(double));
                printf("\n\tList of Data Sizes where receive count at Master was not equal to sum of all worker transmit counts:\n"
                        "\t\t\t"); 
                for (i = 0; i < rxFailedIndex; i++)
                    printf("%ld ", rxFailedDataSizes[i]*sizeof(double));
                printf("\n");
            }
            printf("The PAPI event value test has ");
            if (eventValueTestPass == 1) printf("PASSED\n");
            else if (eventValueTestPass == -1) printf("PASSED WITH WARNING\n");
            else printf("FAILED\n"); 
        } else {
            printf("Transmit or receive events were NOT found!\n");
        }

        // compute values
        printf("************************ TEST SUMMARY (COMPUTE VALUES) **********************\n");
        if (computeTestPassCount != Nmax) {
            printf("No. of times sanity check FAILED on the floating point computation: %d (%.1f%%)\n"
                    "Total No. of floating point computations performed: %d \n",
                    Nmax-computeTestPassCount, ((float) (Nmax-computeTestPassCount)/Nmax)*100.0, Nmax); 
        } else {
            printf("Sanity check PASSED on all floating point computations.\n"
                    "Note: this may pass even if one event was tested successfully!\n");
        }
        printf("The overall test took %.3f secs.\n\n", elapsedTime);           
    } // end -- print summary of test results.

    /* finialize MPI */
    MPI_Finalize();

    /* determine success of overall test based on all tests */
    if (computeTestPass == 1 && eventValueTestPass == 1 && eventNumTestPass == 1) {
        // all has to be good for the test to pass. 
        // note: test will generate a warning if tx/rx events are not available.
        test_pass( __FILE__ );
    } 
    else if ( (eventValueTestPass < 0 && (eventNumTestPass < 0 || eventNumTestPass == 1) ) ||
            (eventValueTestPass == 1 && eventNumTestPass < 0) || 
            (eventValueTestPass == 1 && eventNumTestPass == 1 && computeTestPass == 0) ) {
        test_warn(__FILE__,__LINE__,"A warning was generated during any PAPI related tests or sanity check on computation failed", 0);
        test_pass(__FILE__);
    }
    else { 
        // fail, in case any of eventValueTest and eventNumTest have failed, 
        // irrespective of the result of computeTest. 
        test_fail(__FILE__, __LINE__,"Any of PAPI event related tests have failed", 0);
    }

} // end main
