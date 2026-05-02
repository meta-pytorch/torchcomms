/**
 * @author Philip Vaccaro
 * Test case for powercap component
 * @brief
 *   Tests basic functionality of powercap component
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

#define MAX_powercap_EVENTS 64

int main ( int argc, char **argv )
{
    (void) argv;
    (void) argc;
    int retval,cid,powercap_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long *values;
    int num_events=0;
    int code;
    char event_names[MAX_powercap_EVENTS][PAPI_MAX_STR_LEN];
    int r,i;
    int quiet = 1, passed = 0;

    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;

    if (argc >= 2) quiet = atoi(argv[1]);

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT )
        fprintf( stderr, "PAPI_library_init failed\n" );

    numcmp = PAPI_num_components();

    for( cid=0; cid<numcmp; cid++ ) {
        if ( ( cmpinfo = PAPI_get_component_info( cid ) ) == NULL )
            fprintf(stderr, "PAPI_get_component_info failed\n");

        if ( strstr( cmpinfo->name,"powercap_ppc" ) ) {
            powercap_cid=cid;
            if ( !quiet ) fprintf(stdout, "Found powercap_ppc component at cid %d\n",powercap_cid );
            if ( cmpinfo->disabled ) {
                if ( !quiet ) {
                    printf( "powercap_ppc component disabled: %s\n",
                            cmpinfo->disabled_reason );
                }
                fprintf(stderr, "powercap_ppc component disabled\n");
            }
            break;
        }
    }

    /* Component not found */
    if ( cid==numcmp )
        fprintf(stderr, "No powercap_ppc component found\n" );

    /* Skip if component has no counters */
    if ( cmpinfo->num_cntrs==0 )
        fprintf(stderr, "No counters in the powercap_ppc component\n" );

    /* Create EventSet */
    retval = PAPI_create_eventset( &EventSet );
    if ( retval != PAPI_OK )
        fprintf(stderr, "PAPI_create_eventset()\n");

    /* Add all events */
    code = PAPI_NATIVE_MASK;
    r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, powercap_cid );
    while ( r == PAPI_OK ) {
        retval = PAPI_event_code_to_name( code, event_names[num_events] );
        if ( retval != PAPI_OK )
            fprintf(stdout, "Error from PAPI_event_code_to_name\n");

        retval = PAPI_get_event_info( code,&evinfo );
        if ( retval != PAPI_OK )
            fprintf(stderr, "Error getting event info\n");

        retval = PAPI_add_event( EventSet, code );

        if (retval != PAPI_OK)
            break; /* We've hit an event limit */
        num_events++;

        r = PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, powercap_cid);
    }

    PAPI_start(EventSet);

    values=calloc(num_events,sizeof(long long));
    if ( values==NULL ) { fprintf(stdout, "No memory for values"); return -1; }

    if ( !quiet ) fprintf(stdout, "\nBefore actual test...\n" );

    long long Pmin = 424242, Pmax = 42, Pcurrent = 42, Ptarget, Pold;

    retval = PAPI_read(EventSet, values);
    int iPcurrent = -1;
    for ( i = 0; i < num_events; ++i ) {
        if (!quiet && strstr( event_names[i], "POWER"))
            fprintf( stdout, "%-45s > %lldW\n",
                    event_names[i], values[i]);

        if ( strstr( event_names[i], "MIN_POWER"))
            Pmin = values[i];

        if ( strstr( event_names[i], "MAX_POWER"))
            Pmax = values[i];

        if ( strstr( event_names[i], "CURRENT_POWER")) {
            iPcurrent = i;
            Pcurrent = values[i];
        }
    }


    if (Pmin <= Pmax) {
        Pold = Pcurrent;
        /* Let's try to cap at 40% */
        Ptarget = Pmin + 0.4 * (Pmax - Pmin);
        /* Ok, current cap was 40%, so let's make it 60% */
        if (Pold == Ptarget) Ptarget = Pmin + 0.6 * (Pmax - Pmin);

        if ( !quiet )
            fprintf(stdout, "Current capping is Pcurrent = %lld W.\nCapping with Ptarget = %lld W\n", Pold, Ptarget);

        values[iPcurrent] = Ptarget;

        long long before = PAPI_get_real_nsec();
        long long after = before;

        PAPI_write(EventSet, values);

        if (!quiet) fprintf(stdout, "Changing the power capping might take some time.\nThe test will time out after 10 seconds.\n");
        do {
            /* Give everyone some time to realize it */
            usleep(100000);
            after = PAPI_get_real_nsec();
            PAPI_read(EventSet, values);
            if (!quiet) fprintf(stdout, ".");
        } while (values[iPcurrent] != Ptarget && (after-before) < 10e10);

        if (values[iPcurrent] != Ptarget) {
            /* test failure */
            if (!quiet) fprintf(stdout, "\nPcurrent read = %lld W, target was %lld W\n", values[iPcurrent], Ptarget);
        } else {
            /* we have a success here */
            if (!quiet) fprintf(stdout, "\nPcurrent read = %lld W, target was %lld W\n", values[iPcurrent], Ptarget);
            /* let's clean behind us, revert to previous capping */
            values[iPcurrent] = Pold;
            PAPI_write( EventSet, values);
            if (!quiet) fprintf(stdout, "Reverting back to previous capping P = %lld W\n", Pold);

            do {
                usleep(100000);
                PAPI_read (EventSet, values);
            } while (values[iPcurrent] != Pold);
            passed = 1;
        }
    }
    else {
        passed = 0;
        fprintf(stderr, "Power capping values read seems wrong: Pmin = %lld W; Pmax = %lld W; Pcurrent = %lld W\n",
            Pmin, Pmax, Pcurrent);
    }

    PAPI_stop(EventSet, values);

    if (passed)
        fprintf(stdout, "TEST SUCCESS\n");
    else
        fprintf(stdout, "TEST FAILED\n");

    /* Done, clean up */
    retval = PAPI_cleanup_eventset(EventSet);
    if ( retval != PAPI_OK )
        fprintf(stdout, "PAPI_cleanup_eventset()\n");

    retval = PAPI_destroy_eventset(&EventSet);
    if ( retval != PAPI_OK )
        fprintf(stdout, "PAPI_destroy_eventset()\n");

  return 0;
}

