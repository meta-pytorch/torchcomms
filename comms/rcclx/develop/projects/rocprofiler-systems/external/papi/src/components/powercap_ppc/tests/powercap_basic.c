/**
 * @author PAPI team UTK/ICL
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

int
main( int argc, char **argv )
{
    (void) argv;
    (void) argc;
    int retval,cid,powercap_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long *values;
    int num_events=0;
    int code;
    char event_names[MAX_powercap_EVENTS][PAPI_MAX_STR_LEN];
    char event_descrs[MAX_powercap_EVENTS][PAPI_HUGE_STR_LEN];
    char units[MAX_powercap_EVENTS][PAPI_MIN_STR_LEN];
    int data_type[MAX_powercap_EVENTS];
    int r,i, quiet = 1, passed = 0;

    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;

    if (2 == argc) quiet = atoi(argv[1]);

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT )
        fprintf( stderr, "PAPI_library_init failed\n" );

    if (!quiet) fprintf( stdout, "Trying all powercap_ppc events\n" );

    numcmp = PAPI_num_components();

    for( cid=0; cid<numcmp; cid++ ) {

        if ( ( cmpinfo = PAPI_get_component_info( cid ) ) == NULL )
            fprintf(stderr, "PAPI_get_component_info failed\n");

        if ( strstr( cmpinfo->name,"powercap_ppc" ) ) {
            powercap_cid=cid;
            if ( !quiet ) fprintf( stdout, "Found powercap_ppc component at cid %d\n",powercap_cid );
            if ( cmpinfo->disabled ) {
                if ( !quiet ) {
                    fprintf(stderr, "powercap_ppc component disabled: %s\n",
                            cmpinfo->disabled_reason);
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

        strncpy( event_descrs[num_events],evinfo.long_descr,sizeof( event_descrs[0] ) );
        strncpy( units[num_events],evinfo.units,sizeof( units[0] ) );
        data_type[num_events] = evinfo.data_type;

        retval = PAPI_add_event( EventSet, code );

        if ( retval != PAPI_OK )
            break; /* We've hit an event limit */
        num_events++;

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, powercap_cid );
    }

    passed = 1;
    PAPI_start( EventSet );

    values = calloc( num_events,sizeof( long long ) );
    if (!values) { fprintf(stderr, "No enough memory for allocation of values array.\n"); return -1; }

    retval |= PAPI_read( EventSet, values );
    for (i = 0; i < num_events; ++i) {
        if (!quiet && strstr( event_names[i], "POWER") && data_type[i] == PAPI_DATATYPE_INT64)
            fprintf( stdout, "%-45s%-20s > %lldW\n",
                    event_names[i], event_descrs[i], values[i]);
        if (1 > values[0] || values[0] > values[1] || values[1] > 10000)
            passed = 0;
        if (values[0] > values[2] || values[2] > values[1])
            passed = 0;
    }

    PAPI_stop( EventSet, values );

    if (passed && PAPI_OK == retval)
        fprintf(stdout, "TEST PASSED\n");
    else
        fprintf(stdout, "TESTS FAILED\n");

    /* Done, clean up */
    retval |= PAPI_cleanup_eventset( EventSet );
    if ( retval != PAPI_OK )
        fprintf(stderr, "PAPI_cleanup_eventset()\n");

    retval |= PAPI_destroy_eventset( &EventSet );
    if ( retval != PAPI_OK )
        fprintf(stderr, "PAPI_destroy_eventset()\n");

    return 0;
}

