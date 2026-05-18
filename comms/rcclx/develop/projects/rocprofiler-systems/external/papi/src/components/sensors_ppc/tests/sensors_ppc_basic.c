/**
 * @author PAPI team UTK/ICL
 * Test case for sensors_ppc component
 * @brief
 *   Tests basic functionality of sensors_ppc component
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"

#define MAX_sensors_ppc_EVENTS 64

int main ( int argc, char **argv )
{
    (void) argv;
    (void) argc;
    int retval,cid,sensors_ppc_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long *values;
    int num_events=0;
    int code;
    char event_names[MAX_sensors_ppc_EVENTS][PAPI_MAX_STR_LEN];
    char units[MAX_sensors_ppc_EVENTS][PAPI_MIN_STR_LEN];
    int r,i;

    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT )
        fprintf(stderr, "PAPI_library_init failed %d\n",retval );

    numcmp = PAPI_num_components();

    for( cid=0; cid<numcmp; cid++ ) {
        if ( ( cmpinfo = PAPI_get_component_info( cid ) ) == NULL )
            fprintf(stderr, "PAPI_get_component_info failed cid=%d\n", cid);
        if ( strstr( cmpinfo->name,"sensors_ppc" ) ) {
            sensors_ppc_cid=cid;
            break;
        }
    }

    /* Component not found */
    if ( cid==numcmp )
        fprintf(stderr, "No sensors_ppc component found\n");

    /* Skip if component has no counters */
    if ( cmpinfo->num_cntrs==0 )
        fprintf(stderr, "No counters in the sensors_ppc component\n");

    /* Create EventSet */
    retval = PAPI_create_eventset( &EventSet );
    if ( retval != PAPI_OK )
        fprintf(stderr, "PAPI_create_eventset()\n");

    /* Add all events */
    code = PAPI_NATIVE_MASK;
    r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, sensors_ppc_cid );
    while ( r == PAPI_OK ) {
        retval = PAPI_event_code_to_name(code, event_names[num_events]);
        if ( retval != PAPI_OK )
            fprintf(stderr, "Error from PAPI_event_code_to_name, error = %d\n", retval);

        retval = PAPI_get_event_info(code,&evinfo);
        if ( retval != PAPI_OK )
            fprintf(stderr, "Error getting event info, error = %d\n",retval);

        char *evt = "sensors_ppc:::VOLTVDD:occ=0";
        if (!strncmp(event_names[num_events], evt, strlen(evt))) {
            retval = PAPI_add_event( EventSet, code );
            strcpy(units[num_events], evinfo.units);
            num_events++;
        }

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, sensors_ppc_cid );
    }

    values=calloc( num_events,sizeof( long long ) );
    if ( values==NULL )
        fprintf(stderr, "No memory");

    PAPI_start(EventSet);

    PAPI_stop( EventSet, values );

    for (i = 0; i < num_events; ++i)
        fprintf(stdout, "%s > %lld %s\n", event_names[i], values[i], units[i]);

    /* Done, clean up */
    retval = PAPI_cleanup_eventset( EventSet );
    if ( retval != PAPI_OK )
        fprintf(stderr, "PAPI_cleanup_eventset(), error=%d\n",retval );

    retval = PAPI_destroy_eventset( &EventSet );
    if ( retval != PAPI_OK )
        fprintf(stderr,  "PAPI_destroy_eventset(), error=%d\n",retval );

    return 0;
}
