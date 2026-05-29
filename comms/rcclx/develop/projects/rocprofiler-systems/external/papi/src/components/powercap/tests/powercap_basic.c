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

#ifdef BASIC_TEST

void run_test( int quiet )
{
    if ( !quiet ) {
        printf( "Sleeping 1 second...\n" );
    }
    sleep( 1 );
}

#else  /* NOT BASIC_TEST */

#define MATRIX_SIZE 1024
static double a[MATRIX_SIZE][MATRIX_SIZE];
static double b[MATRIX_SIZE][MATRIX_SIZE];
static double c[MATRIX_SIZE][MATRIX_SIZE];

/* Naive matrix multiply */
void run_test( int quiet )
{
    double s;
    int i,j,k;

    if ( !quiet ) printf( "Doing a naive %dx%d MMM...\n",MATRIX_SIZE,MATRIX_SIZE );

    for( i=0; i<MATRIX_SIZE; i++ ) {
        for( j=0; j<MATRIX_SIZE; j++ ) {
            a[i][j]=( double )i*( double )j;
            b[i][j]=( double )i/( double )( j+5 );
        }
    }

    for( j=0; j<MATRIX_SIZE; j++ ) {
        for( i=0; i<MATRIX_SIZE; i++ ) {
            s=0;
            for( k=0; k<MATRIX_SIZE; k++ ) {
                s+=a[i][k]*b[k][j];
            }
            c[i][j] = s;
        }
    }

    s=0.0;
    for( i=0; i<MATRIX_SIZE; i++ ) {
        for( j=0; j<MATRIX_SIZE; j++ ) {
            s+=c[i][j];
        }
    }

    if ( !quiet ) printf( "Matrix multiply sum: s=%lf\n",s );
}

#endif

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
    char event_descrs[MAX_powercap_EVENTS][PAPI_HUGE_STR_LEN];
    char units[MAX_powercap_EVENTS][PAPI_MIN_STR_LEN];
    int data_type[MAX_powercap_EVENTS];
    int r,i;

    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;
    long long before_time,after_time;
    double elapsed_time;

    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

    /* Currently unimplemented? */
#if 0
    int do_wrap = 0;
    if ( argc > 1 ) {
        if ( strstr( argv[1], "-w" ) ) {
            do_wrap = 1;
        }
    }
#endif

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT )
        test_fail( __FILE__, __LINE__,"PAPI_library_init failed\n",retval );

    if ( !TESTS_QUIET ) printf( "Trying all powercap events\n" );

    numcmp = PAPI_num_components();

    for( cid=0; cid<numcmp; cid++ ) {

        if ( ( cmpinfo = PAPI_get_component_info( cid ) ) == NULL )
            test_fail( __FILE__, __LINE__,"PAPI_get_component_info failed\n", 0 );

        if ( strstr( cmpinfo->name,"powercap" ) ) {
            powercap_cid=cid;
            if ( !TESTS_QUIET ) printf( "Found powercap component at cid %d\n",powercap_cid );
            if ( cmpinfo->disabled ) {
                if ( !TESTS_QUIET ) {
                    printf( "powercap component disabled: %s\n",
                            cmpinfo->disabled_reason );
                }
                test_skip( __FILE__,__LINE__,"powercap component disabled",0 );
            }
            break;
        }
    }

    /* Component not found */
    if ( cid==numcmp )
        test_skip( __FILE__,__LINE__,"No powercap component found\n",0 );

    /* Skip if component has no counters */
    if ( cmpinfo->num_cntrs==0 )
        test_skip( __FILE__,__LINE__,"No counters in the powercap component\n",0 );

    /* Create EventSet */
    retval = PAPI_create_eventset( &EventSet );
    if ( retval != PAPI_OK )
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset()",retval );

    /* Add all events */
    code = PAPI_NATIVE_MASK;
    r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, powercap_cid );
    while ( r == PAPI_OK ) {
        retval = PAPI_event_code_to_name( code, event_names[num_events] );
        if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__,"Error from PAPI_event_code_to_name", retval );

        retval = PAPI_get_event_info( code,&evinfo );
        if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__, "Error getting event info\n",retval );

        strncpy( event_descrs[num_events],evinfo.long_descr,PAPI_HUGE_STR_LEN );
        strncpy( units[num_events],evinfo.units,PAPI_MIN_STR_LEN );
        // buffer must be null terminated to safely use strstr operation on it below
        event_names[num_events][PAPI_MAX_STR_LEN-1] = '\0';
        data_type[num_events] = evinfo.data_type;
        retval = PAPI_add_event( EventSet, code );

        if ( retval != PAPI_OK )
            break; /* We've hit an event limit */
        num_events++;

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, powercap_cid );
    }

    values=calloc( num_events,sizeof( long long ) );
    if ( values==NULL )
        test_fail( __FILE__, __LINE__,"No memory",retval );

    if ( !TESTS_QUIET ) printf( "\nStarting measurements...\n\n" );

    /* Start Counting */
    before_time=PAPI_get_real_nsec();
    retval = PAPI_start( EventSet );
    if ( retval != PAPI_OK )
        test_fail( __FILE__, __LINE__, "PAPI_start()",retval );

    /* Run test */
    run_test( TESTS_QUIET );

    /* Stop Counting */
    after_time=PAPI_get_real_nsec();
    retval = PAPI_stop( EventSet, values );
    if ( retval != PAPI_OK )
        test_fail( __FILE__, __LINE__, "PAPI_stop()",retval );

    elapsed_time=( ( double )( after_time-before_time ) )/1.0e9;

    if ( !TESTS_QUIET ) {
        printf( "\nStopping measurements, took %.3fs, gathering results...\n\n", elapsed_time );

        printf( "\n" );
        printf( "scaled energy measurements:\n" );
        for( i=0; i<num_events; i++ ) {
            if ( strstr( event_names[i],"ENERGY_UJ" ) ) {
                if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                    printf( "%-45s%-20s%4.6f J (Average Power %.1fW)\n",
                            event_names[i], event_descrs[i],
                            ( double )values[i]/1.0e6,
                            ( ( double )values[i]/1.0e6 )/elapsed_time );
                }
            }
        }

        printf( "\n" );
        printf( "energy counts:\n" );
        for( i=0; i<num_events; i++ ) {
            if ( strstr( event_names[i],"ENERGY_UJ" ) ) {
                if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                    printf( "%-45s%-20s%12lld\t%#08llx\n", event_names[i],
                            event_descrs[i],
                            values[i], values[i] );
                }
            }
        }

        printf( "\n" );
        printf( "long term time window values:\n" );
        for( i=0; i<num_events; i++ ) {
            if ( strstr( event_names[i],"TIME_WINDOW_A_US" ) ) {
                if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                    printf( "%-45s%-20s%4f (secs)\n",
                            event_names[i], event_descrs[i],
                            ( double )values[i]/1.0e6 );
                }
            }
        }

        printf( "\n" );
        printf( "short term time window values:\n" );
        for( i=0; i<num_events; i++ ) {
            if ( strstr( event_names[i],"TIME_WINDOW_B_US" ) ) {
                if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                    printf( "%-45s%-20s%4f (secs)\n",
                            event_names[i], event_descrs[i],
                            ( double )values[i]/1.0e6 );
                }
            }
        }

        printf( "\n" );
        printf( "long term power limit:\n" );
        for( i=0; i<num_events; i++ ) {
            if ( strstr( event_names[i],"POWER_LIMIT_A_UW" ) ) {
                if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                    printf( "%-45s%-20s%4f (watts)\n",
                            event_names[i], event_descrs[i],
                            ( double )values[i]/1.0e6 );
                }
            }
        }

        printf( "\n" );
        printf( "short term power limit:\n" );
        for( i=0; i<num_events; i++ ) {
            if ( strstr( event_names[i],"POWER_LIMIT_B_UW" ) ) {
                if ( data_type[i] == PAPI_DATATYPE_UINT64 ) {
                    printf( "%-45s%-20s%4f (watts)\n",
                            event_names[i], event_descrs[i],
                            ( double )values[i]/1.0e6 );
                }
            }
        }

    }

    /* Done, clean up */
    free(values);

    retval = PAPI_cleanup_eventset( EventSet );
    if ( retval != PAPI_OK )
        test_fail( __FILE__, __LINE__,"PAPI_cleanup_eventset()",retval );

    retval = PAPI_destroy_eventset( &EventSet );
    if ( retval != PAPI_OK )
        test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset()",retval );

    test_pass( __FILE__ );

    return 0;
}

