/**
 * @author PAPI team UTK/ICL
 * Test case for powercap component
 * @brief
 *   Tests basic reading functionality of powercap component
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

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
    long long values[1];
    int code;
    char event_names[1][PAPI_MAX_STR_LEN];
    char event_descrs[1][PAPI_HUGE_STR_LEN];
    char units[1][PAPI_MIN_STR_LEN];
    int r;

    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;

    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

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

    /* Add all events, but one at a time */
    printf( "\nThis test may take a few minutes to complete.\n\n" );
    code = PAPI_NATIVE_MASK;
    r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, powercap_cid );
    while ( r == PAPI_OK ) {
        retval = PAPI_event_code_to_name( code, event_names[0] );
        if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__,"Error from PAPI_event_code_to_name", retval );

        retval = PAPI_get_event_info( code,&evinfo );
        if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__, "Error getting event info\n",retval );

        strncpy( event_descrs[0],evinfo.long_descr,PAPI_HUGE_STR_LEN );
        strncpy( units[0],evinfo.units,PAPI_MIN_STR_LEN );

        retval = PAPI_add_event( EventSet, code );

        if ( retval != PAPI_OK )
            break; /* We've hit an event limit */

        /* Start Counting */
        retval = PAPI_start( EventSet );
        if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__, "PAPI_start()",retval );

        /* Run test */
        run_test( TESTS_QUIET );

        /* Stop Counting */
        retval = PAPI_stop( EventSet, values );
        if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__, "PAPI_stop()",retval );

        if ( !TESTS_QUIET ) {
            printf( "\n" );
            printf( "%-50s%4.6f %s\n",
                    event_names[0], ( double )values[0]/1.0e0, units[0]);
        }

        /* Clean-up event set before the next event is added */
        retval = PAPI_cleanup_eventset( EventSet );
        if ( retval != PAPI_OK )
            test_fail( __FILE__, __LINE__,"PAPI_cleanup_eventset()",retval );

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, powercap_cid );
    }

    /* Done, clean up */
    retval = PAPI_destroy_eventset( &EventSet );
    if ( retval != PAPI_OK )
        test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset()",retval );

    test_pass( __FILE__ );

    return 0;
}

