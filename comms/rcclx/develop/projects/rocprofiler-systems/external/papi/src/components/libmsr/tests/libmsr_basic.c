/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @author  Asim YarKhan
 * @author  Vince Weaver (original rapl version)
 *
 * Test case for libmsr component
 *
 * @brief
 *   Tests basic functionality of libmsr component
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define MAX_LIBMSR_EVENTS 64

typedef union { long long ll; double dbl; } ll_dbl_union_t;

#ifdef SLEEP_TEST
void run_test( int quiet )
{
    if ( !quiet )
        printf( "Sleeping 2 second...\n" );
    sleep( 2 );
}

#else  /* e.g. nothing defined or BUSY_TEST / WRAP_TEST */

#define MATRIX_SIZE 1024

static double a[MATRIX_SIZE][MATRIX_SIZE];
static double b[MATRIX_SIZE][MATRIX_SIZE];
static double c[MATRIX_SIZE][MATRIX_SIZE];

/* Naive matrix multiply */
void run_test( int quiet )
{
    double s;
    int i,j,k;
    if ( !quiet ) 
        printf( "Doing a naive %dx%d MMM...\n",MATRIX_SIZE,MATRIX_SIZE );
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
    int retval,cid,libmsr_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long *values;
    int num_events=0;
    int code;
    char event_names[MAX_LIBMSR_EVENTS][PAPI_MAX_STR_LEN];
    char units[MAX_LIBMSR_EVENTS][PAPI_MIN_STR_LEN];
    int data_type[MAX_LIBMSR_EVENTS];
    int r,i, do_wrap = 0;
    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;
    long long before_time,after_time;
    double elapsed_time;
    ll_dbl_union_t tmp_ll_dbl;
    int repeat;

    (void) do_wrap; // 'wrap' option is not implemented.
    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );
    if ( argc > 1 && strstr( argv[1], "-w" ) )
        do_wrap = 1;

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) 
        test_fail( __FILE__, __LINE__,"PAPI_library_init failed\n",retval );
    
    if ( !TESTS_QUIET ) 
        printf( "Trying all LIBMSR events\n" );

    numcmp = PAPI_num_components();

    for( cid=0; cid<numcmp; cid++ ) {

        if ( ( cmpinfo = PAPI_get_component_info( cid ) ) == NULL ) 
            test_fail( __FILE__, __LINE__,"PAPI_get_component_info failed\n", 0 );
        
        if ( strstr( cmpinfo->name,"libmsr" ) ) {
            libmsr_cid=cid;
            if ( !TESTS_QUIET ) printf( "Found libmsr component at cid %d\n",libmsr_cid );
            if ( cmpinfo->disabled ) {
                if ( !TESTS_QUIET ) printf( "libmsr component disabled: %s\n",cmpinfo->disabled_reason );
                test_skip( __FILE__,__LINE__, "libmsr component disabled", 0 );
            }
            break;
        }
    }
    
    /* Component not found */
    if ( cid==numcmp ) 
        test_skip( __FILE__,__LINE__,"No libmsr component found\n",0 );

    /* Create EventSet */
    retval = PAPI_create_eventset( &EventSet );
    if ( retval != PAPI_OK ) 
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset()",retval );

    /* Add all events */
    num_events = 0;
    code = PAPI_NATIVE_MASK;
    r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, libmsr_cid );
    while ( r == PAPI_OK ) {
        retval = PAPI_event_code_to_name( code, event_names[num_events] );
        if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
        retval = PAPI_get_event_info( code,&evinfo );
        if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__,"Error getting event info\n",retval );
        strncpy( units[num_events], evinfo.units, sizeof( units[0] )-1 );
        // buffer must be null terminated to safely use strstr operation on it below
        units[num_events][sizeof( units[0] )-1] = '\0';
        data_type[num_events] = evinfo.data_type;

        retval = PAPI_add_event( EventSet, code );
        if ( retval != PAPI_OK ) 
            break; /* We've hit an event limit */
        num_events++;

        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, libmsr_cid );
    }

    values=calloc( num_events,sizeof( long long ) );
    if ( values==NULL ) 
        test_fail( __FILE__, __LINE__,"No memory - calloc failed",retval );
    
    if ( !TESTS_QUIET ) printf( "Starting measurements...\n" );

    /* Start Counting */
    before_time=PAPI_get_real_nsec();
    retval = PAPI_start( EventSet );
    if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_start()",retval );

    for ( repeat=0; repeat<3; repeat++ ) {

        /* Run test */
        run_test( TESTS_QUIET );
        
        /* Stop Counting */
        after_time=PAPI_get_real_nsec();
        retval = PAPI_read( EventSet, values );
        if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_read()",retval );
        
        elapsed_time=( ( double )( after_time-before_time ) )/1.0e9;
        
        if ( !TESTS_QUIET ) {
            printf( "Stopping measurements, took %.3fs, gathering results...\n", elapsed_time );
            for( i=0; i<num_events; i++ ) {
                tmp_ll_dbl.ll = values[i];
                if ( data_type[i]==PAPI_DATATYPE_FP64 ) {
                    printf( "%-40s %12.6f %s\n", event_names[i], tmp_ll_dbl.dbl, units[i] );
                } else if ( data_type[i]==PAPI_DATATYPE_UINT64 ) {
                    printf( "%-40s %12lld %s\n", event_names[i], tmp_ll_dbl.ll, units[i] );
                }
            }            
            printf( "\n" );
        }
    }

    retval = PAPI_stop( EventSet, values );
    if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_stop()",retval );
     
    retval = PAPI_cleanup_eventset( EventSet );
    if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset()",retval );

    retval = PAPI_destroy_eventset( &EventSet );
    if ( retval != PAPI_OK ) test_fail( __FILE__, __LINE__,"PAPI_destroy_eventset()",retval );

    test_pass( __FILE__ );

    return 0;
}

