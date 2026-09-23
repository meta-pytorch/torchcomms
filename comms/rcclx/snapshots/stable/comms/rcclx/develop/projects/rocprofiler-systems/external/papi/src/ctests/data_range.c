/*
* File:    data_range.c
* Author:  Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: */
/*     exercise the Itanium data address range interface */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define NUM 16384

static void init_array( void );
static int do_malloc_work( long loop );
static int do_static_work( long loop );
static void measure_load_store( vptr_t start, vptr_t end );
static void measure_event( int index, PAPI_option_t * option );

int *parray1, *parray2, *parray3;
int array1[NUM], array2[NUM], array3[NUM];
char event_name[2][PAPI_MAX_STR_LEN];
int PAPI_event[2];
int EventSet = PAPI_NULL;

int
main( int argc, char **argv )
{
	int retval;
	const PAPI_exe_info_t *prginfo = NULL;
	const PAPI_hw_info_t *hw_info;

	/* Set TESTS_QUIET variable */
	tests_quiet( argc, argv );

#if !defined(ITANIUM2) && !defined(ITANIUM3)
	test_skip( __FILE__, __LINE__, "Currently only works on itanium2", 0 );
	exit( 1 );
#endif

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	init_array(  );
	printf( "Malloc'd array  pointers: %p   %p   %p\n", &parray1, &parray2,
			&parray3 );
	printf( "Malloc'd array addresses: %p   %p   %p\n", parray1, parray2,
			parray3 );
	printf( "Static   array addresses: %p   %p   %p\n", &array1, &array2,
			&array3 );

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	prginfo = PAPI_get_executable_info(  );
	if ( prginfo == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_executable_info", 1 );

#if defined(linux) && defined(__ia64__)
	sprintf( event_name[0], "loads_retired" );
	sprintf( event_name[1], "stores_retired" );
	PAPI_event_name_to_code( event_name[0], &PAPI_event[0] );
	PAPI_event_name_to_code( event_name[1], &PAPI_event[1] );
#else
	test_skip( __FILE__, __LINE__, "only works for Itanium", PAPI_ENOSUPP );
#endif

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );

/***************************************************************************************/
	printf
		( "\n\nMeasure loads and stores on the pointers to the allocated arrays\n" );
	printf( "Expected loads: %d; Expected stores: 0\n", NUM * 2 );
	printf
		( "These loads result from accessing the pointers to compute array addresses.\n" );
	printf
		( "They will likely disappear with higher levels of optimization.\n" );

	measure_load_store( ( vptr_t ) & parray1, ( vptr_t ) ( &parray1 + 1 ) );
	measure_load_store( ( vptr_t ) & parray2, ( vptr_t ) ( &parray2 + 1 ) );
	measure_load_store( ( vptr_t ) & parray3, ( vptr_t ) ( &parray3 + 1 ) );
/***************************************************************************************/
	printf
		( "\n\nMeasure loads and stores on the allocated arrays themselves\n" );
	printf( "Expected loads: %d; Expected stores: %d\n", NUM, NUM );

	measure_load_store( ( vptr_t ) parray1, ( vptr_t ) ( parray1 + NUM ) );
	measure_load_store( ( vptr_t ) parray2, ( vptr_t ) ( parray2 + NUM ) );
	measure_load_store( ( vptr_t ) parray3, ( vptr_t ) ( parray3 + NUM ) );
/***************************************************************************************/
	printf( "\n\nMeasure loads and stores on the static arrays\n" );
	printf
		( "These values will differ from the expected values by the size of the offsets.\n" );
	printf( "Expected loads: %d; Expected stores: %d\n", NUM, NUM );

	measure_load_store( ( vptr_t ) array1, ( vptr_t ) ( array1 + NUM ) );
	measure_load_store( ( vptr_t ) array2, ( vptr_t ) ( array2 + NUM ) );
	measure_load_store( ( vptr_t ) array3, ( vptr_t ) ( array3 + NUM ) );
/***************************************************************************************/

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy", retval );

	free( parray1 );
	free( parray2 );
	free( parray3 );

	test_pass( __FILE__ );

	return 0;

}

static void
measure_load_store( vptr_t start, vptr_t end )
{
	PAPI_option_t option;
	int retval;

	/* set up the optional address structure for starting and ending data addresses */
	option.addr.eventset = EventSet;
	option.addr.start = start;
	option.addr.end = end;

	if ( ( retval = PAPI_set_opt( PAPI_DATA_ADDRESS, &option ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_set_opt(PAPI_DATA_ADDRESS)",
				   retval );

	measure_event( 0, &option );
	measure_event( 1, &option );
}

static void
measure_event( int index, PAPI_option_t * option )
{
	int retval;
	long long value;

	if ( ( retval = PAPI_add_event( EventSet, PAPI_event[index] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );

	if ( index == 0 ) {
/*	    if ((retval = PAPI_get_opt(PAPI_DATA_ADDRESS, option)) != PAPI_OK)
	      test_fail(__FILE__, __LINE__, "PAPI_get_opt(PAPI_DATA_ADDRESS)", retval);
*/
		printf
			( "Requested Start Address: %p; Start Offset: %#5x; Actual Start Address: %p\n",
			  option->addr.start, option->addr.start_off,
			  option->addr.start - option->addr.start_off );
		printf
			( "Requested End   Address: %p; End   Offset: %#5x; Actual End   Address: %p\n",
			  option->addr.end, option->addr.end_off,
			  option->addr.end + option->addr.end_off );
	}

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}
	do_malloc_work( NUM );
	do_static_work( NUM );
	retval = PAPI_stop( EventSet, &value );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	printf( "%s:  %lld\n", event_name[index], value );

	if ( ( retval =
		   PAPI_remove_event( EventSet, PAPI_event[index] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_remove_event", retval );
}

static void
init_array( void )
{
	parray1 = ( int * ) malloc( NUM * sizeof ( int ) );
	if ( parray1 == NULL )
		test_fail( __FILE__, __LINE__, "No memory available!\n", 0 );
	memset( parray1, 0x0, NUM * sizeof ( int ) );

	parray2 = ( int * ) malloc( NUM * sizeof ( int ) );
	if ( parray2 == NULL )
		test_fail( __FILE__, __LINE__, "No memory available!\n", 0 );
	memset( parray2, 0x0, NUM * sizeof ( int ) );

	parray3 = ( int * ) malloc( NUM * sizeof ( int ) );
	if ( parray3 == NULL )
		test_fail( __FILE__, __LINE__, "No memory available!\n", 0 );
	memset( parray3, 0x0, NUM * sizeof ( int ) );

}

static int
do_static_work( long loop )
{
	int i;
	int sum = 0;

	for ( i = 0; i < loop; i++ ) {
		array1[i] = i;
		sum += array1[i];
	}

	for ( i = 0; i < loop; i++ ) {
		array2[i] = i;
		sum += array2[i];
	}

	for ( i = 0; i < loop; i++ ) {
		array3[i] = i;
		sum += array3[i];
	}

	return sum;
}

static int
do_malloc_work( long loop )
{
	int i;
	int sum = 0;

	for ( i = 0; i < loop; i++ ) {
		parray1[i] = i;
		sum += parray1[i];
	}

	for ( i = 0; i < loop; i++ ) {
		parray2[i] = i;
		sum += parray2[i];
	}

	for ( i = 0; i < loop; i++ ) {
		parray3[i] = i;
		sum += parray3[i];
	}

	return sum;
}
