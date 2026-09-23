/* This file performs the following test: valid fields in hw_info */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	int retval, i, j;
	const PAPI_hw_info_t *hwinfo = NULL;
	const PAPI_mh_info_t *mh;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if (!TESTS_QUIET) {
		printf( "Test case hwinfo.c: "
			"Check output of PAPI_get_hardware_info.\n");
	}
	hwinfo=PAPI_get_hardware_info();
	if ( hwinfo == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

	mh = &hwinfo->mem_hierarchy;

	validate_string( hwinfo->vendor_string, "vendor_string" );
	validate_string( hwinfo->model_string, "model_string" );

	if ( hwinfo->vendor == PAPI_VENDOR_UNKNOWN )
		test_fail( __FILE__, __LINE__, "Vendor unknown", 0 );

	if ( hwinfo->cpu_max_mhz == 0.0 )
		test_fail( __FILE__, __LINE__, "Mhz unknown", 0 );

	if ( hwinfo->ncpu < 1 )
		test_fail( __FILE__, __LINE__, "ncpu < 1", 0 );

	if ( hwinfo->totalcpus < 1 )
		test_fail( __FILE__, __LINE__, "totalcpus < 1", 0 );

	/*	if ( PAPI_get_opt( PAPI_MAX_HWCTRS, NULL ) < 1 )
		test_fail( __FILE__, __LINE__, "get_opt(MAX_HWCTRS) < 1", 0 );

	if ( PAPI_get_opt( PAPI_MAX_MPX_CTRS, NULL ) < 1 )
	test_fail( __FILE__, __LINE__, "get_opt(MAX_MPX_CTRS) < 1", 0 );*/

	if ( mh->levels < 0 )
		test_fail( __FILE__, __LINE__, "max mh level < 0", 0 );

	if (!TESTS_QUIET) {
	printf( "Max level of TLB or Cache: %d\n", mh->levels );
	for ( i = 0; i < mh->levels; i++ ) {
		for ( j = 0; j < PAPI_MH_MAX_LEVELS; j++ ) {
			const PAPI_mh_cache_info_t *c = &mh->level[i].cache[j];
			const PAPI_mh_tlb_info_t *t = &mh->level[i].tlb[j];
			printf( "Level %d, TLB %d: %d, %d, %d\n", i, j, t->type,
					t->num_entries, t->associativity );
			printf( "Level %d, Cache %d: %d, %d, %d, %d, %d\n", i, j, c->type,
					c->size, c->line_size, c->num_lines, c->associativity );
		}
	}
	}

	test_pass( __FILE__ );

	return 0;
}
