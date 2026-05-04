/* 
* File:    overflow_values.c
* CVS:     $Id$
* Author:  Harald Servat
*          harald@cepba.upc.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: overflow values check

     The Eventset contains:
     + PAPI_TOT_INS (overflow monitor)
     + PAPI_TOT_CYC
     + PAPI_L1_DCM

   - Start eventset
   - Read and report event counts mod 1000
   - report overflow event counts
   - visually inspect for consistency
   - Stop eventset
*/

#include "papi_test.h"

#define OVRFLOW 5000000
#define LOWERFLOW (OVRFLOW - (OVRFLOW/100))
#define UPPERFLOW (OVRFLOW/100)
#define ERRORFLOW (UPPERFLOW/5)
static long long ovrflow = 0;

void
handler( int EventSet, void *address, long long overflow_vector, void *context )
{
	int ret;
	int i;
	long long vals[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	printf( "\nOverflow at %p! bit=%#llx \n", address, overflow_vector );
	ret = PAPI_read( EventSet, vals );
	printf( "Overflow read vals :" );
	for ( i = 0; i < 3 /* 8 */ ; i++ )
		printf( "%lld ", vals[i] );
	printf( "\n\n" );
	ovrflow = vals[0];
}

int
main( int argc, char *argv[] )
{
	int EventSet = PAPI_NULL;
	int retval, i, dash = 0, evt3 = PAPI_L1_DCM;
	PAPI_option_t options;
	PAPI_option_t options2;
	const PAPI_hw_info_t *hwinfo;
	long long lwrflow = 0, error, max_error = 0;
	long long vals[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT && retval > 0 )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	retval = PAPI_get_opt( PAPI_HWINFO, &options );
	if ( retval < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_get_opt", retval );
	printf( "ovf_info = %d (%#x)\n", options.ovf_info.type,
			options.ovf_info.type );

	retval = PAPI_get_opt( PAPI_SUBSTRATEINFO, &options2 );
	if ( retval < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_get_opt", retval );
	printf( "sub_info->hardware_intr = %d\n\n",
			options2.sub_info->hardware_intr );

	if ( ( hwinfo = PAPI_get_hardware_info(  ) ) == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", PAPI_EMISC );

	printf( "Architecture %s, %d\n", hwinfo->model_string, hwinfo->model );

/* processing exceptions is a pain */
#if ((defined(linux) && (defined(__i386__) || (defined __x86_64__))) )
	if ( !strncmp( hwinfo->model_string, "Intel Pentium 4", 15 ) ) {
		evt3 = PAPI_L2_TCM;
	} else if ( !strncmp( hwinfo->model_string, "AMD K7", 6 ) ) {
		/* do nothing */
	} else if ( !strncmp( hwinfo->model_string, "AMD K8", 6 ) ) {
		/* do nothing */
	} else if ( !strncmp( hwinfo->model_string, "Intel Core", 10 ) ) {
		evt3 = 0;
	} else
		evt3 = 0;			 /* for default PIII */
#endif

	retval = PAPI_create_eventset( &EventSet );
	if ( retval < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	retval = PAPI_add_event( EventSet, PAPI_TOT_INS );
	if ( retval < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_add_event:PAPI_TOT_INS", retval );
	retval = PAPI_add_event( EventSet, PAPI_TOT_CYC );
	if ( retval < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_add_event:PAPI_TOT_CYC", retval );
	if ( evt3 ) {
		retval = PAPI_add_event( EventSet, evt3 );
		if ( retval < 0 )
			test_fail( __FILE__, __LINE__, "PAPI_add_event:evt3", retval );
	}
	retval = PAPI_overflow( EventSet, PAPI_TOT_INS, OVRFLOW, 0, handler );
	if ( retval < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	retval = PAPI_start( EventSet );
	if ( retval < 0 )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	for ( i = 0; i < 1000000; i++ ) {
		if ( i % 1000 == 0 ) {
			int i;

			PAPI_read( EventSet, vals );
			if ( vals[0] % OVRFLOW > LOWERFLOW ||
				 vals[0] % OVRFLOW < UPPERFLOW ) {
				dash = 0;
				printf( "Main loop read vals :" );
				for ( i = 0; i < 3 /* 8 */ ; i++ )
					printf( "%lld ", vals[i] );
				printf( "\n" );
				if ( ovrflow ) {
					error = ovrflow - ( lwrflow + vals[0] ) / 2;
					printf( "Difference: %lld\n", error );
					ovrflow = 0;
					if ( abs( error ) > max_error )
						max_error = abs( error );
				}
				lwrflow = vals[0];
			} else if ( vals[0] % OVRFLOW > UPPERFLOW && !dash ) {
				dash = 1;
				printf( "---------------------\n" );
			}
		}
	}

	retval = PAPI_stop( EventSet, vals );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	printf( "Verification:\n" );
	printf
		( "Maximum absolute difference between overflow value\nand adjacent measured values is: %lld\n",
		  max_error );
	if ( max_error >= ERRORFLOW ) {
		printf( "This exceeds the error limit: %d\n", ERRORFLOW );
		test_fail( __FILE__, __LINE__, "Overflows", 1 );
	}
	printf( "This is within the error limit: %d\n", ERRORFLOW );
	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
