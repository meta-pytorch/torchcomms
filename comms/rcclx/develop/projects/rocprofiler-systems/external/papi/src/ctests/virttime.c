#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "papi.h"
#include "papi_test.h"

int
main( int argc, char **argv )
{
	int retval;
	long long elapsed_us, elapsed_cyc;
	const PAPI_hw_info_t *hw_info;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	elapsed_us = PAPI_get_virt_usec(  );
	elapsed_cyc = PAPI_get_virt_cyc(  );

	if (!TESTS_QUIET) {
		printf( "Testing virt time clock. (CPU Max %d MHz, CPU Min %d MHz)\n",
				hw_info->cpu_max_mhz, hw_info->cpu_min_mhz );
		printf( "Sleeping for 10 seconds.\n" );
	}

	sleep( 10 );

	elapsed_us = PAPI_get_virt_usec(  ) - elapsed_us;
	elapsed_cyc = PAPI_get_virt_cyc(  ) - elapsed_cyc;

	if (!TESTS_QUIET) {
		printf( "%lld us. %lld cyc.\n", elapsed_us, elapsed_cyc );
	}

/* Elapsed microseconds and elapsed cycles are not as unambiguous as they appear.
   On Pentium III and 4, for example, cycles is a measured value, while useconds 
   is computed from cycles and mhz. MHz is read from /proc/cpuinfo (on linux).
   Thus, any error in MHz is propagated to useconds.
   Conversely, on ultrasparc useconds are extracted from a system call (gethrtime())
   and cycles are computed from useconds. Also, MHz comes from a scan of system info,
   Thus any error in gethrtime() propagates to both cycles and useconds, and cycles
   can be further impacted by errors in reported MHz.
   Without knowing the error bars on these system values, we can't really specify
   error ranges for our reported values, but we *DO* know that errors for at least
   one instance of Pentium 4 (torc17@utk) are on the order of one part per thousand.
*/

	/* We'll accept 1.5 part per thousand error here (to allow Pentium 4
	   and Alpha to pass) */
	if ( elapsed_us > 100000 )
		test_fail( __FILE__, __LINE__, "Virt time greater than .1 seconds!",
				   PAPI_EMISC );

	test_pass( __FILE__ );

	return 0;

}
