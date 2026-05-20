/* zero.c */

/* This is possibly the most important PAPI tests, and is the one */
/* that is often used as a quick test that PAPI is working.       */
/* We should make sure that it always passes, if possible.        */

/* Traditionally it used FLOPS, due to the importance of this to HPC.       */
/* This has been changed to use Instructions/Cycles as some recent          */
/* major Intel chips do not have good floating point events and would fail. */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#define NUM_EVENTS	2

#define NUM_LOOPS	200

int main( int argc, char **argv ) {

	int retval, tmp, result, i;
	int EventSet1 = PAPI_NULL;
	long long values[NUM_EVENTS];
	long long elapsed_us, elapsed_cyc, elapsed_virt_us, elapsed_virt_cyc;
	double ipc;
	int quiet=0;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Initialize the EventSet */
	retval=PAPI_create_eventset(&EventSet1);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* Add PAPI_TOT_CYC */
	retval=PAPI_add_named_event(EventSet1,"PAPI_TOT_CYC");
	if (retval!=PAPI_OK) {
		if (!quiet) {
			printf("Trouble adding PAPI_TOT_CYC: %s\n",
				PAPI_strerror(retval));
		}
		test_skip( __FILE__, __LINE__, "adding PAPI_TOT_CYC", retval );
	}

	/* Add PAPI_TOT_INS */
	retval=PAPI_add_named_event(EventSet1,"PAPI_TOT_INS");
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "adding PAPI_TOT_INS", retval );
	}

	/* warm up the processor to pull it out of idle state */
	for(i=0;i<100;i++) {
		result=instructions_million();
	}

	if (result==CODE_UNIMPLEMENTED) {
		if (!quiet) printf("Instructions testcode not available\n");
		test_skip( __FILE__, __LINE__, "No instructions code", retval );
	}

	/* Gather before stats */
	elapsed_us = PAPI_get_real_usec(  );
	elapsed_cyc = PAPI_get_real_cyc(  );
	elapsed_virt_us = PAPI_get_virt_usec(  );
	elapsed_virt_cyc = PAPI_get_virt_cyc(  );

	/* Start PAPI */
	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* our work code */
	for(i=0;i<NUM_LOOPS;i++) {
		instructions_million();
	}

	/* Stop PAPI */
	retval = PAPI_stop( EventSet1, values );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	/* Calculate total values */
	elapsed_virt_us = PAPI_get_virt_usec(  ) - elapsed_virt_us;
	elapsed_virt_cyc = PAPI_get_virt_cyc(  ) - elapsed_virt_cyc;
	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	/* Shutdown the EventSet */
	retval = PAPI_remove_named_event( EventSet1, "PAPI_TOT_CYC" );
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_remove_named_event", retval );
	}

	retval = PAPI_remove_named_event( EventSet1, "PAPI_TOT_INS" );
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_remove_named_event", retval );
	}

	retval=PAPI_destroy_eventset( &EventSet1 );
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );
	}

	/* Calculate Instructions per Cycle, avoiding division by zero */
	if (values[0]!=0) {
		ipc = (double)values[1]/(double)values[0];
	}
	else {
		ipc=0.0;
	}

	/* Print the results */
	if ( !quiet ) {
		printf( "Test case 0: start, stop.\n" );
		printf( "-----------------------------------------------\n" );
		tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
		printf( "Default domain is: %d (%s)\n", tmp,
				stringify_all_domains( tmp ) );
		tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
		printf( "Default granularity is: %d (%s)\n", tmp,
				stringify_granularity( tmp ) );
		printf( "Using %d iterations 1 million instructions\n", NUM_LOOPS );
		printf( "-------------------------------------------------------------------------\n" );

		printf( "Test type    : \t           1\n" );

		/* cycles is first, other event second */
		printf( "%-12s %12lld\n", "PAPI_TOT_CYC : \t", values[0] );
		printf( "%-12s %12lld\n", "PAPI_TOT_INS : \t", values[1] );
		printf( "%-12s %12.2lf\n",  "IPC          : \t", ipc );

		printf( "%-12s %12lld\n", "Real usec    : \t", elapsed_us );
		printf( "%-12s %12lld\n", "Real cycles  : \t", elapsed_cyc );
		printf( "%-12s %12lld\n", "Virt usec    : \t", elapsed_virt_us );
		printf( "%-12s %12lld\n", "Virt cycles  : \t", elapsed_virt_cyc );

		printf( "-------------------------------------------------------------------------\n" );


		printf( "Verification: PAPI_TOT_INS should be roughly %d\n", NUM_LOOPS*1000000 );

	}

	/* Check that TOT_INS is reasonable */
	if (llabs(values[1] - (1000000*NUM_LOOPS)) > (1000000*NUM_LOOPS)) {
		printf("%s Error of %.2f%%\n", "PAPI_TOT_INS", (100.0 * (double)(values[1] - (1000000*NUM_LOOPS)))/(1000000*NUM_LOOPS));
		test_fail( __FILE__, __LINE__, "Instruction validation", 0 );
	}

	/* Check that TOT_CYC is non-zero */
	if(values[0]==0) {
		printf("Cycles is zero\n");
		test_fail( __FILE__, __LINE__, "Cycles validation", 0 );
	}

	/* Unless you have an amazing processor, IPC should be < 100 */
	if ((ipc <=0.01 ) || (ipc >=100.0)) {
		printf("Unlikely IPC of %.2f%%\n", ipc);
		test_fail( __FILE__, __LINE__, "IPC validation", 0 );
	}

	test_pass( __FILE__ );

	return 0;
}
