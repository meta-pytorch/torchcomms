/* flops.c, based on the hl_rates.c ctest
 *
 * This test runs a "classic" matrix multiply
 * and then runs it again with the inner loop swapped.
 * the swapped version should have better MFLIPS/MFLOPS/IPC and we test that.
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

int
main( int argc, char **argv )
{
	int retval;
	double rtime, ptime, mflips, mflops, ipc;
	long long flips=0, flops=0, ins[2];

	double rtime_start,rtime_end;
	double ptime_start,ptime_end;

	double rtime_classic,rtime_swapped;
	double mflips_classic,mflips_swapped;
	double mflops_classic,mflops_swapped;
	double ipc_classic,ipc_swapped;

	int quiet,event_added_flips,event_added_flops,event_added_ipc;

	int eventset=PAPI_NULL;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );


	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Create the eventset */
	retval=PAPI_create_eventset(&eventset);
	if (retval!=PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	/* Initialize the test matrix */
	flops_float_init_matrix();

	/************************/
	/* FLIPS		*/
	/************************/

	if (!quiet) {
		printf( "\n----------------------------------\n" );
		printf( "PAPI_flips\n");
	}

	/* Add FP_INS event */
	retval=PAPI_add_named_event(eventset,"PAPI_FP_INS");
	if (retval!=PAPI_OK) {
		if (!quiet) fprintf(stderr,"PAPI_FP_INS not available!\n");
		event_added_flips=0;
	}
	else {
		event_added_flips=1;
	}

	if (event_added_flips) {
		PAPI_start(eventset);
	}

	rtime_start=PAPI_get_real_usec();
	ptime_start=PAPI_get_virt_usec();

	// Flips classic
	flops_float_matrix_matrix_multiply();

	rtime_end=PAPI_get_real_usec();
	ptime_end=PAPI_get_virt_usec();

	if (event_added_flips) {
		PAPI_stop(eventset,&flips);
	}

	rtime=rtime_end-rtime_start;
	ptime=ptime_end-ptime_start;

	mflips=flips/rtime;

	if (!quiet) {
		printf( "\nClassic\n");
		printf( "real time:       %lf\n", rtime);
		printf( "process time:    %lf\n", ptime);
		printf( "FP Instructions: %lld\n", flips);
		printf( "MFLIPS           %lf\n", mflips);
	}
	mflips_classic=mflips;


	// Flips swapped
	rtime_start=PAPI_get_real_usec();
	ptime_start=PAPI_get_virt_usec();

	if (event_added_flips) {
		PAPI_reset(eventset);
		PAPI_start(eventset);
	}

	flops_float_swapped_matrix_matrix_multiply();

	rtime_end=PAPI_get_real_usec();
	ptime_end=PAPI_get_virt_usec();

	if (event_added_flips) {
		PAPI_stop(eventset,&flips);
	}

	rtime=rtime_end-rtime_start;
	ptime=ptime_end-ptime_start;

	mflips=flips/rtime;

	if (!quiet) {
		printf( "\nSwapped\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Instructions: %lld\n", flips);
		printf( "MFLIPS           %f\n", mflips);
	}
	mflips_swapped=mflips;

	// turn off flips
	if (event_added_flips) {
		retval=PAPI_remove_named_event(eventset,"PAPI_FP_INS");
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"PAPI_remove_named_event", retval );
		}
	}

	/************************/
	/* FLOPS		*/
	/************************/

	if (!quiet) {
		printf( "\n----------------------------------\n" );
		printf( "PAPI_flops\n");
	}

	/* Add FP_OPS event */
	retval=PAPI_add_named_event(eventset,"PAPI_FP_OPS");
	if (retval!=PAPI_OK) {
		if (!quiet) fprintf(stderr,"PAPI_FP_OPS not available!\n");
		event_added_flops=0;
	}
	else {
		event_added_flops=1;
	}

	if (event_added_flops) {
		PAPI_start(eventset);
	}

	rtime_start=PAPI_get_real_usec();
	ptime_start=PAPI_get_virt_usec();

	// Classic flops
	flops_float_matrix_matrix_multiply();

	rtime_end=PAPI_get_real_usec();
	ptime_end=PAPI_get_virt_usec();

	if (event_added_flops) {
		PAPI_stop(eventset,&flops);
	}

	rtime=rtime_end-rtime_start;
	ptime=ptime_end-ptime_start;

	mflops=flops/rtime;

	if (!quiet) {
		printf( "\nClassic\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Operations:   %lld\n", flops);
		printf( "MFLOPS           %f\n", mflops);
	}
	mflops_classic=mflops;

	// Swapped flops

	rtime_start=PAPI_get_real_usec();
	ptime_start=PAPI_get_virt_usec();

	if (event_added_flops) {
		PAPI_reset(eventset);
		PAPI_start(eventset);
	}

	flops_float_swapped_matrix_matrix_multiply();

	rtime_end=PAPI_get_real_usec();
	ptime_end=PAPI_get_virt_usec();

	if (event_added_flops) {
		PAPI_stop(eventset,&flops);
	}

	rtime=rtime_end-rtime_start;
	ptime=ptime_end-ptime_start;

	mflops=flops/rtime;

	if (!quiet) {
		printf( "\nSwapped\n");
		printf( "real time:       %f\n", rtime);
		printf( "process time:    %f\n", ptime);
		printf( "FP Operations:   %lld\n", flops);
		printf( "MFLOPS           %f\n", mflops);
	}
	mflops_swapped=mflops;

	// turn off flops
	if (event_added_flops) {
		retval=PAPI_remove_named_event(eventset,"PAPI_FP_OPS");
		if (retval!=PAPI_OK) {
			test_fail( __FILE__, __LINE__,
				"PAPI_remove_named_event", retval );
		}
	}

	/************************/
	/* IPC  		*/
	/************************/

	if (!quiet) {
		printf( "\n----------------------------------\n" );
		printf( "PAPI_ipc\n");
	}

	/* Add PAPI_TOT_INS event */
	retval=PAPI_add_named_event(eventset,"PAPI_TOT_INS");
	if (retval!=PAPI_OK) {
		if (!quiet) fprintf(stderr,"PAPI_TOT_INS not available!\n");
		event_added_ipc=0;
	}
	else {
		event_added_ipc=1;
	}

	if (event_added_ipc) {
		/* Add PAPI_TOT_CYC event */
		retval=PAPI_add_named_event(eventset,"PAPI_TOT_CYC");
		if (retval!=PAPI_OK) {
			if (!quiet) fprintf(stderr,"PAPI_TOT_CYC not available!\n");
			event_added_ipc=0;
		}
		else {
			event_added_ipc=1;
		}
	}

	if (event_added_ipc) {
		PAPI_start(eventset);
	}

	rtime_start=PAPI_get_real_usec();
	ptime_start=PAPI_get_virt_usec();

	// Classic ipc
	flops_float_matrix_matrix_multiply();

	rtime_end=PAPI_get_real_usec();
	ptime_end=PAPI_get_virt_usec();

	if (event_added_ipc) {
		PAPI_stop(eventset,ins);
	}

	rtime=rtime_end-rtime_start;
	ptime=ptime_end-ptime_start;

	ipc=(double)ins[0]/(double)ins[1];

	if (!quiet) {
		printf( "\nClassic\n");
		printf( "real time:       %lf\n", rtime);
		printf( "process time:    %lf\n", ptime);
		printf( "Instructions:    %lld\n", ins[0]);
		printf( "Cycles:          %lld\n", ins[1]);
		printf( "IPC              %lf\n", ipc);
	}
	ipc_classic=ipc;
	rtime_classic=rtime;

	// Swapped ipc

	if (event_added_ipc) {
		PAPI_reset(eventset);
		PAPI_start(eventset);
	}

	rtime_start=PAPI_get_real_usec();
	ptime_start=PAPI_get_virt_usec();


	flops_float_swapped_matrix_matrix_multiply();

	rtime_end=PAPI_get_real_usec();
	ptime_end=PAPI_get_virt_usec();

	if (event_added_ipc) {
		PAPI_stop(eventset,ins);
	}

	rtime=rtime_end-rtime_start;
	ptime=ptime_end-ptime_start;

	ipc=(double)ins[0]/(double)ins[1];

	if (!quiet) {
		printf( "\nSwapped\n");
		printf( "real time:       %lf\n", rtime);
		printf( "process time:    %lf\n", ptime);
		printf( "Instructions:    %lld\n", ins[0]);
		printf( "Cycles:          %lld\n", ins[1]);
		printf( "IPC              %lf\n", ipc);
	}
	ipc_swapped=ipc;
	rtime_swapped=rtime;


	/* Validate */
	if (event_added_flips) {
		if (mflips_swapped<mflips_classic) {
			test_fail(__FILE__,__LINE__,
				"FLIPS should be better when swapped",0);
		}
	}

	if (event_added_flops) {
		if (mflops_swapped<mflops_classic) {
			test_fail(__FILE__,__LINE__,
				"FLOPS should be better when swapped",0);
		}
	}

	if (event_added_ipc) {
		if (ipc_swapped<ipc_classic) {
			test_fail(__FILE__,__LINE__,
				"IPC should be better when swapped",0);
		}
	}

	if (rtime_swapped>rtime_classic) {
		test_fail(__FILE__,__LINE__,
				"time should be better when swapped",0);
	}

	test_pass( __FILE__ );

	return 0;
}
