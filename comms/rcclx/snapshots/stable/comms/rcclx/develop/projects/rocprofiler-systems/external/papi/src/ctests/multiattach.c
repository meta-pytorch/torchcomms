/* This file performs the following test: start, stop and timer functionality for
   multiple attached processes.

   - It attempts to use the following two counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS
     + PAPI_TOT_CYC
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <inttypes.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

#ifdef _AIX
#define _LINUX_SOURCE_COMPAT
#endif

#if defined(__FreeBSD__)
# define PTRACE_ATTACH PT_ATTACH
# define PTRACE_CONT PT_CONTINUE
#endif

#define MULTIPLIER	5

static int
wait_for_attach_and_loop( int num )
{
	kill( getpid(  ), SIGSTOP );
	do_flops( NUM_FLOPS * num );
	kill( getpid(  ), SIGSTOP );
	return 0;
}

int
main( int argc, char **argv )
{
	int status, retval, num_tests = 2, tmp;
	int EventSet1 = PAPI_NULL, EventSet2 = PAPI_NULL;
	int PAPI_event, PAPI_event2, mask1, mask2;
	int num_events1, num_events2;
	long long **values;
	long long elapsed_us, elapsed_cyc, elapsed_virt_us, elapsed_virt_cyc;
	char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_2MAX_STR_LEN];
	const PAPI_component_info_t *cmpinfo;
	pid_t pid, pid2;
	double ratio1,ratio2;

	/* Set TESTS_QUIET variable */
	tests_quiet( argc, argv );

	/* Initialize the library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* get the component info and check if we support attach */
	if ( ( cmpinfo = PAPI_get_component_info( 0 ) ) == NULL ) {
	   test_fail( __FILE__, __LINE__, "PAPI_get_component_info", 0 );
	}

	if ( cmpinfo->attach == 0 ) {
	   test_skip( __FILE__, __LINE__,
		      "Platform does not support attaching", 0 );
	}

	/* fork off first child */
	pid = fork(  );
	if ( pid < 0 ) {
	   test_fail( __FILE__, __LINE__, "fork()", PAPI_ESYS );
	}
	if ( pid == 0 ) {
	   exit( wait_for_attach_and_loop( 1 ) );
	}

	/* fork off second child, does twice as much */
	pid2 = fork(  );
	if ( pid2 < 0 ) {
	   test_fail( __FILE__, __LINE__, "fork()", PAPI_ESYS );
	}
	if ( pid2 == 0 ) {
	   exit( wait_for_attach_and_loop( MULTIPLIER ) );
	}

	/* add PAPI_TOT_CYC and one of the events in
           PAPI_FP_INS, PAPI_FP_OPS or PAPI_TOT_INS,
           depending on the availability of the event
           on the platform                            */
	EventSet1 = add_two_events( &num_events1, &PAPI_event, &mask1 );
	EventSet2 = add_two_events( &num_events2, &PAPI_event2, &mask2 );

	if ( cmpinfo->attach_must_ptrace ) {
	   if ( ptrace( PTRACE_ATTACH, pid, NULL, NULL ) == -1 ) {
	      perror( "ptrace(PTRACE_ATTACH)" );
	      return 1 ;
	   }
	   if ( waitpid( pid, &status, 0 ) == -1 ) {
	      perror( "waitpid()" );
	      exit( 1 );
	   }
	   if ( WIFSTOPPED( status ) == 0 ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didnt return true to WIFSTOPPED", 0 );
	   }

	   if ( ptrace( PTRACE_ATTACH, pid2, NULL, NULL ) == -1 ) {
	      perror( "ptrace(PTRACE_ATTACH)" );
	      return 1;
	   }
	   if ( waitpid( pid2, &status, 0 ) == -1 ) {
	      perror( "waitpid()" );
	      exit( 1 );
	   }
	   if ( WIFSTOPPED( status ) == 0 ) {
 	      test_fail( __FILE__, __LINE__,
			"Child process didnt return true to WIFSTOPPED", 0 );
	   }
	}

	retval = PAPI_attach( EventSet1, ( unsigned long ) pid );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_attach", retval );
	}

	retval = PAPI_attach( EventSet2, ( unsigned long ) pid2 );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_attach", retval );
	}

	strcpy(event_name, "PAPI_TOT_INS");
	sprintf( add_event_str, "PAPI_add_event[%s]", event_name );

	/* num_events1 is greater than num_events2 so don't worry. */

	values = allocate_test_space( num_tests, num_events1 );

	/* Gather before values */
	elapsed_us = PAPI_get_real_usec(  );
	elapsed_cyc = PAPI_get_real_cyc(  );
	elapsed_virt_us = PAPI_get_virt_usec(  );
	elapsed_virt_cyc = PAPI_get_virt_cyc(  );

	/* Wait for the SIGSTOP. */
	if ( cmpinfo->attach_must_ptrace ) {
	   if ( ptrace( PTRACE_CONT, pid, NULL, NULL ) == -1 ) {
	      perror( "ptrace(PTRACE_CONT)" );
	      return 1;
	   }
	   if ( waitpid( pid, &status, 0 ) == -1 ) {
	      perror( "waitpid()" );
	      exit( 1 );
	   }
	   if ( WIFSTOPPED( status ) == 0 ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't return true to WIFSTOPPED", 0 );
	   }
	   if ( WSTOPSIG( status ) != SIGSTOP ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't stop on SIGSTOP", 0 );
	   }

	   if ( ptrace( PTRACE_CONT, pid2, NULL, NULL ) == -1 ) {
	      perror( "ptrace(PTRACE_CONT)" );
	      return 1;
	   }
	   if ( waitpid( pid2, &status, 0 ) == -1 ) {
	      perror( "waitpid()" );
	      exit( 1 );
	   }
	   if ( WIFSTOPPED( status ) == 0 ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't return true to WIFSTOPPED", 0 );
	   }
	   if ( WSTOPSIG( status ) != SIGSTOP ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't stop on SIGSTOP", 0 );
	   }
	}

	/* start measuring in first child */
	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	/* start measuring in second child */
	retval = PAPI_start( EventSet2 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

		/* Start first child and Wait for the SIGSTOP. */
	if ( cmpinfo->attach_must_ptrace ) {
	   if ( ptrace( PTRACE_CONT, pid, NULL, NULL ) == -1 ) {
	      perror( "ptrace(PTRACE_ATTACH)" );
	      return 1;
	   }
	   if ( waitpid( pid, &status, 0 ) == -1 ) {
	      perror( "waitpid()" );
	      exit( 1 );
	   }
	   if ( WIFSTOPPED( status ) == 0 ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't return true to WIFSTOPPED", 0 );
	   }
	   if ( WSTOPSIG( status ) != SIGSTOP ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't stop on SIGSTOP", 0 );
	   }

		/* Start second child and Wait for the SIGSTOP. */
	   if ( ptrace( PTRACE_CONT, pid2, NULL, NULL ) == -1 ) {
	       perror( "ptrace(PTRACE_ATTACH)" );
	       return 1;
	   }
	   if ( waitpid( pid2, &status, 0 ) == -1 ) {
	      perror( "waitpid()" );
	      exit( 1 );
	   }
	   if ( WIFSTOPPED( status ) == 0 ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't return true to WIFSTOPPED", 0 );
	   }
	   if ( WSTOPSIG( status ) != SIGSTOP ) {
	      test_fail( __FILE__, __LINE__,
			"Child process didn't stop on SIGSTOP", 0 );
	   }
	}

	elapsed_virt_us = PAPI_get_virt_usec(  ) - elapsed_virt_us;
	elapsed_virt_cyc = PAPI_get_virt_cyc(  ) - elapsed_virt_cyc;
	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	/* stop measuring and read first child */
	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval != PAPI_OK ) {
	   printf( "Warning: PAPI_stop returned error %d, probably ok.\n",
				retval );
	}

	/* stop measuring and read second child */
	retval = PAPI_stop( EventSet2, values[1] );
	if ( retval != PAPI_OK ) {
	   printf( "Warning: PAPI_stop returned error %d, probably ok.\n",
				retval );
	}

	/* close down the measurements */
	remove_test_events( &EventSet1, mask1 );
	remove_test_events( &EventSet2, mask2 );

	/* restart events so they can end */
	if ( cmpinfo->attach_must_ptrace ) {
	   if ( ptrace( PTRACE_CONT, pid, NULL, NULL ) == -1 ) {
	      perror( "ptrace(PTRACE_CONT)" );
	      return 1;
	   }
	   if ( ptrace( PTRACE_CONT, pid2, NULL, NULL ) == -1 ) {
	      perror( "ptrace(PTRACE_CONT)" );
	      return 1;
	   }
	}

	if ( waitpid( pid, &status, 0 ) == -1 ) {
	   perror( "waitpid()" );
	   exit( 1 );
	}
	if ( WIFEXITED( status ) == 0 ) {
	   test_fail( __FILE__, __LINE__,
		     "Child process didn't return true to WIFEXITED", 0 );
	}

	if ( waitpid( pid2, &status, 0 ) == -1 ) {
	   perror( "waitpid()" );
	   exit( 1 );
	}
	if ( WIFEXITED( status ) == 0 ) {
		test_fail( __FILE__, __LINE__,
			  "Child process didn't return true to WIFEXITED", 0 );
	}

	/* This code isn't necessary as we know the child has exited, */
	/* it *may* return an error if the component so chooses. You  */
        /* should use read() instead. */

	if (!TESTS_QUIET) {
		printf( "Test case: multiple 3rd party attach start, stop.\n" );
		printf( "-----------------------------------------------\n" );
		tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
		printf( "Default domain is: %d (%s)\n", tmp,
			stringify_all_domains( tmp ) );
		tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
		printf( "Default granularity is: %d (%s)\n", tmp,
			stringify_granularity( tmp ) );
		printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
		printf( "-------------------------------------------------------------------------\n" );

		sprintf( add_event_str, "(PID %jd) %-12s : \t", ( intmax_t ) pid,
				 event_name );
		printf( TAB1, add_event_str, values[0][1] );
		sprintf( add_event_str, "(PID %jd) PAPI_TOT_CYC : \t",
			 ( intmax_t ) pid );
		printf( TAB1, add_event_str, values[0][0] );
		sprintf( add_event_str, "(PID %jd) %-12s : \t", ( intmax_t ) pid2,
			 event_name );
		printf( TAB1, add_event_str,values[1][1] );
		sprintf( add_event_str, "(PID %jd) PAPI_TOT_CYC : \t",
			 ( intmax_t ) pid2 );
		printf( TAB1, add_event_str, values[1][0] );
		printf( TAB1, "Real usec    : \t", elapsed_us );
		printf( TAB1, "Real cycles  : \t", elapsed_cyc );
		printf( TAB1, "Virt usec    : \t", elapsed_virt_us );
		printf( TAB1, "Virt cycles  : \t", elapsed_virt_cyc );

		printf( "-------------------------------------------------------------------------\n" );

		printf("Verification: pid %d results should be %dx pid %d\n",
			pid2,MULTIPLIER,pid );
	}

	/* FLOPS ratio */
	ratio1=(double)values[1][0]/(double)values[0][0];

	/* CYCLES ratio */
	ratio2=(double)values[1][1]/(double)values[0][1];

	if (!TESTS_QUIET) {
		printf("\tFLOPS ratio %lld/%lld = %lf\n",
			values[1][0],values[0][0],ratio1);
	}

	double ratio1_high,ratio1_low,ratio2_high,ratio2_low;

	ratio1_high=(double)MULTIPLIER *1.10;
	ratio1_low=(double)MULTIPLIER * 0.90;

	if ((ratio1 > ratio1_high ) || (ratio1 < ratio1_low)) {
	  printf("Ratio out of range, should be ~%lf not %lf\n",
		(double)MULTIPLIER, ratio1);
	  test_fail( __FILE__, __LINE__,
		    "Error: Counter ratio not two", 0 );
	}

	if (!TESTS_QUIET) {
		printf("\tCycles ratio %lld/%lld = %lf\n",
			values[1][1],values[0][1],ratio2);
	}

	ratio2_high=(double)MULTIPLIER *1.20;
	ratio2_low=(double)MULTIPLIER * 0.80;

	if ((ratio2 > ratio2_high ) || (ratio2 < ratio2_low )) {
	  printf("Ratio out of range, should be ~%lf, not %lf\n",
		(double)MULTIPLIER, ratio2);
	  test_fail( __FILE__, __LINE__,
		    "Known issue: Counter ratio not two", 0 );
	}

	test_pass( __FILE__ );

	return 0;
}
