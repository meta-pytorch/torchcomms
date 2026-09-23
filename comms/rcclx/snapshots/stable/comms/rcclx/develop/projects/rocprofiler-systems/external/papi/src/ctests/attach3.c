/* This file performs the following test: start, stop and timer functionality for
   attached processes.

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
#include <unistd.h>
#include <string.h>

#include <limits.h>
#include <sys/ptrace.h>
#include <sys/wait.h>

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"


#ifdef _AIX
#define _LINUX_SOURCE_COMPAT
#endif

#if defined(__FreeBSD__)
# define PTRACE_ATTACH PT_ATTACH
# define PTRACE_TRACEME PT_TRACE_ME
#endif

static int
wait_for_attach_and_loop( void )
{
  char *path;
  char newpath[PATH_MAX];
  path = getenv("PATH");

  sprintf(newpath, "PATH=./:%s", (path)?path:"\0" );
  putenv(newpath);

  if (ptrace(PTRACE_TRACEME, 0, 0, 0) == 0) {
    execlp("attach_target","attach_target","100000000",NULL);
    perror("execl(attach_target) failed");
  }
  perror("PTRACE_TRACEME");
  return ( 1 );
}

int
main( int argc, char **argv )
{
	int status, retval, tmp;
	int EventSet1 = PAPI_NULL;
	long long **values;
	long long elapsed_us, elapsed_cyc, elapsed_virt_us, elapsed_virt_cyc;
	char event_name[PAPI_MAX_STR_LEN];;
	const PAPI_hw_info_t *hw_info;
	const PAPI_component_info_t *cmpinfo;
	pid_t pid;
	int quiet;

	/* Fork before doing anything with the PMU */

	setbuf(stdout,NULL);
	pid = fork(  );
	if ( pid < 0 )
		test_fail( __FILE__, __LINE__, "fork()", PAPI_ESYS );
	if ( pid == 0 )
		exit( wait_for_attach_and_loop(  ) );

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );


	/* Master only process below here */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( cmpinfo = PAPI_get_component_info( 0 ) ) == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_component_info", 0 );

	if ( cmpinfo->attach == 0 )
		test_skip( __FILE__, __LINE__, "Platform does not support attaching",
				   0 );

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 0 );

	/* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
	   PAPI_TOT_INS, depending on the availability of the event on the
	   platform */
	retval = PAPI_create_eventset(&EventSet1);
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_attach", retval );
	}

	/* Force addition of component */

	retval = PAPI_assign_eventset_component( EventSet1, 0 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
				   retval );

	/* The following call causes this test to fail for perf_events */

	retval = PAPI_attach( EventSet1, ( unsigned long ) pid );
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Cannot attach: %s\n",PAPI_strerror(retval));
		test_skip( __FILE__, __LINE__, "PAPI_attach", retval );
	}


	retval = PAPI_add_event(EventSet1, PAPI_TOT_CYC);
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Could not add PAPI_TOT_CYC\n");
		test_skip( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	strcpy(event_name,"PAPI_FP_INS");
	retval = PAPI_add_named_event(EventSet1, event_name);
	if ( retval == PAPI_ENOEVNT ) {
		strcpy(event_name,"PAPI_TOT_INS");
		retval = PAPI_add_named_event(EventSet1, event_name);
	}

	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	values = allocate_test_space( 1, 2);

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	elapsed_virt_us = PAPI_get_virt_usec(  );

	elapsed_virt_cyc = PAPI_get_virt_cyc(  );

	if (!quiet) printf("must_ptrace is %d\n",cmpinfo->attach_must_ptrace);
	pid_t  child = wait( &status );
	if (!quiet) printf( "Debugger exited wait() with %d\n",child );
	  if (WIFSTOPPED( status ))
	    {
	      if (!quiet) printf( "Child has stopped due to signal %d (%s)\n",
		      WSTOPSIG( status ), strsignal(WSTOPSIG( status )) );
	    }
	  if (WIFSIGNALED( status ))
	    {
	      if (!quiet) printf( "Child %ld received signal %d (%s)\n",
		      (long)child,
		      WTERMSIG(status) , strsignal(WTERMSIG( status )) );
	    }
	if (!quiet) printf("After %d\n",retval);

	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	if (!quiet) printf("Continuing\n");
#if defined(__FreeBSD__)
	if ( ptrace( PT_CONTINUE, pid, (vptr_t) 1, 0 ) == -1 ) {
#else
	if ( ptrace( PTRACE_CONT, pid, NULL, NULL ) == -1 ) {
#endif
	  perror( "ptrace(PTRACE_CONT)" );
	  return 1;
	}


	do {
	  child = wait( &status );
	  if (!quiet) printf( "Debugger exited wait() with %d\n", child);
	  if (WIFSTOPPED( status ))
	    {
	      if (!quiet) printf( "Child has stopped due to signal %d (%s)\n",
		      WSTOPSIG( status ), strsignal(WSTOPSIG( status )) );
	    }
	  if (WIFSIGNALED( status ))
	    {
	      if (!quiet) printf( "Child %ld received signal %d (%s)\n",
		      (long)child,
		      WTERMSIG(status) , strsignal(WTERMSIG( status )) );
	    }
	} while (!WIFEXITED( status ));

	if (!quiet) printf("Child exited with value %d\n",WEXITSTATUS(status));
	if (WEXITSTATUS(status) != 0) {
	  test_fail( __FILE__, __LINE__, "Exit status of child to attach to", PAPI_EMISC);
	}
	retval = PAPI_stop( EventSet1, values[0] );
	if ( retval != PAPI_OK )
	  test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	elapsed_virt_us = PAPI_get_virt_usec(  ) - elapsed_virt_us;

	elapsed_virt_cyc = PAPI_get_virt_cyc(  ) - elapsed_virt_cyc;

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	retval = PAPI_cleanup_eventset(EventSet1);
	if (retval != PAPI_OK)
	  test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset(&EventSet1);
	if (retval != PAPI_OK)
	  test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	if (!quiet) {
	printf( "Test case: 3rd party attach start, stop.\n" );
	printf( "-----------------------------------------------\n" );
	tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
	printf( "Default domain is: %d (%s)\n", tmp, stringify_all_domains( tmp ) );
	tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
	printf( "Default granularity is: %d (%s)\n", tmp,
			stringify_granularity( tmp ) );
	printf( "Using %d iterations of c += a*b\n", NUM_FLOPS );
	printf( "-------------------------------------------------------------------------\n" );

	printf( "Test type    : \t           1\n" );

	printf( TAB1, "PAPI_TOT_CYC : \t", ( values[0] )[0] );
	printf( "%s : \t %12lld\n", event_name, ( values[0] )[1] );
	printf( TAB1, "Real usec    : \t", elapsed_us );
	printf( TAB1, "Real cycles  : \t", elapsed_cyc );
	printf( TAB1, "Virt usec    : \t", elapsed_virt_us );
	printf( TAB1, "Virt cycles  : \t", elapsed_virt_cyc );

	printf( "-------------------------------------------------------------------------\n" );

	printf( "Verification: none\n" );
	}
	test_pass( __FILE__ );

	return 0;

}
