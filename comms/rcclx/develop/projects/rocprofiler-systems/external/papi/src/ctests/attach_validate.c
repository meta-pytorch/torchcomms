/* This test attempts to attach to a child and makes sure we are */
/* getting the expected results for the child only, not local.   */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <sys/ptrace.h>
#include <sys/wait.h>

#include "papi.h"
#include "papi_test.h"

#include "testcode.h"

#ifdef _AIX
#define _LINUX_SOURCE_COMPAT
#endif

#if defined(__FreeBSD__)
# define PTRACE_ATTACH PT_ATTACH
# define PTRACE_TRACEME PT_TRACE_ME
#endif

static int wait_for_attach_and_loop( int quiet ) {

	int i,result;

	if (ptrace(PTRACE_TRACEME, 0, 0, 0) == 0) {
		raise(SIGSTOP);

		if (!quiet) printf("Child running 50 million instructions\n");

		/* Run 50 million instructions */
		for(i=0;i<50;i++) {
			result=instructions_million();
		}
	}
	perror("PTRACE_TRACEME");
	(void)result;

	return 0;
}

int main( int argc, char **argv ) {

	int status, retval, tmp;
	int EventSet1 = PAPI_NULL;
	long long values[1];
	const PAPI_hw_info_t *hw_info;
	const PAPI_component_info_t *cmpinfo;
	pid_t pid;
	int quiet;
	int i,result;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );

	/* Fork before doing anything with the PMU */
	setbuf(stdout,NULL);
	pid = fork(  );
	if ( pid < 0 ) {
		test_fail( __FILE__, __LINE__, "fork()", PAPI_ESYS );
	}

	/* If child */
	if ( pid == 0 ) {
		exit(wait_for_attach_and_loop(quiet) );
	}


	/* Parent process below here */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	if ( ( cmpinfo = PAPI_get_component_info( 0 ) ) == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_component_info", 0 );
	}

	if ( cmpinfo->attach == 0 ) {
		test_skip( __FILE__, __LINE__, "Platform does not support attaching",
				   0 );
	}

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 0 );
	}

	/* Create Eventset */
	retval = PAPI_create_eventset(&EventSet1);
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	retval = PAPI_assign_eventset_component( EventSet1, 0 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_assign_eventset_component",
		retval );
	}

	/* Attach to our child */
	retval = PAPI_attach( EventSet1, ( unsigned long ) pid );
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Cannot attach: %s\n",PAPI_strerror(retval));
		test_skip( __FILE__, __LINE__, "PAPI_attach", retval );
	}

	/* Add instructions event */
	retval = PAPI_add_event(EventSet1, PAPI_TOT_INS);
	if ( retval != PAPI_OK ) {
		if (!quiet) printf("Problem adding PAPI_TOT_INS\n");
		test_skip( __FILE__, __LINE__, "PAPI_add_event", retval );
	}


	if (!quiet) {
		printf("must_ptrace is %d\n",cmpinfo->attach_must_ptrace);
	}

	/* Wait for child to stop for debugging */
	pid_t  child = wait( &status );

	if (!quiet) printf( "Debugger exited wait() with %d\n",child );
	if (WIFSTOPPED( status )) {
		if (!quiet) {
			printf( "Child has stopped due to signal %d (%s)\n",
				WSTOPSIG( status ),
				strsignal(WSTOPSIG( status )) );
		}
	}

	if (WIFSIGNALED( status )) {
	      if (!quiet) {
			printf( "Child %ld received signal %d (%s)\n",
				(long)child,
				WTERMSIG(status),
				strsignal(WTERMSIG( status )) );
		}
	}
	if (!quiet) {
		printf("After %d\n",retval);
	}

	/* Start eventset */
	retval = PAPI_start( EventSet1 );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	if (!quiet) {
		printf("Continuing\n");
	}

#if defined(__FreeBSD__)
	if ( ptrace( PT_CONTINUE, pid, (vptr_t) 1, 0 ) == -1 ) {
		perror( "ptrace(PTRACE_CONT)" );
		test_fail( __FILE__, __LINE__,
			"Continuing", PAPI_EMISC);
		return 1;
	}
#else
	if ( ptrace( PTRACE_CONT, pid, NULL, NULL ) == -1 ) {
		perror( "ptrace(PTRACE_CONT)" );
		test_fail( __FILE__, __LINE__,
			"Continuing", PAPI_EMISC);
	}
#endif


	/* Run a billion instructions, should not appear in count */

	for(i=0;i<1000;i++) {
		result=instructions_million();
	}

	/* Wait for child to finish */
	do {
		child = wait( &status );
		if (!quiet) {
			printf( "Debugger exited wait() with %d\n", child);
		}
		if (WIFSTOPPED( status )) {
			if (!quiet) {
				printf( "Child has stopped due to signal "
					"%d (%s)\n",
					WSTOPSIG( status ),
					strsignal(WSTOPSIG( status )) );
			}
		}
		if (WIFSIGNALED( status )) {
			if (!quiet) {
				printf( "Child %ld received signal "
					"%d (%s)\n",
					(long)child,
					WTERMSIG(status),
					strsignal(WTERMSIG( status )) );
			}
		}
	} while (!WIFEXITED( status ));

	if (!quiet) {
		printf("Child exited with value %d\n",WEXITSTATUS(status));
	}

	if (WEXITSTATUS(status) != 0) {
		test_fail( __FILE__, __LINE__,
			"Exit status of child to attach to", PAPI_EMISC);
	}

	/* Stop counts */
	retval = PAPI_stop( EventSet1, &values[0] );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	}

	retval = PAPI_cleanup_eventset(EventSet1);
	if (retval != PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );
	}

	retval = PAPI_destroy_eventset(&EventSet1);
	if (retval != PAPI_OK) {
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );
	}

	if (!quiet) {
	printf( "Test case: attach validation.\n" );
	printf( "-----------------------------------------------\n" );
	tmp = PAPI_get_opt( PAPI_DEFDOM, NULL );
	printf( "Default domain is: %d (%s)\n", tmp, stringify_all_domains( tmp ) );
	tmp = PAPI_get_opt( PAPI_DEFGRN, NULL );
	printf( "Default granularity is: %d (%s)\n", tmp,
			stringify_granularity( tmp ) );
	printf( "Using 50 million instructions\n");
	printf( "-------------------------------------------------------------------------\n" );

	printf( "Test type    : \t           1\n" );

	printf( TAB1, "PAPI_TOT_INS : \t", ( values[0] ) );

	printf( "-------------------------------------------------------------------------\n" );

	}

	if (values[0]<100) {
		test_fail( __FILE__, __LINE__, "wrong result", PAPI_EMISC );
	}

	if (values[0]>60000000) {
		test_fail( __FILE__, __LINE__, "wrong result", PAPI_EMISC );
	}

	(void)result;

	test_pass( __FILE__ );

	return 0;

}
