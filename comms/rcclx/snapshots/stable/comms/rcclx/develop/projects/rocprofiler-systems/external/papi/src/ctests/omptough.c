#include <pthread.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <stdio.h>
#include <omp.h>

#include "papi.h"
#include "papi_test.h"

#define NITER (100000)

int
main( int argc, char *argv[] )
{
	int i;
	int ret;
	int nthreads;
	int *evtset;
	int *ctrcode;

	nthreads = omp_get_max_threads(  );
	evtset = ( int * ) malloc( sizeof ( int ) * nthreads );
	ctrcode = ( int * ) malloc( sizeof ( int ) * nthreads );

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	ret = PAPI_library_init( PAPI_VER_CURRENT );
	if ( ret != PAPI_VER_CURRENT && ret > 0 ) {
		fprintf( stderr, "PAPI library version mismatch '%s'\n",
				 PAPI_strerror( ret ) );
		exit( 1 );
	}

	if ( ret < 0 ) {
		fprintf( stderr, "PAPI initialization error '%s'\n",
				 PAPI_strerror( ret ) );
		exit( 1 );
	}

	if ( ( ret =
		   PAPI_thread_init( ( unsigned long ( * )( void ) ) pthread_self ) ) !=
		 PAPI_OK ) {
		fprintf( stderr, "PAPI thread initialization error '%s'\n",
				 PAPI_strerror( ret ) );
		exit( 1 );
	}

	for ( i = 0; i < nthreads; i++ ) {
		evtset[i] = PAPI_NULL;

		if ( ( ret = PAPI_event_name_to_code( "PAPI_TOT_INS", &ctrcode[i] ) )
			 != PAPI_OK ) {
			fprintf( stderr, "PAPI evt-name-to-code error '%s'\n",
					 PAPI_strerror( ret ) );
		}

	}

	for ( i = 0; i < NITER; i++ ) {
#pragma omp parallel
		{
			int tid;
			int pid;
			tid = omp_get_thread_num(  );

			pid = pthread_self(  );

			if ( ( ret = PAPI_register_thread(  ) ) != PAPI_OK ) {
				if ( !TESTS_QUIET ) {
					fprintf( stderr,
							 "[%5d] Error in register thread (tid=%d pid=%d) '%s'\n",
							 i, tid, pid, PAPI_strerror( ret ) );
					test_fail( __FILE__, __LINE__, "omptough", 1 );
				}
			}

			evtset[tid] = PAPI_NULL;
			if ( ( ret = PAPI_create_eventset( &( evtset[tid] ) ) ) != PAPI_OK ) {
				if ( !TESTS_QUIET ) {
					fprintf( stderr,
							 "[%5d] Error creating eventset (tid=%d pid=%d) '%s'\n",
							 i, tid, pid, PAPI_strerror( ret ) );
					test_fail( __FILE__, __LINE__, "omptough", 1 );
				}
			}


			if ( ( ret =
				   PAPI_destroy_eventset( &( evtset[tid] ) ) ) != PAPI_OK ) {
				if ( !TESTS_QUIET ) {
					fprintf( stderr,
							 "[%5d] Error destroying eventset (tid=%d pid=%d) '%s'\n",
							 i, tid, pid, PAPI_strerror( ret ) );
					evtset[tid] = PAPI_NULL;
					test_fail( __FILE__, __LINE__, "omptough", 1 );
				}
			}

			if ( ( ret = PAPI_unregister_thread(  ) ) != PAPI_OK ) {
				if ( !TESTS_QUIET ) {
					fprintf( stderr,
							 "[%5d] Error in unregister thread (tid=%d pid=%d) ret='%s'\n",
							 i, tid, pid, PAPI_strerror( ret ) );
					test_fail( __FILE__, __LINE__, "omptough", 1 );
				}
			}
		}
	}
	test_pass( __FILE__ );

	return 0;
}
