/* From Dave McNamara at PSRV. Thanks! */

/* If an event is countable but you've exhausted the counter resources
and you try to add an event, it seems subsequent PAPI_start and/or
PAPI_stop will causes a Seg. Violation.

   I got around this by calling PAPI to get the # of countable events,
then making sure that I didn't try to add more than these number of
events. I still have a problem if someone adds Level 2 cache misses
and then adds FLOPS 'cause I didn't count FLOPS as actually requiring
2 counters. */

#include "papi_test.h"
#include <mpi.h>
#include <stdio.h>
#include <math.h>

extern int TESTS_QUIET;				   /* Declared in test_utils.c */
char *netevents[] =
	{ "LO_RX_PACKETS", "LO_TX_PACKETS", "ETH0_RX_PACKETS", "ETH0_TX_PACKETS" };

double
f( double a )
{
	return ( 4.0 / ( 1.0 + a * a ) );
}

int
main( int argc, char **argv )
{
	int EventSet = PAPI_NULL, EventSet1 = PAPI_NULL;
	int evtcode;
	int retval, i, ins = 0;
	long long g1[2], g2[2];

	int done = 0, n, myid, numprocs;
	double PI25DT = 3.141592653589793238462643;
	double mypi, pi, h, sum, x;
	double startwtime = 0.0, endwtime;
	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	if ( ( retval = PAPI_create_eventset( &EventSet1 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );

	PAPI_event_name_to_code( netevents[2], &evtcode );
	if ( ( retval = PAPI_query_event( evtcode ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_aquery_event", retval );
	}
	if ( ( retval = PAPI_add_event( EventSet, evtcode ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	PAPI_event_name_to_code( netevents[3], &evtcode );
	if ( ( retval = PAPI_query_event( evtcode ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_aquery_event", retval );
	}
	if ( ( retval = PAPI_add_event( EventSet, evtcode ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	if ( ( retval = PAPI_query_event( PAPI_FP_INS ) ) != PAPI_OK ) {
		if ( ( retval = PAPI_query_event( PAPI_FP_OPS ) ) == PAPI_OK ) {
			ins = 2;
			if ( ( retval =
				   PAPI_add_event( EventSet1, PAPI_FP_OPS ) ) != PAPI_OK ) {
				if ( retval != PAPI_ECNFLCT )
					test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
			}
		}
	} else {
		ins = 1;
		if ( ( retval = PAPI_add_event( EventSet1, PAPI_FP_INS ) ) != PAPI_OK ) {
			if ( retval != PAPI_ECNFLCT )
				test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
		}
	}

	if ( ( retval = PAPI_add_event( EventSet1, PAPI_TOT_CYC ) ) != PAPI_OK ) {
		if ( retval != PAPI_ECNFLCT )
			test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	MPI_Init( &argc, &argv );

	MPI_Comm_size( MPI_COMM_WORLD, &numprocs );
	MPI_Comm_rank( MPI_COMM_WORLD, &myid );
	MPI_Get_processor_name( processor_name, &namelen );

	fprintf( stdout, "Process %d of %d on %s\n",
			 myid, numprocs, processor_name );
	fflush( stdout );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	if ( ( retval = PAPI_start( EventSet1 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	n = 0;
	while ( !done ) {
		if ( myid == 0 ) {
			if ( n == 0 )
				n = 1000000;
			else
				n = 0;

			startwtime = MPI_Wtime(  );
		}
		MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD );
		if ( n == 0 )
			done = 1;
		else {
			h = 1.0 / ( double ) n;
			sum = 0.0;
			/* A slightly better approach starts from large i and works back */
			for ( i = myid + 1; i <= n; i += numprocs ) {
				x = h * ( ( double ) i - 0.5 );
				sum += f( x );
			}
			mypi = h * sum;

			MPI_Reduce( &mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );

			if ( myid == 0 ) {
				printf( "pi is approximately %.16f, Error is %.16f\n",
						pi, fabs( pi - PI25DT ) );
				endwtime = MPI_Wtime(  );
				printf( "wall clock time = %f\n", endwtime - startwtime );
				fflush( stdout );
			}
		}
	}

	if ( ( retval = PAPI_stop( EventSet1, g1 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	if ( ( retval = PAPI_stop( EventSet, g2 ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	MPI_Finalize(  );


	printf( "ETH0_RX_BYTES: %lld   ETH0_TX_BYTES: %lld\n", g2[0], g2[1] );
	if ( ins == 0 ) {
		printf( "PAPI_TOT_CYC : %lld\n", g1[0] );
	} else if ( ins == 1 ) {
		printf( "PAPI_FP_INS  : %lld   PAPI_TOT_CYC : %lld\n", g1[0], g1[1] );
	} else if ( ins == 2 ) {
		printf( "PAPI_FP_OPS  : %lld   PAPI_TOT_CYC : %lld\n", g1[0], g1[1] );
	}
	test_pass( __FILE__, NULL, 0 );
	return 0;
}
