#include <stdio.h>
#include <sys/time.h>

/* Repeat doing some busy-work floating point */
/* Until at least len seconds have passed */

double
do_cycles( int minimum_time )
{
	struct timeval start, now;
	double x, sum;

	gettimeofday( &start, NULL );

	for ( ;; ) {
		sum = 1.0;
		for ( x = 1.0; x < 250000.0; x += 1.0 ) {
			sum += x;
		}
		if ( sum < 0.0 ) {
			printf( "==>>  SUM IS NEGATIVE !!  <<==\n" );
		}

		gettimeofday( &now, NULL );
		if ( now.tv_sec >= start.tv_sec + minimum_time ) {
			break;
		}
	}
	return sum;
}

