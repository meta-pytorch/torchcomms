#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "papi.h"

#include "clockcore.h"

#define NUM_ITERS  1000000

static char *func_name[] = {
	"PAPI_get_real_cyc",
	"PAPI_get_real_usec",
	"PAPI_get_virt_cyc",
	"PAPI_get_virt_usec"
};
static int CLOCK_ERROR = 0;

static int
clock_res_check( int flag, int quiet )
{
	if ( CLOCK_ERROR ) {
		return -1;
	}

	long long *elapsed_cyc, total_cyc = 0, uniq_cyc = 0, diff_cyc = 0;
	int i;
	double min, max, average, std, tmp;

	elapsed_cyc = ( long long * ) calloc( NUM_ITERS, sizeof ( long long ) );

	/* Real */
	switch ( flag ) {
	case 0:
		for ( i = 0; i < NUM_ITERS; i++ )
			elapsed_cyc[i] = ( long long ) PAPI_get_real_cyc(  );
		break;
	case 1:
		for ( i = 0; i < NUM_ITERS; i++ )
			elapsed_cyc[i] = ( long long ) PAPI_get_real_usec(  );
		break;
	case 2:
		for ( i = 0; i < NUM_ITERS; i++ )
			elapsed_cyc[i] = ( long long ) PAPI_get_virt_cyc(  );
		break;
	case 3:
		for ( i = 0; i < NUM_ITERS; i++ )
			elapsed_cyc[i] = ( long long ) PAPI_get_virt_usec(  );
		break;
	default:
      free(elapsed_cyc);
		return -1;

	}

	min = max = ( double ) ( elapsed_cyc[1] - elapsed_cyc[0] );

	for ( i = 1; i < NUM_ITERS; i++ ) {
		if ( elapsed_cyc[i] - elapsed_cyc[i - 1] < 0 ) {
			CLOCK_ERROR = 1;
			fprintf(stderr,"Error! Negative elapsed time\n");
			free( elapsed_cyc );
			return -1;
		}

		diff_cyc = elapsed_cyc[i] - elapsed_cyc[i - 1];
		if ( min > diff_cyc )
			min = ( double ) diff_cyc;
		if ( max < diff_cyc )
			max = ( double ) diff_cyc;
		if ( diff_cyc != 0 )
			uniq_cyc++;
		total_cyc += diff_cyc;
	}

	average = ( double ) total_cyc / ( NUM_ITERS - 1 );
	std = 0;

	for ( i = 1; i < NUM_ITERS; i++ ) {
		tmp = ( double ) ( elapsed_cyc[i] - elapsed_cyc[i - 1] );
		tmp = tmp - average;
		std += tmp * tmp;
	}

	if ( !quiet ) {
		std = sqrt( std / ( NUM_ITERS - 2 ) );
		printf( "%s: min %.3lf  max %.3lf \n", func_name[flag], min, max );
		printf( "                   average %.3lf std %.3lf\n", average, std );

		if ( uniq_cyc == NUM_ITERS - 1 ) {
			printf( "%s : %7.3f   <%7.3f\n", func_name[flag],
					( double ) total_cyc / ( double ) ( NUM_ITERS ),
					( double ) total_cyc / ( double ) uniq_cyc );
		} else if ( uniq_cyc ) {
			printf( "%s : %7.3f    %7.3f\n", func_name[flag],
					( double ) total_cyc / ( double ) ( NUM_ITERS ),
					( double ) total_cyc / ( double ) uniq_cyc );
		} else {
			printf( "%s : %7.3f   >%7.3f\n", func_name[flag],
					( double ) total_cyc / ( double ) ( NUM_ITERS ),
					( double ) total_cyc );
		}
	}

	free( elapsed_cyc );

	return PAPI_OK;
}

int
clockcore( int quiet )
{
	/* check PAPI_get_real_cyc */
	clock_res_check( 0, quiet );
	/* check PAPI_get_real_usec */
	clock_res_check( 1, quiet );

	/* check PAPI_get_virt_cyc */
	/* Virtual */
	if ( PAPI_get_virt_cyc(  ) != -1 ) {
		clock_res_check( 2, quiet );
	} else {
		return CLOCKCORE_VIRT_CYC_FAIL;
	}

	/* check PAPI_get_virt_usec */
	if ( PAPI_get_virt_usec(  ) != -1 ) {
		clock_res_check( 3, quiet );
	} else {
		return CLOCKCORE_VIRT_USEC_FAIL;
	}

	return PAPI_OK;
}
