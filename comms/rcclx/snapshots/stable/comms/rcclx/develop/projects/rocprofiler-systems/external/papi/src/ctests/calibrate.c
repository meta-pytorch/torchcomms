/*
   Calibrate.c
        A program to perform one or all of three tests to count flops.
        Test 1. Inner Product:                          2*n operations
                for i = 1:n; a = a + x(i)*y(i); end
        Test 2. Matrix Vector Product:          2*n^2 operations
                for i = 1:n; for j = 1:n; x(i) = x(i) + a(i,j)*y(j); end; end;
        Test 3. Matrix Matrix Multiply:         2*n^3 operations
                for i = 1:n; for j = 1:n; for k = 1:n; c(i,j) = c(i,j) + a(i,k)*b(k,j); end; end; end;

  Supply a command line argument of 1, 2, or 3 to perform each test, or
  no argument to perform all three.

  Each test initializes PAPI and presents a header with processor information.
  Then it performs 500 iterations, printing result lines containing:
  n, measured counts, theoretical counts, (measured - theory), % error
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "papi_test.h"

#define INDEX1 100
#define INDEX5 500

#define MAX_WARN 10
#define MAX_ERROR 80
#define MAX_DIFF  14

/*
        Extract and display hardware information for this processor.
        (Re)Initialize PAPI_flops() and begin counting floating ops.
*/
static void
headerlines( const char *title, int quiet )
{

	if ( !quiet ) {
		printf( "\n%s:\n%8s %12s %12s %8s %8s\n", title, "i", "papi", "theory",
				"diff", "%error" );
		printf( "-------------------------------------------------------------------------\n" );
	}
}

/*
  Read PAPI_flops.
  Format and display results.
  Compute error without using floating ops.
*/
#if defined(mips)
#define FMA 1
#elif (defined(sparc) && defined(sun))
#define FMA 1
#else
#define FMA 0
#endif

static void
resultline( int i, int j, int EventSet, int fail, int quiet )
{
	float ferror = 0;
	long long flpins = 0;
	long long papi, theory;
	int diff, retval;
	char err_str[PAPI_MAX_STR_LEN];

	retval = PAPI_stop( EventSet, &flpins );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	i++;					 /* convert to 1s base  */
	theory = 2;
	while ( j-- )
		theory *= i;		 /* theoretical ops   */
	papi = flpins << FMA;

	diff = ( int ) ( papi - theory );

	ferror = ( ( float ) abs( diff ) ) / ( ( float ) theory ) * 100;

	if (!quiet) {
		printf( "%8d %12lld %12lld %8d %10.4f\n", i, papi, theory, diff, ferror );
	}

	if ( ferror > MAX_WARN && abs( diff ) > MAX_DIFF && i > 20 ) {
		sprintf( err_str, "Calibrate: difference exceeds %d percent", MAX_WARN );
		test_warn( __FILE__, __LINE__, err_str, 0 );
	}
	if (fail) {
		if ( ferror > MAX_ERROR && abs( diff ) > MAX_DIFF && i > 20 ) {
			sprintf( err_str, "Calibrate: error exceeds %d percent", MAX_ERROR );
			test_fail( __FILE__, __LINE__, err_str, PAPI_EMISC );
		}
	}
}


static void
print_help( char **argv )
{
	printf( "Usage: %s [-ivmdh] [-e event]\n", argv[0] );
	printf( "Options:\n\n" );
	printf( "\t-i            Inner Product test.\n" );
	printf( "\t-v            Matrix-Vector multiply test.\n" );
	printf( "\t-m            Matrix-Matrix multiply test.\n" );
	printf( "\t-d            Double precision data. Default is float.\n" );
	printf( "\t-e event      Use <event> as PAPI event instead of PAPI_FP_OPS\n" );
	printf( "\t-f            Suppress failures\n" );
	printf( "\t-h            Print this help message\n" );
	printf( "\n" );
	printf( "This test measures floating point operations for the specified test.\n" );
	printf( "Operations can be performed in single or double precision.\n" );
	printf( "Default operation is all three tests in single precision.\n" );
}

static float
inner_single( int n, float *x, float *y )
{
	float aa = 0.0;
	int i;

	for ( i = 0; i <= n; i++ )
		aa = aa + x[i] * y[i];
	return ( aa );
}

static double
inner_double( int n, double *x, double *y )
{
	double aa = 0.0;
	int i;

	for ( i = 0; i <= n; i++ )
		aa = aa + x[i] * y[i];
	return ( aa );
}

static void
vector_single( int n, float *a, float *x, float *y )
{
	int i, j;

	for ( i = 0; i <= n; i++ )
		for ( j = 0; j <= n; j++ )
			y[i] = y[i] + a[i * n + j] * x[i];
}

static void
vector_double( int n, double *a, double *x, double *y )
{
	int i, j;

	for ( i = 0; i <= n; i++ )
		for ( j = 0; j <= n; j++ )
			y[i] = y[i] + a[i * n + j] * x[i];
}

static void
matrix_single( int n, float *c, float *a, float *b )
{
	int i, j, k;

	for ( i = 0; i <= n; i++ )
		for ( j = 0; j <= n; j++ )
			for ( k = 0; k <= n; k++ )
				c[i * n + j] = c[i * n + j] + a[i * n + k] * b[k * n + j];
}

static void
matrix_double( int n, double *c, double *a, double *b )
{
	int i, j, k;

	for ( i = 0; i <= n; i++ )
		for ( j = 0; j <= n; j++ )
			for ( k = 0; k <= n; k++ )
				c[i * n + j] = c[i * n + j] + a[i * n + k] * b[k * n + j];
}

static void
reset_flops( const char *title, int EventSet )
{
	int retval;
	char err_str[PAPI_MAX_STR_LEN];

	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
		sprintf( err_str, "%s: PAPI_start", title );
		test_fail( __FILE__, __LINE__, err_str, retval );
	}
}

int
main( int argc, char *argv[] )
{
	extern void dummy( void * );

	float aa, *a=NULL, *b=NULL, *c=NULL, *x=NULL, *y=NULL;
	double aad, *ad=NULL, *bd=NULL, *cd=NULL, *xd=NULL, *yd=NULL;
	int i, j, n;
	int inner = 0;
	int vector = 0;
	int matrix = 0;
	int double_precision = 0;
	int fail = 1;
	int retval = PAPI_OK;
	char papi_event_str[PAPI_MIN_STR_LEN] = "PAPI_FP_OPS";
	int papi_event;
	int EventSet = PAPI_NULL;
	int quiet;

	/* Parse the input arguments */
	for ( i = 0; i < argc; i++ ) {
		if ( strstr( argv[i], "-i" ) )
			inner = 1;
		else if ( strstr( argv[i], "-f" ) )
			fail = 0;
		else if ( strstr( argv[i], "-v" ) )
			vector = 1;
		else if ( strstr( argv[i], "-m" ) )
			matrix = 1;
		else if ( strstr( argv[i], "-e" ) ) {
			if ( ( argv[i + 1] == NULL ) || ( strlen( argv[i + 1] ) == 0 ) ) {
				print_help( argv );
				exit( 1 );
			}
			strncpy( papi_event_str, argv[i + 1], sizeof ( papi_event_str ) - 1);
			papi_event_str[sizeof ( papi_event_str )-1] = '\0';
			i++;
		} else if ( strstr( argv[i], "-d" ) )
			double_precision = 1;
		else if ( strstr( argv[i], "-h" ) ) {
			print_help( argv );
			exit( 1 );
		}
	}

	/* if no options specified, set all tests to TRUE */
	if ( inner + vector + matrix == 0 )
		inner = vector = matrix = 1;


	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	if ( !quiet ) {
		printf( "Initializing..." );
	}

	/* Initialize PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Translate name */
	retval = PAPI_event_name_to_code( papi_event_str, &papi_event );
	if ( retval != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", retval );
	}

	if ( PAPI_query_event( papi_event ) != PAPI_OK ) {
		test_skip( __FILE__, __LINE__, "PAPI_query_event", PAPI_ENOEVNT );
	}

	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
	}

	if ( ( retval = PAPI_add_event( EventSet, papi_event ) ) != PAPI_OK ) {
		test_fail( __FILE__, __LINE__, "PAPI_add_event", retval );
	}

	if (!quiet) printf( "\n" );

	retval = PAPI_OK;

	/* Inner Product test */
	if ( inner ) {
		/* Allocate the linear arrays */
	   if (double_precision) {
	        xd = malloc( INDEX5 * sizeof(double) );
	        yd = malloc( INDEX5 * sizeof(double) );
		if ( !( xd && yd ) )
			retval = PAPI_ENOMEM;
	   }
	   else {
	        x = malloc( INDEX5 * sizeof(float) );
		y = malloc( INDEX5 * sizeof(float) );
		if ( !( x && y ) )
			retval = PAPI_ENOMEM;
	   }

		if ( retval == PAPI_OK ) {
			headerlines( "Inner Product Test", quiet );

			/* step through the different array sizes */
			for ( n = 0; n < INDEX5; n++ ) {
				if ( n < INDEX1 || ( ( n + 1 ) % 50 ) == 0 ) {

					/* Initialize the needed arrays at this size */
					if ( double_precision ) {
						for ( i = 0; i <= n; i++ ) {
							xd[i] = ( double ) rand(  ) * ( double ) 1.1;
							yd[i] = ( double ) rand(  ) * ( double ) 1.1;
						}
					} else {
						for ( i = 0; i <= n; i++ ) {
							x[i] = ( float ) rand(  ) * ( float ) 1.1;
							y[i] = ( float ) rand(  ) * ( float ) 1.1;
						}
					}

					/* reset PAPI flops count */
					reset_flops( "Inner Product Test", EventSet );

					/* do the multiplication */
					if ( double_precision ) {
						aad = inner_double( n, xd, yd );
						dummy( ( void * ) &aad );
					} else {
						aa = inner_single( n, x, y );
						dummy( ( void * ) &aa );
					}
					resultline( n, 1, EventSet, fail, quiet );
				}
			}
		}
		if (double_precision) {
			free( xd );
			free( yd );
		} else {
			free( x );
			free( y );
		}
	}

	/* Matrix Vector test */
	if ( vector && retval != PAPI_ENOMEM ) {
		/* Allocate the needed arrays */
	  if (double_precision) {
	        ad = malloc( INDEX5 * INDEX5 * sizeof(double) );
	        xd = malloc( INDEX5 * sizeof(double) );
	        yd = malloc( INDEX5 * sizeof(double) );
		if ( !( ad && xd && yd ) )
			retval = PAPI_ENOMEM;
	  } else {
	        a = malloc( INDEX5 * INDEX5 * sizeof(float) );
	        x = malloc( INDEX5 * sizeof(float) );
	        y = malloc( INDEX5 * sizeof(float) );
		if ( !( a && x && y ) )
			retval = PAPI_ENOMEM;
	  }

		if ( retval == PAPI_OK ) {
			headerlines( "Matrix Vector Test", quiet );

			/* step through the different array sizes */
			for ( n = 0; n < INDEX5; n++ ) {
				if ( n < INDEX1 || ( ( n + 1 ) % 50 ) == 0 ) {

					/* Initialize the needed arrays at this size */
					if ( double_precision ) {
						for ( i = 0; i <= n; i++ ) {
							yd[i] = 0.0;
							xd[i] = ( double ) rand(  ) * ( double ) 1.1;
							for ( j = 0; j <= n; j++ )
								ad[i * n + j] =
									( double ) rand(  ) * ( double ) 1.1;
						}
					} else {
						for ( i = 0; i <= n; i++ ) {
							y[i] = 0.0;
							x[i] = ( float ) rand(  ) * ( float ) 1.1;
							for ( j = 0; j <= n; j++ )
								a[i * n + j] =
									( float ) rand(  ) * ( float ) 1.1;
						}
					}

					/* reset PAPI flops count */
					reset_flops( "Matrix Vector Test", EventSet );

					/* compute the resultant vector */
					if ( double_precision ) {
						vector_double( n, ad, xd, yd );
						dummy( ( void * ) yd );
					} else {
						vector_single( n, a, x, y );
						dummy( ( void * ) y );
					}
					resultline( n, 2, EventSet, fail, quiet );
				}
			}
		}
		if (double_precision) {
			free( ad );
			free( xd );
			free( yd );
		} else {
			free( a );
			free( x );
			free( y );
		}
	}

	/* Matrix Multiply test */
	if ( matrix && retval != PAPI_ENOMEM ) {
		/* Allocate the needed arrays */
	  if (double_precision) {
	        ad = malloc( INDEX5 * INDEX5 * sizeof(double) );
	        bd = malloc( INDEX5 * INDEX5 * sizeof(double) );
	        cd = malloc( INDEX5 * INDEX5 * sizeof(double) );
		if ( !( ad && bd && cd ) )
			retval = PAPI_ENOMEM;
	  } else {
	        a = malloc( INDEX5 * INDEX5 * sizeof(float) );
	        b = malloc( INDEX5 * INDEX5 * sizeof(float) );
	        c = malloc( INDEX5 * INDEX5 * sizeof(float) );
		if ( !( a && b && c ) )
			retval = PAPI_ENOMEM;
	  }


		if ( retval == PAPI_OK ) {
			headerlines( "Matrix Multiply Test", quiet );

			/* step through the different array sizes */
			for ( n = 0; n < INDEX5; n++ ) {
				if ( n < INDEX1 || ( ( n + 1 ) % 50 ) == 0 ) {

					/* Initialize the needed arrays at this size */
					if ( double_precision ) {
						for ( i = 0; i <= n * n + n; i++ ) {
							cd[i] = 0.0;
							ad[i] = ( double ) rand(  ) * ( double ) 1.1;
							bd[i] = ( double ) rand(  ) * ( double ) 1.1;
						}
					} else {
						for ( i = 0; i <= n * n + n; i++ ) {
							c[i] = 0.0;
							a[i] = ( float ) rand(  ) * ( float ) 1.1;
							b[i] = ( float ) rand(  ) * ( float ) 1.1;
						}
					}

					/* reset PAPI flops count */
					reset_flops( "Matrix Multiply Test", EventSet );

					/* compute the resultant matrix */
					if ( double_precision ) {
						matrix_double( n, cd, ad, bd );
						dummy( ( void * ) c );
					} else {
						matrix_single( n, c, a, b );
						dummy( ( void * ) c );
					}
					resultline( n, 3, EventSet, fail, quiet );
				}
			}
		}
		if (double_precision) {
			free( ad );
			free( bd );
			free( cd );
		} else {
			free( a );
			free( b );
			free( c );
		}
	}

	/* exit with status code */
	if ( retval == PAPI_ENOMEM ) {
		test_fail( __FILE__, __LINE__, "malloc", retval );
	}

	test_pass( __FILE__ );

	return 0;
}

