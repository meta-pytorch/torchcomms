/* This includes various workloads that had been scattered all over */
/* the various ctests.  The goal is to have them in one place, and */
/* share them, as well as maybe have only one file that has to be */
/* compiled with reduced optimizations */

#include <stdio.h>
#include <stdlib.h>

#include "testcode.h"

#define ROWS	1000
#define COLUMNS	1000

static float float_matrixa[ROWS][COLUMNS],
		float_matrixb[ROWS][COLUMNS],
		float_mresult[ROWS][COLUMNS];

static double double_matrixa[ROWS][COLUMNS],
		double_matrixb[ROWS][COLUMNS],
		double_mresult[ROWS][COLUMNS];


int flops_float_init_matrix(void) {

	int i,j;

	/* Initialize the Matrix arrays */
	/* Non-optimail row major.  Intentional? */
	for ( i = 0; i < ROWS; i++ ) {
		for ( j = 0; j < COLUMNS; j++) {
			float_mresult[j][i] = 0.0;
			float_matrixa[j][i] = ( float ) rand() * ( float ) 1.1;
			float_matrixb[j][i] = ( float ) rand() * ( float ) 1.1;
		}
	}

#if defined(__powerpc__)
	/* Has fused multiply-add */
	return ROWS*ROWS*ROWS;
#else
	return ROWS*ROWS*ROWS*2;
#endif

}

float flops_float_matrix_matrix_multiply(void) {

	int i,j,k;

	/* Matrix-Matrix multiply */
	for ( i = 0; i < ROWS; i++ ) {
		for ( j = 0; j < COLUMNS; j++ ) {
			for ( k = 0; k < COLUMNS; k++ ) {
				float_mresult[i][j] += float_matrixa[i][k] * float_matrixb[k][j];
			}
		}
	}

	return float_mresult[10][10];
}

float flops_float_swapped_matrix_matrix_multiply(void) {

	int i, j, k;

	/* Matrix-Matrix multiply */
	/* With inner loops swapped */

	for (i = 0; i < ROWS; i++) {
		for (k = 0; k < COLUMNS; k++) {
			for (j = 0; j < COLUMNS; j++) {
				float_mresult[i][j] += float_matrixa[i][k] * float_matrixb[k][j];
			}
		}
	}
	return float_mresult[10][10];
}



int flops_double_init_matrix(void) {

	int i,j;

	/* Initialize the Matrix arrays */
	/* Non-optimail row major.  Intentional? */
	for ( i = 0; i < ROWS; i++ ) {
		for ( j = 0; j < COLUMNS; j++) {
			double_mresult[j][i] = 0.0;
			double_matrixa[j][i] = ( double ) rand() * ( double ) 1.1;
			double_matrixb[j][i] = ( double ) rand() * ( double ) 1.1;
		}
	}

#if defined(__powerpc__)
		/* has fused multiply-add */
		return ROWS*ROWS*ROWS;
#else
	return ROWS*ROWS*ROWS*2;
#endif

}

double flops_double_matrix_matrix_multiply(void) {

	int i,j,k;

	/* Matrix-Matrix multiply */
	for ( i = 0; i < ROWS; i++ ) {
		for ( j = 0; j < COLUMNS; j++ ) {
			for ( k = 0; k < COLUMNS; k++ ) {
				double_mresult[i][j] += double_matrixa[i][k] * double_matrixb[k][j];
			}
		}
	}

	return double_mresult[10][10];
}

double flops_double_swapped_matrix_matrix_multiply(void) {

	int i, j, k;

	/* Matrix-Matrix multiply */
	/* With inner loops swapped */

	for (i = 0; i < ROWS; i++) {
		for (k = 0; k < COLUMNS; k++) {
			for (j = 0; j < COLUMNS; j++) {
				double_mresult[i][j] += double_matrixa[i][k] * double_matrixb[k][j];
			}
		}
	}
	return double_mresult[10][10];
}


/* This was originally called "dummy3" in the various sdsc tests */
/* Does a lot of floating point ops near 1.0 */
/* In theory returns a value roughly equal to the number of flops */
double
do_flops3( double x, int iters, int quiet )
{
	int i;
	double w, y, z, a, b, c, d, e, f, g, h;
	double result;
	double one;
	one = 1.0;
	w = x;
	y = x;
	z = x;
	a = x;
	b = x;
	c = x;
	d = x;
	e = x;
	f = x;
	g = x;
	h = x;
	for ( i = 1; i <= iters; i++ ) {
		w = w * 1.000000000001 + one;
		y = y * 1.000000000002 + one;
		z = z * 1.000000000003 + one;
		a = a * 1.000000000004 + one;
		b = b * 1.000000000005 + one;
		c = c * 0.999999999999 + one;
		d = d * 0.999999999998 + one;
		e = e * 0.999999999997 + one;
		f = f * 0.999999999996 + one;
		g = h * 0.999999999995 + one;
		h = h * 1.000000000006 + one;
	}
	result = 2.0 * ( a + b + c + d + e + f + w + x + y + z + g + h );

	if (!quiet) printf("Result = %lf\n", result);

	return result;
}


volatile double a = 0.5, b = 2.2;

double
do_flops( int n, int quiet )
{
        int i;
        double c = 0.11;

        for ( i = 0; i < n; i++ ) {
                c += a * b;
        }

	if (!quiet) printf("%lf\n",c);

	return c;
}

