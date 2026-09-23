#include <stdio.h>

#define NUM_RUNS 3

#define MATRIX_SIZE 512
static double a[MATRIX_SIZE][MATRIX_SIZE];
static double b[MATRIX_SIZE][MATRIX_SIZE];
static double c[MATRIX_SIZE][MATRIX_SIZE];

long long naive_matrix_multiply_estimated_flops(int quiet) {

	long long muls,divs,adds;

	/* setup */
	muls=MATRIX_SIZE*MATRIX_SIZE;
	divs=MATRIX_SIZE*MATRIX_SIZE;
	adds=MATRIX_SIZE*MATRIX_SIZE;

	/* multiply */
	muls+=MATRIX_SIZE*MATRIX_SIZE*MATRIX_SIZE;
	adds+=MATRIX_SIZE*MATRIX_SIZE*MATRIX_SIZE;

	/* sum */
	adds+=MATRIX_SIZE*MATRIX_SIZE;

	if (!quiet) {
		printf("Estimated flops: adds: %lld muls: %lld divs: %lld\n",
			adds,muls,divs);
	}

	return adds+muls+divs;
}


long long naive_matrix_multiply_estimated_loads(int quiet) {

	long long loads=0;

	/* setup */
	loads+=0;

	/* multiply */
	loads+=MATRIX_SIZE*MATRIX_SIZE*MATRIX_SIZE*2;

	/* sum */
	loads+=MATRIX_SIZE*MATRIX_SIZE;

	if (!quiet) {
		printf("Estimated loads: %lld\n",loads);
	}

	return loads;
}

long long naive_matrix_multiply_estimated_stores(int quiet) {

	long long stores=0;

	/* setup */
	stores+=MATRIX_SIZE*MATRIX_SIZE*2;

	/* multiply */
	stores+=MATRIX_SIZE*MATRIX_SIZE;

	/* sum */
	stores+=1;

	if (!quiet) {
		printf("Estimated stores: %lld\n",stores);
	}

	return stores;
}


double naive_matrix_multiply(int quiet) {

	double s;
	int i,j,k;

	for(i=0;i<MATRIX_SIZE;i++) {
		for(j=0;j<MATRIX_SIZE;j++) {
			a[i][j]=(double)i*(double)j;
			b[i][j]=(double)i/(double)(j+5);
		}
	}

	for(j=0;j<MATRIX_SIZE;j++) {
		for(i=0;i<MATRIX_SIZE;i++) {
			s=0;
			for(k=0;k<MATRIX_SIZE;k++) {
				s+=a[i][k]*b[k][j];
			}
			c[i][j] = s;
		}
	}

	s=0.0;

	for(i=0;i<MATRIX_SIZE;i++) {
		for(j=0;j<MATRIX_SIZE;j++) {
			s+=c[i][j];
		}
	}

	if (!quiet) printf("Matrix multiply sum: s=%lf\n",s);

	return s;
}

