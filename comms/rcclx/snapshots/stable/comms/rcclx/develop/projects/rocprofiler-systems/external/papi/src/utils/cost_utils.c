#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_ITERS	1000000

int num_iters = NUM_ITERS;

/* computes min, max, and mean for an array; returns std deviation */
double
do_stats( long long *array, long long *min, long long *max, double *average )
{
	int i;
	double std, tmp;

	*min = *max = array[0];
	*average = 0;
	for ( i = 0; i < num_iters; i++ ) {
		*average += ( double ) array[i];
		if ( *min > array[i] )
			*min = array[i];
		if ( *max < array[i] )
			*max = array[i];
	}
	*average = *average / ( double ) num_iters;
	std = 0;
	for ( i = 0; i < num_iters; i++ ) {
		tmp = ( double ) array[i] - ( *average );
		std += tmp * tmp;
	}
	std = sqrt( std / ( num_iters - 1 ) );
	return ( std );
}

void
do_std_dev( long long *a, int *s, double std, double ave )
{
	int i, j;
	double dev[10];

	for ( i = 0; i < 10; i++ ) {
		dev[i] = std * ( i + 1 );
		s[i] = 0;
	}

	for ( i = 0; i < num_iters; i++ ) {
		for ( j = 0; j < 10; j++ ) {
			if ( ( ( double ) a[i] - dev[j] ) > ave )
				s[j]++;
		}
	}
}

void
do_dist( long long *a, long long min, long long max, int bins, int *d )
{
	int i, j;
	int dmax = 0;
	int range = ( int ) ( max - min + 1 );	/* avoid edge conditions */

	/* clear the distribution array */
	for ( i = 0; i < bins; i++ ) {
		d[i] = 0;
	}

	/* scan the array to distribute cost per bin */
	for ( i = 0; i < num_iters; i++ ) {
		j = ( ( int ) ( a[i] - min ) * bins ) / range;
		d[j]++;
		if ( j && ( dmax < d[j] ) )
			dmax = d[j];
	}

	/* scale each bin to a max of 100 */
	for ( i = 1; i < bins; i++ ) {
		d[i] = ( d[i] * 100 ) / dmax;
	}
}

/* Long Long compare function for qsort */
static int cmpfunc (const void *a, const void *b) {

	if ( *(long long *)a - *(long long *)b < 0 ) {
		return -1;
	}

	if ( *(long long int*)a - *(long long int*)b > 0 ) {
		return 1;
	}

	return 0;
}

/* Calculate the percentiles for making boxplots */
int do_percentile(long long *a,
		long long *percent25,
		long long *percent50,
		long long *percent75,
		long long *percent99) {

	long long *a_sort;
	int i_25,i_50,i_75,i_99;

	/* Allocate room for a copy of the results */
	a_sort = calloc(num_iters,sizeof(long long));
	if (a_sort==NULL) {
		fprintf(stderr,"Memory allocation error!\n");
		return -1;
	}

	/* Make a copy of the results */
	memcpy(a_sort,a,num_iters*sizeof(long long));

	/* Calculate indices */
	i_25=(int)num_iters/4;
	i_50=(int)num_iters/2;
	// index for  75%, not quite accurate because it doesn't
	// take even or odd into consideration
	i_75=((int)num_iters*3)/4;
	i_99=((int)num_iters*99)/100;

	qsort(a_sort,num_iters-1,sizeof(long long),cmpfunc);

	*percent25=a_sort[i_25];
	*percent50=a_sort[i_50];
	*percent75=a_sort[i_75];
	*percent99=a_sort[i_99];

	free(a_sort);

	return 0;
}
