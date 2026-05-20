#include <stdio.h>
#include <stdlib.h>

#include "testcode.h"

int cache_write_test(double *array, int size) {
	int i;

	for(i=0; i<size; i++) {
		array[i]=(double)i;
	}

	return 0;
}

double cache_read_test(double *array, int size) {

	int i;
	double sum=0;

	for(i=0; i<size; i++) {
		sum+= array[i];
	}

	return sum;
}

int cache_random_write_test(double *array, int size, int count) {
	int i;

	for(i=0; i<count; i++) {
		array[random()%size]=(double)i;
	}

	return 0;
}

double cache_random_read_test(double *array, int size, int count) {

	int i;
	double sum=0;

	for(i=0; i<count; i++) {
		sum+= array[random()%size];
	}

	return sum;
}
