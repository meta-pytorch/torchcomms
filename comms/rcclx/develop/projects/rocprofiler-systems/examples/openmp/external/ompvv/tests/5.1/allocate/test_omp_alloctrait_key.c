//---------------------test_omp_alloctrait_key.c---------------------
//
// OpenMP API Version 5.1 March 2023
//
// This check intends to check the function of omp_alloctrait_key
// enum. The test creates two allocations which are then passed into
// a parrallel region to enusre the allocations work as expected.
//
//-------------------------------------------------------------------

#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "ompvv.h"
#define N 1000

int test_case(){
	
	int *x, *y;
	
	int errors = 0;
	int test_x = 0;
	int test_y = 0;

	omp_memspace_handle_t xy_memspace = omp_default_mem_space;
	omp_alloctrait_t xy_traits[1] = {omp_atk_alignment, 64};
	omp_allocator_handle_t xy_alloc = omp_init_allocator(xy_memspace,1,xy_traits);
	
	x = (int *) omp_alloc(N*sizeof(int), xy_alloc);
	y = (int *) omp_alloc(N*sizeof(int), xy_alloc);

	#pragma omp parallel
	{
		#pragma omp for simd simdlen(16) aligned(x,y:64)
		for(int i=0;i<N; i++){ x[i]=0; y[i]=0;}
		
		#pragma omp for simd simdlen(16) aligned(x,y:64)
		for(int i=0; i<N; i++){x[i] = 1; y[i] = 1;}
	}
	
	for(int i=0; i<N; i++){
		test_x += x[i];
		test_y += y[i];
	}

	omp_free(x, xy_alloc);
	omp_free(y,  xy_alloc);
	omp_destroy_allocator(xy_alloc);

	int test_all = test_x + test_y;
	OMPVV_TEST_AND_SET_VERBOSE(errors,test_all != 2000);
	OMPVV_INFOMSG_IF(test_x != 1000, "test x not properly allocated");
	OMPVV_INFOMSG_IF(test_y != 1000, "test y not properly allocated");
	return errors;
}

int main(){
	int errors = 0;
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_case() != 0);
	OMPVV_REPORT_AND_RETURN(errors);
	return errors;
}
