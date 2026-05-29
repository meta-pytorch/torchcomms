//------------------------------------------- test_taskloop_collapse.c --------------------------------------//
// OpenMP API Version 4.5 September 2015
//
// This test checks the 'taskloop' directive with the 'collapse' clause specified.
// The 'taskloop' construct parallelizes loops with independent iterations by creating tasks. 
// It allows for efficient parallel execution of loop iterations by distributing them among multiple threads. 
// The 'collapse' clause ensures that multiple nested loops into a single loop.      
//----------------------------------------------------------------------------------------------------------//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define NUM_THREADS 100
#define NUM_COLLAPSE 2

int test_taskloop_collapse() {

  int errors = 0;

  int size = 200;
  int a[(size*size)];
  
  int sum_without_openmp = 0, sum_with_openmp = 0;
   
  for(int i = 0; i < (size*size); i++) {
   	a[i] = i;
   }

   //get valid sum without openmp
   for(int i = 0; i < (size); i++) {
	for(int j = 0; j < size; j++) {
	     sum_without_openmp = sum_without_openmp + (a[i]*a[j]);
	}
   }
   
   #pragma omp parallel num_threads(NUM_THREADS)
   {
      #pragma omp single
      {
	#pragma omp taskloop collapse(NUM_COLLAPSE)  
        for(int i = 0; i < size; i++) 
	{
	    for(int j = 0; j < size; j++) 
	    {
		    #pragma omp atomic
		    sum_with_openmp = sum_with_openmp + (a[i]*a[j]); 
            }
	}
      }	   
   }
   
   OMPVV_TEST_AND_SET_VERBOSE(errors, sum_with_openmp != sum_without_openmp);

   return errors;
}

int main() {
  
  int errors = 0;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, (test_taskloop_collapse()) != 0);
  OMPVV_REPORT_AND_RETURN(errors);

  return 0;
}
