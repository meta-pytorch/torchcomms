//------------------------------------------- test_taskloop_shared.c --------------------------------------//
//
// OpenMP API Version 4.5 November 2015
//
// This test checks the 'taskloop' directive with the 'shared' clause specified.
// The 'taskloop' construct parallelize loops with independent iterations by creating tasks. 
// It allows for efficient parallel execution of loop iterations by distributing them among multiple threads. 
// The 'shared' clause ensures that vaiable is shared with all the tasks.  
//----------------------------------------------------------------------------------------------------------//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define NUM_THREADS 50
#define NUM_TASKS 1000

int test_taskloop_shared() {

  int errors = 0;

  long int all_thread_sum, real_sum = 0; 
  long int shared_var = 0; //This variable is shared with all the tasks.  

   //get valid sum without openmp
   for(int i = 0; i < NUM_TASKS; i++) {
     real_sum = real_sum + i;
   }
   
   #pragma omp parallel num_threads(NUM_THREADS)
   {
      #pragma omp single
      {
	#pragma omp taskloop shared(shared_var) 
        for(int i = 0; i < NUM_TASKS; i++) 
	{
	    #pragma omp atomic 
	    shared_var = shared_var + i;
	}
      }	   
   }

   all_thread_sum = shared_var;
   OMPVV_TEST_AND_SET_VERBOSE(errors, real_sum != all_thread_sum);

   return errors;
}

int main() {
  
  int errors = 0;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, (test_taskloop_shared()) != 0);
  OMPVV_REPORT_AND_RETURN(errors);

  return 0;
}
