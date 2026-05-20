//---------------------------------------- test_taskloop_private.c -----------------------------------------//
//
// OpenMP API Version 4.5 August 2015
//
// This test checks the taskloop directive with the 'private' clause specified.
// The 'taskloop' construct parallels loops with independent iterations by creating tasks.
// It allows for efficient parallel execution of loop iterations by distributing them among multiple threads. 
// The 'private' clause ensures that each task should have private copies. 
//----------------------------------------------------------------------------------------------------------//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define NUM_THREADS 100
#define NUM_TASKS 5

int test_taskloop_private() {

   int errors = 0;

   int private_var;
   int shared_var_sum = 0;

   #pragma omp parallel num_threads(NUM_THREADS)
   {
	    #pragma omp single
   	    {
		    #pragma omp taskloop private(private_var)
            for(int i = 0; i < NUM_TASKS; i++)
		    {
            	int sum = 0;
	    		for(private_var = 0; private_var < 5; private_var++)
	    			sum += private_var;

			    #pragma omp atomic
	    		shared_var_sum += sum;
	 	    }
   	    }
   }

   //check if the value of shared_var_sum is equal to (NUM_TASKS * 10)
   OMPVV_TEST_AND_SET_VERBOSE(errors, shared_var_sum != (NUM_TASKS * 10));

   return errors;
}


int main() {
  
  int errors = 0;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, (test_taskloop_private()) != 0);
  OMPVV_REPORT_AND_RETURN(errors);

  return 0;
}

