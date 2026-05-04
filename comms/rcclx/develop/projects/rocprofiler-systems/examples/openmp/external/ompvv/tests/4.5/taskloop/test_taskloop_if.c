//--------------------------------------------- test_taskloop_if.c ----------------------------------------//
//
// OpenMP API Version 4.5 November 2015
//
// This test checks the taskloop directive with the if clause specified.
// The 'taskloop' construct parallelize loops with independent iterations by creating tasks. 
// It allows for efficient parallel execution of loop iterations by distributing them among multiple threads. 
// The 'if' clause used to check the condition and taskloop will take into consideration only,      
// if the condition is TRUE.
//----------------------------------------------------------------------------------------------------------//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define NUM_TASKS 1000
#define NUM_THREADS 1000

#define M 100

//if all thread ids are same, return 1 else return 0
int thread_ids_are_same(int a[], int len) {

	int first = a[0];

	for(int i = 1; i < len; i++) {
		if(first != a[i]) 
		  return 0; //if all the thread ids are not same
	}

	return 1; //if all the thread ids are same	
}

int test_taskloop_if(int THRESHOLD) {
  
  int errors = 0;
  
  int thread_ids[NUM_TASKS];

  #pragma omp parallel num_threads(NUM_THREADS)
  {
  	#pragma omp single
    	{	
        	#pragma omp taskloop if(M == THRESHOLD)
      		for (int i = 0; i < NUM_TASKS; i++) {
	 		thread_ids[i] = omp_get_thread_num();
      		} 
   	}
  } 
  
  if (M ==  THRESHOLD) {
    if (thread_ids_are_same(thread_ids, NUM_TASKS) != 0) {
      errors++; 
    }
  } else if (M != THRESHOLD) {
    if (thread_ids_are_same(thread_ids, NUM_TASKS) == 0) {
      errors++; 
    }
  }

  return errors;            

}

int main() {
 
   int errors = 0;
   
   int THRESHOLD = 100; //Equal to M
   OMPVV_TEST_AND_SET_VERBOSE(errors, (test_taskloop_if(THRESHOLD) != 0));

   THRESHOLD = 110; // Not equal to M
   OMPVV_TEST_AND_SET_VERBOSE(errors, (test_taskloop_if(THRESHOLD) != 0));
   
   OMPVV_REPORT_AND_RETURN(errors);

   return 0;
}
