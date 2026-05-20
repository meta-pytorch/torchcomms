//----------------------------------- test_taskloop_firstprivate.c ----------------------------------------//
//
// OpenMP API Version 4.5 November 2015
//
// This test checks the taskloop directive with the 'firstprivate' clause specified.
// The 'taskloop' construct parallelize loops with independent iterations by creating tasks. 
// It allows for efficient parallel execution of loop iterations by distributing them among multiple threads. 
// The 'firstprivate' clause ensures that each task should have private copies and initialized                          // private copies with the value from the master thread. 
// -------------------------------------------------------------------------------------------------------//

#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define NUM_THREADS 500
#define NUM_TASKS 1000

int is_task_values_equal_to_firstprivate(int first_priv_value, int a[])
{
    for(int i = 0; i < NUM_TASKS; i++) {
    	if(first_priv_value != a[i])
        	return 0; //if all array values are not same. 
    }	

    return 1; //if all array values are same
}

int test_taskloop_firstprivate() {
    
    int errors = 0;
    
    int private_var = 11; 
    int var_not_in_openmp = private_var;

    int task_vals[NUM_TASKS];
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp single
        {
            #pragma omp taskloop firstprivate(private_var)
            for (int i = 0; i < NUM_TASKS; i++) {
		task_vals[i] = private_var;
            }
        }
    }
    
    //if all the tasks values are same as first private value, get TRUE else FALSE 
    OMPVV_TEST_AND_SET_VERBOSE(errors, (is_task_values_equal_to_firstprivate(var_not_in_openmp, task_vals)) != 1);
    
    return errors;
}

int main() {
  
    int errors = 0;
  
    OMPVV_TEST_AND_SET_VERBOSE(errors, (test_taskloop_firstprivate()) != 0);
    OMPVV_REPORT_AND_RETURN(errors);

    return 0;
}

