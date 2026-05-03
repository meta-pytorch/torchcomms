//-------------------------------- test_taskloop_lastprivate.c ------------------------------------------------//
//
// OpenMP API Version 4.5 November 2015
//
// This test checks the 'taskloop' directive with the 'lastprivate' clause specified.
// The 'taskloop' construct parallelize loops with independent iterations by creating tasks. 
// It allows for efficient parallel execution of loop iterations by distributing them among multiple threads. 
// The 'lastprivate' clause ensures that the last value of variable in a loop is available after loop completion.  
//-------------------------------------------------------------------------------------------------------------//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define NUM_THREADS 1000
#define NUM_TASKS 1000 

int test_taskloop_lastprivate() {
    
    int errors = 0;
   
    int val = 100;
    int task_vals[NUM_TASKS];

    int last_itr = NUM_TASKS - 1;
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp single
        {
            #pragma omp taskloop lastprivate(val)
            for (int i = 0; i < NUM_TASKS; i++) {
                val = i;
                task_vals[i] = val; 
            }
        }
    }
   
    //NOTE: lastprivate_var depends on NUM_TASK and not on Input value
    //it is the last task iteration which is kept, not the last operation value.
     OMPVV_TEST_AND_SET_VERBOSE(errors, (task_vals[last_itr] != val));

    return errors;    
}

int main() {
   
    int errors = 0;

    OMPVV_TEST_AND_SET_VERBOSE(errors, (test_taskloop_lastprivate() != 0));
    OMPVV_REPORT_AND_RETURN(errors);
    
    return 0;
}

