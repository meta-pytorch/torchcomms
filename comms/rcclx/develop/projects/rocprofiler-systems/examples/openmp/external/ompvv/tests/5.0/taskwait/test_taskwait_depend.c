//===---- test_taskwait_depend.c ---------------------------------------------===//
// 
// OpenMP API Version 5.0 
// 
// This test uses the taskwait directive with the depend clause, which
// ensures that the item listed in depend shares a depend with a
// previous task to ensure sychronization
//
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
int errors = 0;
int test_wrapper() { //wrapper for taskwait depend function
    #pragma omp parallel for
        for (int i=1; i<N; i++){
          int x,y,err = 0;
          #pragma omp task depend(inout: x) shared(x) // 1st Task
          x=i;
          #pragma omp task depend(inout: y) shared(y) // 2nd Task
          y=i;
          #pragma omp taskwait depend(in: x) //Requires the completion of the 1st task
          OMPVV_TEST_AND_SET(err, x!= i);
          #pragma omp taskwait depend(in: x,y) //Requires the completion of both tasks
          OMPVV_TEST_AND_SET(err, y!=i || x!=i);
          #pragma omp atomic
          errors += err;
        }
    return errors;
}


int main () {
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
  OMPVV_REPORT_AND_RETURN(errors);
}  
