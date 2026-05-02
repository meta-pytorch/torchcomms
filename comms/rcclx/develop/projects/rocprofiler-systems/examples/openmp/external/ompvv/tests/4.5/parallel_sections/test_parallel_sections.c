//===================-test_parallel_sections.c-=============================//
//
// OpenMP API Version 4.5 Nov 2015
//
// testing the combined construct 'parallel sections'
//
//===---------------------------------------------------------------------===//



#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include "ompvv.h"

int Var = 0;

// The following function waits for Var value to become 1 and then increments
// it to Var = 1
void function1(int *Var) {
  int temp = -1;
  while(1) {
   #pragma omp atomic read
    temp = *Var;
    
    if (temp == 1) {
      #pragma omp atomic update
      *Var += 1;
      break;
    }
  }
}

// The following function check if Var == 0 and then increments to Var = 1
// and then immediately it loops until Var == 3 and then increments it to
// Var = 4
void function2(int *Var) {
  int temp = -1;
  while(1) {
    #pragma omp atomic read
    temp = *Var;
    
    if (temp == 0) {
      #pragma omp atomic update
      *Var += 1;
      break;
    }
  }

  while(1) {
    #pragma omp atomic read
    temp = *Var;
    
    if (temp == 3) {
      #pragma omp atomic update
      *Var += 1;
      break;
    }
  }
}

// The following function checks if Var == 2 and then increments to Var = 3
void function3(int *Var) {
  int temp = -1;
  while(1) {
    #pragma omp atomic read
    temp = *Var;
    
    if (*Var == 2) {
      #pragma omp atomic update
      *Var += 1;
      break;
    }
  }
}


int main() {
  int errors = 0;
  
  #pragma omp parallel sections
  {
    if(omp_get_num_threads() == 1) 
    {
      OMPVV_WARNING("Sections are executed by a single thread, test will be aborted.")
      exit(0);
    }
      
    #pragma omp section
    function1(&Var);

    #pragma omp section
    function2(&Var);

    #pragma omp section
    function3(&Var);
  }
 
  
  // The final expected value of Var is '4'
  OMPVV_TEST_AND_SET_VERBOSE(errors, (Var != 4));
  OMPVV_REPORT_AND_RETURN(errors);
}
