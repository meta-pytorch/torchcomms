//===-- test_task_critical.c ------------------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// Description:
// testTaskWithCriticalBasic()
// This is a basic test to demonstrate how a shared resource
// can be accessed and written to in multiple thread environment.
//
//testTaskWithCriticalAdvanced()
// This is a advanced test to demonstrate how a shared resource
// can be accessed and written to in multiple thread environment.
// In this a buffer is filled with random integers. The parallel
// region counts the number of prime integers.
//===----------------------------------------------------------------------===//



#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include "ompvv.h"

int testTaskWithCriticalBasic() {
  int errors = 0;
  int count = 0;
  int NThrds = -2;
  omp_set_num_threads(OMPVV_NUM_THREADS_HOST);
  
#pragma omp parallel
  {
      if(omp_get_thread_num() == 0) NThrds = omp_get_num_threads();
       
#pragma omp task
    {
#pragma omp critical
      {
        count = count + 1;
      }
    }
  }
  int ret = 0;
  if (count == NThrds) {
    ret = 0;
  } else {
    ret = -1;
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, ret != 0);
  return errors;
}

/**
  Local Function check if number is prime
*/
int isPrime(unsigned int number) {
  int ret = 0;
  if (number < 2) {
    return 1;
  }
  for (int num = 2; (num*num) <= number; num++) {
    if ((number % num) == 0) {
      ret = 1;
      break;
    }
  }
  return ret;
}

int testTaskWithCriticalAdvanced(int numThreads, int expectedVal) {
  int errors = 0;
  int countPrime = 0, count = 0;
  unsigned int *A = (unsigned int*) (malloc(numThreads*sizeof(unsigned int)));
  omp_set_num_threads(numThreads);
  // fill data
  for (int i = 0; i < numThreads; i++) {
    A[i] = i;
  }
  // Calculate number of prime numbers 
#pragma omp parallel
  {
#pragma omp task
    {
      int idx = 0;
#pragma omp critical
      {
        count++;
        idx = count - 1;
      }
      int ret = isPrime(A[idx]);
#pragma omp critical
      {
        if (ret == 0) {
          countPrime++;
        }
      }
    }
  }
  
  free(A);

  OMPVV_TEST_AND_SET_VERBOSE(errors, countPrime != expectedVal);
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithCriticalBasic());

  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithCriticalAdvanced(1, 0));

  OMPVV_REPORT_AND_RETURN(errors);
}
