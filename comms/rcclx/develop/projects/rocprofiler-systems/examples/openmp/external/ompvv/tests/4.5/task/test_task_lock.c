//===-- test_task_lock.c ------------------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// Description
// testTaskWithLockBasic()
// This is a basic test to demonstrate how a shared resource
// can be accessed and written to in multiple thread environment.
// 
// testTaskWithLockAdvanced()
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

int testTaskWithLockBasic(int numThreads) {
  int errors = 0;
  int count = 0;
  omp_lock_t lock;
  omp_init_lock(&lock);
  omp_set_num_threads(numThreads);
#pragma omp parallel
  {
#pragma omp task
    {
    omp_set_lock(&lock);
    {
      count = count + 1;
    }
    omp_unset_lock(&lock);
    }
  }
  omp_destroy_lock(&lock);
  int ret = 0;
  if (count == numThreads) {
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

int testTaskWithLockAdvanced(int numThreads, int expectedVal) {
  int errors = 0;
  omp_lock_t lock;
  omp_init_lock(&lock);
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
      omp_set_lock(&lock);
      {
        count++;
        idx = count - 1;
      }
      omp_unset_lock(&lock);
      int ret = isPrime(A[idx]);
      omp_set_lock(&lock);
      {
        if (ret == 0) {
          countPrime++;
        }
      }
      omp_unset_lock(&lock);
    }
  }
  free(A);
  omp_destroy_lock(&lock);
  OMPVV_TEST_AND_SET_VERBOSE(errors, countPrime != expectedVal);
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockBasic(4));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockBasic(8));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockBasic(16));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockBasic(32));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockBasic(64));

  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockAdvanced(1, 0));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockAdvanced(4, 2));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockAdvanced(8, 4));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockAdvanced(16, 6));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockAdvanced(32, 11));
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTaskWithLockAdvanced(64, 18));

  OMPVV_REPORT_AND_RETURN(errors);
}
