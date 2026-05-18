//===--- test_default_firstprivate_parallel.c -----------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  This test checks behavior of the default clause when the specified data-sharing-attribute  
//  is firstprivate. The constructs allowed for a default clause are parallel, teams, task, and taskloop.
//  This test utilizes the parallel construct, where firstprivate states that one ore more list items are
//  private to the parallel task, and each of them are initialized with the value of the corresponding
//  original item.
//
////===--------------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int errors, i;

int test_default_firstprivate_parallel() {
  int scalar_var = 5;
  int arr[N];
  int sum = 0;
  for (int i=0; i<N; i++){
	arr[i] = i;
	sum += arr[i];
  }
  #pragma omp parallel default(firstprivate) num_threads(OMPVV_NUM_THREADS_HOST)
  {
  	scalar_var += 10;
  	for (int i=0; i<N; i++){
		arr[i] = i+2;
  	}	
  }
  int newsum = 0;
  int wrongsum = 0;
  for(int i=0; i<N; i++){
	newsum += arr[i];
	wrongsum += i+2;
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_var != 5);
  OMPVV_INFOMSG_IF(scalar_var == 0, "Scalar was not initialized in parallel region & not updated");
  OMPVV_INFOMSG_IF(scalar_var == 15, "Scalar was not firstprivate, changes made in parallel affected original copy");
  OMPVV_TEST_AND_SET_VERBOSE(errors, sum != newsum);
  OMPVV_INFOMSG_IF(newsum == 0, "Array was not initialized in parallel region properly");
  OMPVV_INFOMSG_IF(newsum == wrongsum, "Array was not first private, changes made in parallel affected original copy");
  return errors;
}

int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_default_firstprivate_parallel() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}           

