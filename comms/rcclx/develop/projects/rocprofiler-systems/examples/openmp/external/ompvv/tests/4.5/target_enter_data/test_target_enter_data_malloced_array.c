//===---- test_target_enter_data_malloced_array.c -------------------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// Tests for target enter data with heap allocated arrays. The test begins by creating
// a pointer to a block of memory that is of size n*sizeof(int) and is allocated using malloc(). 
// A check is made to ensure that the pointer is not null, variable x is set equal to the dereferenced 
// pointer and is filled with int value 10. Global array B[10] is filled with int value 0. 
// The values of x[n] are mapped onto the device using enter data map, and values in array B[10]
// are set equal to values of x[n] to ensure that values of x[n] were properly mapped to device.
// Back on the host, a final check is made to confirm values of array B[10] are all integer 10.
// 
//===----------------------------------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ompvv.h"


// global variables
int n=10, B[10];
int *x;


void init(int **A) {
  int i;
  *A = (int *) malloc(n*sizeof(int));
  if (NULL == *A){ 
    OMPVV_ERROR("This Test Has Failed, disregard other messages, array A is not properly allocated");
    exit(-1);
  }
  x = *A;
  for (i = 0; i < n; i++){
    x[i] = 10;
    B[i] = 0;
}
 

#pragma omp target enter data map(to:x[:n])//Note:Mapping *A[:n] is incorrect as OpenMP doesn't support arbitrary expressions
}

int main () {

//check offloading
  int is_offloading;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
  int i, errors = 0;
  int *A;
  init(&A);

#pragma omp target map(to: n) map(tofrom: B)
{
  for (i = 0; i < n; i++)
    B[i] = x[i];
}

// finalize();
  for (i = 0; i < n; i++)
    if (B[i] != 10) {
     errors += 1;
   }
 

  OMPVV_TEST_AND_SET_VERBOSE(errors, (errors != 0));

  OMPVV_REPORT_AND_RETURN(errors);
}
