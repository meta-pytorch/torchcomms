//===---- test_target_update_if.c - check the if clause of target update ------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// The if clause determines if the section should be updated on  
// the device. There are two scenarios to test here: 
// (a) with offloading when 'if' clause evaluates to true then 
// associated data is updated depending on the motion clause.
// (b) with offloading when 'if' clause evaluates to false 
// then there is no update
// The if clause is evaluated on runtime which means that variables could
// determine this behavior. 
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include "ompvv.h"

#define N 100

int a[N];
int b[N];
int c[N];
int count, toggle=0;

int init_b(){
  if(toggle % 2){
    int i;
    for (i = 0; i < N; i++) {
      b[i] = b[i] * 2; 
    }
    toggle++;
    return 1;
  }
  else{
    toggle++;
    return 0;
  }
}
  
// Test for OpenMP 4.5 target update with if
int main() {
  int errors[2]={0,0}, i = 0, report_errors = 0, change_flag = 0;

  for (i = 0; i < N; i++) {
    a[i] = 10;
  }

  // We test for offloading
  int is_offloading;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading); 
  
  if (!is_offloading)
  OMPVV_WARNING("It is not possible to test conditional data transfers "
                 "if the environment is shared or offloading is off. Not testing "
                 "anything");
    
  for(count = 0; count < 4; count++){
    for (i = 0; i < N; i++) {
      b[i] = 2; 
      c[i] = 0;
  }
#pragma omp target data map(to: a[:N], b[:N]) map(tofrom: c)
{
  #pragma omp target 
  {
        int j = 0;
        for (j = 0; j < N; j++) {
          c[j] = (a[j] + b[j]);//c=12 
        }
  } // end target

  change_flag = init_b();
  #pragma omp target update if (change_flag) to(b[:N]) //update b=4 for all odd iterations

  #pragma omp target //default mapping is tofrom on b and c
  {
        int j = 0;
        for (j = 0; j < N; j++) {
          c[j] = (c[j] + b[j]);//if b is updated c=16 else c=14
        }
  } // end target

}// end target-data

    // checking results 
  if (change_flag) {
    for (i = 0; i < N; i++) {
        if (c[i] != 16) {
          errors[0] += 1;
        }
    }
  }
  else {
    for (i = 0; i < N; i++) {
        if (c[i] != 14) {
          errors[1] += 1;
        }
    }
  }
}//end for

  OMPVV_TEST_AND_SET_VERBOSE(report_errors, errors[0] > 0);
  OMPVV_INFOMSG_IF(errors[0] > 0, "Target update test when if clause is true failed");
  OMPVV_TEST_AND_SET_VERBOSE(report_errors, errors[1] > 0);
  OMPVV_INFOMSG_IF(errors[1] > 0,  "Target update test when if clause is false failed");

  OMPVV_REPORT_AND_RETURN(report_errors);
}
