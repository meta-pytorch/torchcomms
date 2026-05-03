//===------ FILE_NAME.c ---------------------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// 
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

int test_function() {
  int errors = 1;

#pragma omp target map(from: errors)
  {
    errors = 0;
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING; // This should be at the beginning of all tests

  // Here are some examples on how to use the VERBOSE_MODE messages
  OMPVV_INFOMSG("THIS IS IS AN EXAMPLE OF AN INFOMSG");
  OMPVV_INFOMSG("THIS IS IS AN EXAMPLE OF AN INFOMSG WITH PARAMETERS (%d, %s, %f)", 3, "ompvv", 3.3);
  OMPVV_INFOMSG_IF(1==1, "THIS IS IS AN EXAMPLE OF AN INFOMSG_IF 1==1");
  OMPVV_INFOMSG_IF(1==0, "THIS IS IS AN EXAMPLE OF AN INFOMSG_IF 1==0 (THIS SHOULD NOT BE DISPLAYED)");
  OMPVV_WARNING("THIS IS IS AN EXAMPLE OF A WARNING");
  OMPVV_WARNING("THIS IS IS AN EXAMPLE OF A WARNING WITH PARAMETERS (%d, %s, %f)", 3, "ompvv", 3.3);
  OMPVV_WARNING_IF(1==1, "THIS IS IS AN EXAMPLE OF AN INFOMSG_IF 1==1");
  OMPVV_WARNING_IF(1==0, "THIS IS IS AN EXAMPLE OF AN INFOMSG_IF 1==0 (THIS SHOULD NOT BE DISPLAYED)");
  OMPVV_ERROR("THIS IS IS AN EXAMPLE OF AN ERROR");
  OMPVV_ERROR("THIS IS IS AN EXAMPLE OF AN ERROR WITH PARAMETERS (%d, %s, %f)", 3, "ompvv", 3.3);
  OMPVV_ERROR_IF(1==1, "THIS IS IS AN EXAMPLE OF AN ERROR_IF 1==1");
  OMPVV_ERROR_IF(1==0, "THIS IS IS AN EXAMPLE OF AN ERROR_IF 1==0 (THIS SHOULD NOT BE DISPLAYED)");

  int errors = 0;
  OMPVV_TEST_AND_SET(errors, 1!=0); // Condition to generate an error
  OMPVV_TEST_AND_SET_VERBOSE(errors, 1!=0); // Condition to generate an error and display
  OMPVV_REPORT(errors); // This should display test fail

  errors = 0;
  OMPVV_TEST_AND_SET(errors, 1==0); // Errors will not be set, the result should still be 0
  OMPVV_TEST_AND_SET_VERBOSE(errors, 1==0); // Errors will not be set, the result should still be 0, NO ERROR SHOULD BE DISPLAYED
  
  errors = test_function();
  // CALL OTHER FUNCTIONS HERE

  // These two lines could be replaced with OMPVV_REPORT_AND_RETURN(errors)
  // OMPVV_REPORT requires OMPVV_TEST_OFFLOADING TO BE CALLED BEFOREHAND
  OMPVV_REPORT(errors);  // Tests must pass
  OMPVV_RETURN(errors); 
}
