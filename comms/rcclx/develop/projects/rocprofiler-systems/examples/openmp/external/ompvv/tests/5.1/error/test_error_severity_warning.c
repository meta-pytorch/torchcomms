//===--- test_error_severity_warning.c ----------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  Test uses error directive with message clause to print a message,
//  if error condition is reached then a warning error is printed,
//  otherwise is simply used to print a message. 
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

int errors, i;

int test_error_severity_warning() {
    i = 0;
    OMPVV_INFOMSG("If successful, test should print a \"error directive encountered\" message at the beginning of output");
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp error severity(warning)
            i+=5;
        }
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, i != 5);
    OMPVV_ERROR_IF(errors > 0, "Error directive caused execution error");
    return errors;
}

int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_error_severity_warning() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}
