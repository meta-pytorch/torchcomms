//===------------ test_depend_inout_omp_all_memory.c --------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// This test checks that the special omp_all_memory locator keyword works
// as intended. The test checks this by having two variables passed into the 
// target parallel region, depending on both values. If both variables are
// properly passed through the tasks then it passes. If not, the test fails.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int main() {
	int errors = 0;
	int x[N];
  	int y = 5;

	for (int i = 0; i < N; i++) {
		x[i] = i;
	}

	OMPVV_TEST_OFFLOADING;

	#pragma omp target map(tofrom: errors) map(to: x, y)
	{
		#pragma omp parallel
		#pragma omp single
		{
       			#pragma omp task depend(out: x)
       			{
         			for (int i = 0; i < N; i++) {
        	    			x[i] += 1;
       				}
        		}
        		#pragma omp task depend(out: y)
        		{   
          			y += 5;
        		}
	      		#pragma omp task depend(inout: omp_all_memory)
	      		{
          			for(int i = 0; i < N; i++) {
            				OMPVV_TEST_AND_SET(errors, x[i] != i + 1);
          			}

          			OMPVV_TEST_AND_SET(errors, y != 10 );
	      		}
        		#pragma omp task depend(out: x)
        		{
          			for (int i = 0; i < N; i++) {
            				x[i] += 1;
          			}
        		}
        		#pragma omp task depend(out: y)
        		{
          			y += 5;
        		}
		}

      		for (int i = 0; i < N; i++) {
        		OMPVV_TEST_AND_SET(errors, x[i] != i + 2);
      		}

      		OMPVV_TEST_AND_SET(errors, y != 15);
	}

	OMPVV_REPORT_AND_RETURN(errors);
}
