//===------ test_parallel_for_lastprivate_conditional.c ------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test evaluates the lastprivate directive using the conditional clause,
// ensuring proper behavior and variable updates in parallel regions.
//===-------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include "ompvv.h"

#define N 1024

int test_lastprivate_conditional() {
        int errors = 0;
        int a[N];
        int x = 0;

        #pragma omp parallel for lastprivate(conditional: x)
        for (int k = 0; k < N; k++) {
                a[k] = k;
                if (k == 24 || k == 123) {
                        x = a[k];
		}
        }
        // Check to make sure x is set.
        OMPVV_TEST_AND_SET(errors, x!= 123);
        return errors;
}

int main() {
        int errors = 0;

        OMPVV_TEST_AND_SET_VERBOSE(errors, test_lastprivate_conditional() != 0);

        OMPVV_REPORT_AND_RETURN(errors);
}

