//===--- test_target_update_iterator.c --------------------------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//  This test usese the target update directive with the iterator clause
//  to update the iterative values.
//
//  This example has been adapted from the 5.2 OpenMP Examples document,
//  "Multidependences Using Iterators" and "Simple target data and target update
//  constructs"
//
////===--------------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

typedef struct test_struct{
    size_t len;
    int *data;
} test_struct_t;

void init(struct test_struct *s ){
    s->len = N;
    s->data = (int *) malloc(sizeof(int) * N);
    for(size_t i = 0; i < s->len; i++){
        s->data[i] = i;
    }
}

void init_again(struct test_struct *s ){
    s->len = N;
    for(size_t i = 0; i < s->len; i++){
        s->data[i] = i+1;
    }
}

int test_target_update_iterator() {
    int errors = 0;
    int A[N]; // array for testing

    for(int i = 0; i < N; i++){
        A[i] = 0;
    }
    test_struct_t new_struct;

    init(&new_struct);

    #pragma omp target enter data map(to: new_struct.data[:N])
    #pragma omp target map(to: new_struct, new_struct.data[:N]) map(tofrom: A[:N]) // pointer attachment
    {
        for(int i = 0; i < N; i++){
            A[i] = new_struct.data[i];
        }
    }
    init_again(&new_struct);
    // update with new values, (i*2)+1
    #pragma omp target update to(iterator(it = 0:N): new_struct.data[it])
    #pragma omp target map(tofrom: A[:N])
    {
        for(int i = 0; i < N; i++){
            A[i] += new_struct.data[i];
        }
    }

    #pragma omp target exit data map(delete:new_struct.data[:N])

    for(int i = 0; i < N; i++){
        OMPVV_TEST_AND_SET(errors, A[i] != (i*2)+1);
    }
    
    return errors;
}

int main() {
   int errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_update_iterator() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}           

