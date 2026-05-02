//===--- test_depobj_depend_update_destroy.cpp ------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//  
// Tests the standalone depobj construct using the depend, update, 
// and destory clause. For the first call to go function, the first task
// must execute first due to inout dependence-type. In the second call to go
// function, both tasks can execute simultaneously due to in dependence-type.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1028
#define TURN1 1
#define TURN2 2

void go(int turn, int a[], int b[], omp_depend_t *obj);
int errors = 0;


int main () {
   int a[N], b[N];
   omp_depend_t obj;

   for (int i = 0; i < N; i++) {
      a[i] = i;
      b[i] = 0;
   }
   
   #pragma omp depobj(obj) depend(inout: a)
   
   go(TURN1, a, b, &obj);

   #pragma omp depobj(obj) update(in)

   go(TURN2, a, b, &obj);

    
   #pragma omp depobj(obj) destroy
   
   for (int i = 0; i < N; i++) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, b[i] != 5);
      OMPVV_TEST_AND_SET_VERBOSE(errors, a[i] != i+2);
   }
 
   OMPVV_REPORT_AND_RETURN(errors);

}

void go(int turn, int a[], int b[], omp_depend_t *obj) {
   #pragma omp parallel num_threads(2)
   #pragma omp single 
   {
      #pragma omp task depend(depobj: *obj)
      {      
         for (int i = 0; i < N; i++) {
              a[i] += 1;
         }
      }
      
      #pragma omp task depend(in: a[:N])
      { 
         if (turn == TURN1) {
            for (int i = 0; i < N; i++) {
               OMPVV_TEST_AND_SET_VERBOSE(errors, a[i]!=(i+turn));
            }
         }  
         if (turn == TURN2) {
            for (int i = 0; i < N; i++) {
               b[i] = 5;
            }
         }
      }
   }   
}
