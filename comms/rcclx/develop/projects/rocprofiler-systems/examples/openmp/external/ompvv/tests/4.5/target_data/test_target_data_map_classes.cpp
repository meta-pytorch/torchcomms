//===---- test_target_data_map_classes.cpp -----------------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// This test checks the mapping of c++ objects on both the stack and heap. The 'new'
// operater is utilized in the case where memory allocation is on heap. Objects are first 
// initalized on the host and then mapped to device. Inside the target region, the object's 
// array data member, which was also mapped to device, is modified through the device array
// pointer. After target region, we swap back to the host array pointer to verify that the 
// host array was properly updated.
//
//===-------------------------------------------------------------------------------------===//

#include <iostream>
#include <omp.h>
#include "ompvv.h"

using namespace std;

#define N 1000

class A {

public:
  // breaks encapsulation
  int *h_array;
  int size;
  int sum;

  A(int *array, const int s) : h_array(array), size(s) { sum = 0; }

  // TODO: Add virtual once supported 
  ~A() { h_array = NULL; }
};

// Test for OpenMP 4.5 target data mapping objects in the heap
int test_map_tofrom_class_heap() {

  OMPVV_INFOMSG("test_map_tofrom_class_heap");

  int sum = 0, errors = 0;

  int *array = new int[N];
  A *obj = new A(array, N);

  // mapping an object + array: it is shallow copy thus 
  // pointers are not translated automatically
#pragma omp target data map(from: array[0:N]) map(tofrom: obj[0:1])
  {
#pragma omp target
    {
      // assign device array ptr to device obj 
      int *tmp_h_array = obj->h_array;
      obj->h_array = array;
      int tmp = 0;
      for (int i = 0; i < N; ++i) {
        obj->h_array[i] = 1;
        tmp += 1;
      }
      // swap array device ptr to host ptr 
      obj->h_array = tmp_h_array;

      obj->sum = tmp;
    } // end target
  } // end target data

  // checking results 
  for (int i = 0; i < N; ++i)
    sum += obj->h_array[i];

  OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (N != obj->sum));

  delete obj;
  delete[] array;

  return errors;
}

// Test for OpenMP 4.5 target data mapping objects on the stack
int test_map_tofrom_class_stack() {

  OMPVV_INFOMSG("test_map_tofrom_class_stack");

  int sum = 0, errors = 0;

  int array[N];
  A obj(array, N);

  // mapping an object + array: it is shallow copy thus 
  // pointers are not translated automatically
#pragma omp target data map(from: array[0:N]) map(tofrom: obj)
  {
#pragma omp target
    {
      // assign device array ptr to device obj 
      int *tmp_h_array = obj.h_array;
      obj.h_array = array;
      int tmp = 0;
      for (int i = 0; i < N; ++i) {
        obj.h_array[i] = 1;
        tmp += 1;
      }
      // swap array device ptr to host ptr 
      obj.h_array = tmp_h_array;

      obj.sum = tmp;
    } // end target
  } // end target data

  // checking results 
  for (int i = 0; i < N; ++i)
    sum += obj.h_array[i];

  OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (N != obj.sum));

  return errors;
}

int main() {

  int errors = 0;
  
  OMPVV_TEST_OFFLOADING;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_tofrom_class_heap());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_tofrom_class_stack());

  OMPVV_REPORT_AND_RETURN(errors);
}
