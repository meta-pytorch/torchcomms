//===--test_targe_enter_exit_data_classes_simple.c - test a simple class ----===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// This test uses target enter data and exit data to map a whole class in the 
// constructor and destructor. It requires the use of helper_ variables since it
// is not legal to use the "this" pointer (implicitely or explicitely) in 4.5. 
//
// We use a modify method to assign values to the array and we use a method to 
// obtain a copy of the values from the device. The disctintion between methods 
// allows to show that data is mapped and remains mapped in the device memory. 
// This test does not use any inheritance or anything similar, therefore 
// the simple connotation
//
// Contrary to the test of target_enter_data, this test checks if during the exit
// data it successfully copies back the attributes of the object
//
////===----------------------------------------------------------------------===//

#include <iostream>
#include <omp.h>
#include "ompvv.h"

#define N 1000

class Simple {
  private:
    int *h_array;
    int size;
    int sum;
    int *errors; 

  public:
    Simple(int s, int *err) : size(s) { 
      sum = 0;
      h_array = new int[size];

      // Initialize the array in the host
      for (int i = 0; i < size; i ++)
        h_array[i] = i;

      // To obtain the error counter variable that is external
      errors = err;
      int * helper_harray = this->h_array;
      Simple * mySelf = this;
#pragma omp target enter data map(to: mySelf[0:1])
#pragma omp target enter data map(to: helper_harray[0:size])
    }

    ~Simple() { 
      // Modify again to see if changes are copied over
      int *helper_harray = this->h_array;
      Simple * mySelf = this;
#pragma omp target exit data map(from: helper_harray[0:size])

      // checking results that are coming back from the target exit data
      for (int i = 0; i < N; ++i) {
        OMPVV_TEST_AND_SET_VERBOSE(*errors, h_array[i] != 3*i);
      }

#pragma omp target exit data map(from: mySelf[0:1])
      OMPVV_TEST_AND_SET_VERBOSE(*errors, sum != 3*N*(N-1)/2);
      delete[] h_array; 
    }

    void modify() {
      int * helper_harray = this->h_array;
      int &helper_sum = this->sum;
      int &helper_size = this->size;
#pragma omp target defaultmap(tofrom:scalar)
      {
        helper_sum = 0;
        for (int i = 0; i < helper_size; ++i) {
          helper_harray[i] += i;
          helper_sum += helper_harray[i];
        }
      }
    }

    void getDeviceAttributes(int * array_copy, int & sum_copy) {
      int * helper_harray = this->h_array;
      int &helper_sum = this->sum;
      int &helper_size = this->size;
#pragma omp target map(from:array_copy[0:N], sum_copy) defaultmap(tofrom:scalar)
      {
        for (int i = 0; i < helper_size; ++i) {
          array_copy[i] = helper_harray[i];
        }
        sum_copy = helper_sum;
      }
    }
};

int test_simple_class() {

  OMPVV_INFOMSG("Testing simple class mapping");

  int sum = 0, errors = 0, h_sum = 0;
  int* h_array = new int[N];

  // allocation on the device
  Simple *obj = new Simple(N, &errors);

  obj->modify();

  obj->getDeviceAttributes(h_array, h_sum);
  
  // checking results
  for (int i = 0; i < N; ++i) {
    sum += h_array[i];
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, N*(N-1) != sum);
  OMPVV_TEST_AND_SET_VERBOSE(errors, N*(N-1) != h_sum);

  obj->modify();
  delete obj;
  delete[] h_array;

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET(errors, test_simple_class());

  OMPVV_REPORT_AND_RETURN(errors);
}
