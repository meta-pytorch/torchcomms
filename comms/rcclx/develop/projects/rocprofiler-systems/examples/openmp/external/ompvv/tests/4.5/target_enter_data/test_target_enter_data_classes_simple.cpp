//===---- test_target_enter_data_classes_simple.c -- target enter data with classes===//
//
// OpenMP API Version 4.5 Nov 2015
//
//
// This test checks for interoperability between C++ classes and device offloading through the use of 
// target enter and exit data clauses. It considers that 4.5 does not support
// mapping of attributes directly, as the implicit use of the this-> pointer when 
// using attributes inside the target region is restrictive.
//
// The description of the map clause says that the map clause receives a list item:
//
// Section 2.15.5.1, page 216, line 17
//
// The syntax of the map clause is as follows:
// map([ [map-type-modifier[,]] map-type : ] list)
// And the definition of list item is
//
// Section 2.1, page 27, line 20
//
// A list item is a variable or array section. An extended list item is a list item or a function name.
//
// This test creates a class that, during construction, it maps an attribute through helper 
// variables that remove the direct use to the attributes. And during destruction of the 
// object it maps the data back to the devices. Additionally, there is a modifier 
// method that uses values from the class indirectly through the use of helper references
// finally there is a synchronization  clause that will obtain the values on demand
//===----------------------------------------------------------------------------------===//
//
//

#include <iostream>
#include <omp.h>
#include <cassert>
#include "ompvv.h"
#include <cmath>

#define N 1000

class Simple {
private:
  int *d_array;
  int size;
  int sum;

public:
  // Constructor. Maps the data into the device
  Simple(int s) : size(s) { 
    this->sum = 0;
    this->d_array = new int[size];
    // Initialize array 
    std::fill(d_array, d_array+size, 0);

    // Removing the direct use of attributes to avoid problems
    // with 4.5 specifications 
    int* helper = d_array;
    int &hs = size;
    int &hsum = sum;
#pragma omp target enter data map(to: helper[0:hs]) map(to: hs) map(to:hsum)
  }

  // Destructor, removes the data from the device
  ~Simple() { 
    // Removing the direct use of attributes to avoid problems
    // with 4.5 specifications 
    int* helper = d_array;
    int &hs = size;
    int &hsum = sum;
#pragma omp target exit data map(delete: helper[0:hs]) map(delete: hs) map(delete: hsum)
    delete[] d_array; 
  }
  
  // Modify the device data directly
  void modify() {
    // Removing the direct use of attributes to avoid problems
    // with 4.5 specifications 
    int *helper = d_array;
    int &hsize = size;
    int &hsum = sum;
#pragma omp target map(alloc:hsum, hsize) 
    {
      hsum = 0;
      for (int i = 0; i < hsize; ++i) {
        helper[i] += 1;
        hsum += helper[i];
      }
    }
  }

  // Get the values from the device through a second array
  void getValues(int &h_sum, int* h_array) {
    int* helper = d_array;
    int &hsize = size;
    int &help_sum = sum;
#pragma omp target map(from: h_array[0:hsize]) map(alloc: help_sum, hsize) map(from:h_sum)
    {
      h_sum = help_sum;
      for (int i = 0; i < hsize; i++)
        h_array[i] = helper[i];
    }
  }
};

int test_simple_class() {
  
  OMPVV_INFOMSG("Testing enter exit data with a simple class");
  int errors = 0, h_sum = 0, sum = 0;
  int* h_array = new int[N];

  // allocation on the device
  Simple *obj = new Simple(N);

  obj->modify();
  obj->modify();
  obj->modify();
  obj->getValues(h_sum, h_array);

  // checking results
  for (int i = 0; i < N; ++i) {
    sum += h_array[i];
  }

  delete obj;
  delete[] h_array;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, (3*N != sum));
  OMPVV_TEST_AND_SET_VERBOSE(errors, (3*N != h_sum));
  OMPVV_ERROR_IF(errors != 0, "N = %d, sum = %d, h_sum = %d", N, sum, h_sum);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_simple_class());
  
  OMPVV_REPORT_AND_RETURN(errors)
}

