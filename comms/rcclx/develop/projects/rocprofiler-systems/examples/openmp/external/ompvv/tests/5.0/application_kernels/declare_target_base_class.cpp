//===--- declare_target_base_class.cpp --------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
//
// This test was suggested by members of NERSC. This test defines a declare
// target region which includes only a base class and a 'concrete' device
// pointer. 
// 
// Test suggestion comes from Chris Daily and Rahulkumar Gayatri from NERSC
////===----------------------------------------------------------------------===//

#include <new>
#include <vector>
#include <iostream>
#include "ompvv.h"
#include <omp.h>

#pragma omp declare target
class S {
public:
  S() : _devPtr(nullptr) {}
  double sag(double x, double y) {
    return x + y;
  }
  S* cloneToDevice() {
    S* ptr;
#pragma omp target map(ptr)
    {
      ptr = new S();
    }
    _devPtr = ptr;
    return ptr;
  }
private:
  S* _devPtr;
};
#pragma omp end declare target

int main() {
  int errors = 0;
  
  OMPVV_TEST_OFFLOADING;

  S s;
  S* devPtr = s.cloneToDevice();

  std::vector<double> in(10, 0.0);
  for(int i = 0; i < 10; i++) {
    in[i] = i;
  }

  std::vector<double> out(10, 0.0);

  double* inptr = in.data();
  double* outptr = out.data();

#pragma omp target teams distribute parallel for map(inptr[:10], outptr[:10]) is_device_ptr(devPtr)
  for(int i = 0; i < 10; i++) {
    outptr[i] = devPtr->sag(inptr[i], inptr[i]);
  }

  for(int i = 0; i < 10; i++) {
    OMPVV_TEST_AND_SET(errors, out[i] != i*2)
  }

  OMPVV_REPORT_AND_RETURN(errors);
}
