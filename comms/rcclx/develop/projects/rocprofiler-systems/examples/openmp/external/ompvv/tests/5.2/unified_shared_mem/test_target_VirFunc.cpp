//===-- test_target_VirFunc.cpp -------------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) Testing pure virtual function on gpu
//===----------------------------------------------------------------------===//

#include <iostream>
#include <omp.h>
#include "ompvv.h"


#pragma omp requires unified_shared_memory

class Base {
  protected:
    int x;
  public:
    virtual int GetX() = 0;
    Base(int val) {
      x = val;
    }
};

class Derived: public Base {
  int y;
  public:
    Derived(int m, int n):Base(m) {
      y = n;
    }
    int GetX() {
      return x;
    }
};

int main() {
  OMPVV_TEST_OFFLOADING;
  int *Val = nullptr, errors = 0;
  Val = reinterpret_cast<int *>(omp_target_alloc(sizeof(int), 0));
#pragma omp target
  {
    Derived Dobj(1, 2);
    *Val = Dobj.GetX();
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, (*Val != 1));
  // Val value
  *Val = 0;
  // object creation using pointer of base class
#pragma omp target
  {
    Base *ptr = new Derived(10, 20);
    *Val = ptr->GetX();
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (*Val != 10));
  omp_target_free(Val, 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
