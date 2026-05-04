//===---- test_target_enter_data_classes_inheritance.c - enter data and classes-===//
//
// OpenMP API Version 4.5 Nov 2015
//
//
// This test checks for interoperability between C++ classes and device 
// offloading in the presence of inheritance and templates, and when using 
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
// A list item is a variable or array section. An extended list item 
// is a list item or a function name.
//
// This test creates a base class that, during construction, it maps an 
// attribute through helper 
// variables that remove the direct use to the attributes. And during 
// destruction of the 
//
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

template<typename T>
class Mapper {
private:
  T* ptr;
  bool not_mapped;
public:
  Mapper (T* p) : ptr(p) {
    not_mapped = !omp_target_is_present(ptr, omp_get_default_device());
    T* solutionPtr = ptr;
    // maps if target is not present
#pragma omp target enter data map(to:solutionPtr[0:1]) if(not_mapped)
  }
  ~Mapper() {
    T* solutionPtr = ptr;
    // unmaps iff this mapper mapped the target
#pragma omp target exit data map(delete: solutionPtr[0:1]) if(not_mapped)
    ptr = NULL;
  }
};

class B : public Mapper<B> {
protected:
  int n;
private:
  double* x;
  double sumB; 

public:
  B(int nn) : Mapper<B>(this), n(nn) {
    x = new double[n];
    std::fill(x, x+n, 0);
    // This is a work around to avoid referring to 
    // the class members in the map() clause. see
    // description
    double* solutionX = x;
    int &cpy_n = n;
#pragma omp target enter data map(to:solutionX[0:n], cpy_n)
  }

  void modifyB() {
    double * cpy_x = x; 
    int &cpy_n = n;
    double &cpy_sumB = sumB;
#pragma omp target defaultmap(tofrom: scalar)
    {
      sumB = 0.0;
      for (int i = 0; i < cpy_n; ++i) {
        cpy_x[i] += 1.0;
        sumB += cpy_x[i];
      }
    } 
  }

  void getValuesB(double& b_sum, double* b_array) {
    double * cpy_x = x;
    int &cpy_n = n;
    double &cpy_sum = sumB;
#pragma omp target defaultmap(tofrom: scalar) map(from: b_array[0:n])
    {
      b_sum = cpy_sum;
      for (int i = 0; i < cpy_n; i++)
        b_array[i] = cpy_x[i];
    }
  }

};

class A : public Mapper<A>, public B {
private:
  int sumA; 
  int* y;

public:
  A(int s) : Mapper<A>(this), B(s){ 
    y = new int[n];
    std::fill(y, y+n, 0);
    int *solutionY = y;
    int &cpy_n = n;
    #pragma omp target update to(cpy_n)
    #pragma omp target enter data map(to:solutionY[0:n])
  }

  void modifyA() {
    modifyB();
    int *cpy_y = y;
    int &cpy_n = n;
    int &cpy_sumA = sumA;

#pragma omp target defaultmap(tofrom: scalar)
    {
      cpy_sumA = 0;
      for (int i = 0; i < cpy_n; ++i) {
        cpy_y[i] += 1;
        cpy_sumA += cpy_y[i];
      }
    }
  }

  void getValuesA(int &a_sum, int* a_array, double& b_sum, double* b_array) {
    getValuesB(b_sum, b_array);
    int* cpy_y = y;
    int &cpy_n = n;
    int &cpy_sumA = sumA;
#pragma omp target defaultmap(tofrom:scalar) map(tofrom: a_array[0:n])
    {
      a_sum = cpy_sumA;
      for (int i = 0; i < cpy_n; i++) {
        a_array[i] = cpy_y[i];
      }
    }
  }
  
};

int test_complex_class() {
  OMPVV_INFOMSG("test_complex_class");

  int sumY = 0, errors = 0, check_sumY = 0;
  double sumX = 0.0, check_sumX = 0.0;
  double * h_array_x = new double[N];
  int * h_array_y = new int[N];

  // allocation on the device
  A *obj = new A(N);

  // Each modify adds 1 to the array that is already mappend in the constructor
  obj->modifyA();
  obj->modifyA();
  obj->modifyA();
  obj->getValuesA(sumY, h_array_y, sumX, h_array_x);

  // checking results
  for (int i = 0; i < N; ++i) {
    check_sumY += h_array_y[i];
    check_sumX += h_array_x[i];
  }

  delete obj;
  delete[] h_array_x;
  delete[] h_array_y;
  
  OMPVV_TEST_AND_SET(errors, check_sumY != 3*N);
  OMPVV_TEST_AND_SET(errors, sumY != 3*N);
  OMPVV_TEST_AND_SET(errors, std::abs(check_sumX - 3*N) > 0.00001);
  OMPVV_TEST_AND_SET(errors, std::abs(sumX - 3*N) > 0.00001);
  OMPVV_ERROR_IF(errors != 0, "N = %d, sumX = %f, check_sumX = %f, sumY=%d, check_sumY = %d", N, sumX, check_sumX, sumY, check_sumY);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_complex_class());
  
  OMPVV_REPORT_AND_RETURN(errors)
}

