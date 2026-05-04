//===--test_targe_enter_exit_data_classes_complex.c - test inheritance class -===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
//
// This test uses target enter data and exit data to map multiple levels of 
// inheritance in classes and templates. The target enter and exit data are used
// in the constructor and destructor of all the levels of the hierarchy. 
// It requires the use of helper_ variables since it
// is not legal to use the "this" pointer (implicitly or explicitly) in 4.5. 
//
// We use a modify method, which is propagated through the hierarchy of classes,
// to assign values to the array. Following we use a method to obtain a copy of the values 
// from the device. The disctintion between methods allows to show that
// data is mapped and remains mapped in the device memory. This test does
// not use any inheritance or anything similar, therefore the simple connotation
//
// Contrary to the test of target_enter_data, this test checks if during the exit
// data it successfully copies back the attributes of the object
////===----------------------------------------------------------------------===//

#include <iostream>
#include <typeinfo>
#include <omp.h>
#include <cmath>
#include "ompvv.h"


using namespace std;

#define N 1000

template<typename T>
class Mapper {

  private:
    T* ptr;
    bool not_mapped;

  public:
    Mapper (T* p) : ptr(p) {
      not_mapped = !omp_target_is_present(ptr, omp_get_default_device());
      // maps if target is not present
      T* helper_ptr = ptr;
      OMPVV_INFOMSG_IF(not_mapped, "Mapping class %s", typeid(T).name());
#pragma omp target enter data map(to: helper_ptr[0:1]) if(not_mapped)
    }
    ~Mapper() {
      // unmaps iff this mapper mapped the target
      T* helper_ptr = ptr;
      OMPVV_INFOMSG_IF(not_mapped, "Unmapping class %s", typeid(T).name());
#pragma omp target exit data map(delete: helper_ptr[0:1]) if(not_mapped)
      ptr = NULL;
    }
};

class B : public Mapper<B> {

  protected:
    int n;
    double* x;
    int* errors;

  public:
    B(int n, int* err) : Mapper<B>(this), n(n), errors(err) {
      x = new double[n];
      for (int i = 0; i < n; i ++) {
        x[i] = (double) i;
      }
      int &helper_n = this->n;
      double *helper_x = this->x;

      OMPVV_INFOMSG("Mapping B attributes");
#pragma omp target update to(helper_n)
#pragma omp target enter data map(to:helper_x[0:n])
    }

    ~B() {
      double *helper_x = this->x;
      OMPVV_INFOMSG("Unmapping B attributes");
#pragma omp target exit data map(from:helper_x[0:n])
      // checking results
      for (int i = 0; i < N; ++i) {
        OMPVV_TEST_AND_SET_VERBOSE(*errors, std::abs(x[i] - (double)(3*i)) > 0.0001);
      }
    }

    void modifyB() {
      OMPVV_INFOMSG("Modifying B");
      int &helper_n = this->n;
      double *helper_x = this->x;
#pragma omp target defaultmap(tofrom:scalar)
      {
        for (int i = 0; i < helper_n; ++i)
          helper_x[i] += (double) i;
      } 
    }
};

class A : public Mapper<A>, public B {
  private:
    int* y;

  public:
    A(int s, int* err) : Mapper<A>(this), B(s, err) { 
      OMPVV_INFOMSG("Mapping A attributes");
      y = new int[n];
      // Initialize y
      for (int i = 0; i < n; i ++) {
        y[i] = i;
      }
      int *helper_y = this->y;
#pragma omp target enter data map(to: helper_y[0:n])
    }
    ~A() {
      OMPVV_INFOMSG("Unmapping A attributes");
      int *helper_y = this->y;
#pragma omp target exit data map(from: helper_y[0:n])
      // checking results
      for (int i = 0; i < N; ++i) {
        OMPVV_TEST_AND_SET_VERBOSE(*errors, y[i] != 3*i);
      }
    }

    void modifyA() {
      modifyB();
      OMPVV_INFOMSG("Modifying A");
      int &helper_n = this->n;      
      int *helper_y = this->y;
#pragma omp target defaultmap(tofrom:scalar)
      {
        for (int i = 0; i < helper_n; ++i) {
          helper_y[i] += i;
        }
      }
    }

    void getAllAttributesDevice(double * copy_x, int * copy_y) {
      int &helper_n = this->n;      
      int *helper_y = this->y;
      double *helper_x = this->x;
#pragma omp target defaultmap(tofrom:scalar) map(from:copy_x[0:n], copy_y[0:n])
      {
        for (int i = 0; i < helper_n; ++i) {
          copy_x[i] = helper_x[i];
          copy_y[i] = helper_y[i];
        }
      }

    }

};


int test_complex_class() {
  OMPVV_INFOMSG("Testing complex class");
  int sumY = 0, errors = 0;
  double sumX = 0.0;
  int *h_y = new int[N];
  double *h_x = new double[N];
  // allocation on the device
  A *obj = new A(N, &errors);

  obj->modifyA();

  obj->getAllAttributesDevice(h_x, h_y);

  // checking results
  for (int i = 0; i < N; ++i) {
    sumY += h_y[i];
    sumX += h_x[i];
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, N*(N-1) != sumY); 
  OMPVV_TEST_AND_SET_VERBOSE(errors, std::abs(sumX - (double) N*(N-1))>0.0001);

  obj->modifyA();

  delete obj;
  delete[] h_x;
  delete[] h_y;

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET(errors, test_complex_class());

  OMPVV_REPORT_AND_RETURN(errors);
}
