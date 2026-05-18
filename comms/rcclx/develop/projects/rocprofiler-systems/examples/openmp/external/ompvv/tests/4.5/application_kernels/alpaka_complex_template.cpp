//===--- alpaka_complex_template.cpp ----------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This is a stripped-down application code which sets up two template
// structs which use standard library functions. A templated struct
// containing the other struct is declared on the device. This test does
// not check any results as it is only a test of declaring a templated
// struct on the device. This test was provided by Jonas Hahnfeld from
// https://bugs.llvm.org/show_bug.cgi?id=43771.
//
////===----------------------------------------------------------------------===//

#include <type_traits>
#include "omp.h"
#include "ompvv.h"

template<int Dim> struct V {
  int version_called;

  template<bool B = (Dim == 0),
           typename = typename std::enable_if<B>::type>
  V() {
    version_called = 1;
  }

  template<typename TArg0,
           typename = typename std::enable_if<(std::is_same<unsigned long,
                                               typename std::decay<TArg0>::type>::value)>::type>
  V(TArg0 && arg0) {
    version_called = 2;
  }
};

template<int Dim> struct S {
  V<Dim> v;
};

int main(int argc, char *argv[]) {
  OMPVV_TEST_OFFLOADING;

  OMPVV_TEST_SHARED_ENVIRONMENT;

  int errors = 0;
  int version_set[2] = {-1};

#pragma omp target map(from: version_set[0:2])
  {
    S<0> s;
    version_set[0] = s.v.version_called;
    V<1> v2((unsigned long) 1);
    version_set[1] = v2.version_called;
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, version_set[0] != 1);
  OMPVV_TEST_AND_SET_VERBOSE(errors, version_set[1] != 2);

  OMPVV_REPORT_AND_RETURN(errors);
}
