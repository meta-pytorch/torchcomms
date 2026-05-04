//===--  test_target_OpertrOvrld.cpp-------------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) Testing operator overloading on gpu
//===----------------------------------------------------------------------===//

#include <iostream>
#include <omp.h>
#include "ompvv.h"

using namespace std;

#pragma omp requires unified_shared_memory
class Fraction {
  int num;
  int denom;
  public:
    Fraction(int num, int denom) {
      this->num = num;
      this->denom = denom;
    }

  Fraction operator+(Fraction const &f1) {
    int lcm = denom * f1.denom;
    int x = lcm/denom;
    int y = lcm/f1.denom;
    int Rnum = x*num + y*f1.num;
    int Rdenom = lcm;
    int gcd = 1;
    for (int i = 1; i <= Rnum; ++i) {
      if (((Rnum % i) == 0) && ((Rdenom % i) == 0)) {
        gcd = i;
      }
    }

  Fraction New(Rnum/gcd, Rdenom/gcd);
    return New;
  }

  Fraction operator*(Fraction const &f1) {
    int Nnum = num * f1.num;
    int Ndenom = denom * f1.denom;
    int gcd = 1;
    for (int i = 1; i <= Nnum; ++i) {
      if (((Nnum % i) == 0) && ((Ndenom % i) == 0)) {
         gcd = i;
      }
    }
  Fraction New(Nnum/gcd, Ndenom/gcd);
    return New;
  }

  pair<int, int> GetVals() {
    pair<int, int> p(num, denom);
    return p;
  }
};


int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  int Errs = 0;
#pragma omp target map(tofrom: Errs)
  {
    Fraction f1(2, 3);
    Fraction f2(3, 4);

    Fraction f4 = f1 + f2;
    pair<int, int> p1 = f4.GetVals();
    // Verifying the result 2/3 + 3/4 = 17/12
    if (!((p1.first == 17) && (p1.second == 12))) {
      Errs++;
    }

    Fraction f5 = f1 * f2;
    pair<int, int> p2 = f5.GetVals();
    // Verifying the result 2/3 * 3/4 = 1/2
    if (!((p2.first == 1) && (p2.second == 2))) {
      Errs++;
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, (Errs != 0));

  OMPVV_REPORT_AND_RETURN(errors);
}
