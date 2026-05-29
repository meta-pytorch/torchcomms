//===-- gemv_target.cpp - General Matrix Vector Multiplication target -------===!
//
// OpenMP API Version 4.5 Nov 2015
//
// Taken from the suggestions of the QMCPack team. This test has a simple
// target region that performs a GEMV
//!===----------------------------------------------------------------------===!
#include <chrono>
#include <string>
#include <cmath>
#include "ompvv.h"
#include "omp.h"

#define N 8192

class Timer
{
  const std::chrono::time_point<std::chrono::system_clock> start;
  const std::string name;

public:
  Timer(const std::string& name_in): start(std::chrono::system_clock::now()), name(name_in) {};
  ~Timer()
  {
    auto end = std::chrono::system_clock::now();
    OMPVV_INFOMSG("Function %s takes %lf us ",name.c_str(), std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count());
  }
};

template <typename T>
void gemv(int n, T alpha, const T*  A, const T*  V, T*  Vout)
{
  #pragma omp target map(to:A[:n*n], V[:n]) map(from:Vout[:n])
  for(int row=0; row<n; row++)
  {
    T sum = T(0);
    const T * A_row = A+row*n;
    for(int col=0; col<n; col++) {
      sum += A_row[col]*V[col];
    }
    Vout[row] = sum*alpha;
  }
}

template <class T>
T* allocate(size_t n)
{
  T* ptr = new T[n];
  std::fill_n(ptr, n, T(1));
  return ptr;
}

template <class T>
void deallocate(T* ptr, size_t n)
{
  delete[] ptr;
}

int main()
{
  OMPVV_TEST_OFFLOADING;
  auto* A = allocate<float>(N*N);
  auto* V = allocate<float>(N);
  auto* Vout = allocate<float>(N);
  int errors = 0;

  {
    Timer local("GEMV");
    gemv(N, 1.0f, A, V, Vout);
  }

  for(int i=0; i<N; i++)
  {
    OMPVV_TEST_AND_SET(errors, std::abs(Vout[i]-N) > 0.0001);
    OMPVV_ERROR_IF(std::abs(Vout[i]-N) > 0.0001, "Error for Vout[%d] = %f, Should be %d", i, Vout[i], N);
  }

  deallocate(A, N*N);
  deallocate(V, N);
  deallocate(Vout, N);
  OMPVV_REPORT_AND_RETURN(errors);
}
