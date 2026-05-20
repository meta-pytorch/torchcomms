//===-- gemv_target_many_matrices.cpp - GEMV on many matrices in parallel ---===!
// 
// OpenMP API Version 4.5 Nov 2015
//
// Taken from the suggestions of the QMCPack team. This test uses a target teams
// distribute combined construct Additionally, it spawns multiple target 
// regions in parallel. Each on a different matrix
//!===----------------------------------------------------------------------===!
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include "ompvv.h"
#include "omp.h"

#define N 4096
#define NUM_CALC 8

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
  #pragma omp target teams distribute map(to:A[:n*n], V[:n]) map(from:Vout[:n])
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
  #pragma omp target enter data map(to:ptr[:n])
  return ptr;
}

template <class T>
void deallocate(T* ptr, size_t n)
{
  #pragma omp target exit data map(delete:ptr[:n])
  delete[] ptr;
}

int main()
{
  OMPVV_TEST_OFFLOADING;
  std::vector<float*> manyA;
  std::vector<float*> manyV;
  std::vector<float*> manyVout;
  int errors = 0;

  // Initializing matrices 
  for(int i=0; i < NUM_CALC; i++)
  {
    manyA.push_back(allocate<float>(N*N));
    manyV.push_back(allocate<float>(N));
    manyVout.push_back(allocate<float>(N));
  }

  // Doing the computation
  {
    Timer local("GEMV");
    #pragma omp parallel for
    for(int i=0; i < NUM_CALC; i++) {
      gemv(N, 1.0f, manyA[i], manyV[i], manyVout[i]);
    }
  }

  for(int i=0; i < NUM_CALC; i++) {
    auto*  Vout = manyVout[i];
    #pragma omp target update from(Vout[:N])
    for(int i=0; i<N; i++)
    {
      OMPVV_TEST_AND_SET(errors, std::abs(Vout[i]-N) > 0.0001);
      OMPVV_ERROR_IF( std::abs(Vout[i]-N) > 0.0001, "Error for Vout[%d] = %f, Should be %d", i, Vout[i], N);
    }
    deallocate(manyA[i], N*N);
    deallocate(manyV[i], N);
    deallocate(manyVout[i], N);
  }

  OMPVV_REPORT_AND_RETURN(errors);
}
