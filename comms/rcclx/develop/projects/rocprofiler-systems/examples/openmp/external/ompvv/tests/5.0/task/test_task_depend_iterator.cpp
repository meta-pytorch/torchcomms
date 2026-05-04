//===-- test_task_depend_iterator.cpp - test of the iterator clause inside the task depend clause ----===//
// 
// OpenMP API Version 5.0 Nov 2020
// 
// This file is a test for the iterator modifier when used with the task depend
// clause. This modifier should create an iterator that expand into multiple values
// inside the clause they appear. In this particular test case the iterator expands into
// several values creating several dependencies at the same time.
//
//===-------------------------------------------------------------------------------------------------===//

#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include "ompvv.h"

int test_task_depend_iterator() {
  int ptr[] = {0, 4, 5, 6, 7, 8, 9, 10, 11}, cols[] = {1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8};
  std::vector<int> threadOrder;
  bool threadOrderError = false;
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
  {
#pragma omp single
    {
      for (int i = 0; i < 8; ++i) {
        int pos = ptr[i], size = ptr[i + 1] - ptr[i];
#pragma omp task depend(iterator(it = 0 : size), in : ptr[cols[pos + it]]) depend(out : ptr[i])
        {
#pragma omp critical
          {
            threadOrder.push_back(i);
          } // end critical section
        } // end task depend
      }
    } // end single
  } // end parallel

  std::vector<int>::iterator idx[8];
  for (int i = 0; i < 8; ++i)
    idx[i] = std::find (threadOrder.begin(), threadOrder.end(), i);
  if (idx[0] != threadOrder.begin())
    threadOrderError |= true;
  if (idx[1] > idx[5] || idx[2] > idx[5])
    threadOrderError |= true;
  if (idx[3] > idx[6] || idx[4] > idx[6])
    threadOrderError |= true;
  if (idx[5] > idx[7] || idx[5] > idx[7])
    threadOrderError |= true;

  std::sort(threadOrder.begin(), threadOrder.end());
  for(int i = 0; i < 8; ++i)
    threadOrderError = (threadOrder[i] != i) || threadOrderError;
  OMPVV_ERROR_IF(threadOrderError, "The dependencies between tasks were not enforced in the correct order.");
  return threadOrderError ? 1 : 0;
}


int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_depend_iterator());
  OMPVV_REPORT_AND_RETURN(errors);
}
