//===--- test_target_update_depend.c ----------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This is a test of the target update construct with the depend clause.
// The test_async_between_hosts_tasks() functions additionatly tests if
// the target enter data and exit data constructs work properly with the
// depend clause. Bits are used for each task in order to determine where
// failure occurs. 
//
////===----------------------------------------------------------------------===//






#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ompvv.h"

#define N 1000
#define HOST_TASK1_BIT 0x1
#define HOST_TASK2_BIT 0x2
#define DEVICE_TASK1_BIT 0x4
#define HOST_TASK3_BIT 0x8
#define ALL_TASKS_BITS 0xF

int test_async_between_hosts_tasks() {
  OMPVV_INFOMSG("test_async_between_hosts_tasks");

  int errors = 0;
  bool isHost = true;
  int sum = 0.0;
  int* h_array = (int *) malloc(N * sizeof(int));
  int* in_1 = (int *) malloc(N * sizeof(int));
  int* in_2 = (int *) malloc(N * sizeof(int));


// We allocate the arrays in the device
#pragma omp target enter data map(alloc: h_array[0:N], in_1[0:N], in_2[0:N]) depend(out: h_array, in_1, in_2)

  // host task 1
#pragma omp task depend(out: in_1) shared(in_1)
  {
    for (int i = 0; i < N; ++i) {
      in_1[i] = HOST_TASK1_BIT; // 0b01
    }
  }

  // host task 2
#pragma omp task depend(out: in_2) shared(in_2)
  {
    for (int i = 0; i < N; ++i) {
      in_2[i] = HOST_TASK2_BIT; // 0b10
    }
  }

 // Testing the update to
#pragma omp target update depend(in: in_1, in_2) depend(out: in_1, in_2) to(in_1[0:N], in_2[0:N])


  // Device task waiting for update
#pragma omp task shared (isHost, h_array, in_1, in_2) depend(inout: h_array) depend(in: in_1) depend(in: in_2)
  {
#pragma omp target map(tofrom: isHost) map(alloc: in_1[0:N]) map(alloc: in_2[0:N]) map(alloc: h_array[0:N])
    {
      isHost = omp_is_initial_device();
      for (int i = 0; i < N; ++i) {
        h_array[i] = DEVICE_TASK1_BIT | in_1[i] | in_2[i]; // Expected = 0b111
      }
    }
  }
 
 // Testing the update from 
#pragma omp target update depend(inout: h_array) from(h_array[0:N])

  // host task 3
#pragma omp task depend(in: h_array) shared(sum, h_array)
  {
    // checking results
    for (int i = 0; i < N; ++i) {
      // Identify which task was problematic
      h_array[i] |= HOST_TASK3_BIT;
      sum += (h_array[i] & ALL_TASKS_BITS); // AND with 0b111 should produce sum
    }
  }
#pragma omp taskwait

  // Garbage collection
#pragma omp target exit data map(delete: h_array[0:N], in_1[0:N], in_2[0:N])

  // We verify all the tasks without a task
  int h_task1 = 0;
  int h_task2 = 0;
  int h_task3 = 0;
  int d_task1 = 0;
  for (int i = 0; i < N; ++i) {
    h_task1 |= !(h_array[i] & HOST_TASK1_BIT);
    h_task2 |= !(h_array[i] & HOST_TASK2_BIT);
    h_task3 |= !(h_array[i] & HOST_TASK3_BIT);
    d_task1 |= !(h_array[i] & DEVICE_TASK1_BIT);
  }
  OMPVV_ERROR_IF(h_task1 != 0, "Error in host task 1");
  OMPVV_ERROR_IF(h_task2 != 0, "Error in host task 2");
  OMPVV_ERROR_IF(h_task3 != 0, "Error in host task 3");
  OMPVV_ERROR_IF(d_task1 != 0, "Error in device task 1");

  OMPVV_TEST_AND_SET(errors, (N * ALL_TASKS_BITS != sum));
  OMPVV_INFOMSG("Test test_async_between_task_target ran on the %s", (isHost ? "host" : "device"));
 
  free(h_array);
  free(in_1);
  free(in_2);

  return errors;
}

/*
 * Test if it is possible to:
 * 3. use nowait for async
 */
int test_async_between_host_and_device() {
  OMPVV_INFOMSG("test_async_between_host_and_device");
  
  int errors = 0;
  bool isHost = true;
  int sum = 0.0;
  int* h_array = (int *) malloc(N * sizeof(int));
  int* in_1 = (int *) malloc(N * sizeof(int));
  int* in_2 = (int *) malloc(N * sizeof(int));


// We allocate the arrays in the device
#pragma omp target enter data map(alloc: h_array[0:N], in_1[0:N], in_2[0:N]) depend(out: h_array, in_1, in_2)

  // host task 1
#pragma omp task depend(out: in_1) shared(in_1)
  {
    for (int i = 0; i < N; ++i) {
      in_1[i] = HOST_TASK1_BIT; // 0b01
    }
  }

  // host task 2
#pragma omp task depend(out: in_2) shared(in_2)
  {
    for (int i = 0; i < N; ++i) {
      in_2[i] = HOST_TASK2_BIT; // 0b10
    }
  }

 // Testing the update to
#pragma omp target update depend(inout: in_1, in_2) to(in_1[0:N], in_2[0:N])


  // Device task waiting for update
#pragma omp target depend(inout: h_array) depend(in: in_1) depend(in: in_2) \
    map(tofrom: isHost) map(alloc: in_1[0:N]) map(alloc: in_2[0:N]) map(alloc: h_array[0:N])
  {
    isHost = omp_is_initial_device();
    for (int i = 0; i < N; ++i) {
      h_array[i] = DEVICE_TASK1_BIT | in_1[i] | in_2[i]; // Expected = 0b111
    }
  }
 
 // Testing the update from 
#pragma omp target update depend(inout: h_array) from(h_array[0:N])

  // host task 3
#pragma omp task depend(in: h_array) shared(sum, h_array)
  {
    // checking results
    for (int i = 0; i < N; ++i) {
      // Identify which task was problematic
      h_array[i] |= HOST_TASK3_BIT;
      sum += (h_array[i] & ALL_TASKS_BITS); // AND with 0b111 should produce sum
    }
  }
#pragma omp taskwait

  // Garbage collection
#pragma omp target exit data map(delete: h_array[0:N], in_1[0:N], in_2[0:N])

  // We verify all the tasks without a task
  int h_task1 = 0;
  int h_task2 = 0;
  int h_task3 = 0;
  int d_task1 = 0;
  for (int i = 0; i < N; ++i) {
    h_task1 |= !(h_array[i] & HOST_TASK1_BIT);
    h_task2 |= !(h_array[i] & HOST_TASK2_BIT);
    h_task3 |= !(h_array[i] & HOST_TASK3_BIT);
    d_task1 |= !(h_array[i] & DEVICE_TASK1_BIT);
  }
  OMPVV_ERROR_IF(h_task1 != 0, "Error in host task 1");
  OMPVV_ERROR_IF(h_task2 != 0, "Error in host task 2");
  OMPVV_ERROR_IF(h_task3 != 0, "Error in host task 3");
  OMPVV_ERROR_IF(d_task1 != 0, "Error in device task 1");

  OMPVV_TEST_AND_SET(errors, (N * ALL_TASKS_BITS != sum));
  OMPVV_INFOMSG("Test test_async_between_task_target ran on the %s", (isHost ? "host" : "device"));
 
  free(h_array);
  free(in_1);
  free(in_2);

  return errors;
}

int main(){
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET(errors, test_async_between_hosts_tasks());
  OMPVV_TEST_AND_SET(errors, test_async_between_host_and_device());

  OMPVV_REPORT_AND_RETURN(errors);
}
