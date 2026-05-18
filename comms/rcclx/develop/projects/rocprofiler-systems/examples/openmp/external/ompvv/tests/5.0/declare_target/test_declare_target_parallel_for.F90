!===--- test_declare_target_parallel_for.F90 -----------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Test of a function that is declared target and contains a parallel for
! construct. The function performs simple array operations on the device
! which are then checked for correctness on the host. The number of
! threads used to run the parallel for is also checked.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_declare_target_parallel_for
    USE iso_fortran_env
    USE ompvv_lib
    USE omp_lib
    implicit none
    
    !$omp declare target to(parallel_for_fun)
    
    OMPVV_TEST_OFFLOADING

    OMPVV_TEST_VERBOSE(test_parallel_for() .NE. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS  
    INTEGER FUNCTION parallel_for_fun(a, b, c)
      INTEGER :: a(:)
      INTEGER :: b(:)
      INTEGER :: c(:)
      INTEGER :: i
      INTEGER :: num_threads = -1
      !$omp declare target to(parallel_for_fun)
      DO i = 1, N
        a(i) = b(i) * c(i)
        IF (omp_get_thread_num() .EQ. 0) THEN
          num_threads = omp_get_num_threads()
        END IF
      END DO
      parallel_for_fun = num_threads
    END FUNCTION parallel_for_fun

    INTEGER FUNCTION test_parallel_for() 
      INTEGER :: errors = 0
      INTEGER, DIMENSION(N) :: x, y, z
      INTEGER :: num_threads = -1
      INTEGER :: i 

      OMPVV_INFOMSG("test_declare_target_parallel_for")

      DO i = 1, N
        x(i) = 0
        y(i) = 1
        z(i) = i
      END DO

      !$omp target map(tofrom: x, y, z, num_threads)
        num_threads = parallel_for_fun(x, y, z)
      !$omp end target

      DO i = 1, N
        OMPVV_TEST_AND_SET_VERBOSE(errors, x(i) .NE. y(i)*z(i))
      END DO

      OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .LT. 1)
      OMPVV_ERROR_IF(num_threads .LT. 1, "Device returned invalid number of threads.")
      OMPVV_WARNING_IF(num_threads .EQ. 1, "Device ran target function with parallel for using one thread, so parallelism cannot be guaranteed.")

      test_parallel_for= errors
    END FUNCTION test_parallel_for

END PROGRAM test_declare_target_parallel_for
