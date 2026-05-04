!===--- test_loop_collapse_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the collapse clause with the loop directive and tests that
! for loops out of the scope of the collapsed loops are not parallelized.
! This test tests using one and two collapsed loops. This test checks these
! features in an offloading (target) context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

! Array Size of 128 uses 16MB target memory and
! scales n^3 in test_collapse2()
#define N 128

PROGRAM test_loop_collapse_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_collapse1() .ne. 0)
  OMPVV_TEST_VERBOSE(test_collapse2() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_collapse1()
    INTEGER,DIMENSION(N,N):: a
    INTEGER,DIMENSION(N+1,N):: b
    INTEGER:: errors, x, y, temp_total

    errors = 0

    ! a and b array initialization
    DO x = 1, N
       b(1,x) = 0
       DO y = 1, N
          a(y,x) = x + y
          b(y+1,x) = 0 
       END DO
    END DO

    !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: a, b)
    !$omp loop collapse(1)
    DO x = 1, N
       DO y = 1, N
          b(y+1,x) = b(y,x) + a(y,x)
       END DO
    END DO
    !$omp end loop
    !$omp end target parallel

    DO x = 1, N
       temp_total = 0
       DO y = 1, N+1
          OMPVV_TEST_AND_SET(errors, (temp_total - b(y,x)) .ne. 0)
          IF ( y .ne. N+1 ) THEN
             temp_total = temp_total + a(y,x)
          END IF
       END DO
    END DO

    test_collapse1 = errors
  END FUNCTION test_collapse1

  INTEGER FUNCTION test_collapse2()
    INTEGER,DIMENSION(N,N,N):: a
    INTEGER,DIMENSION(N+1,N,N):: b
    INTEGER:: errors, num_threads, x, y, z, temp_total

    errors = 0
    num_threads = 0

    ! a and b array initialization
    DO x = 1, N
       DO y = 1, N
          b(1,y,x) = 0
          DO z = 1, N
             a(z,y,x) = x + y + z
             b(z+1,y,x) = 0 
          END DO
       END DO
    END DO

    !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: a, b, num_threads)
    IF (omp_get_thread_num() .eq. 0) THEN
       num_threads = omp_get_num_threads()
    END IF
    !$omp loop collapse(2)
    DO x = 1, N
       DO y = 1, N
          DO z = 1, N
             b(z+1,y,x) = b(z,y,x) + a(z,y,x)
          END DO
       END DO
    END DO
    !$omp end loop
    !$omp end target parallel

    DO x = 1, N
       DO y = 1, N
          temp_total = 0
          DO z = 1, N+1
             OMPVV_TEST_AND_SET(errors, (temp_total - b(z,y,x)) .ne. 0)
             IF ( z .ne. N+1 ) THEN
                temp_total = temp_total + a(z,y,x)
             END IF
          END DO
       END DO
    END DO

    IF (num_threads == 1) THEN
       OMPVV_WARNING("Test operated with one thread.  Parallelism of loop directive in parallel region can't be guaranteed.")
    END IF
    test_collapse2 = errors
  END FUNCTION test_collapse2
END PROGRAM test_loop_collapse_device
