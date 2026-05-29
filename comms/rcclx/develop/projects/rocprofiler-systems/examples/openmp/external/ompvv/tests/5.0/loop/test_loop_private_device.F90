!===--- test_loop_private_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the private clause on a loop directive to indicate that the
! variable in the private clause should be made private to each thread
! executing the loop region.  The test then operates on the privatized
! variable in such a way that would most likely cause competing operations
! if the variable is not privatized.  If the computation completes without
! errors, we assume that the privatization occured. This test checks the
! above in a target context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define NSIZE 1024

PROGRAM test_loop_private_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  INTEGER,DIMENSION(NSIZE):: a, b, c, d
  INTEGER:: privatized, num_threads, x, y

  OMPVV_TEST_OFFLOADING

  num_threads = -1

  DO x = 1, NSIZE
     a(x) = 1
     b(x) = x
     c(x) = 2*x
     d(x) = 0
  END DO

  !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: a, b, c, d, num_threads)
  !$omp loop private(privatized)
  DO x = 1, NSIZE
     privatized = 0
     DO y = 1, a(x) + b(x)
        privatized = privatized + 1 
     END DO
     d(x) = c(x) * privatized
  END DO
  !$omp end loop
  IF (omp_get_thread_num() .eq. 0 ) THEN
     num_threads = omp_get_num_threads()
  END IF
  !$omp end target parallel

  DO x = 1, NSIZE
     OMPVV_TEST_VERBOSE(d(x) .ne. (1 + x)*2*x)
     IF (d(x) .ne. (1 + x)*2*x) THEN
        exit
     END IF
  END DO

  OMPVV_WARNING_IF(num_threads .eq. 1, "Test ran with one thread. Results of private test are inconclusive.")
  OMPVV_TEST_VERBOSE(num_threads .lt. 1)
  OMPVV_REPORT_AND_RETURN()
END PROGRAM test_loop_private_device
