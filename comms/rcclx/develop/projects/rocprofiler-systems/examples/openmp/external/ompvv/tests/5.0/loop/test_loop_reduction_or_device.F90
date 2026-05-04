!===--- test_loop_reduction_or_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a loop directive, testing that the
! variable in the reduction clause is properly reduced using the or
! operator. This test checks the above in a target context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024
#define THRESHOLD 512

PROGRAM test_loop_reduction_or_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_or() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_or()
    LOGICAL,DIMENSION(N):: a
    INTEGER,DIMENSION(N):: num_threads
    DOUBLE PRECISION:: true_margin
    INTEGER:: errors, itr_count, x
    LOGICAL:: tested_true, tested_false
    LOGICAL:: test_result, host_result
    CHARACTER(len=400) :: msgHelper
    INTEGER :: seedSize
    INTEGER,ALLOCATABLE :: seed(:)
    DOUBLE PRECISION:: randomNumber

    errors = 0
    tested_true = .FALSE.
    tested_false = .FALSE.
    itr_count = 0
    ! See the 'and' operator test for
    ! an explanation of this math.
    true_margin = EXP(LOG(0.5)/N)
    CALL random_seed(size=seedSize)
    ALLOCATE(seed(seedSize))
    seed = 1
    CALL random_seed(put=seed)
    DEALLOCATE(seed)

    DO WHILE ( ((.not. tested_true) .or. (.not. tested_false)) .and. (itr_count .lt. THRESHOLD) ) 
       DO x = 1, N
          !random_number() generates a real number, r, uniformly distributed in 0 <= r < 1.
          CALL random_number(randomNumber)
          a(x) = (randomNumber .gt. true_margin)
          num_threads(x) = -1 * x
       END DO

       test_result = .FALSE.
       host_result = .FALSE.

       !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: test_result, a, num_threads)
       !$omp loop reduction(.or.:test_result)
       DO x = 1, N
         test_result = test_result .or. a(x)
       END DO
       !$omp end loop
       !$omp do
       DO x = 1, N
         num_threads(x) = omp_get_num_threads()
       END DO
       !$omp end do
       !$omp end target parallel

       DO x = 1, N 
         host_result = host_result .or. a(x)
       END DO

       IF (itr_count .eq. 0) THEN
          DO x = 2, N
             OMPVV_WARNING_IF(num_threads(x - 1) .ne. num_threads(x), "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.")
          END DO
          OMPVV_WARNING_IF(num_threads(1) .eq. 1, "Test operated with one thread.  Reduction clause cannot be tested.")
          OMPVV_WARNING_IF(num_threads(1) .le. 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.")
       END IF

       OMPVV_TEST_AND_SET_VERBOSE(errors, host_result .neqv. test_result)
       WRITE(msgHelper, *) "Result from loop directive is ", test_result, " but expected result is ", host_result, "." 
       OMPVV_ERROR_IF(host_result .neqv. test_result, msgHelper)

       IF (host_result) THEN
          tested_true = .TRUE.
       ELSE
          tested_false = .TRUE.
       END IF

       IF (host_result .neqv. test_result) THEN
          exit
       END IF

       itr_count = itr_count + 1
    END DO

    OMPVV_WARNING_IF(.not. tested_true, "Did not test a case in which final result was true.")
    OMPVV_WARNING_IF(.not. tested_false, "Did not test a case in which final result was false.")

    test_or = errors
  END FUNCTION test_or
END PROGRAM test_loop_reduction_or_device
