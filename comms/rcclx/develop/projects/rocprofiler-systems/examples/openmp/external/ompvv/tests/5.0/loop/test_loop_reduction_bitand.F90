!===--- test_loop_reduction_bitand.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the reduction clause on a loop directive, testing that the
! variable in the reduction clause is properly reduced using the bitand
! operator.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024
#define THRESHOLD 512

PROGRAM test_loop_reduction_bitand
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_bitand() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_bitand()
    INTEGER,DIMENSION(N):: a
    INTEGER,DIMENSION(N):: num_threads
    DOUBLE PRECISION:: false_margin
    INTEGER:: b, host_b
    INTEGER:: errors, num_attempts, x, y
    LOGICAL:: have_true, have_false
    CHARACTER(len=400) :: msgHelper
    INTEGER :: seedSize
    INTEGER,ALLOCATABLE :: seed(:)
    DOUBLE PRECISION:: randomNumber

    errors = 0
    num_attempts = 0
    have_true = .FALSE.
    have_false = .FALSE.
    ! See the 'and' operator test for an exaplantion of this math.
    false_margin = EXP(LOG(0.5)/N)
    CALL random_seed(size=seedSize)
    ALLOCATE(seed(seedSize))
    seed = 1
    CALL random_seed(put=seed)
    DEALLOCATE(seed)

    DO WHILE ( ((.not. have_true) .or. (.not. have_false)) .and. (num_attempts .lt. THRESHOLD) ) 
       have_true = .FALSE.
       have_false = .FALSE.
       DO x = 1, N
          IF (num_attempts .eq. 0) THEN
             a(x) = 0
          END IF
          DO y = 0, 15
             !random_number() generates a real number, r, uniformly distributed in 0 <= r < 1.
             CALL random_number(randomNumber)
             IF (randomNumber .lt. false_margin) THEN
                a(x) = a(x) + ISHFT(1, y)
                have_true = .TRUE.
             ELSE
                have_false = .TRUE.
             END IF
          END DO
          num_threads(x) = -1 * x
       END DO
       num_attempts = num_attempts + 1
    END DO

    OMPVV_WARNING_IF(.not. have_true, "No true bits were generated to test")
    OMPVV_WARNING_IF(.not. have_false, "No false bits were generated to test")

    b = 0
    DO x = 0, 15
       b = b + ISHFT(1, x)
    END DO

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
    !$omp loop reduction(iand:b)
    DO x = 1, N
       b = IAND(b, a(x))
    END DO
    !$omp end loop
    !$omp do
    DO x = 1, N
       num_threads(x) = omp_get_num_threads()
    END DO
    !$omp end do
    !$omp end parallel

    host_b = a(1)
    DO x = 1, N 
       host_b = IAND(host_b, a(x))
    END DO

    DO x = 2, N
       OMPVV_WARNING_IF(num_threads(x - 1) .ne. num_threads(x), "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.")
    END DO
    OMPVV_WARNING_IF(num_threads(1) .eq. 1, "Test operated with one thread.  Reduction clause cannot be tested.")
    OMPVV_WARNING_IF(num_threads(1) .le. 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.")

    OMPVV_TEST_AND_SET_VERBOSE(errors, b .ne. host_b)
    WRITE(msgHelper, *) "Bit from loop directive is ", b, " but expected bit is ", host_b, "." 
    OMPVV_ERROR_IF(host_b .ne. b, msgHelper)

    test_bitand = errors
  END FUNCTION test_bitand
END PROGRAM test_loop_reduction_bitand
