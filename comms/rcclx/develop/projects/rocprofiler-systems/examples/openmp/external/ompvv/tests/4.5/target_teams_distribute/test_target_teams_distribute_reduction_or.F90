!===--- test_target_teams_distribute_reduction_or.F90-----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the or operator, that the variable in the
! reduction clause is properly reduced.
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024
#define THRESHOLD 512

PROGRAM test_target_teams_distribute_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  errors = 0

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_or() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_or()
    LOGICAL,DIMENSION(N):: a
    REAL(8),DIMENSION(N):: randoms
    REAL(8):: true_margin
    LOGICAL:: result, host_result, tested_true, tested_false
    INTEGER:: x, errors, itr_count
    errors = 0

    tested_true = .FALSE.
    tested_false = .FALSE.
    itr_count = 0

    true_margin = exp(log(.5) / N)
    CALL RANDOM_SEED()

    DO WHILE ((.NOT. tested_true .OR. .NOT. tested_false) &
         & .AND. (itr_count .lt. THRESHOLD))
       CALL RANDOM_NUMBER(randoms)
       host_result = .FALSE.
       result = .FALSE.
       DO x = 1, N
          IF (randoms(x) .gt. true_margin) THEN
             a(x) = .TRUE.
          ELSE
             a(x) = .FALSE.
          END IF
       END DO

       DO x = 1, N
          host_result = a(x) .OR. host_result
       END DO

       !$omp target teams distribute reduction(.or.: &
       !$omp& result) defaultmap(tofrom:scalar)
       DO x = 1, N
          result = a(x) .OR. result
       END DO

       IF (host_result) THEN
          tested_true = .TRUE.
       ELSE
          tested_false = .TRUE.
       END IF

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .neqv. host_result)

       itr_count = itr_count + 1
    END DO

    OMPVV_WARNING_IF(.NOT. tested_true, "Did not test true case")
    OMPVV_WARNING_IF(.NOT. tested_false, "Did not test false case")

    test_or = errors
  END FUNCTION test_or
END PROGRAM test_target_teams_distribute_device
