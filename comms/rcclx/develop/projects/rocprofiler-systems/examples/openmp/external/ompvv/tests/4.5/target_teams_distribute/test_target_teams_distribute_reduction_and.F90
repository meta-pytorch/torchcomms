!===--- test_target_teams_distribute_reduction_and.F90----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the and operator, that the variable in the
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

  OMPVV_TEST_VERBOSE(test_and() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_and()
    LOGICAL,DIMENSION(N):: a
    REAL(8),DIMENSION(N):: randoms
    REAL(8):: false_margin
    LOGICAL:: result, host_result, tested_true, tested_false
    INTEGER:: x, errors, itr_count
    errors = 0

    tested_true = .FALSE.
    tested_false = .FALSE.
    itr_count = 0

    false_margin = exp(log(.5) / N)
    CALL RANDOM_SEED()

    DO WHILE ((.NOT. tested_true .OR. .NOT. tested_false) &
         & .AND. (itr_count .lt. THRESHOLD))
       CALL RANDOM_NUMBER(randoms)
       DO x = 1, N
          IF (randoms(x) .lt. false_margin) THEN
             a(x) = .TRUE.
          ELSE
             a(x) = .FALSE.
          END IF
       END DO

       result = .TRUE.
       host_result = .TRUE.

       DO x = 1, N
          host_result = host_result .AND. a(x)
       END DO

       !$omp target teams distribute map(to: a(1:N)) &
       !$omp& reduction(.and.: result) defaultmap( &
       !$omp& tofrom: scalar)
       DO x = 1, N
          result = a(x) .AND. result
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

    test_and = errors
  END FUNCTION test_and
END PROGRAM test_target_teams_distribute_device
