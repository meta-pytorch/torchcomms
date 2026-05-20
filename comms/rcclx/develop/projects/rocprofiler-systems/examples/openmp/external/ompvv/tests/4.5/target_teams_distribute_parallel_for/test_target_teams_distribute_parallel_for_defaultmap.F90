!===---- test_target_teams_distribute_parallel_for_defaultmap.F90 -----------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! Testing defaultmap of different scalar values. We check when it is off and 
! when it is on. The first one should not copy values back from the device of 
! scalars. The second should copy the values back even if they are not mapped
! explicitly.
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define ITERATIONS 1000

PROGRAM target_teams_distribute_parallel_for_defaultmap
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(test_defaultmap_on() .ne. 0)
   OMPVV_TEST_VERBOSE(test_defaultmap_off() .ne. 0)
   OMPVV_REPORT_AND_RETURN()

CONTAINS 
   INTEGER FUNCTION test_defaultmap_on()
      INTEGER :: errors, i
      CHARACTER :: scalar_char
      CHARACTER, DIMENSION(ITERATIONS) :: scalar_char_cpy
      INTEGER(1) :: scalar_short 
      INTEGER(1), DIMENSION(ITERATIONS) :: scalar_short_cpy
      INTEGER :: scalar_int
      INTEGER, DIMENSION(ITERATIONS) :: scalar_int_cpy
      REAL :: scalar_float
      REAL, DIMENSION(ITERATIONS) :: scalar_float_cpy
      DOUBLE PRECISION :: scalar_double 
      DOUBLE PRECISION, DIMENSION(ITERATIONS) :: scalar_double_cpy 
      errors = 0
      scalar_char = 'a' 
      scalar_short = 10
      scalar_int = 11
      scalar_float = 5.5
      scalar_double = 10.45

      OMPVV_INFOMSG("test_defaultmap_on()");

      ! Testing the to behavior of the tofrom. We use an array to avoid
      ! data races and check that all threads receive the proper value
      !$omp target teams distribute parallel do defaultmap(tofrom: scalar)
      DO i = 1, ITERATIONS
         scalar_char_cpy(i) = scalar_char
         scalar_short_cpy(i) = scalar_short
         scalar_int_cpy(i) = scalar_int
         scalar_float_cpy(i) = scalar_float
         scalar_double_cpy(i) = scalar_double
      END DO

      DO i = 1, ITERATIONS
         OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char_cpy(i) .ne. 'a')
         OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short_cpy(i) .ne. 10)
         OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int_cpy(i) .ne.  11)
         OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_float_cpy(i) - 5.5) .gt. 0.0001);
         OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_double_cpy(i) - 10.45) .gt. 0.00001);
      END DO

      !$omp target teams distribute parallel do defaultmap (tofrom:scalar)
      DO i = 1, ITERATIONS
         IF (omp_get_team_num() .eq. 0) THEN
            IF (omp_get_thread_num() .eq. 0) THEN
               scalar_char = 'b'
               scalar_short = 20;
               scalar_int = 33
               scalar_float = 6.5
               scalar_double = 20.45
            END IF
         END IF
      END DO

      OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char .ne. 'b')
      OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short .ne. 20)
      OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int .ne. 33)
      OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_float - 6.5) .gt. 0.0001);
      OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_double - 20.45) .gt. 0.00001);

   test_defaultmap_on = errors
   END FUNCTION test_defaultmap_on

   INTEGER FUNCTION test_defaultmap_off()
      INTEGER :: errors, i
      CHARACTER :: scalar_char
      CHARACTER, DIMENSION(ITERATIONS) :: scalar_char_cpy
      INTEGER(1) :: scalar_short
      INTEGER(1), DIMENSION(ITERATIONS) :: scalar_short_cpy
      INTEGER :: scalar_int
      INTEGER, DIMENSION(ITERATIONS) :: scalar_int_cpy
      REAL :: scalar_float
      REAL, DIMENSION(ITERATIONS) :: scalar_float_cpy
      DOUBLE PRECISION :: scalar_double
      DOUBLE PRECISION, DIMENSION(ITERATIONS) :: scalar_double_cpy
      errors = 0
      scalar_char = 'a'
      scalar_short = 10
      scalar_int = 11
      scalar_float = 5.5
      scalar_double = 10.45
 
      OMPVV_INFOMSG("test_defaultmap_off()");

      ! Testing the copy behavior of the firstprivatization. We use an array
      ! to avoid data races and check that all threads get the value
      !$omp target teams distribute parallel do 
      DO i = 1, ITERATIONS
         scalar_char_cpy(i) = scalar_char
         scalar_short_cpy(i) = scalar_short
         scalar_int_cpy(i) = scalar_int
         scalar_float_cpy(i) = scalar_float
         scalar_double_cpy(i) = scalar_double
      END DO

      DO i = 1, ITERATIONS
         OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char_cpy(i) .ne. 'a')
         OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short_cpy(i) .ne. 10)
         OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int_cpy(i) .ne.  11)
         OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_float_cpy(i) - 5.5) .gt. 0.0001);
         OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_double_cpy(i) - 10.45) .gt. 0.00001);
      END DO
 
      !$omp target teams distribute parallel do
      DO i = 1, 1 ! Unlike previous function, these values should not change on host following end target
         scalar_char = 'b'
         scalar_short = 20
         scalar_int = 33
         scalar_float = 6.5
         scalar_double = 20.45
      END DO
      
      OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char .ne. 'b') ! The Fortran character type is not a scalar variable in OpenMP
      OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short .ne. 10)
      OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int .ne. 11)
      OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_float - 5.5) > 0.0001)
      OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_double - 10.45) > 0.0001)

   test_defaultmap_off = errors
   END FUNCTION test_defaultmap_off
END PROGRAM target_teams_distribute_parallel_for_defaultmap
