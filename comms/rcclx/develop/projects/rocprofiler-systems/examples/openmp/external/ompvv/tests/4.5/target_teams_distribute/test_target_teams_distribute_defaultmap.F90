!===--- test_target_teams_distribute_defaultmap.F90-------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the defaultmap clause on a target teams distribute directive.
! This tests the following scalars: char, short, int, float, double, and enum.
! Both using the clause defaultmap(tofrom:scalar) is used. When it is used,
! the test tests the to nature by setting arrays to the value.  Then it is also
! tested that, as opposed to the default action on scalars which is to first-
! privatize them, they are shared and returned to the host.
!
! It also tests the default operation of treating scalars without the defaultmap
! clause.  The test first tests the privatization of the firstprivatized
! scalars and then separately tests the proper initialization of them separately
!
!===------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_defaultmap
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  LOGICAL :: isSharedEnv
  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedEnv)
  IF (isSharedEnv) THEN
     OMPVV_WARNING("Shared memory environment. Scalars are not copied")
     OMPVV_WARNING("but modified. Defaultmap off test is inconclusive")
  END IF
  OMPVV_TEST_VERBOSE(test_defaultmap_on() .ne. 0)
  OMPVV_TEST_VERBOSE(test_defaultmap_off() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_defaultmap_on()
    INTEGER :: errors, x
    INTEGER(kind = int8) :: scalar_byte
    INTEGER(kind = int16) :: scalar_short
    INTEGER(kind = int32) :: scalar_int
    INTEGER(kind = int64) :: scalar_long_int
    REAL(kind = real32) :: scalar_float
    REAL(kind = real64) :: scalar_double
    LOGICAL :: scalar_logical
    LOGICAL(kind = 4) :: scalar_logical_kind4
    LOGICAL(kind = 8) :: scalar_logical_kind8
    COMPLEX :: scalar_complex

    INTEGER(kind = int8),DIMENSION(N) :: byte_array
    INTEGER(kind = int16),DIMENSION(N) :: short_array
    INTEGER(kind = int32),DIMENSION(N) :: int_array
    INTEGER(kind = int64),DIMENSION(N) :: long_int_array
    REAL(kind = real32),DIMENSION(N) :: float_array
    REAL(kind = real64),DIMENSION(N) :: double_array
    LOGICAL,DIMENSION(N) :: logical_array
    LOGICAL(kind = 4),DIMENSION(N) :: logical_kind4_array
    LOGICAL(kind = 8),DIMENSION(N) :: logical_kind8_array
    COMPLEX,DIMENSION(N) :: complex_array

    scalar_byte = 8
    scalar_short = 23
    scalar_int = 56
    scalar_long_int = 90000
    scalar_float = 4.5
    scalar_double = 4.9
    scalar_logical = .TRUE.
    scalar_logical_kind4 = .TRUE.
    scalar_logical_kind8 = .TRUE.
    scalar_complex = (1, 1)

    errors = 0

    !$omp target teams distribute defaultmap(tofrom: scalar) map(tofrom: &
    !$omp& byte_array(1:N), short_array(1:N), &
    !$omp& int_array(1:N), long_int_array(1:N), float_array(1:N), &
    !$omp& double_array(1:N), logical_array(1:N), &
    !$omp& logical_kind4_array(1:N),  logical_kind8_array(1:N), &
    !$omp& complex_array(1:N))
    DO x = 1, N
       byte_array(x) = scalar_byte
       short_array(x) = scalar_short
       int_array(x) = scalar_int
       long_int_array(x) = scalar_long_int
       float_array(x) = scalar_float
       double_array(x) = scalar_double
       logical_array(x) = scalar_logical
       logical_kind4_array(x) = scalar_logical_kind4
       logical_kind8_array(x) = scalar_logical_kind8
       complex_array(x) = scalar_complex
    END DO
    !$omp end target teams distribute

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, byte_array(x) .ne. scalar_byte)
       OMPVV_TEST_AND_SET_VERBOSE(errors, short_array(x) .ne. scalar_short)
       OMPVV_TEST_AND_SET_VERBOSE(errors, int_array(x) .ne. scalar_int)
       OMPVV_TEST_AND_SET_VERBOSE(errors, long_int_array(x) .ne. scalar_long_int)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(float_array(x) - scalar_float) .gt. .00001)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(double_array(x) - scalar_double) .gt. .0000000001)
       OMPVV_TEST_AND_SET_VERBOSE(errors, logical_array(x) .neqv. scalar_logical)
       OMPVV_TEST_AND_SET_VERBOSE(errors, logical_kind4_array(x) .neqv. scalar_logical_kind4)
       OMPVV_TEST_AND_SET_VERBOSE(errors, LOGICAL(logical_kind8_array(x) .neqv. scalar_logical_kind8, 4))
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(REAL(complex_array(x)) - REAL(scalar_complex)) .gt. .00001)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(AIMAG(complex_array(x)) - AIMAG(scalar_complex)) .gt. .00001)
    END DO

    !$omp target teams distribute defaultmap(tofrom: scalar)
    DO x = 1, N
       IF (omp_get_team_num() .eq. 0) THEN
          scalar_byte = 5
          scalar_short = 83
          scalar_int = 49
          scalar_long_int = 12345
          scalar_float  = 11.2_real32
          scalar_double  = 9.5_real64
          scalar_logical = .false.
          scalar_logical_kind4 = .false.
          scalar_logical_kind8 = .false.
          scalar_complex = (5, 5)
       END IF
    END DO
    !$omp end target teams distribute

    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_byte .ne. 5)
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short .ne. 83)
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int .ne. 49)
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_long_int .ne. 12345)
    OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_float - 11.2_real32) .gt. .00001)
    OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_double - 9.5_real64) .gt. .0000000001)
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_logical .neqv. .false.)
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_logical_kind4 .neqv. .false.)
    OMPVV_TEST_AND_SET_VERBOSE(errors, LOGICAL(scalar_logical_kind8 .neqv. .false., 4))
    OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(REAL(scalar_complex) - 5) .gt. .00001)
    OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(AIMAG(scalar_complex) - 5) .gt. .00001)

    test_defaultmap_on = errors
  END FUNCTION test_defaultmap_on

  INTEGER FUNCTION test_defaultmap_off()
    INTEGER :: errors, x
    INTEGER(kind = int64) :: y
    INTEGER(kind = int8) :: scalar_byte, scalar_byte_copy
    INTEGER(kind = int16) :: scalar_short, scalar_short_copy
    INTEGER(kind = int32) :: scalar_int, scalar_int_copy
    INTEGER(kind = int64) :: scalar_long_int, scalar_long_int_copy
    REAL(kind = real32) :: scalar_float, scalar_float_copy
    REAL(kind = real64) :: scalar_double, scalar_double_copy
    LOGICAL :: scalar_logical, scalar_logical_copy
    LOGICAL(kind = 4) :: scalar_logical_kind4, scalar_logical_kind4_copy
    LOGICAL(kind = 8) :: scalar_logical_kind8, scalar_logical_kind8_copy
    COMPLEX :: scalar_complex, scalar_complex_copy

    INTEGER(kind = int8),DIMENSION(N) :: byte_array_a, byte_array_b
    INTEGER(kind = int16),DIMENSION(N) :: short_array_a, short_array_b
    INTEGER(kind = int32),DIMENSION(N) :: int_array_a, int_array_b
    INTEGER(kind = int64),DIMENSION(N) :: long_int_array_a, long_int_array_b
    REAL(kind = real32),DIMENSION(N) :: float_array_a, float_array_b
    REAL(kind = real64),DIMENSION(N) :: double_array_a, double_array_b
    LOGICAL,DIMENSION(N, 16) :: logical_array_a
    LOGICAL,DIMENSION(N) :: logical_array_b
    LOGICAL(kind = 4),DIMENSION(N, 16) :: logical_kind4_array_a
    LOGICAL(kind = 4),DIMENSION(N) :: logical_kind4_array_b
    LOGICAL(kind = 8),DIMENSION(N, 16) :: logical_kind8_array_a
    LOGICAL(kind = 8),DIMENSION(N) :: logical_kind8_array_b
    COMPLEX,DIMENSION(N) :: complex_array_a, complex_array_b

    scalar_byte = 8
    scalar_short = 23
    scalar_int = 56
    scalar_long_int = 90000
    scalar_float = 4.5_real32
    scalar_double = 4.9_real64
    scalar_logical = .TRUE.
    scalar_logical_kind4 = .TRUE.
    scalar_logical_kind8 = .TRUE.
    scalar_complex = (1, 1)

    DO x = 1, N
       byte_array_a(x) = 9
       short_array_a(x) = 50
       int_array_a(x) = 70
       long_int_array_a(x) = 150
       float_array_a(x) = 15.5_real32
       double_array_a(x) = 52.45_real64
       DO y = 1, 16
          logical_array_a(x, y) = .TRUE.
          logical_kind4_array_a(x, y) = .TRUE.
          logical_kind8_array_a(x, y) = .TRUE.
       END DO
       complex_array_a(x) = (20, 40)
    END DO


    errors = 0

    !$omp target teams distribute map(tofrom: &
    !$omp & byte_array_a(1:N), byte_array_b(1:N), &
    !$omp & short_array_a(1:N), short_array_b(1:N), int_array_a(1:N),  &
    !$omp & int_array_b(1:N), float_array_a(1:N), &
    !$omp & float_array_b(1:N), double_array_a(1:N), &
    !$omp & double_array_b(1:N), logical_array_a(1:N, 1:16), &
    !$omp & logical_array_b(1:N), logical_kind4_array_a(1:N, 1:16), &
    !$omp & logical_kind4_array_b(1:N), logical_kind8_array_a(1:N, 1:16),&
    !$omp & logical_kind8_array_b(1:N), complex_array_a(1:N), &
    !$omp & complex_array_b(1:N))
    DO x = 1, N
       scalar_byte = 0
       DO y = 1, byte_array_a(x)
          scalar_byte = scalar_byte + 1_int8
       END DO
       byte_array_b(x) = scalar_byte

       scalar_short = 0
       DO y = 1, short_array_a(x)
          scalar_short = scalar_short + 1_int16
       END DO
       short_array_b(x) = scalar_short

       scalar_int = 0
       DO y = 1, int_array_a(x)
          scalar_int = scalar_int + 1_int32
       END DO
       int_array_b(x) = scalar_int

       scalar_long_int = 0
       DO y = 1, long_int_array_a(x)
          scalar_long_int = scalar_long_int + 1_int64
       END DO
       long_int_array_b(x) = scalar_long_int

       scalar_float = 0
       DO y = 1, INT(float_array_a(x))
          scalar_float = scalar_float + .7_real32
       END DO
       float_array_b(x) = scalar_float

       scalar_double = 0
       DO y = 1, INT(double_array_a(x))
          scalar_double = scalar_double + .7_real64
       END DO
       double_array_b(x) = scalar_double

       scalar_logical = .FALSE.
       DO y = 1, 16
          scalar_logical = scalar_logical .NEQV. logical_array_a(x, y)
       END DO
       logical_array_b(x) = scalar_logical

       scalar_logical_kind4 = .FALSE.
       DO y = 1, 16
          scalar_logical_kind4 = scalar_logical_kind4 .NEQV. logical_&
               &kind4_array_a(x, y)
       END DO
       logical_kind4_array_b(x) = scalar_logical_kind4

       scalar_logical_kind8 = .FALSE.
       DO y = 1, 16
          scalar_logical_kind8 = scalar_logical_kind8 .NEQV. logical_&
               &kind8_array_a(x, y)
       END DO
       logical_kind8_array_b(x) = scalar_logical_kind8

       scalar_complex = (0, 0)
       DO WHILE (ABS(scalar_complex) .lt. ABS(complex_array_a(x)))
          scalar_complex = scalar_complex + (1, 1)
       END DO
       complex_array_b(x) = scalar_complex
    END DO



    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, byte_array_a(x) .ne. byte_array_b(x))
       OMPVV_TEST_AND_SET_VERBOSE(errors, short_array_a(x) .ne. short_array_b(x))
       OMPVV_TEST_AND_SET_VERBOSE(errors, int_array_a(x) .ne. int_array_b(x))
       OMPVV_TEST_AND_SET_VERBOSE(errors, long_int_array_a(x) .ne. long_int_array_b(x))
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS((INT(float_array_a(x)) * .7_real32) - float_array_b(x)) .gt. .00001_real32)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS((INT(double_array_a(x)) * .7_real64) - double_array_b(x)) .gt. .00001_real64)
       OMPVV_TEST_AND_SET_VERBOSE(errors, logical_array_b(x) .neqv. .FALSE.)
       OMPVV_TEST_AND_SET_VERBOSE(errors, logical_kind4_array_b(x) .neqv. .FALSE.)
       OMPVV_TEST_AND_SET_VERBOSE(errors, LOGICAL(logical_kind8_array_b(x) .neqv. .FALSE., 4))
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(complex_array_b(x)) - ABS(complex_array_a(x)) .lt. 0)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(complex_array_b(x) - (1, 1)) - (ABS(complex_array_a(x))) .gt. 0)
    END DO

    scalar_byte = 8
    scalar_short = 23
    scalar_int = 56
    scalar_long_int = 90000
    scalar_float = 4.5_real32
    scalar_double = 4.9_real64
    scalar_logical = .TRUE.
    scalar_logical_kind4 = .TRUE.
    scalar_logical_kind8 = .TRUE.
    scalar_complex = (1, 1)

    scalar_byte_copy = scalar_byte
    scalar_short_copy = scalar_short
    scalar_int_copy = scalar_int
    scalar_long_int_copy = scalar_long_int
    scalar_float_copy = scalar_float
    scalar_double_copy = scalar_double
    scalar_logical_copy = scalar_logical
    scalar_logical_kind4_copy = scalar_logical_kind4
    scalar_logical_kind8_copy = scalar_logical_kind8
    scalar_complex_copy = scalar_complex

    !$omp target teams distribute map(tofrom: byte_array_a(1:N), &
    !$omp &short_array_a(1:N), int_array_a(1:N), long_int_&
    !$omp &array_a(1:N), float_array_a(1:N), double_array_a(1:N), &
    !$omp & logical_array_b(1:N), logical_kind4_array_b(1:N), logical_&
    !$omp &kind8_array_b(1:N), complex_array_a(1:N))
    DO x = 1, N
       byte_array_a(x) = scalar_byte
       short_array_a(x) = scalar_short
       int_array_a(x) = scalar_int
       long_int_array_a(x) = scalar_long_int
       float_array_a(x) = scalar_float
       double_array_a(x) = scalar_double
       logical_array_b(x) = scalar_logical
       logical_kind4_array_b(x) = scalar_logical_kind4
       logical_kind8_array_b(x) = scalar_logical_kind8
       complex_array_a(x) = scalar_complex
    END DO

    !$omp target teams distribute
    DO x = 1, N
       scalar_byte = 0
       scalar_short = 0
       scalar_int = 0
       scalar_long_int = 0
       scalar_float = 0
       scalar_double = 0
       scalar_logical = .FALSE.
       scalar_logical_kind4 = .FALSE.
       scalar_logical_kind8 = .FALSE.
       scalar_complex = (0, 0)
    END DO

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, byte_array_a(x) .ne. scalar_byte_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, short_array_a(x) .ne. scalar_short_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, int_array_a(x) .ne. scalar_int_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, long_int_array_a(x) .ne. scalar_long_int_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(float_array_a(x) - scalar_float_copy) .gt. .000001_real32)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(double_array_a(x) - scalar_double_copy) .gt. .0000000001_real64)
       OMPVV_TEST_AND_SET_VERBOSE(errors, logical_array_b(x) .neqv. scalar_logical_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, logical_kind4_array_b(x) .neqv. scalar_logical_kind4_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, LOGICAL(logical_kind8_array_b(x) .neqv. scalar_logical_kind8_copy, 4))
       OMPVV_TEST_AND_SET_VERBOSE(errors, complex_array_a(x) .ne. scalar_complex_copy)
    END DO

    IF (isSharedEnv .neqv. .TRUE.) THEN
       OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_byte .ne. scalar_byte_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short .ne. scalar_short_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int .ne. scalar_int_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_long_int .ne. scalar_long_int_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_float - scalar_float_copy) .gt. .000001_real32)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(scalar_double - scalar_double_copy) .gt. .0000000001_real64)
       OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_logical .neqv. scalar_logical_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_logical_kind4 .neqv. scalar_logical_kind4_copy)
       OMPVV_TEST_AND_SET_VERBOSE(errors, LOGICAL(scalar_logical_kind8 .neqv. scalar_logical_kind8_copy, 4))
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(REAL(scalar_complex) - REAL(scalar_complex_copy)) .gt. .000001)
       OMPVV_TEST_AND_SET_VERBOSE(errors, ABS(AIMAG(scalar_complex) - AIMAG(scalar_complex_copy)) .gt. .000001)
    END IF
    test_defaultmap_off = errors
  END FUNCTION test_defaultmap_off
END PROGRAM test_target_teams_distribute_defaultmap
