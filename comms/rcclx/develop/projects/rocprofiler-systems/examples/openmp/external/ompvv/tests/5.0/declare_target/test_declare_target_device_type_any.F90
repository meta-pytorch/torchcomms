!===--- test_declare_target_device_type_any.F90 -----------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! The declare target directive specifies that variables, functions(C,C++ and Fortran),
! and subroutines (Fortran) are mapped to a device. If a device_type
! clause is present on the contained declare target directive, then its argument 
! determines which versions are made available. When device_type(any) is specified 
! both device and host versions of the procedure are made available.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

MODULE m_dat
    !$omp declare target to(a, b, c, i)
    INTEGER, DIMENSION(N) :: a, b, c
    INTEGER :: i = 0
END MODULE m_dat

PROGRAM test_declare_target_device_type_any

    USE iso_fortran_env
    USE ompvv_lib
    USE omp_lib
    USE m_dat
    implicit none
    INTEGER :: errors = 0
    
    ! initialize arrays on host
    DO i = 1, N
      a(i) = i
      b(i) = i + 1
      c(i) = i + 2
    END DO
     
    OMPVV_TEST_VERBOSE(test_device_type_any() .ne. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS  
    SUBROUTINE update()
      !$omp declare target to(update) device_type(any)
      DO i = 1, N
        a(i) = a(i) + 1
        b(i) = b(i) + 2
        c(i) = c(i) + 3
      END DO
    END SUBROUTINE update

    INTEGER FUNCTION test_device_type_any() 
      !$omp target update to(a, b, c)
      !$omp target
      CALL update()
      !$omp end target

      !$omp target update from(a, b, c)

      ! check array values on host
      DO i = 1, N
        IF ( a(i) .NE. i+1 .OR. b(i) .NE. i+3 .OR. c(i) .NE. i+5  ) THEN
          errors = errors + 1
        END IF
      END DO
     
      ! on host
      CALL update()

      ! check array values on host
      DO i = 1, N
        IF ( a(i) .NE. i+2 .OR. b(i) .NE. i+5 .OR. c(i) .NE. i+8  ) THEN
          errors = errors + 1
        END IF
      END DO

      test_device_type_any= errors
    END FUNCTION test_device_type_any

END PROGRAM test_declare_target_device_type_any
