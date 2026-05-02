!===--- test_declare_target_device_type_host.F90 -----------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! The declare target directive specifies that variables, functions(C,C++ and Fortran),
! and subroutines (Fortran) are mapped to a device. If a device_type
! clause is present on the contained declare target directive, then its argument 
! determines which versions are made available. If device_type(host) is present 
! only a host version of the procedure is made available. 
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

MODULE m_dat
    !$omp declare target to(a, b, c, i)
    INTEGER, DIMENSION(N) :: a, b, c
    INTEGER :: i = 0
END MODULE m_dat

PROGRAM test_declare_target_device_type_host

    USE iso_fortran_env
    USE ompvv_lib
    USE omp_lib
    USE m_dat
    implicit none
    INTEGER :: errors = 0
    
    ! initialize arrays on host
    DO i = 1, N
      a(i) = 0
      b(i) = 0
      c(i) = 0
    END DO
     
    OMPVV_TEST_VERBOSE(test_device_type_host() .ne. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS  
    SUBROUTINE update()
      !$omp declare target to(update) device_type(host)
      DO i = 1, N
        a(i) = a(i) + 1
        b(i) = b(i) + 2
        c(i) = c(i) + 3
      END DO
    END SUBROUTINE update

    INTEGER FUNCTION test_device_type_host() 
      !$omp target update to(a, b, c)
      !$omp target
      DO i = 1, N
        a(i) = a(i) + i
        b(i) = b(i) + 2*i
        c(i) = c(i) + 3*i
      END DO
      !$omp end target

      !$omp target update from(a, b, c)

      ! on host
      CALL update()

      ! check array values on host
      DO i = 1, N
        IF ( a(i) .NE. i+1 .OR. b(i) .NE. 2*i+2 .OR. c(i) .NE. 3*i+3  ) THEN
          errors = errors + 1
        END IF
      END DO
     
      test_device_type_host= errors
    END FUNCTION test_device_type_host

END PROGRAM test_declare_target_device_type_host
