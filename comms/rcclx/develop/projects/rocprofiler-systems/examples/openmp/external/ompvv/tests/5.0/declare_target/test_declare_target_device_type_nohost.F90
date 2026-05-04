!===--- test_declare_target_device_type_nohost.F90 -----------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! The declare target directive specifies that variables, functions(C,C++ and Fortran),
! and subroutines (Fortran) are mapped to a device. If a device_type
! clause is present on the contained declare target directive, then its argument 
! determines which versions are made available. If device_type(nohost) is present 
! only a device version of the procedure is made available. 
! The device version of the function is specified via declare variant.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 10

MODULE m_dat
    !$omp declare target to(a, b, c, i)
    INTEGER, DIMENSION(N) :: a, b, c
    INTEGER :: i = 0
END MODULE m_dat

PROGRAM test_declare_target_device_type_nohost

    USE iso_fortran_env
    USE ompvv_lib
    USE omp_lib
    USE m_dat
    implicit none
    INTEGER :: errors = 0
    
    OMPVV_TEST_OFFLOADING

    ! initialize arrays on host
    DO i = 1, N
      a(i) = 0
      b(i) = 0
      c(i) = 0
    END DO
     
    OMPVV_TEST_VERBOSE(test_device_type_nohost() .ne. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS  
    SUBROUTINE target_function()
      !$omp declare target to(target_function) device_type(nohost)
      DO i = 1, N
        a(i) = a(i) + 1
        b(i) = b(i) + 2
        c(i) = c(i) + 3
      END DO
    END SUBROUTINE target_function

    SUBROUTINE update()
      !$omp declare variant(target_function) match(device={kind(nohost)})
      DO i = 1, N
        a(i) = a(i) + 5
        b(i) = b(i) + 5
        c(i) = c(i) + 5
      END DO
    END SUBROUTINE update

    INTEGER FUNCTION test_device_type_nohost() 
      CALL update()

      !$omp target update to(a, b, c)

      !$omp target
      CALL update()
      !$omp end target

      !$omp target update from(a, b, c)

      IF ( omp_get_default_device() .GE. 0 .AND. omp_get_default_device() .LT. omp_get_num_devices() ) THEN
        ! check array values on host
        DO i = 1, N
          IF ( a(i) .NE. 6 .OR. b(i) .NE. 7 .OR. c(i) .NE. 8  ) THEN
            errors = errors + 1
          END IF
        END DO
      ELSE
        OMPVV_WARNING("Default device is the host device. Thus, test only ran on the host")
        ! check array values on host
        DO i = 1, N
          IF ( a(i) .NE. 10 .OR. b(i) .NE. 10 .OR. c(i) .NE. 10  ) THEN
            errors = errors + 1
          END IF
        END DO
      END IF
     
      test_device_type_nohost= errors
    END FUNCTION test_device_type_nohost

END PROGRAM test_declare_target_device_type_nohost
