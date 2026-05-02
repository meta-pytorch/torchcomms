!===--- test_target_update_to_present.F90    -----------------------------===//
!
!  OpenMP API Version 5.1 Aug 2021
!
!  This test checks behavior of the target update clause when the specified motion-modifier  
!  is present. Tests 1. A corresponding list item and an original list item exist for each 
!  list item in a to or from clause. If the corresponding list item is not present in the
!  device data environment and the present modifier is not specified in the clause then no
!  assignment occurs to or from the original list item. Also tests 2. Otherwise, each
!  corresponding list item in the device data environment has an original list item in the
!  current task's data environment. 
!
!===-----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

MODULE my_struct
  TYPE test_struct
    INTEGER :: s
    INTEGER :: Sdata(N)
  END TYPE
END MODULE

PROGRAM main
    USE my_struct
    USE iso_fortran_env
    USE, INTRINSIC :: iso_c_binding
    USE ompvv_lib
    USE omp_lib
    implicit none
    
    OMPVV_TEST_VERBOSE(test_target_update_to_present() .ne. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS
    INTEGER FUNCTION test_target_update_to_present()
        INTEGER :: errors
        INTEGER, TARGET :: scalar_var ! scalar
        INTEGER, TARGET :: A(N) ! aggregate
        INTEGER, POINTER :: ptr(:) ! scalar, pointer
        TYPE(test_struct), TARGET :: new_struct
        INTEGER (C_INT) :: t
        TYPE (C_PTR) :: scalar_var_cptr, A_cptr, new_struct_cptr
        LOGICAL :: isOffloadingOn

        scalar_var_cptr = c_loc(scalar_var)
        A_cptr = c_loc(A(1))
        new_struct_cptr = c_loc(new_struct)

        errors = 0 
        scalar_var = 1
        isOffloadingOn = .FALSE.
        A(1) = 0
        A(51) = 50
        new_struct%s = 10
        new_struct%Sdata(1) = 10
        new_struct%Sdata(2) = 10
        ptr => A
        ptr(51) = 50
        ptr(52) = 51

        OMPVV_TEST_AND_SET_OFFLOADING(isOffloadingOn)

        ! Tests OpenMP 5.1 Specification pp. 207 lines 2-4
        !$omp target update to(scalar_var, A, new_struct)
        t = omp_get_default_device()
        IF (isOffloadingOn) THEN
            ! Skip this test if target offloading is not enabled (running on the host).
            IF (omp_target_is_present(scalar_var_cptr, t) .NE. 0) THEN
                errors = errors + 1
            END IF
            IF (omp_target_is_present(A_cptr, t) .NE. 0) THEN
                errors = errors + 1
            END IF
            IF (omp_target_is_present(new_struct_cptr, t) .NE. 0) THEN
                errors = errors + 1
            END IF
        END IF

        ! Tests OpenMP 5.1 Specification pp. 207 lines 5-6
        !$omp target enter data map(alloc: scalar_var, A, new_struct)
        !$omp target update to(present: scalar_var, A, new_struct) 
        IF (omp_target_is_present(scalar_var_cptr, t) .EQ. 0) THEN
            errors = errors + 1
        END IF
        IF (omp_target_is_present(A_cptr, t) .EQ. 0) THEN
            errors = errors + 1
        END IF
        IF (omp_target_is_present(new_struct_cptr, t) .EQ. 0) THEN
            errors = errors + 1
        END IF

        !$omp target map(tofrom: errors) defaultmap(none) map(to: scalar_var, A, new_struct)
        IF(scalar_var .NE. 1) THEN
            errors = errors + 1
        END IF
        IF((A(1) .NE. 0) .OR. (A(51) .NE. 50)) THEN
            errors = errors + 1
        END IF
        IF((A(51) .NE. 50) .OR. (A(52) .NE. 51)) THEN
            errors = errors + 1
        END IF
        IF(new_struct%s .NE. 10) THEN
            errors = errors + 1
        END IF
        IF(new_struct%Sdata(1) .NE. 10) THEN
            errors = errors + 1
        END IF
        !$omp end target
        !$omp target exit data map(release: scalar_var, A, new_struct)

        test_target_update_to_present = errors
    END FUNCTION test_target_update_to_present
END PROGRAM main
