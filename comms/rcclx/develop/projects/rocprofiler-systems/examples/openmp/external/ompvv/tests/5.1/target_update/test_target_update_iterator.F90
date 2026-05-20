!===--- test_target_update_iterator.F90    -----------------------------===//
!
!  OpenMP API Version 5.1 Aug 2021
!
!  This test uses the target update directive with the iterator clause
!  to update the iterative values.
!
!  This example has been adapted from the 5.2 OpenMP Examples document,
!  "Multidependences Using Iterators" and "Simple target data and target update
!  constructs"
!
!===-----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

MODULE my_struct
  TYPE test_struct
    INTEGER :: len
    INTEGER, POINTER:: data(:)
  END TYPE
END MODULE

PROGRAM main
    USE my_struct
    USE iso_fortran_env
    USE ompvv_lib
    USE omp_lib
    implicit none
    
    OMPVV_TEST_OFFLOADING

    OMPVV_TEST_VERBOSE(test_target_update_iterator() .NE. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS
    SUBROUTINE init(s)
        TYPE(test_struct), INTENT(IN), POINTER :: s
        INTEGER :: i
        s%len = N
        ALLOCATE(s%data(N))
        DO i = 1, s%len
            s%data(i) = i
        END DO
    END SUBROUTINE

    SUBROUTINE init_again(s)
        TYPE(test_struct), INTENT(IN), POINTER :: s
        INTEGER :: i
        s%len = N
        DO i = 1, s%len
            s%data(i) = i + 1
        END DO
    END SUBROUTINE

    INTEGER FUNCTION test_target_update_iterator()
        INTEGER :: errors, i
        INTEGER, TARGET :: A(N)
        TYPE(test_struct), TARGET :: new_struct

        errors = 0

        CALL init(new_struct)

        !$omp target enter data map(to: new_struct%data)
        !$omp target map(to: new_struct, new_struct%data) map(tofrom: A) 
        ! pointer attachment
        DO i = 1, N
            A(i) = new_struct%data(i)
        END DO
        !$omp end target

        CALL init_again(new_struct)
        ! update with new values, (i*2)+1
        !$omp target update to(iterator(it = 1:N): new_struct%data(it))
        !$omp target map(tofrom: A)
        DO i = 1, N
            A(i) = A(i) + new_struct%data(i)
        END DO
        !$omp end target

        !$omp target exit data map(delete: new_struct%data)

        DO i = 1, N
            OMPVV_TEST_AND_SET(errors, A(i) .NE. (i*2)+1)
        END DO

        test_target_update_iterator = errors
    END FUNCTION test_target_update_iterator
END PROGRAM main
