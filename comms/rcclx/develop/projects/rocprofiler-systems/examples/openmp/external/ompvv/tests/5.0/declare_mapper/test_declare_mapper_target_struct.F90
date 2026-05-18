!===--- test_declare_mapper_target_struct.F90 -----------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
!  This example has been adapted from the 5.0 OpenMP Examples document.
!  The declare mapper directive specifies that any structure of type
!  myvec_t for which implicit data-mapping rules apply will be mapped 
!  according to its map clause. The variable v is used for referencing 
!  the structure and its elements within the map clause. Within the map
!  clause the v variable specifies that all elements of the structure are 
!  to be mapped.
!
!===-----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1000

module my_struct
  type newvec
    integer                     :: len
    double precision, pointer   :: data(:)
  end type
end module

PROGRAM main
    USE my_struct
    USE iso_fortran_env
    USE ompvv_lib
    USE omp_lib
    implicit none
    INTEGER :: errors
    
    errors = 0 

    OMPVV_TEST_OFFLOADING

    OMPVV_TEST_VERBOSE(test_declare_mapper() .ne. 0)

    OMPVV_REPORT_AND_RETURN()

CONTAINS
    INTEGER FUNCTION test_declare_mapper()
        INTEGER :: i 
        !$omp declare mapper(newvec :: v)&
        !$omp& map(v, v%data(1:v%len))

        type(newvec) :: s
        allocate(s%data(N))
        s%data(1:N) = 0.0d0
        s%len = N

        !$omp target
        CALL init(s) 
        !$omp end target 

        DO i=1, N
            IF (s%data(i) .ne. i) THEN
                errors = errors + 1
            END IF
        END DO

        test_declare_mapper = errors
    END FUNCTION test_declare_mapper
END PROGRAM main

SUBROUTINE init(s)
    use my_struct
    type(newvec) :: s

    !$omp declare target

    s%data = [(i, i = 1, s%len)]
END SUBROUTINE
