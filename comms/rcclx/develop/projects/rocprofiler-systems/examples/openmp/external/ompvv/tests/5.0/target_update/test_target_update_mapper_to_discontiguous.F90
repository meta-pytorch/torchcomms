!===--- test_target_update_mapper_to_discontiguous.F90 ----------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
!  This test seeks to ensure that target update with motion-clause "to"
!  can properly map data to the device by specifying a user-defined
!  mapper. Additionally, the test checks a new addition to target update
!  in OpenMP 5.0 that states "List items in the to or from clauses may
!  include array sections with stride expressions."
!
!  Adopted from OpenMP 5.0 Example target_mapper.1.F90
!===-----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

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

    OMPVV_TEST_VERBOSE(test_target_update_mapper() .ne. 0)
 
    OMPVV_REPORT_AND_RETURN()

CONTAINS
    INTEGER FUNCTION test_target_update_mapper()
        INTEGER :: i 
        !$omp declare mapper(newvec :: v)&
        !$omp& map(v, v%data(1:v%len))

        type(newvec) :: s
        allocate(s%data(N))
        s%data(1:N) = 0.0d0
        s%len = N

        !$omp target data map(tofrom:s)
        s%data = [(i, i=1, s%len)] 
 
        ! update odd array position values from host
        ! This should set them to i
        !$omp target update to(s%data(1:s%len:2))

        ! update array on the device
        !$omp target
        DO i=1, N
            s%data(i) = s%data(i) + i
        END DO
        !$omp end target
        !$omp end target data

        DO i=1, s%len
           IF (modulo(i,2) .EQ. 0) THEN
                ! even positions should be result[i] = i
                OMPVV_TEST_AND_SET(errors, s%data(i) /= i)
            ELSE
                ! odd positions should be result[i] = 2*i
                OMPVV_TEST_AND_SET(errors, s%data(i) /= 2*i)
            END IF
        END DO

        test_target_update_mapper = errors

    END FUNCTION test_target_update_mapper
END PROGRAM main

