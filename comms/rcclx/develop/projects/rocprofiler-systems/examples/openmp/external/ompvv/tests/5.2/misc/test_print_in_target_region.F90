!===-------- test_print_in_target_region ----------------------------------===!
!
! OpenMP API Version 5.2 Nov 2021
!
! This tests check that print inside of a target region is support for 
! several different common data types in fortran: integer, real, character,
! and logical.
!
!===-----------------------------------------------------------------------===!

#include "ompvv.F90"

PROGRAM target_print
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(test_print_in_target() .ne. 0 )

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION test_print_in_target()
      INTEGER :: errors 
      INTEGER(kind = 2) :: shortval
      INTEGER(kind = 4) :: largeval
      INTEGER(kind = 8) :: verylongval
      INTEGER(kind = 16) :: veryverylongval
      REAL :: p, q, real_temp
      CHARACTER (len=40) :: statement
      LOGICAL :: isOffloading

      errors = 0
      OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)      
      OMPVV_WARNING_IF(isOffloading .eqv. .false., "Support for PRINT * in target region cannot be evaluated, no devices are available")
      p = 5.0
      q = 2.0
      real_temp = p/q
      statement = "This should print"

      !omp target map(tofrom: short_val, largeval, verylongval, veryverylongval, real_temp, statement, isOffloading)
         PRINT *, huge(shortval)
         PRINT *, huge(largeval)
         PRINT *, huge(verylongval)
         PRINT *, huge(veryverylongval)
         PRINT *, real_temp
         PRINT *, statement
         PRINT *, isOffloading
      !omp end target
      
      OMPVV_GET_ERRORS(test_print_in_target)
   END FUNCTION test_print_in_target 
END PROGRAM target_print
