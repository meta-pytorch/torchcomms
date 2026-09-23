#include "fpapi_test.h"
#define ITERS 100000

      program clockres
      IMPLICIT integer (p)

      integer*8, allocatable, dimension(:) :: elapsed_usec, elapsed_cyc
      INTEGER*8 total_usec, total_cyc
      INTEGER i,handle
      INTEGER retval
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

      total_usec=0
      total_cyc=0
      handle=0
      allocate(elapsed_usec(ITERS))
      allocate(elapsed_cyc(ITERS))

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .     'PAPI_library_init', retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, 'Test case: Clock resolution.'
      print *,'-----------------------------------------------'
      end if

      do i=1,ITERS
         call PAPIf_get_real_cyc( elapsed_cyc(i) )
      end do

      do i=2,ITERS
         if ((elapsed_cyc(i)-elapsed_cyc(i-1)).LT.0 ) stop 
         total_cyc =total_cyc+(elapsed_cyc(i) - elapsed_cyc(i-1))
      end do

      do i=1,ITERS
         call PAPIf_get_real_usec(elapsed_usec(i))
      end do

      do i=2,ITERS
        if ((elapsed_usec(i) - elapsed_usec(i-1)).LT.0) stop
        total_usec=total_usec+(elapsed_usec(i) - elapsed_usec(i-1))
      end do
 
      if (tests_quiet .EQ. 0) then
      print *,'PAPIf_get_real_cyc : ',(total_cyc/(ITERS-1))
      print *,'PAPIf_get_real_usec: ',(total_usec/(ITERS-1))
      end if

      deallocate(elapsed_usec, elapsed_cyc)
      call ftests_pass(__FILE__)
      end
