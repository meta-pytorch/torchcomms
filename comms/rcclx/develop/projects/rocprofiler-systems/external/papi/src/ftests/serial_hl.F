#include "fpapi.h"

      program flops
      integer retval
      integer i

      do i = 1, 4
         call PAPIf_hl_region_begin("main", retval)
         if ( retval .NE. PAPI_OK ) then
            write (*,*) "PAPIf_hl_region_begin failed!"
         end if

         write (*,*) 'Round', i
         call fdo_flops(NUM_FLOPS)

         call PAPIf_hl_region_end("main", retval)
         if ( retval .NE. PAPI_OK ) then
            write (*,*) "PAPIf_hl_region_end failed!"
         end if
      end do

      call ftests_hl_pass(__FILE__)
      end program flops
