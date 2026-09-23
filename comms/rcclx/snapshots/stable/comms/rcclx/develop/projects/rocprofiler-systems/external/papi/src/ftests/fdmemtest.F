#include "fpapi_test.h"
      program dmemtest
      IMPLICIT integer (p)
      
      INTEGER retval
      INTEGER*8 dmeminfo(PAPIF_DMEM_MAXVAL)
      
      integer tests_quiet, get_quiet
      external get_quiet
      real EventSet

      tests_quiet = get_quiet()
      EventSet = PAPI_NULL

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_library_init', retval)
      end if


      CALL PAPIf_get_dmem_info(dmeminfo, retval)
      if ( retval.NE.PAPI_OK) then
         stop
      end if
      
      if (tests_quiet .EQ. 0) then
          print *, "Mem Size: ", dmeminfo(PAPIF_DMEM_VMSIZE)
          print *, "Mem Resident: ", dmeminfo(PAPIF_DMEM_RESIDENT)
          print *, "Mem High Water: ", dmeminfo(PAPIF_DMEM_HIGH_WATER)
          print *, "Mem Shared: ", dmeminfo(PAPIF_DMEM_SHARED)
          print *, "Mem Text: ", dmeminfo(PAPIF_DMEM_TEXT)
          print *, "Mem Library: ", dmeminfo(PAPIF_DMEM_LIBRARY)
          print *, "Mem Heap: ", dmeminfo(PAPIF_DMEM_HEAP)
          print *, "Mem Locked: ", dmeminfo(PAPIF_DMEM_LOCKED)
          print *, "Mem Stack: ", dmeminfo(PAPIF_DMEM_STACK)
          print *, "Mem Pagesize: ", dmeminfo(PAPIF_DMEM_PAGESIZE)
      end if
      call ftests_pass(__FILE__)
      end
