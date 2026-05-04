/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:		freebsd-memory.c
* Author:	Harald Servat
*			redcrash@gmail.com
* Mod:		James Ralph 
*			ralph@cs.utk.edu
*/

#include "papi.h"
#include "papi_internal.h"

#include "x86_cpuid_info.h"

#define UNREFERENCED(x) (void)x


#if defined(__i386__)||defined(__x86_64__)
static int
x86_get_memory_info( PAPI_hw_info_t *hw_info )
{
  int retval = PAPI_OK;

  switch ( hw_info->vendor ) {
  case PAPI_VENDOR_AMD:
  case PAPI_VENDOR_INTEL:
    retval = _x86_cache_info( &hw_info->mem_hierarchy );
    break;
  default:
    PAPIERROR( "Unknown vendor in memory information call for x86." );
    return PAPI_ENOIMPL;
  }
  return retval;
}
#endif


int 
_freebsd_get_memory_info( PAPI_hw_info_t *hw_info, int id)
{
	UNREFERENCED(id);
	UNREFERENCED(hw_info);

#if defined(__i386__)||defined(__x86_64__)
        x86_get_memory_info( hw_info );
#endif

	return PAPI_ENOIMPL;
}

int _papi_freebsd_get_dmem_info(PAPI_dmem_info_t *d)
{
  /* TODO */
	d->pagesize = getpagesize();
	return PAPI_OK;
}
 
