#include <string.h>
#include <errno.h>
#include <sys/sysctl.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h" /* papi_calloc() */

#include "x86_cpuid_info.h"

#include "darwin-lock.h"

int
_darwin_get_dmem_info( PAPI_dmem_info_t * d )
{

  int mib[4];
  size_t len;
  char buffer[BUFSIZ];
  long long ll;

  /**********/
  /* memory */
  /**********/
  len = 2;
  sysctlnametomib("hw.memsize", mib, &len);

  len = 8;
  if (sysctl(mib, 2, &ll, &len, NULL, 0) == -1) {
    return PAPI_ESYS;
  }

  d->size=ll;

  d->pagesize = getpagesize(  );

	return PAPI_OK;
}

/*
 * Architecture-specific cache detection code 
 */


#if defined(__i386__)||defined(__x86_64__)
static int
x86_get_memory_info( PAPI_hw_info_t * hw_info )
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
_darwin_get_memory_info( PAPI_hw_info_t * hwinfo, int cpu_type )
{
	( void ) cpu_type;		 /*unused */
	int retval = PAPI_OK;

	x86_get_memory_info( hwinfo );

	return retval;
}

int
_darwin_update_shlib_info( papi_mdi_t *mdi )
{


	return PAPI_OK;
}
