/*
 * File:    linux-bgp-memory.c
 * Author:  Dave Hermsmeier
 *          dlherms@us.ibm.com
 */

#include "papi.h"
#include "papi_internal.h"
#ifdef __LINUX__
#include <limits.h>
#endif
#include <stdio.h>
#include <string.h>

/*
 * Prototypes...
 */
int init_bgp( PAPI_mh_info_t * pMem_Info );

/*
 * Get Memory Information
 *
 * Fills in memory information - effectively set to all 0x00's
 */
extern int
_bgp_get_memory_info( PAPI_hw_info_t * pHwInfo, int pCPU_Type )
{
	int retval = 0;

	switch ( pCPU_Type ) {
	default:
		//fprintf(stderr,"Default CPU type in %s (%d)\n",__FUNCTION__,__LINE__);
		retval = init_bgp( &pHwInfo->mem_hierarchy );
		break;
	}

	return retval;
}

/*
 * Get DMem Information for BG/P
 *
 * NOTE:  Currently, all values set to -1
 */
extern int
_bgp_get_dmem_info( PAPI_dmem_info_t * pDmemInfo )
{

	pDmemInfo->size = PAPI_EINVAL;
	pDmemInfo->resident = PAPI_EINVAL;
	pDmemInfo->high_water_mark = PAPI_EINVAL;
	pDmemInfo->shared = PAPI_EINVAL;
	pDmemInfo->text = PAPI_EINVAL;
	pDmemInfo->library = PAPI_EINVAL;
	pDmemInfo->heap = PAPI_EINVAL;
	pDmemInfo->locked = PAPI_EINVAL;
	pDmemInfo->stack = PAPI_EINVAL;
	pDmemInfo->pagesize = PAPI_EINVAL;

	return PAPI_OK;
}

/*
 * Cache configuration for BG/P
 */
int
init_bgp( PAPI_mh_info_t * pMem_Info )
{
	memset( pMem_Info, 0x0, sizeof ( *pMem_Info ) );

	return PAPI_OK;
}
