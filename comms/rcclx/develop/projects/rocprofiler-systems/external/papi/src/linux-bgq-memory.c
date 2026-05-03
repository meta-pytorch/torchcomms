/* 
 * File:    linux-bgq-memory.c
 * CVS:     $Id$
 * Author:  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:    
 *          
 */



#include "papi.h"
#include "papi_internal.h"
#include "linux-bgq.h"
#ifdef __LINUX__
#include <limits.h>
#endif
#include <stdio.h>

/*
 * Prototypes...
 */
int init_bgq( PAPI_mh_info_t * pMem_Info );

// inline void cpuid(unsigned int *, unsigned int *,unsigned int *,unsigned int *);

/*
 * Get Memory Information
 *
 * Fills in memory information - effectively set to all 0x00's
 */
extern int
_bgq_get_memory_info( PAPI_hw_info_t * pHwInfo, int pCPU_Type )
{
	int retval = 0;

	switch ( pCPU_Type ) {
	default:
		//fprintf(stderr,"Default CPU type in %s (%d)\n",__FUNCTION__,__LINE__);
		retval = init_bgq( &pHwInfo->mem_hierarchy );
		break;
	}

	return retval;
}

/*
 * Get DMem Information for BG/Q
 *
 * NOTE:  Currently, all values set to -1
 */
extern int
_bgq_get_dmem_info( PAPI_dmem_info_t * pDmemInfo )
{
//  pid_t xPID = getpid();
//  prpsinfo_t xInfo;
//  char xFile[256];
//  int xFD;

//  sprintf(xFile, "/proc/%05d", xPID);
//  if ((fd = open(xFile, O_RDONLY)) < 0) {
//     SUBDBG("PAPI_get_dmem_info can't open /proc/%d\n", xPID);
//     return (PAPI_ESYS);
//  }
//  if (ioctl(xFD, PIOCPSINFO, &xInfo) < 0) {
//     return (PAPI_ESYS);
//  }
//  close(xFD);

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
 * Cache configuration for BG/Q
 */
int
init_bgq( PAPI_mh_info_t * pMem_Info )
{
	memset( pMem_Info, 0x0, sizeof ( *pMem_Info ) );
	//fprintf(stderr,"mem_info not est up [%s (%d)]\n",__FUNCTION__,__LINE__);

	return PAPI_OK;
}
