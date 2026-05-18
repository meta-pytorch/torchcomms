/*
* File:    aix-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

#include "aix.h"

int
_aix_get_memory_info( PAPI_hw_info_t * mem_info, int type )
{
	PAPI_mh_level_t *L = mem_info->mem_hierarchy.level;

	/* Not quite sure what bit 30 indicates.
	   I'm assuming it flags a unified tlb */
	if ( _system_configuration.tlb_attrib & ( 1 << 30 ) ) {
		L[0].tlb[0].type = PAPI_MH_TYPE_UNIFIED;
		L[0].tlb[0].num_entries = _system_configuration.itlb_size;
		L[0].tlb[0].type = PAPI_MH_TYPE_UNIFIED;
	} else {
		L[0].tlb[0].type = PAPI_MH_TYPE_INST;
		L[0].tlb[0].num_entries = _system_configuration.itlb_size;
		L[0].tlb[0].associativity = _system_configuration.itlb_asc;
		L[0].tlb[1].type = PAPI_MH_TYPE_DATA;
		L[0].tlb[1].num_entries = _system_configuration.dtlb_size;
		L[0].tlb[1].associativity = _system_configuration.dtlb_asc;
	}
	/* Not quite sure what bit 30 indicates.
	   I'm assuming it flags a unified cache */
	if ( _system_configuration.cache_attrib & ( 1 << 30 ) ) {
		L[0].cache[0].type = PAPI_MH_TYPE_UNIFIED;
		L[0].cache[0].size = _system_configuration.icache_size;
		L[0].cache[0].associativity = _system_configuration.icache_asc;
		L[0].cache[0].line_size = _system_configuration.icache_line;
	} else {
		L[0].cache[0].type = PAPI_MH_TYPE_INST;
		L[0].cache[0].size = _system_configuration.icache_size;
		L[0].cache[0].associativity = _system_configuration.icache_asc;
		L[0].cache[0].line_size = _system_configuration.icache_line;
		L[0].cache[1].type = PAPI_MH_TYPE_DATA;
		L[0].cache[1].size = _system_configuration.dcache_size;
		L[0].cache[1].associativity = _system_configuration.dcache_asc;
		L[0].cache[1].line_size = _system_configuration.dcache_line;
	}
	L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
	L[1].cache[0].size = _system_configuration.L2_cache_size;
	L[1].cache[0].associativity = _system_configuration.L2_cache_asc;
	/* is there a line size for Level 2 cache? */

	/* it looks like we've always got at least 2 levels of info */
	/* what about level 3 cache? */
	mem_info->mem_hierarchy.levels = 2;

	return PAPI_OK;
}

int
_aix_get_dmem_info( PAPI_dmem_info_t * d )
{
	/* This function has been reimplemented 
	   to conform to current interface.
	   It has not been tested.
	   Nor has it been confirmed for completeness.
	   dkt 05-10-06
	 */

	struct procsinfo pi;
	pid_t mypid = getpid(  );
	pid_t pid;
	int found = 0;

	pid = 0;
	while ( 1 ) {
		if ( getprocs( &pi, sizeof ( pi ), 0, 0, &pid, 1 ) != 1 )
			break;
		if ( mypid == pi.pi_pid ) {
			found = 1;
			break;
		}
	}
	if ( !found )
		return ( PAPI_ESYS );

	d->size = pi.pi_size;
	d->resident = pi.pi_drss + pi.pi_trss;
	d->high_water_mark = PAPI_EINVAL;
	d->shared = PAPI_EINVAL;
	d->text = pi.pi_trss;	 /* this is a guess */
	d->library = PAPI_EINVAL;
	d->heap = PAPI_EINVAL;
	d->locked = PAPI_EINVAL;
	d->stack = PAPI_EINVAL;
	d->pagesize = getpagesize(  );

	return ( PAPI_OK );
}
