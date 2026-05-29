/*
* File:    solaris-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    Philip J. Mucci
*          mucci@cs.utk.edu
*
* Mods:    Vince Weaver
*          vweaver1@eecs.utk.edu
*
* Mods:    Fabian Gorsler 
*          fabian.gorsler@smail.inf.h-bonn-rhein-sieg.de
*/

#include "papi.h"
#include "papi_internal.h"


int
_solaris_get_memory_info( PAPI_hw_info_t * hw, int id )
{
        FILE *pipe;
        char line[BUFSIZ];

	PAPI_mh_level_t *mem = hw->mem_hierarchy.level;

	pipe=popen("prtconf -pv","r");
        if (pipe==NULL) {
	   return PAPI_ESYS;
	}

	while(1) {

	   if (fgets(line,BUFSIZ,pipe)==NULL) break;

           if (strstr(line,"icache-size:")) {
	      sscanf(line,"%*s %#x",&mem[0].cache[0].size);
	   }
           if (strstr(line,"icache-line-size:")) {
	      sscanf(line,"%*s %#x",&mem[0].cache[0].line_size);
	   }
           if (strstr(line,"icache-associativity:")) {
	      sscanf(line,"%*s %#x",&mem[0].cache[0].associativity);
	   }

           if (strstr(line,"dcache-size:")) {
	      sscanf(line,"%*s %#x",&mem[0].cache[1].size);
	   }
           if (strstr(line,"dcache-line-size:")) {
	      sscanf(line,"%*s %#x",&mem[0].cache[1].line_size);
	   }
           if (strstr(line,"dcache-associativity:")) {
	      sscanf(line,"%*s %#x",&mem[0].cache[1].associativity);
	   }

           if (strstr(line,"ecache-size:")) {
	      sscanf(line,"%*s %#x",&mem[1].cache[0].size);
	   }
           if (strstr(line,"ecache-line-size:")) {
	      sscanf(line,"%*s %#x",&mem[1].cache[0].line_size);
	   }
           if (strstr(line,"ecache-associativity:")) {
	      sscanf(line,"%*s %#x",&mem[1].cache[0].associativity);
	   }

           if (strstr(line,"#itlb-entries:")) {
	      sscanf(line,"%*s %#x",&mem[0].tlb[0].num_entries);
	   }
           if (strstr(line,"#dtlb-entries:")) {
	      sscanf(line,"%*s %#x",&mem[0].tlb[1].num_entries);
	   }

	}
       

        pclose(pipe);

	/* I-Cache -> L1$ instruction */
	mem[0].cache[0].type = PAPI_MH_TYPE_INST;
	if (mem[0].cache[0].line_size!=0) mem[0].cache[0].num_lines =
		mem[0].cache[0].size / mem[0].cache[0].line_size;

	/* D-Cache -> L1$ data */
	mem[0].cache[1].type =
		PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU;
	if (mem[0].cache[1].line_size!=0) mem[0].cache[1].num_lines =
		mem[0].cache[1].size / mem[0].cache[1].line_size;


	/* ITLB -> TLB instruction */
	mem[0].tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
        /* assume fully associative */
	mem[0].tlb[0].associativity = mem[0].tlb[0].num_entries;

	/* DTLB -> TLB data */
	mem[0].tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
        /* assume fully associative */
	mem[0].tlb[1].associativity = mem[0].tlb[1].num_entries;

	/* L2$ unified */
	mem[1].cache[0].type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_WB
		| PAPI_MH_TYPE_PSEUDO_LRU;
	if (mem[1].cache[0].line_size!=0) mem[1].cache[0].num_lines =
		mem[1].cache[0].size / mem[1].cache[0].line_size;

	/* Indicate we have two levels filled in the hierarchy */
	hw->mem_hierarchy.levels = 2;

	return PAPI_OK;
}




int
_solaris_get_dmem_info( PAPI_dmem_info_t * d )
{

	FILE *fd;
	struct psinfo psi;

	if ( ( fd = fopen( "/proc/self/psinfo", "r" ) ) == NULL ) {
		SUBDBG( "fopen(/proc/self) errno %d", errno );
		return ( PAPI_ESYS );
	}

	fread( ( void * ) &psi, sizeof ( struct psinfo ), 1, fd );
	fclose( fd );

	d->pagesize = sysconf( _SC_PAGESIZE );
	d->size = d->pagesize * sysconf( _SC_PHYS_PAGES );
	d->resident = ( ( 1024 * psi.pr_size ) / d->pagesize );
	d->high_water_mark = PAPI_EINVAL;
	d->shared = PAPI_EINVAL;
	d->text = PAPI_EINVAL;
	d->library = PAPI_EINVAL;
	d->heap = PAPI_EINVAL;
	d->locked = PAPI_EINVAL;
	d->stack = PAPI_EINVAL;

	return PAPI_OK;

}

int
_niagara2_get_memory_info( PAPI_hw_info_t * hw, int id )
{
	PAPI_mh_level_t *mem = hw->mem_hierarchy.level;


	/* I-Cache -> L1$ instruction */
	/* FIXME: The policy used at this cache is unknown to PAPI. LSFR with random
	   replacement. */
	mem[0].cache[0].type = PAPI_MH_TYPE_INST;
	mem[0].cache[0].size = 16 * 1024;	// 16 Kb
	mem[0].cache[0].line_size = 32;
	mem[0].cache[0].num_lines =
		mem[0].cache[0].size / mem[0].cache[0].line_size;
	mem[0].cache[0].associativity = 8;

	/* D-Cache -> L1$ data */
	mem[0].cache[1].type =
		PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU;
	mem[0].cache[1].size = 8 * 1024;	// 8 Kb
	mem[0].cache[1].line_size = 16;
	mem[0].cache[1].num_lines =
		mem[0].cache[1].size / mem[0].cache[1].line_size;
	mem[0].cache[1].associativity = 4;

	/* ITLB -> TLB instruction */
	mem[0].tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
	mem[0].tlb[0].num_entries = 64;
	mem[0].tlb[0].associativity = 64;

	/* DTLB -> TLB data */
	mem[0].tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
	mem[0].tlb[1].num_entries = 128;
	mem[0].tlb[1].associativity = 128;

	/* L2$ unified */
	mem[1].cache[0].type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_WB
		| PAPI_MH_TYPE_PSEUDO_LRU;
	mem[1].cache[0].size = 4 * 1024 * 1024;	// 4 Mb
	mem[1].cache[0].line_size = 64;
	mem[1].cache[0].num_lines =
		mem[1].cache[0].size / mem[1].cache[0].line_size;
	mem[1].cache[0].associativity = 16;

	/* Indicate we have two levels filled in the hierarchy */
	hw->mem_hierarchy.levels = 2;

	return PAPI_OK;
}
