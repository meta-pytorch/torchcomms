/*
 * This file perfoms the following test:  memory info
 *
 * Author: Kevin London
 *         london@cs.utk.edu
 */
/** file papi_mem_info.c
  * @brief papi_mem_info utility.
  *	@page papi_mem_info
  *	@section NAME
  *		papi_mem_info - provides information on the memory architecture of the current processor. 
  *
  *	@section Synopsis
  *
  *	@section Description
  *		papi_mem_info is a PAPI utility program that reports information about 
  *		the cache memory architecture of the current processor, including number, 
  *		types, sizes and associativities of instruction and data caches and 
  *		Translation Lookaside Buffers.
  *
  *	@section Options
  *		This utility has no command line options.
  *
  *	@section Bugs
  *		There are no known bugs in this utility.
  *		If you find a bug, it should be reported to the
  *		PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
  */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"


int
main( int argc, char **argv )
{
	const PAPI_hw_info_t *meminfo = NULL;
	PAPI_mh_level_t *L;
	int i, j, retval;

	(void)argc;
	(void)argv;

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr,"Error! PAPI_library_init\n");
		return retval;
	}

	meminfo = PAPI_get_hardware_info(  );
	if (meminfo == NULL ) {
		fprintf(stderr,"Error! PAPI_get_hardware_info");
		return 2;
	}

	printf( "Memory Cache and TLB Hierarchy Information.\n" );
	printf( "------------------------------------------------------------------------\n" );

	/* Extract and report the tlb and cache information */
	L = ( PAPI_mh_level_t * ) & ( meminfo->mem_hierarchy.level[0] );
	printf( "TLB Information.\n  There may be multiple descriptors for each level of TLB\n" );
	printf( "  if multiple page sizes are supported.\n\n" );
	/* Scan the TLB structures */
	for ( i = 0; i < meminfo->mem_hierarchy.levels; i++ ) {
		for ( j = 0; j < PAPI_MH_MAX_LEVELS; j++ ) {
			switch ( PAPI_MH_CACHE_TYPE( L[i].tlb[j].type ) ) {
				case PAPI_MH_TYPE_UNIFIED:
					printf( "L%d Unified TLB:\n", i + 1 );
					break;
				case PAPI_MH_TYPE_DATA:
					printf( "L%d Data TLB:\n", i + 1 );
					break;
				case PAPI_MH_TYPE_INST:
					printf( "L%d Instruction TLB:\n", i + 1 );
					break;
			}
			if ( L[i].tlb[j].type ) {
				if ( L[i].tlb[j].page_size )
					printf( "  Page Size:         %6d KB\n",
							L[i].tlb[j].page_size >> 10 );
				printf( "  Number of Entries: %6d\n",
						L[i].tlb[j].num_entries );
				switch ( L[i].tlb[j].associativity ) {
					case 0: /* undefined */
						break;
					case 1:
						printf( "  Associativity:      Direct Mapped\n\n" );
						break;
					case SHRT_MAX:
						printf( "  Associativity:       Full\n\n" );
						break;
					default:
						printf( "  Associativity:     %6d\n\n",
								L[i].tlb[j].associativity );
						break;
				}
			}
		}
	}

	/* Scan the Cache structures */
	printf( "\nCache Information.\n\n" );
	for ( i = 0; i < meminfo->mem_hierarchy.levels; i++ ) {
		for ( j = 0; j < 2; j++ ) {
			switch ( PAPI_MH_CACHE_TYPE( L[i].cache[j].type ) ) {
				case PAPI_MH_TYPE_UNIFIED:
					printf( "L%d Unified Cache:\n", i + 1 );
					break;
				case PAPI_MH_TYPE_DATA:
					printf( "L%d Data Cache:\n", i + 1 );
					break;
				case PAPI_MH_TYPE_INST:
					printf( "L%d Instruction Cache:\n", i + 1 );
					break;
				case PAPI_MH_TYPE_TRACE:
					printf( "L%d Trace Buffer:\n", i + 1 );
					break;
				case PAPI_MH_TYPE_VECTOR:
					printf( "L%d Vector Cache:\n", i + 1 );
					break;
			}
			if ( L[i].cache[j].type ) {
				printf( "  Total size:        %6d KB\n  Line size:         %6d B\n  Number of Lines:   %6d\n  Associativity:     %6d\n\n",
						  ( L[i].cache[j].size ) >> 10, L[i].cache[j].line_size,
						  L[i].cache[j].num_lines,
						  L[i].cache[j].associativity );
			}
		}
	}

	return 0;
}
