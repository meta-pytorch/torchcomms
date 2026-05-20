/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papivi.h
* CVS:     $Id$
* Author:  dan terpstra
*          terpstra@cs.utk.edu
* Mods:    your name here
*          yourname@cs.esu.edu
*
* Include this file INSTEAD OF "papi.h" in your application code
* to provide semitransparent version independent PAPI support.
* Follow the rules described below and elsewhere to facilitate
* this support.
*
*/

#ifndef _PAPIVI
#define _PAPIVI

#include "papi.h"

/***************************************************************************
* If PAPI_VERSION is not defined, then papi.h is for PAPI 2.
* The preprocessor block below contains the definitions, data structures,
* macros and code needed to emulate much of the PAPI 3 interface in code
* linking to the PAPI 2 library.
****************************************************************************/
#ifndef PAPI_VERSION


#define PAPI_VERSION_NUMBER(maj,min,rev) (((maj)<<16) | ((min)<<8) | (rev))
#define PAPI_VERSION_MAJOR(x)   	(((x)>>16)    & 0xffff)
#define PAPI_VERSION_MINOR(x)		(((x)>>8)     & 0xff)
#define PAPI_VERSION_REVISION(x)	((x)          & 0xff)

/* This is the PAPI version on which we are running */
#define PAPI_VERSION  			PAPI_VERSION_NUMBER(2,3,4)

/* This is the PAPI 3 version with which we are compatible */
#define PAPI_VI_VERSION  			PAPI_VERSION_NUMBER(3,0,6)

/* PAPI 3 has an error code not defined for PAPI 2 */
#define PAPI_EPERM   PAPI_EMISC	/* You lack the necessary permissions */

/*
* These are defined in papi_internal.h for PAPI 2.
* They need to be exposed for version independent PAPI code to work.
*/
//#define PRESET_MASK     0x80000000
#define PAPI_PRESET_MASK 0x80000000
//#define PRESET_AND_MASK 0x7FFFFFFF
#define PAPI_PRESET_AND_MASK 0x7FFFFFFF
#define PAPI_NATIVE_MASK 0x40000000
#define PAPI_NATIVE_AND_MASK 0x3FFFFFFF

/*
* Some PAPI 3 definitions for PAPI_{set,get}_opt() map
* onto single definitions in PAPI 2. The new definitions
* (shown below) should be used to guarantee PAPI 3 compatibility.
*/
#define PAPI_CLOCKRATE     PAPI_GET_CLOCKRATE
#define PAPI_MAX_HWCTRS    PAPI_GET_MAX_HWCTRS
#define PAPI_HWINFO        PAPI_GET_HWINFO
#define PAPI_EXEINFO       PAPI_GET_EXEINFO
#define PAPI_MAX_CPUS      PAPI_GET_MAX_CPUS
#define PAPI_CPUS          PAPI_GET_CPUS
#define PAPI_THREADS       PAPI_GET_THREADS

/*
* PAPI 2 defined only one string length.
* PAPI 3 defines three. This insures limited compatibility.
*/
#define PAPI_MIN_STR_LEN   PAPI_MAX_STR_LEN
#define PAPI_HUGE_STR_LEN  PAPI_MAX_STR_LEN

/*
* PAPI 2 always profiles into 16-bit buckets.
* PAPI 3 supports multiple bucket sizes.
* Exercise caution if these defines appear in your code.
* There is a potential for data overflow in PAPI 2.
*/
#define PAPI_PROFIL_BUCKET_16 0
#define PAPI_PROFIL_BUCKET_32 0
#define PAPI_PROFIL_BUCKET_64 0

/*
* PAPI 3 defines a new eventcode that can often be emulated
* successfully on PAPI 2. PAPI 3 also deprecates two eventcodes
* found in PAPI 2:
* PAPI_IPS   (instructions per second)
* PAPI_FLOPS (floating point instructions per second)
* Don't use these eventcodes in version independent code
*/
#define PAPI_FP_OPS PAPI_FP_INS

/*
* Two new data structures are introduced in PAPI 3 that are 
* required to support the functionality of:
* PAPI_get_event_info() and
* PAPI_get_executable_info()
* These structures are reproduced below.
* They MUST stay synchronized with their counterparts in papi.h
*/
#define PAPI_MAX_INFO_TERMS 8
typedef struct event_info
{
	unsigned int event_code;
	unsigned int count;
	char symbol[PAPI_MAX_STR_LEN + 3];
	char short_descr[PAPI_MIN_STR_LEN];
	char long_descr[PAPI_HUGE_STR_LEN];
	char derived[PAPI_MIN_STR_LEN];
	char postfix[PAPI_MIN_STR_LEN];
	unsigned int code[PAPI_MAX_INFO_TERMS];
	char name[PAPI_MAX_INFO_TERMS]
		[PAPI_MIN_STR_LEN];
	char note[PAPI_HUGE_STR_LEN];
} PAPI_event_info_t;

/* Possible values for the 'modifier' parameter of the PAPI_enum_event call.
   This enumeration is new in PAPI 3. It will act as a nop in PAPI 2, but
   must be defined for code compatibility.
*/
enum
{
	PAPI_ENUM_ALL = 0,				   /* Always enumerate all events */
	PAPI_PRESET_ENUM_AVAIL,	 /* Enumerate events that exist here */

	/* PAPI PRESET section */
	PAPI_PRESET_ENUM_INS,	 /* Instruction related preset events */
	PAPI_PRESET_ENUM_BR,	 /* branch related preset events */
	PAPI_PRESET_ENUM_MEM,	 /* memory related preset events */
	PAPI_PRESET_ENUM_TLB,	 /* Translation Lookaside Buffer events */
	PAPI_PRESET_ENUM_FP,	 /* Floating Point related preset events */

	/* Pentium 4 specific section */
	PAPI_PENT4_ENUM_GROUPS = 0x100,	/* 45 groups + custom + user */
	PAPI_PENT4_ENUM_COMBOS,	 /* all combinations of mask bits for given group */
	PAPI_PENT4_ENUM_BITS,	 /* all individual bits for given group */

	/* POWER 4 specific section */
	PAPI_PWR4_ENUM_GROUPS = 0x200	/* Enumerate groups an event belongs to */
};

typedef struct _papi_address_map
{
	char mapname[PAPI_HUGE_STR_LEN];
	vptr_t text_start;				   /* Start address of program text segment */
	vptr_t text_end;				   /* End address of program text segment */
	vptr_t data_start;				   /* Start address of program data segment */
	vptr_t data_end;				   /* End address of program data segment */
	vptr_t bss_start;				   /* Start address of program bss segment */
	vptr_t bss_end;				   /* End address of program bss segment */
} PAPI_address_map_t;

/*
 * PAPI 3 beta 3 introduces new structures for static memory description.
 * These include structures for tlb and cache description, a structure
 * to describe a level in the memory hierarchy, and a structure 
 * to describe all levels of the hierarchy.
 * These structures, and the requisite data types are defined below.
 */

   /* All sizes are in BYTES */
   /* Except tlb size, which is in entries */

#define PAPI_MAX_MEM_HIERARCHY_LEVELS 	  3
#define PAPI_MH_TYPE_EMPTY    0x0
#define PAPI_MH_TYPE_INST	   0x1
#define PAPI_MH_TYPE_DATA     0x2
#define PAPI_MH_TYPE_UNIFIED  PAPI_MH_TYPE_INST|PAPI_MH_TYPE_DATA

typedef struct _papi_mh_tlb_info
{
	int type;						   /* Empty, unified, data, instr */
	int num_entries;
	int associativity;
} PAPI_mh_tlb_info_t;

typedef struct _papi_mh_cache_info
{
	int type;						   /* Empty, unified, data, instr */
	int size;
	int line_size;
	int num_lines;
	int associativity;
} PAPI_mh_cache_info_t;

typedef struct _papi_mh_level_info
{
	PAPI_mh_tlb_info_t tlb[2];
	PAPI_mh_cache_info_t cache[2];
} PAPI_mh_level_t;

typedef struct _papi_mh_info
{									   /* mh for mem hierarchy maybe? */
	int levels;
	PAPI_mh_level_t level[PAPI_MAX_MEM_HIERARCHY_LEVELS];
} PAPI_mh_info_t;

/*
* Three data structures are modified in PAPI 3
* These modifications are 
* required to support the functionality of:
* PAPI_get_hardware_info() and
* PAPI_get_executable_info()
* These structures are reproduced below.
* They MUST stay synchronized with their counterparts in papi.h
* To avoid namespace collisions, these structures have been renamed
* to PAPIvi_xxx, and must also be renamed in your code.
*/
typedef struct _papi3_hw_info
{
	int ncpu;						   /* Number of CPUs in an SMP Node */
	int nnodes;						   /* Number of Nodes in the entire system */
	int totalcpus;					   /* Total number of CPUs in the entire system */
	int vendor;						   /* Vendor number of CPU */
	char vendor_string[PAPI_MAX_STR_LEN];	/* Vendor string of CPU */
	int model;						   /* Model number of CPU */
	char model_string[PAPI_MAX_STR_LEN];	/* Model string of CPU */
	float revision;					   /* Revision of CPU */
	float mhz;						   /* Cycle time of this CPU, *may* be estimated at 
									      init time with a quick timing routine */

	PAPI_mh_info_t mem_hierarchy;
} PAPIvi_hw_info_t;

typedef struct _papi3_preload_option
{
	char lib_preload_env[PAPI_MAX_STR_LEN];	/* Model string of CPU */
	char lib_preload_sep;
	char lib_dir_env[PAPI_MAX_STR_LEN];
	char lib_dir_sep;
} PAPIvi_preload_option_t;

typedef struct _papi3_program_info
{
	char fullname[PAPI_MAX_STR_LEN];   /* path+name */
	char name[PAPI_MAX_STR_LEN];	   /* name */
	PAPI_address_map_t address_info;
	PAPIvi_preload_option_t preload_info;
} PAPIvi_exe_info_t;


/*
* The Low Level API
* Functions in this API are classified in 4 basic categories:
* Modified:   13 functions
* New:         8 functions
* Unchanged:  32 functions
* Deprecated:  9 functions
*
* Each of these categories is discussed further below.
*/

/*
* Modified functions are further divided into 4 subcategories:
* Dereferencing changes: 6 functions
*     These functions simply substitute an EventSet value for
*     a pointer to an EventSet. In the case of PAPI_remove_event{s}()
*     there is also a name change.
* Name changes:          1 function
*     This is a simple name change with no change in functionality.
* Parameter changes:     4 functions
*     Several functions have changed functionality reflected in changed
*     parameters:
*     PAPI_{un}lock() supports multiple locks in PAPI 3
*     PAPI_profil() supports multiple bucket sizes in PAPI 3
*     PAPI_thread_init() removes an unused parameter in PAPI 3
* New functionality:     2 functions
*     These functions support new data in revised data structures
*     The code implemented here maps the old structures to the new
*     where possible.
*/

 /* Modified Functons: Dereferencing changes */
#define PAPIvi_add_event(EventSet, Event) \
          PAPI_add_event(&EventSet, Event)
#define PAPIvi_add_events(EventSet, Events, number) \
          PAPI_add_events(&EventSet, Events, number)
#define PAPIvi_cleanup_eventset(EventSet) \
          PAPI_cleanup_eventset(&EventSet)
#define PAPIvi_remove_event(EventSet, EventCode) \
          PAPI_rem_event(&EventSet, EventCode)
#define PAPIvi_remove_events(EventSet, Events, number) \
          PAPI_rem_events(&EventSet, Events, number)
#define PAPIvi_set_multiplex(EventSet) \
          PAPI_set_multiplex(&EventSet)

 /* Modified Functons: Name changes */
#define PAPIvi_is_initialized \
          PAPI_initialized

 /* Modified Functons: Parameter changes */
#define PAPIvi_lock(lck) \
          PAPI_lock()
#define PAPIvi_profil(buf, bufsiz, offset, scale, EventSet, EventCode, threshold, flags) \
          PAPI_profil((unsigned short *)buf, bufsiz, (unsigned long)offset, scale, EventSet, EventCode, threshold, flags)
#define PAPIvi_thread_init(id_fn) \
          PAPI_thread_init(id_fn, 0)
#define PAPIvi_unlock(lck) \
          PAPI_unlock()

 /* Modified Functons: New functionality */
static const PAPIvi_exe_info_t *
PAPIvi_get_executable_info( void )
{
	static PAPIvi_exe_info_t prginfo3;
	const PAPI_exe_info_t *prginfo2 = PAPI_get_executable_info(  );

	if ( prginfo2 == NULL )
		return ( NULL );

	strcpy( prginfo3.fullname, prginfo2->fullname );
	strcpy( prginfo3.name, prginfo2->name );
	prginfo3.address_info.mapname[0] = 0;
	prginfo3.address_info.text_start = prginfo2->text_start;
	prginfo3.address_info.text_end = prginfo2->text_end;
	prginfo3.address_info.data_start = prginfo2->data_start;
	prginfo3.address_info.data_end = prginfo2->data_end;
	prginfo3.address_info.bss_start = prginfo2->bss_start;
	prginfo3.address_info.bss_end = prginfo2->bss_end;
	strcpy( prginfo3.preload_info.lib_preload_env, prginfo2->lib_preload_env );

	return ( &prginfo3 );
}

static const PAPIvi_hw_info_t *
PAPIvi_get_hardware_info( void )
{
	static PAPIvi_hw_info_t papi3_hw_info;
	const PAPI_hw_info_t *papi2_hw_info = PAPI_get_hardware_info(  );
	const PAPI_mem_info_t *papi2_mem_info = PAPI_get_memory_info(  );

	/* Copy the basic hardware info (same in both structures */
	memcpy( &papi3_hw_info, papi2_hw_info, sizeof ( PAPI_hw_info_t ) );

	memset( &papi3_hw_info.mem_hierarchy, 0, sizeof ( PAPI_mh_info_t ) );
	/* check for a unified tlb */
	if ( papi2_mem_info->total_tlb_size &&
		 papi2_mem_info->itlb_size == 0 && papi2_mem_info->dtlb_size == 0 ) {
		papi3_hw_info.mem_hierarchy.level[0].tlb[0].type = PAPI_MH_TYPE_UNIFIED;
		papi3_hw_info.mem_hierarchy.level[0].tlb[0].num_entries =
			papi2_mem_info->total_tlb_size;
	} else {
		if ( papi2_mem_info->itlb_size ) {
			papi3_hw_info.mem_hierarchy.level[0].tlb[0].type =
				PAPI_MH_TYPE_INST;
			papi3_hw_info.mem_hierarchy.level[0].tlb[0].num_entries =
				papi2_mem_info->itlb_size;
			papi3_hw_info.mem_hierarchy.level[0].tlb[0].associativity =
				papi2_mem_info->itlb_assoc;
		}
		if ( papi2_mem_info->dtlb_size ) {
			papi3_hw_info.mem_hierarchy.level[0].tlb[1].type =
				PAPI_MH_TYPE_DATA;
			papi3_hw_info.mem_hierarchy.level[0].tlb[1].num_entries =
				papi2_mem_info->dtlb_size;
			papi3_hw_info.mem_hierarchy.level[0].tlb[1].associativity =
				papi2_mem_info->dtlb_assoc;
		}
	}
	/* check for a unified level 1 cache */
	if ( papi2_mem_info->total_L1_size )
		papi3_hw_info.mem_hierarchy.levels = 1;
	if ( papi2_mem_info->total_L1_size &&
		 papi2_mem_info->L1_icache_size == 0 &&
		 papi2_mem_info->L1_dcache_size == 0 ) {
		papi3_hw_info.mem_hierarchy.level[0].cache[0].type =
			PAPI_MH_TYPE_UNIFIED;
		papi3_hw_info.mem_hierarchy.level[0].cache[0].size =
			papi2_mem_info->total_L1_size << 10;
	} else {
		if ( papi2_mem_info->L1_icache_size ) {
			papi3_hw_info.mem_hierarchy.level[0].cache[0].type =
				PAPI_MH_TYPE_INST;
			papi3_hw_info.mem_hierarchy.level[0].cache[0].size =
				papi2_mem_info->L1_icache_size << 10;
			papi3_hw_info.mem_hierarchy.level[0].cache[0].associativity =
				papi2_mem_info->L1_icache_assoc;
			papi3_hw_info.mem_hierarchy.level[0].cache[0].num_lines =
				papi2_mem_info->L1_icache_lines;
			papi3_hw_info.mem_hierarchy.level[0].cache[0].line_size =
				papi2_mem_info->L1_icache_linesize;
		}
		if ( papi2_mem_info->L1_dcache_size ) {
			papi3_hw_info.mem_hierarchy.level[0].cache[1].type =
				PAPI_MH_TYPE_DATA;
			papi3_hw_info.mem_hierarchy.level[0].cache[1].size =
				papi2_mem_info->L1_dcache_size << 10;
			papi3_hw_info.mem_hierarchy.level[0].cache[1].associativity =
				papi2_mem_info->L1_dcache_assoc;
			papi3_hw_info.mem_hierarchy.level[0].cache[1].num_lines =
				papi2_mem_info->L1_dcache_lines;
			papi3_hw_info.mem_hierarchy.level[0].cache[1].line_size =
				papi2_mem_info->L1_dcache_linesize;
		}
	}

	/* check for level 2 cache info */
	if ( papi2_mem_info->L2_cache_size ) {
		papi3_hw_info.mem_hierarchy.levels = 2;
		papi3_hw_info.mem_hierarchy.level[1].cache[0].type =
			PAPI_MH_TYPE_UNIFIED;
		papi3_hw_info.mem_hierarchy.level[1].cache[0].size =
			papi2_mem_info->L2_cache_size << 10;
		papi3_hw_info.mem_hierarchy.level[1].cache[0].associativity =
			papi2_mem_info->L2_cache_assoc;
		papi3_hw_info.mem_hierarchy.level[1].cache[0].num_lines =
			papi2_mem_info->L2_cache_lines;
		papi3_hw_info.mem_hierarchy.level[1].cache[0].line_size =
			papi2_mem_info->L2_cache_linesize;
	}

	/* check for level 3 cache info */
	if ( papi2_mem_info->L3_cache_size ) {
		papi3_hw_info.mem_hierarchy.levels = 3;
		papi3_hw_info.mem_hierarchy.level[2].cache[0].type =
			PAPI_MH_TYPE_UNIFIED;
		papi3_hw_info.mem_hierarchy.level[2].cache[0].size =
			papi2_mem_info->L3_cache_size << 10;
		papi3_hw_info.mem_hierarchy.level[2].cache[0].associativity =
			papi2_mem_info->L3_cache_assoc;
		papi3_hw_info.mem_hierarchy.level[2].cache[0].num_lines =
			papi2_mem_info->L3_cache_lines;
		papi3_hw_info.mem_hierarchy.level[2].cache[0].line_size =
			papi2_mem_info->L3_cache_linesize;
	}

	return ( &papi3_hw_info );
}

/*
* New functions are either supported or unsupported.
* Of the three supported functions, two replaced deprecated functions
* to describe events, and one is simply a convenience function.
* The five unsupported new functions include three related to thread
* functionality, a convenience function to return the number of events
* in an event set, and a function to query information about shared libraries.
*/

 /* New Supported Functions */
static int
PAPIvi_enum_event( int *EventCode, int modifier )
{
	int i = *EventCode;
	const PAPI_preset_info_t *presets = PAPI_query_all_events_verbose(  );
	i &= PAPI_PRESET_AND_MASK;
	while ( ++i < PAPI_MAX_PRESET_EVENTS ) {
		if ( ( !modifier ) || ( presets[i].avail ) ) {
			*EventCode = i | PAPI_PRESET_MASK;
			if ( presets[i].event_name != NULL )
				return ( PAPI_OK );
			else
				return ( PAPI_ENOEVNT );
		}
	}
	return ( PAPI_ENOEVNT );
}

static int
PAPIvi_get_event_info( int EventCode, PAPI_event_info_t * info )
{
	int i;
	const PAPI_preset_info_t *info2 = PAPI_query_all_events_verbose(  );

	i = EventCode & PAPI_PRESET_AND_MASK;
	if ( ( i >= PAPI_MAX_PRESET_EVENTS ) || ( info2[i].event_name == NULL ) )
		return ( PAPI_ENOTPRESET );

	info->event_code = info2[i].event_code;
	info->count = info2[i].avail;
	if ( info2[i].flags & PAPI_DERIVED ) {
		info->count++;
		strcpy( info->derived, "DERIVED" );
	}
	if ( info2[i].event_name == NULL )
		info->symbol[0] = 0;
	else
		strcpy( info->symbol, info2[i].event_name );
	if ( info2[i].event_label == NULL )
		info->short_descr[0] = 0;
	else
		strcpy( info->short_descr, info2[i].event_label );
	if ( info2[i].event_descr == NULL )
		info->long_descr[0] = 0;
	else
		strcpy( info->long_descr, info2[i].event_descr );
	if ( info2[i].event_note == NULL )
		info->note[0] = 0;
	else
		strcpy( info->note, info2[i].event_note );
	return ( PAPI_OK );
}

/*
static int PAPI_get_multiplex(int EventSet)
{
   PAPI_option_t popt;
   int retval;

   popt.multiplex.eventset = EventSet;
   retval = PAPI_get_opt(PAPI_GET_MULTIPLEX, &popt);
   if (retval < 0)
      retval = 0;
   return retval;
}
*/

 /* New Unsupported Functions */
#define PAPIvi_get_shared_lib_info \
          PAPI_get_shared_lib_info
#define PAPIvi_get_thr_specific(tag, ptr) \
          PAPI_get_thr_specific(tag, ptr)
#define PAPIvi_num_events(EventSet) \
          PAPI_num_events(EventSet)
#define PAPIvi_register_thread \
          PAPI_register_thread
#define PAPIvi_set_thr_specific(tag, ptr) \
          PAPI_set_thr_specific(tag, ptr)

/*
* Over half of the functions in the Low Level API remain unchanged
* These are included in the macro list in case they do change in future
* revisions, and to simplify the naming conventions for writing 
* version independent PAPI code.
*/

#define PAPIvi_accum(EventSet, values) \
          PAPI_accum(EventSet, values)
#define PAPIvi_create_eventset(EventSet) \
          PAPI_create_eventset(EventSet)
#define PAPIvi_destroy_eventset(EventSet) \
          PAPI_destroy_eventset(EventSet)
#define PAPIvi_event_code_to_name(EventCode, out) \
          PAPI_event_code_to_name(EventCode, out)
#define PAPIvi_event_name_to_code(in, out) \
          PAPI_event_name_to_code(in, out)
#define PAPIvi_get_dmem_info(option) \
          PAPI_get_dmem_info(option)
#define PAPIvi_get_opt(option, ptr) \
          PAPI_get_opt(option, ptr)
#define PAPIvi_get_real_cyc \
          PAPI_get_real_cyc
#define PAPIvi_get_real_usec \
          PAPI_get_real_usec
#define PAPIvi_get_virt_cyc \
          PAPI_get_virt_cyc
#define PAPIvi_get_virt_usec \
          PAPI_get_virt_usec
#define PAPIvi_library_init(version) \
          PAPI_library_init(version)
#define PAPIvi_list_events(EventSet, Events, number) \
          PAPI_list_events(EventSet, Events, number)
#define PAPIvi_multiplex_init \
          PAPI_multiplex_init
#define PAPIvi_num_hwctrs \
          PAPI_num_hwctrs
#define PAPIvi_overflow(EventSet, EventCode, threshold, flags, handler) \
          PAPI_overflow(EventSet, EventCode, threshold, flags, handler)
#define PAPIvi_perror( s ) \
          PAPI_perror( s )
#define PAPIvi_query_event(EventCode) \
          PAPI_query_event(EventCode)
#define PAPIvi_read(EventSet, values) \
          PAPI_read(EventSet, values)
#define PAPIvi_reset(EventSet) \
          PAPI_reset(EventSet)
#define PAPIvi_set_debug(level) \
          PAPI_set_debug(level)
#define PAPIvi_set_domain(domain) \
          PAPI_set_domain(domain)
#define PAPIvi_set_granularity(granularity) \
          PAPI_set_granularity(granularity)
#define PAPIvi_set_opt(option, ptr) \
          PAPI_set_opt(option, ptr)
#define PAPIvi_shutdown \
          PAPI_shutdown
#define PAPIvi_sprofil(prof, profcnt, EventSet, EventCode, threshold, flags) \
          PAPI_sprofil(prof, profcnt, EventSet, EventCode, threshold, flags)
#define PAPIvi_start(EventSet) \
          PAPI_start(EventSet)
#define PAPIvi_state(EventSet, status) \
          PAPI_state(EventSet, status)
#define PAPIvi_stop(EventSet, values) \
          PAPI_stop(EventSet, values)
#define PAPIvi_strerror(err) \
          PAPI_strerror(err)
#define PAPIvi_thread_id \
          PAPI_thread_id
#define PAPIvi_write(EventSet, values) \
          PAPI_write(EventSet, values)

/*
* Of the nine functions deprecated from PAPI 2 to PAPI 3,
* three (PAPI_add_pevent, PAPI_restore, and PAPI_save) were
* never implemented, and four dealt with describing events.
* Two remain:
* PAPI_get_overflow_address() must still be used in version specific overflow handlers
* PAPI_profil_hw() was rarely used, and only on platforms supporting hardware overflow.
* The prototypes of these functions are shown below for completeness.
*/
/*
int PAPI_add_pevent(int *EventSet, int code, void *inout);
void *PAPI_get_overflow_address(void *context);
int PAPI_profil_hw(unsigned short *buf, unsigned bufsiz, unsigned long offset, \
          unsigned scale, int EventSet, int EventCode, int threshold, int flags);
const PAPI_preset_info_t *PAPI_query_all_events_verbose(void);
int PAPI_describe_event(char *name, int *EventCode, char *description);
int PAPI_label_event(int EventCode, char *label);
int PAPI_query_event_verbose(int EventCode, PAPI_preset_info_t *info);
int PAPI_restore(void);
int PAPI_save(void);
*/


/*
* The High Level API
* There are 8 functions in this API.
* 6 are unchanged, and 2 are new.
* Of the new functions, one is emulated and one is unsupported.
*/

/* Unchanged Functions */
#define PAPIvi_accum_counters(values, array_len) \
          PAPI_accum_counters(values, array_len)
#define PAPIvi_num_counters \
          PAPI_num_counters
#define PAPIvi_read_counters(values, array_len) \
          PAPI_read_counters(values, array_len)
#define PAPIvi_start_counters(Events, array_len) \
          PAPI_start_counters(Events, array_len)
#define PAPIvi_stop_counters(values, array_len) \
          PAPI_stop_counters(values, array_len)
#define PAPIvi_flops(rtime, ptime, flpops, mflops) \
          PAPI_flops(rtime, ptime, flpops, mflops)

 /* New Supported Functions */
#define PAPIvi_flips(rtime, ptime, flpins, mflips) \
          PAPI_flops(rtime, ptime, flpins, mflips)

 /* New Unupported Functions */
#define PAPIvi_ipc(rtime, ptime, ins, ipc) \
          PAPI_ipc(rtime, ptime, ins, ipc)


/*******************************************************************************
* If PAPI_VERSION is defined, and the MAJOR version number is 3,
* then papi.h is for PAPI 3.
* The preprocessor block below contains definitions and macros needed to 
* allow version independent linking to the PAPI 3 library.
* Other than a handful of definitions to support calls to PAPI_{get,set}_opt(),
* this layer simply converts version independent names to PAPI 3 library calls.
********************************************************************************/
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3)

/*
* The following option definitions reflect the fact that PAPI 2 had separate 
* definitions for options to PAPI_set_opt and PAPI_get_opt, while PAPI 3 has
* only a single set for both. By using the older naming convention, you can 
* create platform independent code for these calls.
*/

#define PAPI_SET_DEBUG     PAPI_DEBUG
#define PAPI_GET_DEBUG     PAPI_DEBUG

#define PAPI_SET_MULTIPLEX PAPI_MULTIPLEX
#define PAPI_GET_MULTIPLEX PAPI_MULTIPLEX

#define PAPI_SET_DEFDOM    PAPI_DEFDOM
#define PAPI_GET_DEFDOM    PAPI_DEFDOM

#define PAPI_SET_DOMAIN    PAPI_DOMAIN
#define PAPI_GET_DOMAIN    PAPI_DOMAIN

#define PAPI_SET_DEFGRN    PAPI_DEFGRN
#define PAPI_GET_DEFGRN    PAPI_DEFGRN

#define PAPI_SET_GRANUL    PAPI_GRANUL
#define PAPI_GET_GRANUL    PAPI_GRANUL

#define PAPI_SET_INHERIT   PAPI_INHERIT
#define PAPI_GET_INHERIT   PAPI_INHERIT

#define PAPI_GET_NUMCTRS   PAPI_NUMCTRS
#define PAPI_SET_NUMCTRS   PAPI_NUMCTRS

#define PAPI_SET_PROFIL    PAPI_PROFIL
#define PAPI_GET_PROFIL    PAPI_PROFIL

/*
* These macros are simple pass-throughs to PAPI 3 structures
*/
#define PAPIvi_hw_info_t   PAPI_hw_info_t
#define PAPIvi_exe_info_t  PAPI_exe_info_t

/*
* The following macros are simple pass-throughs to PAPI 3 library calls
*/
 /* The Low Level API */
#define PAPIvi_accum(EventSet, values) \
          PAPI_accum(EventSet, values)
#define PAPIvi_add_event(EventSet, Event) \
          PAPI_add_event(EventSet, Event)
#define PAPIvi_add_events(EventSet, Events, number) \
          PAPI_add_events(EventSet, Events, number)
#define PAPIvi_cleanup_eventset(EventSet) \
          PAPI_cleanup_eventset(EventSet)
#define PAPIvi_create_eventset(EventSet) \
          PAPI_create_eventset(EventSet)
#define PAPIvi_destroy_eventset(EventSet) \
          PAPI_destroy_eventset(EventSet)
#define PAPIvi_enum_event(EventCode, modifier) \
          PAPI_enum_event(EventCode, modifier)
#define PAPIvi_event_code_to_name(EventCode, out) \
          PAPI_event_code_to_name(EventCode, out)
#define PAPIvi_event_name_to_code(in, out) \
          PAPI_event_name_to_code(in, out)
#define PAPIvi_get_dmem_info(option) \
          PAPI_get_dmem_info(option)
#define PAPIvi_get_event_info(EventCode, info) \
          PAPI_get_event_info(EventCode, info)
#define PAPIvi_get_executable_info \
          PAPI_get_executable_info
#define PAPIvi_get_hardware_info \
          PAPI_get_hardware_info
#define PAPIvi_get_multiplex(EventSet) \
          PAPI_get_multiplex(EventSet)
#define PAPIvi_get_opt(option, ptr) \
          PAPI_get_opt(option, ptr)
#define PAPIvi_get_real_cyc \
          PAPI_get_real_cyc
#define PAPIvi_get_real_usec \
          PAPI_get_real_usec
#define PAPIvi_get_shared_lib_info \
          PAPI_get_shared_lib_info
#define PAPIvi_get_thr_specific(tag, ptr) \
          PAPI_get_thr_specific(tag, ptr)
#define PAPIvi_get_virt_cyc \
          PAPI_get_virt_cyc
#define PAPIvi_get_virt_usec \
          PAPI_get_virt_usec
#define PAPIvi_is_initialized \
          PAPI_is_initialized
#define PAPIvi_library_init(version) \
          PAPI_library_init(version)
#define PAPIvi_list_events(EventSet, Events, number) \
          PAPI_list_events(EventSet, Events, number)
#define PAPIvi_lock(lck) \
          PAPI_lock(lck)
#define PAPIvi_multiplex_init \
          PAPI_multiplex_init
#define PAPIvi_num_hwctrs \
          PAPI_num_hwctrs
#define PAPIvi_num_events(EventSet) \
          PAPI_num_events(EventSet)
#define PAPIvi_overflow(EventSet, EventCode, threshold, flags, handler) \
          PAPI_overflow(EventSet, EventCode, threshold, flags, handler)
#define PAPIvi_perror( s ) \
          PAPI_perror( s )
#define PAPIvi_profil(buf, bufsiz, offset, scale, EventSet, EventCode, threshold, flags) \
          PAPI_profil(buf, bufsiz, offset, scale, EventSet, EventCode, threshold, flags)
#define PAPIvi_query_event(EventCode) \
          PAPI_query_event(EventCode)
#define PAPIvi_read(EventSet, values) \
          PAPI_read(EventSet, values)
#define PAPIvi_register_thread \
          PAPI_register_thread
#define PAPIvi_remove_event(EventSet, EventCode) \
          PAPI_remove_event(EventSet, EventCode)
#define PAPIvi_remove_events(EventSet, Events, number) \
          PAPI_remove_events(EventSet, Events, number)
#define PAPIvi_reset(EventSet) \
          PAPI_reset(EventSet)
#define PAPIvi_set_debug(level) \
          PAPI_set_debug(level)
#define PAPIvi_set_domain(domain) \
          PAPI_set_domain(domain)
#define PAPIvi_set_granularity(granularity) \
          PAPI_set_granularity(granularity)
#define PAPIvi_set_multiplex(EventSet) \
          PAPI_set_multiplex(EventSet)
#define PAPIvi_set_opt(option, ptr) \
          PAPI_set_opt(option, ptr)
#define PAPIvi_set_thr_specific(tag, ptr) \
          PAPI_set_thr_specific(tag, ptr)
#define PAPIvi_shutdown \
          PAPI_shutdown
#define PAPIvi_sprofil(prof, profcnt, EventSet, EventCode, threshold, flags) \
          PAPI_sprofil(prof, profcnt, EventSet, EventCode, threshold, flags)
#define PAPIvi_start(EventSet) \
          PAPI_start(EventSet)
#define PAPIvi_state(EventSet, status) \
          PAPI_state(EventSet, status)
#define PAPIvi_stop(EventSet, values) \
          PAPI_stop(EventSet, values)
#define PAPIvi_strerror(err) \
          PAPI_strerror(err)
#define PAPIvi_thread_id \
          PAPI_thread_id
#define PAPIvi_thread_init(id_fn) \
          PAPI_thread_init(id_fn)
#define PAPIvi_unlock(lck) \
          PAPI_unlock(lck)
#define PAPIvi_write(EventSet, values) \
          PAPI_write(EventSet, values)

   /* The High Level API */

#define PAPIvi_accum_counters(values, array_len) \
          PAPI_accum_counters(values, array_len)
#define PAPIvi_num_counters \
          PAPI_num_counters
#define PAPIvi_read_counters(values, array_len) \
          PAPI_read_counters(values, array_len)
#define PAPIvi_start_counters(Events, array_len) \
          PAPI_start_counters(Events, array_len)
#define PAPIvi_stop_counters(values, array_len) \
          PAPI_stop_counters(values, array_len)
#define PAPIvi_flips(rtime, ptime, flpins, mflips) \
          PAPI_flips(rtime, ptime, flpins, mflips)
#define PAPIvi_flops(rtime, ptime, flpops, mflops) \
          PAPI_flops(rtime, ptime, flpops, mflops)
#define PAPIvi_ipc(rtime, ptime, ins, ipc) \
          PAPI_ipc(rtime, ptime, ins, ipc)


/*******************************************************************************
* If PAPI_VERSION is defined, and the MAJOR version number is not 3, then we
* generate an error message.
* This block allows us to support future version with a 
* version independent syntax.
********************************************************************************/
#else
#error Compiling against a not yet released PAPI version
#endif

#endif /* _PAPIVI */
