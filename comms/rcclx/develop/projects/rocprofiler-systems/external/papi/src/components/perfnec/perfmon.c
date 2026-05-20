/*
* File:    perfmon.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu
*/

/* TODO LIST:
   - Events for all platforms
   - Derived events for all platforms
   - Latency profiling
   - BTB/IPIEAR sampling
   - Test on ITA2, Pentium 4
   - hwd_ntv_code_to_name
   - Make native map carry major events, not umasks
   - Enum event uses native_map not pfnec()
   - Hook up globals to be freed to sub_info
   - Better feature bit support for IEAR
*/
#include <sys/stat.h>
#include <fcntl.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"

#include "perfmon/perfmon_nec.h"
#include "perfnec.h"

#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"

#define PFNECLIB_MAX_PMDS 32


static int num_events=0;

static
int check_pmmr(void)
{
	uint64_t pmmr;
	asm volatile(
		"smir %0, %pmmr":"=r"(pmmr));
	if (pmmr != 0x0000000000001000) {
		fprintf(stderr, "PMMR is not expected: 0x%lx\n", pmmr);
		return -1;
	}
    return PAPI_OK;
}

static uint64_t diff56(uint64_t after, uint64_t before)
{
	return 0x00ffffffffffffff & (after - before);
}

static uint64_t diff52(uint64_t after, uint64_t before)
{
	return 0x000fffffffffffff & (after - before);
}


#define MAX_COUNTERS PFMLIB_MAX_PMCS
#define MAX_COUNTER_TERMS PFMLIB_MAX_PMCS

/* typedefs to conform to PAPI component layer code. */
/* these are void * in the PAPI framework layer code. */
typedef pfnec_control_state_t cmp_control_state_t;
typedef pfnec_context_t cmp_context_t;


/* Advance declarations */
static int _papi_pfnec_set_overflow( EventSetInfo_t * ESI, int EventIndex,
							int threshold );
papi_vector_t _perfnec_vector;


/* Static locals */

static int _perfnec_pfnec_pmu_type = -1;
//static pfneclib_regmask_t _perfmon2_pfnec_unavailable_pmcs;
//static pfneclib_regmask_t _perfmon2_pfnec_unavailable_pmds;

/* Debug functions */

#ifdef DEBUG
static void
dump_smpl_arg( pfnec_dfl_smpl_arg_t * arg )
{
}

static void
dump_sets( pfarg_setdesc_t * set, int num_sets )
{
}

static void
dump_setinfo( pfarg_setinfo_t * setinfo, int num_sets )
{
}

static void
dump_pmc( pfnec_control_state_t * ctl )
{
}

static void
dump_pmd( pfnec_control_state_t * ctl )
{
}

static void
dump_smpl_hdr( pfnec_dfl_smpl_hdr_t * hdr )
{
}

static void
dump_smpl( pfnec_dfl_smpl_entry_t * entry )
{
}
#endif

#define PFM_MAX_PMCDS 20

static int
_papi_pfnec_write_pmcs( pfnec_context_t * ctx, pfnec_control_state_t * ctl )
{
	return PAPI_OK;
}

static int
_papi_pfnec_write_pmds( pfnec_context_t * ctx, pfnec_control_state_t * ctl )
{
	return PAPI_OK;
}

static int
_papi_pfnec_read_pmds( pfnec_context_t * ctx, pfnec_control_state_t * ctl )
{
	return PAPI_OK;
}


/* This routine effectively does argument checking as the real magic will happen
   in compute_kernel_args. This just gets the value back from the kernel. */

static int
check_multiplex_timeout( int ctx_fd, unsigned long *timeout_ns )
{
	return ( PAPI_OK );
}

/* The below function is stolen from libpfnec from Stephane Eranian */
static int
detect_timeout_and_unavail_pmu_regs( pfneclib_regmask_t * r_pmcs,
									 pfneclib_regmask_t * r_pmds,
									 unsigned long *timeout_ns )
{
	return PAPI_OK;
}

/* BEGIN COMMON CODE */

static inline int
compute_kernel_args( hwd_control_state_t * ctl0 )
{
	return ( PAPI_OK );
}


static int
attach( hwd_control_state_t * ctl, unsigned long tid )
{
	return ( PAPI_OK );
}

static int
detach( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	return ( PAPI_OK );
}

static inline int
set_domain( hwd_control_state_t * ctl, int domain )
{
	return ( compute_kernel_args( ctl ) );
}

static inline int
set_granularity( hwd_control_state_t * this_state, int domain )
{
	return PAPI_OK;
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static inline int
set_inherit( int arg )
{
	return PAPI_ECMP;
}

static int
get_string_from_file( char *file, char *str, int len )
{
	return ( PAPI_OK );
}

static int
_papi_pfnec_init_component( int cidx )
{
    int e;
    num_events = 0;
    int status = check_pmmr();
    if (PAPI_OK != status) {
        // do something smart to disable components
        fprintf(stderr, "Placeholder: component disabled, error in check_pmmr\n");
    }
    for ( e = 0; e < PKG_NUM_EVENTS; ++e ) {
        /* compose string to individual event */
        size_t ret;

        ret = snprintf(perfnec_ntv_events[num_events].name,
                       sizeof(perfnec_ntv_events[num_events].name),
                       "%s", pkg_event_names[e]);
        if (ret <= 0 || sizeof(perfnec_ntv_events[num_events].name) <= ret) continue;
        ret = snprintf(perfnec_ntv_events[num_events].description,
                       sizeof(perfnec_ntv_events[num_events].description),
                       "%s", pkg_event_descs[e]);
        if (ret <= 0 || sizeof(perfnec_ntv_events[num_events].description) <= ret) continue;
        ret = snprintf(perfnec_ntv_events[num_events].units,
                       sizeof(perfnec_ntv_events[num_events].name),
                       "%s", pkg_units[e]);
        if (ret < 0 || sizeof(perfnec_ntv_events[num_events].name) <= ret) continue;

        perfnec_ntv_events[num_events].return_type = PAPI_DATATYPE_INT64;
        perfnec_ntv_events[num_events].type = pkg_events[e];

        perfnec_ntv_events[num_events].resources.selector = num_events + 1;

        num_events++;
    }

    // this is statically decided so far.
    _perfnec_vector.cmp_info.num_native_events = PKG_NUM_EVENTS;
    _perfnec_vector.cmp_info.num_cntrs = PKG_NUM_EVENTS;
    _perfnec_vector.cmp_info.num_mpx_cntrs = PKG_NUM_EVENTS;

    _perfnec_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
}

static int
_papi_pfnec_shutdown_component(  )
{
	return PAPI_OK;
}

static int
_papi_pfnec_init_thread( hwd_context_t * thr_ctx )
{
	return ( PAPI_OK );
}

/* reset the hardware counters */
static int
_papi_pfnec_reset( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	return ( PAPI_OK );
}

/* write(set) the hardware counters */
static int
_papi_pfnec_write( hwd_context_t * ctx, hwd_control_state_t * ctl,
				 long long *from )
{
	return ( PAPI_OK );
}

static int
_papi_pfnec_read( hwd_context_t * ctx, hwd_control_state_t * ctl,
				long long **events, int flags )
{
	uint64_t pmc[16];
	asm volatile(
		"smir %0,%pmc00\n"
		"smir %1,%pmc01\n"
		"smir %2,%pmc02\n"
		"smir %3,%pmc03\n"
		"smir %4,%pmc04\n"
		"smir %5,%pmc05\n"
		"smir %6,%pmc06\n"
		"smir %7,%pmc07\n"
		"smir %8,%pmc08\n"
		"smir %9,%pmc09\n"
		"smir %10,%pmc10\n"
		"smir %11,%pmc11\n"
		"smir %12,%pmc12\n"
		"smir %13,%pmc13\n"
		"smir %14,%pmc14\n"
		"smir %15,%pmc15\n"
		:"=r"(pmc[0]),
		 "=r"(pmc[1]),
		 "=r"(pmc[2]),
		 "=r"(pmc[3]),
		 "=r"(pmc[4]),
		 "=r"(pmc[5]),
		 "=r"(pmc[6]),
		 "=r"(pmc[7]),
		 "=r"(pmc[8]),
		 "=r"(pmc[9]),
		 "=r"(pmc[10]),
		 "=r"(pmc[11]),
		 "=r"(pmc[12]),
		 "=r"(pmc[13]),
		 "=r"(pmc[14]),
		 "=r"(pmc[15])
		:);

    (void) flags;
    (void) ctx;
    _perfnec_control_state_t* control = ( _perfnec_control_state_t* ) ctl;

    long long curr_val = 0;

    int c, i;
    for( c = 0; c < control->active_counters; c++ ) {
        i = control->which_counter[c];
        curr_val = pmc[i];
        SUBDBG("%d, current value %lld\n", i, curr_val);
        control->count[c]=curr_val;
    }

    *events = ( ( _perfnec_control_state_t* ) ctl )->count;

    return PAPI_OK;
}

#if defined(__crayxt)
int _papi_hwd_start_create_context = 0;	/* CrayPat checkpoint support */
#endif /* XT */

static int
_papi_pfnec_start( hwd_context_t * ctx0, hwd_control_state_t * ctl0 )
{
	return PAPI_OK;
}

static int
_papi_pfnec_stop( hwd_context_t * ctx0, hwd_control_state_t * ctl0 )
{
	return PAPI_OK;
}

static inline int
round_requested_ns( int ns )
{
	return PAPI_OK;
}

static int
_papi_pfnec_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
		return ( PAPI_OK );
}

static int
_papi_pfnec_shutdown( hwd_context_t * ctx0 )
{
	return ( PAPI_OK );
}

/* This will need to be modified for the Pentium IV */

static inline int
find_profile_index( EventSetInfo_t * ESI, int pmd, int *flags,
					unsigned int *native_index, int *profile_index )
{
			return ( PAPI_OK );
}

#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6
static inline void
pfnec_bv_set( uint64_t * bv, uint16_t rnum )
{
}

static inline int
setup_ear_event( unsigned int native_index, pfarg_pmd_t * pd, int flags )
{
	return ( 0 );
}

static inline int
process_smpl_entry( unsigned int native_pfnec_index, int flags,
					pfnec_dfl_smpl_entry_t ** ent, vptr_t * pc )
{

		return 0;
}

static inline int
process_smpl_buf( int num_smpl_pmds, int entry_size, ThreadInfo_t ** thr )
{
	return ( PAPI_OK );
}


/* This function  used when hardware overflows ARE working 
    or when software overflows are forced					*/

static void
_papi_pfnec_dispatch_timer( int n, hwd_siginfo_t * info, void *uc )
{
}

static int
_papi_pfnec_stop_profiling( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	return ( PAPI_OK );
}

static int
_papi_pfnec_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	return ( PAPI_OK );
}



static int
_papi_pfnec_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	return ( PAPI_OK );
}

static int
_papi_pfnec_init_control_state( hwd_control_state_t * ctl )
{
    _perfnec_control_state_t* control = ( _perfnec_control_state_t* ) ctl;
    memset( control, 0, sizeof ( _perfnec_control_state_t ) );
    return ( PAPI_OK );
}

static int
_papi_pfnec_allocate_registers( EventSetInfo_t * ESI )
{
	return PAPI_OK;
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

static int
_papi_pfnec_update_control_state( hwd_control_state_t * ctl,
				NativeInfo_t * native, int count,
				hwd_context_t * ctx )
{
    (void) ctx;
    int i, index;

    _perfnec_control_state_t* control = ( _perfnec_control_state_t* ) ctl;
    control->active_counters = count;

    for ( i = 0; i < count; ++i ) {
        index = native[i].ni_event;
        control->which_counter[i]=index;
        native[i].ni_position = i;
    }

    return ( PAPI_OK );
}

static int
_perfnec_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    int index;
    switch ( modifier ) {
        case PAPI_ENUM_FIRST:
            *EventCode = 0 | PAPI_NATIVE_MASK;
            return PAPI_OK;
        case PAPI_ENUM_EVENTS:
            index = *EventCode & PAPI_NATIVE_AND_MASK;
            if ( index < num_events - 1 ) {
                *EventCode = (*EventCode + 1) | PAPI_NATIVE_MASK;
                return PAPI_OK;
            } else {
                return PAPI_ENOEVNT;
            }

        default:
            return PAPI_EINVAL;
    }
}

/*
 *
 */
static int
_perfnec_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK;

    if ( index >= 0 && index < num_events ) {
        _local_strlcpy( name, perfnec_ntv_events[index].name, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}

static int
_perfnec_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t *info )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK;
    if ( index < 0 || index >= num_events )
       return PAPI_ENOEVNT;

    _local_strlcpy( info->symbol, perfnec_ntv_events[index].name, sizeof( info->symbol ));
    _local_strlcpy( info->units, perfnec_ntv_events[index].units, sizeof( info->units ) );
    _local_strlcpy( info->long_descr, perfnec_ntv_events[index].description, sizeof( info->long_descr ) );

    info->data_type = perfnec_ntv_events[index].return_type;

    return PAPI_OK;
}

static int
_perfnec_ntv_name_to_code( const char *name, unsigned int *EventCode)
{
    int i;
    for (i = 0; i < PKG_NUM_EVENTS; ++i)
        if (!strcmp(name, pkg_event_names[i])) {
            *EventCode = i;
            return PAPI_OK;
        }
    return PAPI_ENOEVNT;
}


papi_vector_t _perfnec_vector = {
   .cmp_info = {
      /* default component information (unspecified values initialized to 0) */
      .name = "perfnec",
      .description =  "Linux perfnec CPU counters for NEC architecture",
      .version = "3.8",

      .default_domain = PAPI_DOM_USER,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
      .default_granularity = PAPI_GRN_THR,
      .available_granularities = PAPI_GRN_THR,

      .hardware_intr = 1,
      .kernel_multiplex = 1,
      .kernel_profile = 1,
      .num_mpx_cntrs = PFNECLIB_MAX_PMDS,

      /* component specific cmp_info initializations */
      .fast_real_timer = 1,
      .fast_virtual_timer = 0,
      .attach = 1,
      .attach_must_ptrace = 1,
  },

	/* sizes of framework-opaque component-private structures */
  .size = {
       .context = sizeof ( int ),//pfnec_context_t ),
       .control_state = sizeof ( pfnec_control_state_t ),
       .reg_value = sizeof ( int ), //pfnec_register_t ),
       .reg_alloc = sizeof ( int ) //pfnec_reg_alloc_t ),
  },
	/* function pointers in this component */
  .init_control_state =   _papi_pfnec_init_control_state,
  .start =                _papi_pfnec_start,
  .stop =                 _papi_pfnec_stop,
  .read =                 _papi_pfnec_read,
  .shutdown_thread =      _papi_pfnec_shutdown,
  .shutdown_component =   _papi_pfnec_shutdown_component,
  .ctl =                  _papi_pfnec_ctl,
  .update_control_state = _papi_pfnec_update_control_state,	
  .set_domain =           set_domain,
  .reset =                _papi_pfnec_reset,
  .set_overflow =         _papi_pfnec_set_overflow,
  .set_profile =          _papi_pfnec_set_profile,
  .stop_profiling =       _papi_pfnec_stop_profiling,
  .init_component =       _papi_pfnec_init_component,
  .dispatch_timer =       _papi_pfnec_dispatch_timer,
  .init_thread =          _papi_pfnec_init_thread,
  .allocate_registers =   _papi_pfnec_allocate_registers,
  .write =                _papi_pfnec_write,

	/* from the counter name library */
  .ntv_enum_events =      _perfnec_ntv_enum_events,
  .ntv_name_to_code =     _perfnec_ntv_name_to_code,
  .ntv_code_to_name =     _perfnec_ntv_code_to_name,
  .ntv_code_to_info =     _perfnec_ntv_code_to_info,
  .ntv_code_to_descr =    NULL,//_perfnec_ntv_code_to_descr,
  .ntv_code_to_bits =     NULL//_papi_libpfnec_ntv_code_to_bits,

};
