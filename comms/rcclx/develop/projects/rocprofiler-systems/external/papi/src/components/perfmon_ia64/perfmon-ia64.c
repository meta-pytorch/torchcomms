/*
* File:    perfmon-ia64.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:	   Kevin London
*	   london@cs.utk.edu
*          Per Ekman
*          pek@pdc.kth.se
*          Zhou Min
*          min@cs.utk.edu
*/


#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "threads.h"
#include "papi_memory.h"
#include "papi_lock.h"

#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"

#if defined(__INTEL_COMPILER)

#define hweight64(x)    _m64_popcnt(x)

#elif defined(__GNUC__)

static inline int
hweight64( unsigned long x )
{
	unsigned long result;
  __asm__( "popcnt %0=%1": "=r"( result ):"r"( x ) );
	return ( int ) result;
}

#else
#error "you need to provide inline assembly from your compiler"
#endif

extern int _perfmon2_pfm_pmu_type;
extern papi_vector_t _ia64_vector;

#define OVFL_SIGNAL SIGPROF
#define PFMW_PEVT_EVTCOUNT(evt)          (evt->inp.pfp_event_count)
#define PFMW_PEVT_EVENT(evt,idx)         (evt->inp.pfp_events[idx].event)
#define PFMW_PEVT_PLM(evt,idx)           (evt->inp.pfp_events[idx].plm)
#define PFMW_PEVT_DFLPLM(evt)            (evt->inp.pfp_dfl_plm)
#define PFMW_PEVT_PFPPC(evt)             (evt->pc)
#define PFMW_PEVT_PFPPD(evt)             (evt->pd)
#define PFMW_PEVT_PFPPC_COUNT(evt)       (evt->outp.pfp_pmc_count)
#define PFMW_PEVT_PFPPC_REG_NUM(evt,idx) (evt->outp.pfp_pmcs[idx].reg_num)
#define PFMW_PEVT_PFPPC_REG_VAL(evt,idx) (evt->pc[idx].reg_value)
#define PFMW_PEVT_PFPPC_REG_FLG(evt,idx) (evt->pc[idx].reg_flags)
#define PFMW_ARCH_REG_PMCVAL(reg) (reg.pmc_val)
#define PFMW_ARCH_REG_PMDVAL(reg) (reg.pmd_val)

#define PFMON_MONT_MAX_IBRS	8
#define PFMON_MONT_MAX_DBRS	8

#define PFMON_ITA2_MAX_IBRS	8
#define PFMON_ITA2_MAX_DBRS	8
/*
   #if defined(ITANIUM3)
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_mont_counter_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_mont_counter_reg.pmc_es)
      typedef pfm_mont_pmc_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_mont_pmd_reg_t pfmw_arch_pmd_reg_t;
   #elif defined(ITANIUM2)
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita2_counter_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita2_counter_reg.pmc_es)
      typedef pfm_ita2_pmc_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_ita2_pmd_reg_t pfmw_arch_pmd_reg_t;
   #else 
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita_count_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita_count_reg.pmc_es)
      typedef pfm_ita_pmc_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_ita_pmd_reg_t pfmw_arch_pmd_reg_t;
   #endif
*/
typedef pfm_default_smpl_hdr_t pfmw_smpl_hdr_t;
typedef pfm_default_smpl_entry_t pfmw_smpl_entry_t;

static  void
pfmw_start( hwd_context_t * ctx )
{
	pfm_self_start( ( ( ia64_context_t * ) ctx )->fd );
}

static  void
pfmw_stop( hwd_context_t * ctx )
{
	pfm_self_stop( ( ( ia64_context_t * ) ctx )->fd );
}

static  int
pfmw_perfmonctl( pid_t tid, int fd, int cmd, void *arg, int narg )
{
	( void ) tid;			 /*unused */
	return ( perfmonctl( fd, cmd, arg, narg ) );
}

static  int
pfmw_destroy_context( hwd_context_t * thr_ctx )
{
	int ret;
	ret = close( ( ( ia64_context_t * ) thr_ctx )->fd );
	if ( ret )
		return PAPI_ESYS;
	else
		return PAPI_OK;
}

static  int
pfmw_dispatch_events( pfmw_param_t * evt )
{
	int ret;
	unsigned int i;
/*
      PFMW_PEVT_DFLPLM(evt) = PFM_PLM3;
*/
#ifdef PFMLIB_MONTECITO_PMU
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU )
		ret =
			pfm_dispatch_events( &evt->inp,
								 ( pfmlib_mont_input_param_t * ) evt->mod_inp,
								 &evt->outp,
								 ( pfmlib_mont_output_param_t * ) evt->
								 mod_outp );
	else
#endif
		ret =
			pfm_dispatch_events( &evt->inp,
								 ( pfmlib_ita2_input_param_t * ) evt->mod_inp,
								 &evt->outp,
								 ( pfmlib_ita2_output_param_t * ) evt->
								 mod_outp );
	if ( ret ) {
		return PAPI_ESYS;
	} else {
		for ( i = 0; i < evt->outp.pfp_pmc_count; i++ ) {
			evt->pc[i].reg_num = evt->outp.pfp_pmcs[i].reg_num;
			evt->pc[i].reg_value = evt->outp.pfp_pmcs[i].reg_value;
		}
#if defined(HAVE_PFMLIB_OUTPUT_PFP_PMD_COUNT)
		for ( i = 0; i < evt->outp.pfp_pmd_count; i++ ) {
			evt->pd[i].reg_num = evt->outp.pfp_pmds[i].reg_num;
		}
#else
		/* This is really broken */
		for ( i = 0; i < evt->inp.pfp_event_count; i++ ) {
			evt->pd[i].reg_num = evt->pc[i].reg_num;
		}
#endif
		return PAPI_OK;
	}
}

static  int
pfmw_create_ctx_common( hwd_context_t * ctx )
{
	pfarg_load_t load_args;
	int ret;

	memset( &load_args, 0, sizeof ( load_args ) );
	/*
	 * we want to monitor ourself
	 */

	load_args.load_pid = ( ( ia64_context_t * ) ctx )->tid;

	SUBDBG( "PFM_LOAD_CONTEXT FD %d, PID %d\n",
			( ( ia64_context_t * ) ctx )->fd,
			( ( ia64_context_t * ) ctx )->tid );
	if ( perfmonctl
		 ( ( ( ia64_context_t * ) ctx )->fd, PFM_LOAD_CONTEXT, &load_args,
		   1 ) == -1 ) {
		PAPIERROR( "perfmonctl(PFM_LOAD_CONTEXT) errno %d", errno );
		return ( PAPI_ESYS );
	}
	/*
	 * setup asynchronous notification on the file descriptor
	 */
	ret =
		fcntl( ( ( ia64_context_t * ) ctx )->fd, F_SETFL,
			   fcntl( ( ( ia64_context_t * ) ctx )->fd, F_GETFL,
					  0 ) | O_ASYNC );
	if ( ret == -1 ) {
		PAPIERROR( "fcntl(%d,F_SETFL,O_ASYNC) errno %d",
				   ( ( ia64_context_t * ) ctx )->fd, errno );
		return ( PAPI_ESYS );
	}

	/*
	 * get ownership of the descriptor
	 */

	ret =
		fcntl( ( ( ia64_context_t * ) ctx )->fd, F_SETOWN,
			   ( ( ia64_context_t * ) ctx )->tid );
	if ( ret == -1 ) {
		PAPIERROR( "fcntl(%d,F_SETOWN) errno %d",
				   ( ( ia64_context_t * ) ctx )->fd, errno );
		return ( PAPI_ESYS );
	}

	ret =
		fcntl( ( ( ia64_context_t * ) ctx )->fd, F_SETSIG,
			   _ia64_vector.cmp_info.hardware_intr_sig );
	if ( ret == -1 ) {
		PAPIERROR( "fcntl(%d,F_SETSIG) errno %d",
				   ( ( ia64_context_t * ) ctx )->fd, errno );
		return ( PAPI_ESYS );
	}

	/* set close-on-exec to ensure we will be getting the PFM_END_MSG, i.e.,
	 * fd not visible to child. */

	ret = fcntl( ( ( ia64_context_t * ) ctx )->fd, F_SETFD, FD_CLOEXEC );
	if ( ret == -1 ) {
		PAPIERROR( "fcntl(%d,FD_CLOEXEC) errno %d",
				   ( ( ia64_context_t * ) ctx )->fd, errno );
		return ( PAPI_ESYS );
	}

	return ( PAPI_OK );

}

static  int
pfmw_create_context( hwd_context_t * thr_ctx )
{
	pfarg_context_t ctx;
	memset( &ctx, 0, sizeof ( ctx ) );

	SUBDBG( "PFM_CREATE_CONTEXT on 0\n" );
	if ( perfmonctl( 0, PFM_CREATE_CONTEXT, &ctx, 1 ) == -1 ) {
		PAPIERROR( "perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno );
		return ( PAPI_ESYS );
	}
	( ( ia64_context_t * ) thr_ctx )->fd = ctx.ctx_fd;
	( ( ia64_context_t * ) thr_ctx )->tid = mygettid(  );
	SUBDBG( "PFM_CREATE_CONTEXT returns FD %d, TID %d\n",
			( int ) ( ( ia64_context_t * ) thr_ctx )->fd,
			( int ) ( ( ia64_context_t * ) thr_ctx )->tid );

	return ( pfmw_create_ctx_common( thr_ctx ) );
}

static  int
set_pmds_to_write( EventSetInfo_t * ESI, int index, unsigned long value )
{
	int *pos, count, i;
	unsigned int hwcntr;
	ia64_control_state_t *this_state =
		( ia64_control_state_t * ) ESI->ctl_state;
	pfmw_param_t *pevt = &( this_state->evt );

	pos = ESI->EventInfoArray[index].pos;
	count = 0;
	while ( pos[count] != -1 && count < MAX_COUNTERS ) {
		hwcntr = pos[count] + PMU_FIRST_COUNTER;
		for ( i = 0; i < MAX_COUNTERS; i++ ) {
			if ( PFMW_PEVT_PFPPC_REG_NUM( pevt, i ) == hwcntr ) {
				this_state->evt.pc[i].reg_smpl_pmds[0] = value;
				break;
			}
		}
		count++;
	}
	return ( PAPI_OK );
}

static  int
_pfm_decode_native_event( unsigned int EventCode, unsigned int *event,
			  unsigned int *umask );

static  int
pfmw_recreate_context( EventSetInfo_t * ESI, hwd_context_t * thr_ctx,
		       void **smpl_vaddr, int EventIndex )
{
	pfm_default_smpl_ctx_arg_t ctx;
	pfm_uuid_t buf_fmt_id = PFM_DEFAULT_SMPL_UUID;
	int ctx_fd;
	unsigned int native_index, EventCode;
	int pos;
	//hwd_context_t *thr_ctx = (hwd_context_t *) &ESI->master->context;
#ifdef PFMLIB_MONTECITO_PMU
	unsigned int umask;
#endif

	pos = ESI->EventInfoArray[EventIndex].pos[0];
	EventCode = ESI->EventInfoArray[EventIndex].event_code;
#ifdef PFMLIB_MONTECITO_PMU
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU ) {
		if ( _pfm_decode_native_event
			 ( ESI->NativeInfoArray[pos].ni_event, &native_index,
			   &umask ) != PAPI_OK )
			return ( PAPI_ENOEVNT );
	} else
#endif
		native_index =
			ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;

	memset( &ctx, 0, sizeof ( ctx ) );
	/*
	 * We initialize the format specific information.
	 * The format is identified by its UUID which must be copied
	 * into the ctx_buf_fmt_id field.
	 */
	memcpy( ctx.ctx_arg.ctx_smpl_buf_id, buf_fmt_id, sizeof ( pfm_uuid_t ) );
	/*
	 * the size of the buffer is indicated in bytes (not entries).
	 * The kernel will record into the buffer up to a certain point.
	 * No partial samples are ever recorded.
	 */
	ctx.buf_arg.buf_size = 4096;
	/*
	 * now create the context for self monitoring/per-task
	 */
	SUBDBG( "PFM_CREATE_CONTEXT on 0\n" );
	if ( perfmonctl( 0, PFM_CREATE_CONTEXT, &ctx, 1 ) == -1 ) {
		if ( errno == ENOSYS )
			PAPIERROR
				( "Your kernel does not have performance monitoring support" );
		else
			PAPIERROR( "perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno );
		return ( PAPI_ESYS );
	}
	/*
	 * extract the file descriptor we will use to
	 * identify this newly created context
	 */
	ctx_fd = ctx.ctx_arg.ctx_fd;
	/* save the fd into the thread context struct */
	( ( ia64_context_t * ) thr_ctx )->fd = ctx_fd;
	( ( ia64_context_t * ) thr_ctx )->tid = mygettid(  );
	SUBDBG( "PFM_CREATE_CONTEXT returns FD %d, TID %d\n",
			( int ) ( ( ia64_context_t * ) thr_ctx )->fd,
			( int ) ( ( ia64_context_t * ) thr_ctx )->tid );
	/* indicate which PMD to include in the sample */
/* DEAR and BTB events */
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		if ( pfm_ita_is_dear( native_index ) )
			set_pmds_to_write( ESI, EventIndex, DEAR_REGS_MASK );
		else if ( pfm_ita_is_btb( native_index )
				  || EventCode == ( unsigned int ) PAPI_BR_INS )
			set_pmds_to_write( ESI, EventIndex, BTB_REGS_MASK );
		break;
	case PFMLIB_ITANIUM2_PMU:
		if ( pfm_ita2_is_dear( native_index ) )
			set_pmds_to_write( ESI, EventIndex, DEAR_REGS_MASK );
		else if ( pfm_ita2_is_btb( native_index )
				  || EventCode == ( unsigned int ) PAPI_BR_INS )
			set_pmds_to_write( ESI, EventIndex, BTB_REGS_MASK );
		break;
	case PFMLIB_MONTECITO_PMU:
		if ( pfm_mont_is_dear( native_index ) )
			set_pmds_to_write( ESI, EventIndex, MONT_DEAR_REGS_MASK );
		else if ( pfm_mont_is_etb( native_index ) ||
				  EventCode == ( unsigned int ) PAPI_BR_INS )
			set_pmds_to_write( ESI, EventIndex, MONT_ETB_REGS_MASK );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}

	*smpl_vaddr = ctx.ctx_arg.ctx_smpl_vaddr;

	return ( pfmw_create_ctx_common( thr_ctx ) );
}

static  int
pfmw_get_event_name( char *name, unsigned int idx )
{
	unsigned int total;

	pfm_get_num_events( &total );
	if ( idx >= total )
		return PAPI_ENOEVNT;
	if ( pfm_get_event_name( idx, name, PAPI_MAX_STR_LEN ) == PFMLIB_SUCCESS )
		return PAPI_OK;
	else
		return PAPI_ENOEVNT;
}

static  void
pfmw_get_event_description( unsigned int idx, char *dest, int len )
{
	char *descr;

	if ( pfm_get_event_description( idx, &descr ) == PFMLIB_SUCCESS ) {
		strncpy( dest, descr, len );
		free( descr );
	} else
		*dest = '\0';
}

static  int
pfmw_is_dear( unsigned int i )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( pfm_ita_is_dear( i ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( pfm_ita2_is_dear( i ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( pfm_mont_is_dear( i ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

static  int
pfmw_is_iear( unsigned int i )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( pfm_ita_is_iear( i ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( pfm_ita2_is_iear( i ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( pfm_mont_is_iear( i ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

static  int
pfmw_support_darr( unsigned int i )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( pfm_ita_support_darr( i ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( pfm_ita2_support_darr( i ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( pfm_mont_support_darr( i ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

static  int
pfmw_support_iarr( unsigned int i )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( pfm_ita_support_iarr( i ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( pfm_ita2_support_iarr( i ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( pfm_mont_support_iarr( i ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

static  int
pfmw_support_opcm( unsigned int i )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( pfm_ita_support_opcm( i ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( pfm_ita2_support_opcm( i ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( pfm_mont_support_opcm( i ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

static void
check_ibrp_events( hwd_control_state_t * current_state )
{
	ia64_control_state_t *this_state = ( ia64_control_state_t * ) current_state;
	pfmw_param_t *evt = &( this_state->evt );
	unsigned long umasks_retired[4];
	unsigned long umask;
	unsigned int j, i, seen_retired, ibrp, idx;
	int code;
	int retired_code, incr;
	pfmlib_ita2_output_param_t *ita2_output_param;
	pfmlib_mont_output_param_t *mont_output_param;

#if defined(PFMLIB_ITANIUM2_PMU) || defined(PFMLIB_MONTECITO_PMU)
char *retired_events[] = {
	"IA64_TAGGED_INST_RETIRED_IBRP0_PMC8",
	"IA64_TAGGED_INST_RETIRED_IBRP1_PMC9",
	"IA64_TAGGED_INST_RETIRED_IBRP2_PMC8",
	"IA64_TAGGED_INST_RETIRED_IBRP3_PMC9",
	NULL
};
#endif

	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM2_PMU:
		ita2_output_param =
			&( this_state->ita_lib_param.ita2_param.ita2_output_param );
		/*
		 * in fine mode, it is enough to use the event
		 * which only monitors the first debug register
		 * pair. The two pairs making up the range
		 * are guaranteed to be consecutive in rr_br[].
		 */
		incr = pfm_ita2_irange_is_fine( &evt->outp, ita2_output_param ) ? 4 : 2;

		for ( i = 0; retired_events[i]; i++ ) {
			pfm_find_event( retired_events[i], &idx );
			pfm_ita2_get_event_umask( idx, umasks_retired + i );
		}

		pfm_get_event_code( idx, &retired_code );

		/*
		 * print a warning message when the using IA64_TAGGED_INST_RETIRED_IBRP* which does
		 * not completely cover the all the debug register pairs used to make up the range.
		 * This could otherwise lead to misinterpretation of the results.
		 */
		for ( i = 0; i < ita2_output_param->pfp_ita2_irange.rr_nbr_used;
			  i += incr ) {

			ibrp = ita2_output_param->pfp_ita2_irange.rr_br[i].reg_num >> 1;

			seen_retired = 0;
			for ( j = 0; j < evt->inp.pfp_event_count; j++ ) {
				pfm_get_event_code( evt->inp.pfp_events[j].event, &code );
				if ( code != retired_code )
					continue;
				seen_retired = 1;
				pfm_ita2_get_event_umask( evt->inp.pfp_events[j].event,
										  &umask );
				if ( umask == umasks_retired[ibrp] )
					break;
			}
			if ( seen_retired && j == evt->inp.pfp_event_count )
				printf
					( "warning: code range uses IBR pair %d which is not monitored using %s\n",
					  ibrp, retired_events[ibrp] );
		}

		break;
	case PFMLIB_MONTECITO_PMU:
		mont_output_param =
			&( this_state->ita_lib_param.mont_param.mont_output_param );
		/*
		 * in fine mode, it is enough to use the event
		 * which only monitors the first debug register
		 * pair. The two pairs making up the range
		 * are guaranteed to be consecutive in rr_br[].
		 */
		incr = pfm_mont_irange_is_fine( &evt->outp, mont_output_param ) ? 4 : 2;

		for ( i = 0; retired_events[i]; i++ ) {
			pfm_find_event( retired_events[i], &idx );
			pfm_mont_get_event_umask( idx, umasks_retired + i );
		}

		pfm_get_event_code( idx, &retired_code );

		/*
		 * print a warning message when the using IA64_TAGGED_INST_RETIRED_IBRP* which does
		 * not completely cover the all the debug register pairs used to make up the range.
		 * This could otherwise lead to misinterpretation of the results.
		 */
		for ( i = 0; i < mont_output_param->pfp_mont_irange.rr_nbr_used;
			  i += incr ) {

			ibrp = mont_output_param->pfp_mont_irange.rr_br[i].reg_num >> 1;

			seen_retired = 0;
			for ( j = 0; j < evt->inp.pfp_event_count; j++ ) {
				pfm_get_event_code( evt->inp.pfp_events[j].event, &code );
				if ( code != retired_code )
					continue;
				seen_retired = 1;
				pfm_mont_get_event_umask( evt->inp.pfp_events[j].event,
										  &umask );
				if ( umask == umasks_retired[ibrp] )
					break;
			}
			if ( seen_retired && j == evt->inp.pfp_event_count )
				printf
					( "warning: code range uses IBR pair %d which is not monitored using %s\n",
					  ibrp, retired_events[ibrp] );
		}
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
	}
}

static  int
install_irange( hwd_context_t * pctx, hwd_control_state_t * current_state )
{
	ia64_control_state_t *this_state = ( ia64_control_state_t * ) current_state;
	unsigned int i, used_dbr;
	int r;
	int pid = ( ( ia64_context_t * ) pctx )->fd;

	pfmlib_ita2_output_param_t *ita2_output_param;
	pfarg_dbreg_t ita2_dbreg[PFMON_ITA2_MAX_IBRS];
	pfmlib_mont_output_param_t *mont_output_param;
	pfarg_dbreg_t mont_dbreg[PFMON_MONT_MAX_IBRS];

	memset( mont_dbreg, 0, sizeof ( mont_dbreg ) );
	memset( ita2_dbreg, 0, sizeof ( ita2_dbreg ) );
	check_ibrp_events( current_state );

	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM2_PMU:
		ita2_output_param =
			&( this_state->ita_lib_param.ita2_param.ita2_output_param );
		used_dbr = ita2_output_param->pfp_ita2_irange.rr_nbr_used;

		for ( i = 0; i < used_dbr; i++ ) {
			ita2_dbreg[i].dbreg_num =
				ita2_output_param->pfp_ita2_irange.rr_br[i].reg_num;
			ita2_dbreg[i].dbreg_value =
				ita2_output_param->pfp_ita2_irange.rr_br[i].reg_value;
		}

		r = perfmonctl( pid, PFM_WRITE_IBRS, ita2_dbreg,
						ita2_output_param->pfp_ita2_irange.rr_nbr_used );
		if ( r == -1 ) {
			SUBDBG( "cannot install code range restriction: %s\n",
					strerror( errno ) );
			return ( PAPI_ESYS );
		}
		return ( PAPI_OK );
		break;
	case PFMLIB_MONTECITO_PMU:
		mont_output_param =
			&( this_state->ita_lib_param.mont_param.mont_output_param );

		used_dbr = mont_output_param->pfp_mont_irange.rr_nbr_used;

		for ( i = 0; i < used_dbr; i++ ) {
			mont_dbreg[i].dbreg_num =
				mont_output_param->pfp_mont_irange.rr_br[i].reg_num;
			mont_dbreg[i].dbreg_value =
				mont_output_param->pfp_mont_irange.rr_br[i].reg_value;
		}

		r = perfmonctl( pid, PFM_WRITE_IBRS, mont_dbreg,
						mont_output_param->pfp_mont_irange.rr_nbr_used );
		if ( r == -1 ) {
			SUBDBG( "cannot install code range restriction: %s\n",
					strerror( errno ) );
			return ( PAPI_ESYS );
		}
		return ( PAPI_OK );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return PAPI_ENOIMPL;
	}
}

static  int
install_drange( hwd_context_t * pctx, hwd_control_state_t * current_state )
{
	ia64_control_state_t *this_state = ( ia64_control_state_t * ) current_state;
	unsigned int i, used_dbr;
	int r;
	int pid = ( ( ia64_context_t * ) pctx )->fd;

	pfmlib_ita2_output_param_t *ita2_output_param;
	pfarg_dbreg_t ita2_dbreg[PFMON_ITA2_MAX_IBRS];
	pfmlib_mont_output_param_t *mont_output_param;
	pfarg_dbreg_t mont_dbreg[PFMON_MONT_MAX_IBRS];

	memset( mont_dbreg, 0, sizeof ( mont_dbreg ) );
	memset( ita2_dbreg, 0, sizeof ( ita2_dbreg ) );

	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM2_PMU:
		ita2_output_param =
			&( this_state->ita_lib_param.ita2_param.ita2_output_param );
		used_dbr = ita2_output_param->pfp_ita2_drange.rr_nbr_used;

		for ( i = 0; i < used_dbr; i++ ) {
			ita2_dbreg[i].dbreg_num =
				ita2_output_param->pfp_ita2_drange.rr_br[i].reg_num;
			ita2_dbreg[i].dbreg_value =
				ita2_output_param->pfp_ita2_drange.rr_br[i].reg_value;
		}

		r = perfmonctl( pid, PFM_WRITE_DBRS, ita2_dbreg,
						ita2_output_param->pfp_ita2_drange.rr_nbr_used );
		if ( r == -1 ) {
			SUBDBG( "cannot install data range restriction: %s\n",
					strerror( errno ) );
			return ( PAPI_ESYS );
		}
		return ( PAPI_OK );
		break;
	case PFMLIB_MONTECITO_PMU:
		mont_output_param =
			&( this_state->ita_lib_param.mont_param.mont_output_param );
		used_dbr = mont_output_param->pfp_mont_drange.rr_nbr_used;

		for ( i = 0; i < used_dbr; i++ ) {
			mont_dbreg[i].dbreg_num =
				mont_output_param->pfp_mont_drange.rr_br[i].reg_num;
			mont_dbreg[i].dbreg_value =
				mont_output_param->pfp_mont_drange.rr_br[i].reg_value;
		}

		r = perfmonctl( pid, PFM_WRITE_DBRS, mont_dbreg,
						mont_output_param->pfp_mont_drange.rr_nbr_used );
		if ( r == -1 ) {
			SUBDBG( "cannot install data range restriction: %s\n",
					strerror( errno ) );
			return PAPI_ESYS;
		}
		return PAPI_OK;
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return PAPI_ENOIMPL;
	}
}

/* The routines set_{d,i}range() provide places to install the data and / or
   instruction address range restrictions for counting qualified events.
   These routines must set up or clear the appropriate local static data structures.
   The actual work of loading the hardware registers must be done in update_ctl_state().
   Both drange and irange can be set on the same eventset.
   If start=end=0, the feature is disabled. 
*/
static  int
set_drange( hwd_context_t * ctx, hwd_control_state_t * current_state,
			_papi_int_option_t * option )
{
	int ret = PAPI_OK;
	ia64_control_state_t *this_state = ( ia64_control_state_t * ) current_state;
	pfmw_param_t *evt = &( this_state->evt );
	pfmlib_input_param_t *inp = &evt->inp;
	pfmlib_ita2_input_param_t *ita2_inp =
		&( this_state->ita_lib_param.ita2_param.ita2_input_param );
	pfmlib_ita2_output_param_t *ita2_outp =
		&( this_state->ita_lib_param.ita2_param.ita2_output_param );
	pfmlib_mont_input_param_t *mont_inp =
		&( this_state->ita_lib_param.mont_param.mont_input_param );
	pfmlib_mont_output_param_t *mont_outp =
		&( this_state->ita_lib_param.mont_param.mont_output_param );

	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM2_PMU:

		if ( ( unsigned long ) option->address_range.start ==
			 ( unsigned long ) option->address_range.end ||
			 ( ( unsigned long ) option->address_range.start == 0 &&
			   ( unsigned long ) option->address_range.end == 0 ) )
			return ( PAPI_EINVAL );
		/*
		 * set the privilege mode:
		 *  PFM_PLM3 : user level only
		 */
		memset( &ita2_inp->pfp_ita2_drange, 0,
				sizeof ( pfmlib_ita2_input_rr_t ) );
		memset( ita2_outp, 0, sizeof ( pfmlib_ita2_output_param_t ) );
		inp->pfp_dfl_plm = PFM_PLM3;
		ita2_inp->pfp_ita2_drange.rr_used = 1;
		ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start =
			( unsigned long ) option->address_range.start;
		ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end =
			( unsigned long ) option->address_range.end;
		SUBDBG
			( "++++ before data range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start,
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end,
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end -
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start,
			  ita2_outp->pfp_ita2_drange.rr_nbr_used >> 1,
			  ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff,
			  ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff );

		/*
		 * let the library figure out the values for the PMCS
		 */
		if ( ( ret = pfmw_dispatch_events( evt ) ) != PFMLIB_SUCCESS ) {
			SUBDBG( "cannot configure events: %s\n", pfm_strerror( ret ) );
		}

		SUBDBG
			( "++++ data range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start,
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end,
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end -
			  ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start,
			  ita2_outp->pfp_ita2_drange.rr_nbr_used >> 1,
			  ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff,
			  ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff );

/*   if(	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start!=0 || 	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end!=0 )
   if((ret=install_irange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff;
   }
*/
		if ( ( ret = install_drange( ctx, current_state ) ) == PAPI_OK ) {
			option->address_range.start_off =
				ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff;
			option->address_range.end_off =
				ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff;
		}
		return ( ret );

		break;
	case PFMLIB_MONTECITO_PMU:

		if ( ( unsigned long ) option->address_range.start ==
			 ( unsigned long ) option->address_range.end ||
			 ( ( unsigned long ) option->address_range.start == 0 &&
			   ( unsigned long ) option->address_range.end == 0 ) )
			return ( PAPI_EINVAL );
		/*
		 * set the privilege mode:
		 *  PFM_PLM3 : user level only
		 */
		memset( &mont_inp->pfp_mont_drange, 0,
				sizeof ( pfmlib_mont_input_rr_t ) );
		memset( mont_outp, 0, sizeof ( pfmlib_mont_output_param_t ) );
		inp->pfp_dfl_plm = PFM_PLM3;
		mont_inp->pfp_mont_drange.rr_used = 1;
		mont_inp->pfp_mont_drange.rr_limits[0].rr_start =
			( unsigned long ) option->address_range.start;
		mont_inp->pfp_mont_drange.rr_limits[0].rr_end =
			( unsigned long ) option->address_range.end;
		SUBDBG
			( "++++ before data range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_start,
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_end,
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_end -
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_start,
			  mont_outp->pfp_mont_drange.rr_nbr_used >> 1,
			  mont_outp->pfp_mont_drange.rr_infos[0].rr_soff,
			  mont_outp->pfp_mont_drange.rr_infos[0].rr_eoff );
		/*
		 * let the library figure out the values for the PMCS
		 */
		if ( ( ret = pfmw_dispatch_events( evt ) ) != PFMLIB_SUCCESS ) {
			SUBDBG( "cannot configure events: %s\n", pfm_strerror( ret ) );
		}

		SUBDBG
			( "++++ data range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_start,
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_end,
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_end -
			  mont_inp->pfp_mont_drange.rr_limits[0].rr_start,
			  mont_outp->pfp_mont_drange.rr_nbr_used >> 1,
			  mont_outp->pfp_mont_drange.rr_infos[0].rr_soff,
			  mont_outp->pfp_mont_drange.rr_infos[0].rr_eoff );

/*   if(	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start!=0 || 	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end!=0 )
   if((ret=install_irange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff;
   }
*/
		if ( ( ret = install_drange( ctx, current_state ) ) == PAPI_OK ) {
			option->address_range.start_off =
				mont_outp->pfp_mont_drange.rr_infos[0].rr_soff;
			option->address_range.end_off =
				mont_outp->pfp_mont_drange.rr_infos[0].rr_eoff;
		}
		return ( ret );

		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return PAPI_ENOIMPL;
	}
}

static  int
set_irange( hwd_context_t * ctx, hwd_control_state_t * current_state,
			_papi_int_option_t * option )
{
	int ret = PAPI_OK;
	ia64_control_state_t *this_state = ( ia64_control_state_t * ) current_state;
	pfmw_param_t *evt = &( this_state->evt );
	pfmlib_input_param_t *inp = &evt->inp;
	pfmlib_ita2_input_param_t *ita2_inp =
		&( this_state->ita_lib_param.ita2_param.ita2_input_param );
	pfmlib_ita2_output_param_t *ita2_outp =
		&( this_state->ita_lib_param.ita2_param.ita2_output_param );
	pfmlib_mont_input_param_t *mont_inp =
		&( this_state->ita_lib_param.mont_param.mont_input_param );
	pfmlib_mont_output_param_t *mont_outp =
		&( this_state->ita_lib_param.mont_param.mont_output_param );

	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM2_PMU:

		if ( ( unsigned long ) option->address_range.start ==
			 ( unsigned long ) option->address_range.end ||
			 ( ( unsigned long ) option->address_range.start == 0 &&
			   ( unsigned long ) option->address_range.end == 0 ) )
			return ( PAPI_EINVAL );
		/*
		 * set the privilege mode:
		 *    PFM_PLM3 : user level only
		 */
		memset( &ita2_inp->pfp_ita2_irange, 0,
				sizeof ( pfmlib_ita2_input_rr_t ) );
		memset( ita2_outp, 0, sizeof ( pfmlib_ita2_output_param_t ) );
		inp->pfp_dfl_plm = PFM_PLM3;
		ita2_inp->pfp_ita2_irange.rr_used = 1;
		ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start =
			( unsigned long ) option->address_range.start;
		ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end =
			( unsigned long ) option->address_range.end;
		SUBDBG
			( "++++ before code range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start,
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end,
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end -
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start,
			  ita2_outp->pfp_ita2_irange.rr_nbr_used >> 1,
			  ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff,
			  ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff );

		/*
		 * let the library figure out the values for the PMCS
		 */
		if ( ( ret = pfmw_dispatch_events( evt ) ) != PFMLIB_SUCCESS ) {
			SUBDBG( "cannot configure events: %s\n", pfm_strerror( ret ) );
		}

		SUBDBG
			( "++++ code range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start,
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end,
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end -
			  ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start,
			  ita2_outp->pfp_ita2_irange.rr_nbr_used >> 1,
			  ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff,
			  ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff );
		if ( ( ret = install_irange( ctx, current_state ) ) == PAPI_OK ) {
			option->address_range.start_off =
				ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff;
			option->address_range.end_off =
				ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff;
		}

		break;
	case PFMLIB_MONTECITO_PMU:

		if ( ( unsigned long ) option->address_range.start ==
			 ( unsigned long ) option->address_range.end ||
			 ( ( unsigned long ) option->address_range.start == 0 &&
			   ( unsigned long ) option->address_range.end == 0 ) )
			return ( PAPI_EINVAL );
		/*
		 * set the privilege mode:
		 *  PFM_PLM3 : user level only
		 */
		memset( &mont_inp->pfp_mont_irange, 0,
				sizeof ( pfmlib_mont_input_rr_t ) );
		memset( mont_outp, 0, sizeof ( pfmlib_mont_output_param_t ) );
		inp->pfp_dfl_plm = PFM_PLM3;
		mont_inp->pfp_mont_irange.rr_used = 1;
		mont_inp->pfp_mont_irange.rr_limits[0].rr_start =
			( unsigned long ) option->address_range.start;
		mont_inp->pfp_mont_irange.rr_limits[0].rr_end =
			( unsigned long ) option->address_range.end;
		SUBDBG
			( "++++ before code range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_start,
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_end,
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_end -
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_start,
			  mont_outp->pfp_mont_irange.rr_nbr_used >> 1,
			  mont_outp->pfp_mont_irange.rr_infos[0].rr_soff,
			  mont_outp->pfp_mont_irange.rr_infos[0].rr_eoff );

		/*
		 * let the library figure out the values for the PMCS
		 */
		if ( ( ret = pfmw_dispatch_events( evt ) ) != PFMLIB_SUCCESS ) {
			SUBDBG( "cannot configure events: %s\n", pfm_strerror( ret ) );
		}

		SUBDBG
			( "++++ code range  : [%#016lx-%#016lx=%ld]: %d pair of debug registers used\n"
			  "     start_offset:-%#lx end_offset:+%#lx\n",
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_start,
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_end,
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_end -
			  mont_inp->pfp_mont_irange.rr_limits[0].rr_start,
			  mont_outp->pfp_mont_irange.rr_nbr_used >> 1,
			  mont_outp->pfp_mont_irange.rr_infos[0].rr_soff,
			  mont_outp->pfp_mont_irange.rr_infos[0].rr_eoff );
		if ( ( ret = install_irange( ctx, current_state ) ) == PAPI_OK ) {
			option->address_range.start_off =
				mont_outp->pfp_mont_irange.rr_infos[0].rr_soff;
			option->address_range.end_off =
				mont_outp->pfp_mont_irange.rr_infos[0].rr_eoff;
		}

		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return PAPI_ENOIMPL;
	}

	return ret;
}

static  int
pfmw_get_num_counters( int *num )
{
	unsigned int tmp;
	if ( pfm_get_num_counters( &tmp ) != PFMLIB_SUCCESS )
		return ( PAPI_ESYS );
	*num = tmp;
	return ( PAPI_OK );
}

static  int
pfmw_get_num_events( int *num )
{
	unsigned int tmp;
	if ( pfm_get_num_events( &tmp ) != PFMLIB_SUCCESS )
		return ( PAPI_ESYS );
	*num = tmp;
	return ( PAPI_OK );
}


/* Globals declared extern elsewhere */

hwi_search_t *preset_search_map;
extern papi_vector_t _ia64_vector;

unsigned int PAPI_NATIVE_EVENT_AND_MASK = 0x000003ff;
unsigned int PAPI_NATIVE_EVENT_SHIFT = 0;
unsigned int PAPI_NATIVE_UMASK_AND_MASK = 0x03fffc00;
unsigned int PAPI_NATIVE_UMASK_MAX = 16;
unsigned int PAPI_NATIVE_UMASK_SHIFT = 10;

/* Static locals */

int _perfmon2_pfm_pmu_type = -1;

/*
static papi_svector_t _linux_ia64_table[] = {
 {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE}, 
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
 {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {NULL, VEC_PAPI_END}
};
*/

static itanium_preset_search_t ia1_preset_search_map[] = {
	{PAPI_L1_TCM, DERIVED_ADD,
	 {"L1D_READ_MISSES_RETIRED", "L2_INST_DEMAND_READS"}, {0}},
	{PAPI_L1_ICM, 0, {"L2_INST_DEMAND_READS"}, {0}},
	{PAPI_L1_DCM, 0, {"L1D_READ_MISSES_RETIRED"}, {0}},
	{PAPI_L2_TCM, 0, {"L2_MISSES"}, {0}},
	{PAPI_L2_DCM, DERIVED_SUB, {"L2_MISSES", "L3_READS_INST_READS_ALL"}, {0}},
	{PAPI_L2_ICM, 0, {"L3_READS_INST_READS_ALL"}, {0}},
	{PAPI_L3_TCM, 0, {"L3_MISSES"}, {0}},
	{PAPI_L3_ICM, 0, {"L3_READS_INST_READS_MISS"}, {0}},
	{PAPI_L3_DCM, DERIVED_ADD,
	 {"L3_READS_DATA_READS_MISS", "L3_WRITES_DATA_WRITES_MISS"}, {0}},
	{PAPI_L3_LDM, 0, {"L3_READS_DATA_READS_MISS"}, {0}},
	{PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITES_MISS"}, {0}},
	{PAPI_L1_LDM, 0, {"L1D_READ_MISSES_RETIRED"}, {0}},
	{PAPI_L2_LDM, 0, {"L3_READS_DATA_READS_ALL"}, {0}},
	{PAPI_L2_STM, 0, {"L3_WRITES_ALL_WRITES_ALL"}, {0}},
	{PAPI_L3_DCH, DERIVED_ADD,
	 {"L3_READS_DATA_READS_HIT", "L3_WRITES_DATA_WRITES_HIT"}, {0}},
	{PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_RETIRED", "L1D_READ_MISSES_RETIRED"},
	 {0}},
	{PAPI_L1_DCA, 0, {"L1D_READS_RETIRED"}, {0}},
	{PAPI_L2_DCA, 0, {"L2_DATA_REFERENCES_ALL"}, {0}},
	{PAPI_L3_DCA, DERIVED_ADD,
	 {"L3_READS_DATA_READS_ALL", "L3_WRITES_DATA_WRITES_ALL"}, {0}},
	{PAPI_L2_DCR, 0, {"L2_DATA_REFERENCES_READS"}, {0}},
	{PAPI_L3_DCR, 0, {"L3_READS_DATA_READS_ALL"}, {0}},
	{PAPI_L2_DCW, 0, {"L2_DATA_REFERENCES_WRITES"}, {0}},
	{PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITES_ALL"}, {0}},
	{PAPI_L3_ICH, 0, {"L3_READS_INST_READS_HIT"}, {0}},
	{PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCH_READS", "L1I_DEMAND_READS"}, {0}},
	{PAPI_L2_ICR, DERIVED_ADD,
	 {"L2_INST_DEMAND_READS", "L2_INST_PREFETCH_READS"}, {0}},
	{PAPI_L3_ICR, 0, {"L3_READS_INST_READS_ALL"}, {0}},
	{PAPI_TLB_DM, 0, {"DTLB_MISSES"}, {0}},
	{PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH"}, {0}},
	{PAPI_MEM_SCY, 0, {"MEMORY_CYCLE"}, {0}},
	{PAPI_STL_ICY, 0, {"UNSTALLED_BACKEND_CYCLE"}, {0}},
	{PAPI_BR_INS, 0, {"BRANCH_EVENT"}, {0}},
	{PAPI_BR_PRC, 0, {"BRANCH_PREDICTOR_ALL_CORRECT_PREDICTIONS"}, {0}},
	{PAPI_BR_MSP, DERIVED_ADD,
	 {"BRANCH_PREDICTOR_ALL_WRONG_PATH", "BRANCH_PREDICTOR_ALL_WRONG_TARGET"},
	 {0}},
	{PAPI_TOT_CYC, 0, {"CPU_CYCLES"}, {0}},
	{PAPI_FP_OPS, DERIVED_ADD, {"FP_OPS_RETIRED_HI", "FP_OPS_RETIRED_LO"}, {0}},
	{PAPI_TOT_INS, 0, {"IA64_INST_RETIRED"}, {0}},
	{PAPI_LD_INS, 0, {"LOADS_RETIRED"}, {0}},
	{PAPI_SR_INS, 0, {"STORES_RETIRED"}, {0}},
	{PAPI_LST_INS, DERIVED_ADD, {"LOADS_RETIRED", "STORES_RETIRED"}, {0}},
	{0, 0, {0}, {0}}
};

static itanium_preset_search_t ia2_preset_search_map[] = {
	{PAPI_CA_SNP, 0, {"BUS_SNOOPS_SELF"}, {0}},
	{PAPI_CA_INV, DERIVED_ADD,
	 {"BUS_MEM_READ_BRIL_SELF", "BUS_MEM_READ_BIL_SELF"}, {0}},
	{PAPI_TLB_TL, DERIVED_ADD, {"ITLB_MISSES_FETCH_L2ITLB", "L2DTLB_MISSES"},
	 {0}},
	{PAPI_STL_ICY, 0, {"DISP_STALLED"}, {0}},
	{PAPI_STL_CCY, 0, {"BACK_END_BUBBLE_ALL"}, {0}},
	{PAPI_TOT_IIS, 0, {"INST_DISPERSED"}, {0}},
	{PAPI_RES_STL, 0, {"BE_EXE_BUBBLE_ALL"}, {0}},
	{PAPI_FP_STAL, 0, {"BE_EXE_BUBBLE_FRALL"}, {0}},
	{PAPI_L2_TCR, DERIVED_ADD,
	 {"L2_DATA_REFERENCES_L2_DATA_READS", "L2_INST_DEMAND_READS",
	  "L2_INST_PREFETCHES"}, {0}},
	{PAPI_L1_TCM, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L1D_READ_MISSES_ALL"},
	 {0}},
	{PAPI_L1_ICM, 0, {"L2_INST_DEMAND_READS"}, {0}},
	{PAPI_L1_DCM, 0, {"L1D_READ_MISSES_ALL"}, {0}},
	{PAPI_L2_TCM, 0, {"L2_MISSES"}, {0}},
	{PAPI_L2_DCM, DERIVED_SUB, {"L2_MISSES", "L3_READS_INST_FETCH_ALL"}, {0}},
	{PAPI_L2_ICM, 0, {"L3_READS_INST_FETCH_ALL"}, {0}},
	{PAPI_L3_TCM, 0, {"L3_MISSES"}, {0}},
	{PAPI_L3_ICM, 0, {"L3_READS_INST_FETCH_MISS"}, {0}},
	{PAPI_L3_DCM, DERIVED_ADD,
	 {"L3_READS_DATA_READ_MISS", "L3_WRITES_DATA_WRITE_MISS"}, {0}},
	{PAPI_L3_LDM, 0, {"L3_READS_ALL_MISS"}, {0}},
	{PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITE_MISS"}, {0}},
	{PAPI_L1_LDM, DERIVED_ADD, {"L1D_READ_MISSES_ALL", "L2_INST_DEMAND_READS"},
	 {0}},
	{PAPI_L2_LDM, 0, {"L3_READS_ALL_ALL"}, {0}},
	{PAPI_L2_STM, 0, {"L3_WRITES_ALL_ALL"}, {0}},
	{PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_SET1", "L1D_READ_MISSES_ALL"}, {0}},
	{PAPI_L2_DCH, DERIVED_SUB, {"L2_DATA_REFERENCES_L2_ALL", "L2_MISSES"}, {0}},
	{PAPI_L3_DCH, DERIVED_ADD,
	 {"L3_READS_DATA_READ_HIT", "L3_WRITES_DATA_WRITE_HIT"}, {0}},
	{PAPI_L1_DCA, 0, {"L1D_READS_SET1"}, {0}},
	{PAPI_L2_DCA, 0, {"L2_DATA_REFERENCES_L2_ALL"}, {0}},
	{PAPI_L3_DCA, DERIVED_ADD,
	 {"L3_READS_DATA_READ_ALL", "L3_WRITES_DATA_WRITE_ALL"}, {0}},
	{PAPI_L1_DCR, 0, {"L1D_READS_SET1"}, {0}},
	{PAPI_L2_DCR, 0, {"L2_DATA_REFERENCES_L2_DATA_READS"}, {0}},
	{PAPI_L3_DCR, 0, {"L3_READS_DATA_READ_ALL"}, {0}},
	{PAPI_L2_DCW, 0, {"L2_DATA_REFERENCES_L2_DATA_WRITES"}, {0}},
	{PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITE_ALL"}, {0}},
	{PAPI_L3_ICH, 0, {"L3_READS_DINST_FETCH_HIT"}, {0}},
	{PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS"}, {0}},
	{PAPI_L2_ICR, DERIVED_ADD, {"L2_INST_DEMAND_READS", "L2_INST_PREFETCHES"},
	 {0}},
	{PAPI_L3_ICR, 0, {"L3_READS_INST_FETCH_ALL"}, {0}},
	{PAPI_L1_ICA, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS"}, {0}},
	{PAPI_L2_TCH, DERIVED_SUB, {"L2_REFERENCES", "L2_MISSES"}, {0}},
	{PAPI_L3_TCH, DERIVED_SUB, {"L3_REFERENCES", "L3_MISSES"}, {0}},
	{PAPI_L2_TCA, 0, {"L2_REFERENCES"}, {0}},
	{PAPI_L3_TCA, 0, {"L3_REFERENCES"}, {0}},
	{PAPI_L3_TCR, 0, {"L3_READS_ALL_ALL"}, {0}},
	{PAPI_L3_TCW, 0, {"L3_WRITES_ALL_ALL"}, {0}},
	{PAPI_TLB_DM, 0, {"L2DTLB_MISSES"}, {0}},
	{PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH_L2ITLB"}, {0}},
	{PAPI_BR_INS, 0, {"BRANCH_EVENT"}, {0}},
	{PAPI_BR_PRC, 0, {"BR_MISPRED_DETAIL_ALL_CORRECT_PRED"}, {0}},
	{PAPI_BR_MSP, DERIVED_ADD,
	 {"BR_MISPRED_DETAIL_ALL_WRONG_PATH", "BR_MISPRED_DETAIL_ALL_WRONG_TARGET"},
	 {0}},
	{PAPI_TOT_CYC, 0, {"CPU_CYCLES"}, {0}},
	{PAPI_FP_OPS, 0, {"FP_OPS_RETIRED"}, {0}},
	{PAPI_TOT_INS, DERIVED_ADD, {"IA64_INST_RETIRED", "IA32_INST_RETIRED"},
	 {0}},
	{PAPI_LD_INS, 0, {"LOADS_RETIRED"}, {0}},
	{PAPI_SR_INS, 0, {"STORES_RETIRED"}, {0}},
	{PAPI_L2_ICA, 0, {"L2_INST_DEMAND_READS"}, {0}},
	{PAPI_L3_ICA, 0, {"L3_READS_INST_FETCH_ALL"}, {0}},
	{PAPI_L1_TCR, DERIVED_ADD, {"L1D_READS_SET0", "L1I_READS"}, {0}},
	{PAPI_L1_TCA, DERIVED_ADD, {"L1D_READS_SET0", "L1I_READS"}, {0}},
	{PAPI_L2_TCW, 0, {"L2_DATA_REFERENCES_L2_DATA_WRITES"}, {0}},
	{0, 0, {0}, {0}}
};

static itanium_preset_search_t ia3_preset_search_map[] = {
/* not sure */
	{PAPI_CA_SNP, 0, {"BUS_SNOOP_STALL_CYCLES_ANY"}, {0}},
	{PAPI_CA_INV, DERIVED_ADD,
	 {"BUS_MEM_READ_BRIL_SELF", "BUS_MEM_READ_BIL_SELF"}, {0}},
/* should be OK */
	{PAPI_TLB_TL, DERIVED_ADD, {"ITLB_MISSES_FETCH_L2ITLB", "L2DTLB_MISSES"},
	 {0}},
	{PAPI_STL_ICY, 0, {"DISP_STALLED"}, {0}},
	{PAPI_STL_CCY, 0, {"BACK_END_BUBBLE_ALL"}, {0}},
	{PAPI_TOT_IIS, 0, {"INST_DISPERSED"}, {0}},
	{PAPI_RES_STL, 0, {"BE_EXE_BUBBLE_ALL"}, {0}},
	{PAPI_FP_STAL, 0, {"BE_EXE_BUBBLE_FRALL"}, {0}},
/* should be OK */
	{PAPI_L2_TCR, DERIVED_ADD,
	 {"L2D_REFERENCES_READS", "L2I_READS_ALL_DMND", "L2I_READS_ALL_PFTCH"},
	 {0}},
/* what is the correct name here: L2I_READS_ALL_DMND or L2I_DEMANDS_READ ?
 * do not have papi_native_avail at this time, going to use L2I_READS_ALL_DMND always
 * just replace on demand
 */
	{PAPI_L1_TCM, DERIVED_ADD, {"L2I_READS_ALL_DMND", "L1D_READ_MISSES_ALL"},
	 {0}},
	{PAPI_L1_ICM, 0, {"L2I_READS_ALL_DMND"}, {0}},
	{PAPI_L1_DCM, 0, {"L1D_READ_MISSES_ALL"}, {0}},
	{PAPI_L2_TCM, 0, {"L2I_READS_MISS_ALL", "L2D_MISSES"}, {0}},
	{PAPI_L2_DCM, DERIVED_SUB, {"L2D_MISSES"}, {0}},
	{PAPI_L2_ICM, 0, {"L2I_READS_MISS_ALL"}, {0}},
	{PAPI_L3_TCM, 0, {"L3_MISSES"}, {0}},
	{PAPI_L3_ICM, 0, {"L3_READS_INST_FETCH_MISS:M:E:S:I"}, {0}},
	{PAPI_L3_DCM, DERIVED_ADD,
	 {"L3_READS_DATA_READ_MISS:M:E:S:I", "L3_WRITES_DATA_WRITE_MISS:M:E:S:I"},
	 {0}},
	{PAPI_L3_LDM, 0, {"L3_READS_ALL_MISS:M:E:S:I"}, {0}},
	{PAPI_L3_STM, 0, {"L3_WRITES_DATA_WRITE_MISS:M:E:S:I"}, {0}},
/* why L2_INST_DEMAND_READS has been added here for the Itanium II ?
 * OLD:  {PAPI_L1_LDM, DERIVED_ADD, {"L1D_READ_MISSES_ALL", "L2_INST_DEMAND_READS", 0, 0}}
 */
	{PAPI_L1_LDM, 0, {"L1D_READ_MISSES_ALL"}, {0}},
	{PAPI_L2_LDM, 0, {"L3_READS_ALL_ALL:M:E:S:I"}, {0}},
	{PAPI_L2_STM, 0, {"L3_WRITES_ALL_ALL:M:E:S:I"}, {0}},
	{PAPI_L1_DCH, DERIVED_SUB, {"L1D_READS_SET1", "L1D_READ_MISSES_ALL"}, {0}},
	{PAPI_L2_DCH, DERIVED_SUB, {"L2D_REFERENCES_ALL", "L2D_MISSES"}, {0}},
	{PAPI_L3_DCH, DERIVED_ADD,
	 {"L3_READS_DATA_READ_HIT:M:E:S:I", "L3_WRITES_DATA_WRITE_HIT:M:E:S:I"},
	 {0}},
	{PAPI_L1_DCA, 0, {"L1D_READS_SET1"}, {0}},
	{PAPI_L2_DCA, 0, {"L2D_REFERENCES_ALL"}, {0}},
	{PAPI_L3_DCA, 0, {"L3_REFERENCES"}, {0}},
	{PAPI_L1_DCR, 0, {"L1D_READS_SET1"}, {0}},
	{PAPI_L2_DCR, 0, {"L2D_REFERENCES_READS"}, {0}},
	{PAPI_L3_DCR, 0, {"L3_READS_DATA_READ_ALL:M:E:S:I"}, {0}},
	{PAPI_L2_DCW, 0, {"L2D_REFERENCES_WRITES"}, {0}},
	{PAPI_L3_DCW, 0, {"L3_WRITES_DATA_WRITE_ALL:M:E:S:I"}, {0}},
	{PAPI_L3_ICH, 0, {"L3_READS_DINST_FETCH_HIT:M:E:S:I"}, {0}},
	{PAPI_L1_ICR, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS"}, {0}},
	{PAPI_L2_ICR, DERIVED_ADD, {"L2I_READS_ALL_DMND", "L2I_PREFETCHES"}, {0}},
	{PAPI_L3_ICR, 0, {"L3_READS_INST_FETCH_ALL:M:E:S:I"}, {0}},
	{PAPI_L1_ICA, DERIVED_ADD, {"L1I_PREFETCHES", "L1I_READS"}, {0}},
	{PAPI_L2_TCH, DERIVED_SUB, {"L2I_READS_HIT_ALL", "L2D_INSERT_HITS"}, {0}},
	{PAPI_L3_TCH, DERIVED_SUB, {"L3_REFERENCES", "L3_MISSES"}, {0}},
	{PAPI_L2_TCA, DERIVED_ADD, {"L2I_READS_ALL_ALL", "L2D_REFERENCES_ALL"},
	 {0}},
	{PAPI_L3_TCA, 0, {"L3_REFERENCES"}, {0}},
	{PAPI_L3_TCR, 0, {"L3_READS_ALL_ALL:M:E:S:I"}, {0}},
	{PAPI_L3_TCW, 0, {"L3_WRITES_ALL_ALL:M:E:S:I"}, {0}},
	{PAPI_TLB_DM, 0, {"L2DTLB_MISSES"}, {0}},
	{PAPI_TLB_IM, 0, {"ITLB_MISSES_FETCH_L2ITLB"}, {0}},
	{PAPI_BR_INS, 0, {"BRANCH_EVENT"}, {0}},
	{PAPI_BR_PRC, 0, {"BR_MISPRED_DETAIL_ALL_CORRECT_PRED"}, {0}},
	{PAPI_BR_MSP, DERIVED_ADD,
	 {"BR_MISPRED_DETAIL_ALL_WRONG_PATH", "BR_MISPRED_DETAIL_ALL_WRONG_TARGET"},
	 {0}},
	{PAPI_TOT_CYC, 0, {"CPU_OP_CYCLES_ALL"}, {0}},
	{PAPI_FP_OPS, 0, {"FP_OPS_RETIRED"}, {0}},
//   {PAPI_TOT_INS, DERIVED_ADD, {"IA64_INST_RETIRED", "IA32_INST_RETIRED"}, {0}},
	{PAPI_TOT_INS, 0, {"IA64_INST_RETIRED"}, {0}},
	{PAPI_LD_INS, 0, {"LOADS_RETIRED"}, {0}},
	{PAPI_SR_INS, 0, {"STORES_RETIRED"}, {0}},
	{PAPI_L2_ICA, 0, {"L2I_DEMAND_READS"}, {0}},
	{PAPI_L3_ICA, 0, {"L3_READS_INST_FETCH_ALL:M:E:S:I"}, {0}},
	{PAPI_L1_TCR, 0, {"L2I_READS_ALL_ALL"}, {0}},
/* Why are TCA READS+READS_SET0? I used the same as PAPI_L1_TCR, because its an write through cache
 * OLD: {PAPI_L1_TCA, DERIVED_ADD, {"L1D_READS_SET0", "L1I_READS"}, {0}}, 
 */
	{PAPI_L1_TCA, DERIVED_ADD,
	 {"L1I_PREFETCHES", "L1I_READS", "L1D_READS_SET0"}, {0}},
	{PAPI_L2_TCW, 0, {"L2D_REFERENCES_WRITES"}, {0}},
	{0, 0, {0}, {0}}
};

/* This component should never malloc anything. All allocation should be
   done by the high level API. */


/*****************************************************************************
 * Code to support unit masks; only needed by Montecito and above            *
 *****************************************************************************/
static  int _ia64_modify_event( unsigned int event, int modifier );

/* Break a PAPI native event code into its composite event code and pfm mask bits */
static  int
_pfm_decode_native_event( unsigned int EventCode, unsigned int *event,
						  unsigned int *umask )
{
	unsigned int tevent, major, minor;

	tevent = EventCode & PAPI_NATIVE_AND_MASK;
	major = ( tevent & PAPI_NATIVE_EVENT_AND_MASK ) >> PAPI_NATIVE_EVENT_SHIFT;
	if ( major >= ( unsigned int ) _ia64_vector.cmp_info.num_native_events )
		return ( PAPI_ENOEVNT );

	minor = ( tevent & PAPI_NATIVE_UMASK_AND_MASK ) >> PAPI_NATIVE_UMASK_SHIFT;
	*event = major;
	*umask = minor;
	SUBDBG( "EventCode %#08x is event %d, umask %#x\n", EventCode, major,
			minor );
	return ( PAPI_OK );
}

/* This routine is used to step through all possible combinations of umask
    values. It assumes that mask contains a valid combination of array indices
    for this event. */
static  int
encode_native_event_raw( unsigned int event, unsigned int mask )
{
	unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
	SUBDBG( "Old native index was %#08x with %#08x mask\n", tmp, mask );
	tmp = tmp | ( mask << PAPI_NATIVE_UMASK_SHIFT );
	SUBDBG( "New encoding is %#08x\n", tmp | PAPI_NATIVE_MASK );
	return ( tmp | PAPI_NATIVE_MASK );
}

/* convert a collection of pfm mask bits into an array of pfm mask indices */
static  int
prepare_umask( unsigned int foo, unsigned int *values )
{
	unsigned int tmp = foo, i, j = 0;

	SUBDBG( "umask %#x\n", tmp );
	if ( foo == 0 )
		return 0;
	while ( ( i = ffs( tmp ) ) ) {
		tmp = tmp ^ ( 1 << ( i - 1 ) );
		values[j] = i - 1;
		SUBDBG( "umask %d is %d\n", j, values[j] );
		j++;
	}
	return ( j );
}

int
_papi_pfm_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	unsigned int event, umask, num_masks;
	int ret;

	if ( modifier == PAPI_ENUM_FIRST ) {
		*EventCode = PAPI_NATIVE_MASK;	/* assumes first native event is always 0x4000000 */
		return ( PAPI_OK );
	}

	if ( _pfm_decode_native_event( *EventCode, &event, &umask ) != PAPI_OK )
		return ( PAPI_ENOEVNT );

	ret = pfm_get_num_event_masks( event, &num_masks );
	SUBDBG( "pfm_get_num_event_masks: event=%d  num_masks=%d\n", event,
			num_masks );
	if ( ret != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_num_event_masks(%d,%p): %s", event, &num_masks,
				   pfm_strerror( ret ) );
		return ( PAPI_ENOEVNT );
	}
	if ( num_masks > PAPI_NATIVE_UMASK_MAX )
		num_masks = PAPI_NATIVE_UMASK_MAX;
	SUBDBG( "This is umask %d of %d\n", umask, num_masks );

	if ( modifier == PAPI_ENUM_EVENTS ) {
		if ( event < ( unsigned int ) _ia64_vector.cmp_info.num_native_events - 1 ) {
			*EventCode = encode_native_event_raw( event + 1, 0 );
			return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	} else if ( modifier == PAPI_NTV_ENUM_UMASK_COMBOS ) {
		if ( umask + 1 < ( unsigned ) ( 1 << num_masks ) ) {
			*EventCode = encode_native_event_raw( event, umask + 1 );
			return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	} else if ( modifier == PAPI_NTV_ENUM_UMASKS ) {
		int thisbit = ffs( umask );

		SUBDBG( "First bit is %d in %08x\b\n", thisbit - 1, umask );
		thisbit = 1 << thisbit;

		if ( thisbit & ( ( 1 << num_masks ) - 1 ) ) {
			*EventCode = encode_native_event_raw( event, thisbit );
			return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	} else {
		while ( event++ <
				( unsigned int ) _ia64_vector.cmp_info.num_native_events - 1 ) {
			*EventCode = encode_native_event_raw( event + 1, 0 );
			if ( _ia64_modify_event( event + 1, modifier ) )
				return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	}
}

static int
_papi_pfm_ntv_name_to_code( const char *name, unsigned int *event_code )
{
	pfmlib_event_t event;
	unsigned int i, mask = 0;
	int ret;

	SUBDBG( "pfm_find_full_event(%s,%p)\n", name, &event );
	ret = pfm_find_full_event( name, &event );
	if ( ret == PFMLIB_SUCCESS ) {
		/* we can only capture PAPI_NATIVE_UMASK_MAX or fewer masks */
		if ( event.num_masks > PAPI_NATIVE_UMASK_MAX ) {
			SUBDBG( "num_masks (%d) > max masks (%d)\n", event.num_masks,
					PAPI_NATIVE_UMASK_MAX );
			return ( PAPI_ENOEVNT );
		} else {
			/* no mask index can exceed PAPI_NATIVE_UMASK_MAX */
			for ( i = 0; i < event.num_masks; i++ ) {
				if ( event.unit_masks[i] > PAPI_NATIVE_UMASK_MAX ) {
					SUBDBG( "mask index (%d) > max masks (%d)\n",
							event.unit_masks[i], PAPI_NATIVE_UMASK_MAX );
					return ( PAPI_ENOEVNT );
				}
				mask |= 1 << event.unit_masks[i];
			}
			*event_code = encode_native_event_raw( event.event, mask );
			SUBDBG( "event_code: %#x  event: %d  num_masks: %d\n", *event_code,
					event.event, event.num_masks );
			return ( PAPI_OK );
		}
	} else if ( ret == PFMLIB_ERR_UMASK ) {
		ret = pfm_find_event( name, &event.event );
		if ( ret == PFMLIB_SUCCESS ) {
			*event_code = encode_native_event_raw( event.event, 0 );
			return ( PAPI_OK );
		}
	}
	return ( PAPI_ENOEVNT );
}

int
_papi_pfm_ntv_code_to_name( unsigned int EventCode, char *ntv_name, int len )
{
	int ret;
	unsigned int event, umask;
	pfmlib_event_t gete;

	memset( &gete, 0, sizeof ( gete ) );

	if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
		return ( PAPI_ENOEVNT );

	gete.event = event;
	gete.num_masks = prepare_umask( umask, gete.unit_masks );
	if ( gete.num_masks == 0 )
		ret = pfm_get_event_name( gete.event, ntv_name, len );
	else
		ret = pfm_get_full_event_name( &gete, ntv_name, len );
	if ( ret != PFMLIB_SUCCESS ) {
		char tmp[PAPI_2MAX_STR_LEN];
		pfm_get_event_name( gete.event, tmp, sizeof ( tmp ) );
		PAPIERROR
			( "pfm_get_full_event_name(%p(event %d,%s,%d masks),%p,%d): %d -- %s",
			  &gete, gete.event, tmp, gete.num_masks, ntv_name, len, ret,
			  pfm_strerror( ret ) );
		if ( ret == PFMLIB_ERR_FULL )
			return PAPI_EBUF;
		return PAPI_ECMP;
	}
	return PAPI_OK;
}

int
_papi_pfm_ntv_code_to_descr( unsigned int EventCode, char *ntv_descr, int len )
{
	unsigned int event, umask;
	char *eventd, **maskd, *tmp;
	int i, ret, total_len = 0;
	pfmlib_event_t gete;

	memset( &gete, 0, sizeof ( gete ) );

	if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
		return ( PAPI_ENOEVNT );

	ret = pfm_get_event_description( event, &eventd );
	if ( ret != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_event_description(%d,%p): %s",
				   event, &eventd, pfm_strerror( ret ) );
		return ( PAPI_ENOEVNT );
	}

	if ( ( gete.num_masks = prepare_umask( umask, gete.unit_masks ) ) ) {
		maskd = ( char ** ) malloc( gete.num_masks * sizeof ( char * ) );
		if ( maskd == NULL ) {
			free( eventd );
			return ( PAPI_ENOMEM );
		}
		for ( i = 0; i < ( int ) gete.num_masks; i++ ) {
			ret =
				pfm_get_event_mask_description( event, gete.unit_masks[i],
												&maskd[i] );
			if ( ret != PFMLIB_SUCCESS ) {
				PAPIERROR( "pfm_get_event_mask_description(%d,%d,%p): %s",
						   event, umask, &maskd, pfm_strerror( ret ) );
				free( eventd );
				for ( ; i >= 0; i-- )
					free( maskd[i] );
				free( maskd );
				return ( PAPI_EINVAL );
			}
			total_len += strlen( maskd[i] );
		}
		tmp =
			( char * ) malloc( strlen( eventd ) + strlen( ", masks:" ) +
							   total_len + gete.num_masks + 1 );
		if ( tmp == NULL ) {
			for ( i = gete.num_masks - 1; i >= 0; i-- )
				free( maskd[i] );
			free( maskd );
			free( eventd );
		}
		tmp[0] = '\0';
		strcat( tmp, eventd );
		strcat( tmp, ", masks:" );
		for ( i = 0; i < ( int ) gete.num_masks; i++ ) {
			if ( i != 0 )
				strcat( tmp, "," );
			strcat( tmp, maskd[i] );
			free( maskd[i] );
		}
		free( maskd );
	} else {
		tmp = ( char * ) malloc( strlen( eventd ) + 1 );
		if ( tmp == NULL ) {
			free( eventd );
			return ( PAPI_ENOMEM );
		}
		tmp[0] = '\0';
		strcat( tmp, eventd );
		free( eventd );
	}
	strncpy( ntv_descr, tmp, len );
	if ( strlen( tmp ) > ( unsigned int ) len - 1 )
		ret = PAPI_EBUF;
	else
		ret = PAPI_OK;
	free( tmp );
	return ( ret );
}

/*****************************************************************************
 *****************************************************************************/

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

/* Low level functions, should not handle errors, just return codes. */

/* I want to keep the old way to define the preset search map.
   In Itanium2, there are more than 400 native events, if I use the
   index directly, it will be difficult for people to debug, so I
   still keep the old way to define preset search table, but 
   I add this function to generate the preset search map in papi3 
*/
int
generate_preset_search_map( hwi_search_t ** maploc,
							itanium_preset_search_t * oldmap, int num_cnt )
{
	( void ) num_cnt;		 /*unused */
	int pnum, i = 0, cnt;
	char **findme;
	hwi_search_t *psmap;

	/* Count up the presets */
	while ( oldmap[i].preset )
		i++;
	/* Add null entry */
	i++;

	psmap = ( hwi_search_t * ) papi_malloc( i * sizeof ( hwi_search_t ) );
	if ( psmap == NULL )
		return ( PAPI_ENOMEM );
	memset( psmap, 0x0, i * sizeof ( hwi_search_t ) );

	pnum = 0;				 /* preset event counter */
	for ( i = 0; i <= PAPI_MAX_PRESET_EVENTS; i++ ) {
		if ( oldmap[i].preset == 0 )
			break;
		pnum++;
		psmap[i].event_code = oldmap[i].preset;
		psmap[i].data.derived = oldmap[i].derived;
		strcpy( psmap[i].data.operation, oldmap[i].operation );
		findme = oldmap[i].findme;
		cnt = 0;
		while ( *findme != NULL ) {
			if ( cnt == MAX_COUNTER_TERMS ) {
				PAPIERROR( "Count (%d) == MAX_COUNTER_TERMS (%d)\n", cnt,
						   MAX_COUNTER_TERMS );
				papi_free( psmap );
				return ( PAPI_EBUG );
			}
			if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU ) {
				if ( _papi_pfm_ntv_name_to_code
					 ( *findme,
					   ( unsigned int * ) &psmap[i].data.native[cnt] ) !=
					 PAPI_OK ) {
					PAPIERROR( "_papi_pfm_ntv_name_to_code(%s) failed\n",
							   *findme );
					papi_free( psmap );
					return ( PAPI_EBUG );
				} else
					psmap[i].data.native[cnt] ^= PAPI_NATIVE_MASK;
			} else {
				if ( pfm_find_event_byname
					 ( *findme,
					   ( unsigned int * ) &psmap[i].data.native[cnt] ) !=
					 PFMLIB_SUCCESS ) {
					PAPIERROR( "pfm_find_event_byname(%s) failed\n", *findme );
					papi_free( psmap );
					return ( PAPI_EBUG );
				} else
					psmap[i].data.native[cnt] ^= PAPI_NATIVE_MASK;
			}

			findme++;
			cnt++;
		}
		psmap[i].data.native[cnt] = PAPI_NULL;
	}

	*maploc = psmap;
	return ( PAPI_OK );
}


static  char *
search_cpu_info( FILE * f, char *search_str, char *line )
{
	/* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
	/* See the PCL home page for the German version of PAPI. */

	char *s;

	while ( fgets( line, 256, f ) != NULL ) {
		if ( strstr( line, search_str ) != NULL ) {
			/* ignore all characters in line up to : */
			for ( s = line; *s && ( *s != ':' ); ++s );
			if ( *s )
				return ( s );
		}
	}
	return ( NULL );

	/* End stolen code */
}

int
_ia64_ita_set_domain( hwd_control_state_t * this_state, int domain )
{
	int mode = 0, did = 0, i;
	pfmw_param_t *evt = &( ( ia64_control_state_t * ) this_state )->evt;

	if ( domain & PAPI_DOM_USER ) {
		did = 1;
		mode |= PFM_PLM3;
	}

	if ( domain & PAPI_DOM_KERNEL ) {
		did = 1;
		mode |= PFM_PLM0;
	}

	if ( !did )
		return ( PAPI_EINVAL );

	PFMW_PEVT_DFLPLM( evt ) = mode;

	/* Bug fix in case we don't call pfmw_dispatch_events after this code */
	/* Who did this? This sucks, we should always call it here -PJM */

	for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ ) {
		if ( PFMW_PEVT_PFPPC_REG_NUM( evt, i ) ) {
			pfm_ita_pmc_reg_t value;
			SUBDBG( "slot %d, register %lud active, config value %#lx\n",
					i, ( unsigned long ) ( PFMW_PEVT_PFPPC_REG_NUM( evt, i ) ),
					PFMW_PEVT_PFPPC_REG_VAL( evt, i ) );

			PFMW_ARCH_REG_PMCVAL( value ) = PFMW_PEVT_PFPPC_REG_VAL( evt, i );
			value.pmc_ita_count_reg.pmc_plm = mode;
			PFMW_PEVT_PFPPC_REG_VAL( evt, i ) = PFMW_ARCH_REG_PMCVAL( value );

			SUBDBG( "new config value %#lx\n",
					PFMW_PEVT_PFPPC_REG_VAL( evt, i ) );
		}
	}

	return PAPI_OK;
}

int
_ia64_ita2_set_domain( hwd_control_state_t * this_state, int domain )
{
	int mode = 0, did = 0, i;
	pfmw_param_t *evt = &this_state->evt;

	if ( domain & PAPI_DOM_USER ) {
		did = 1;
		mode |= PFM_PLM3;
	}

	if ( domain & PAPI_DOM_KERNEL ) {
		did = 1;
		mode |= PFM_PLM0;
	}

	if ( !did )
		return ( PAPI_EINVAL );

	PFMW_PEVT_DFLPLM( evt ) = mode;

	/* Bug fix in case we don't call pfmw_dispatch_events after this code */
	/* Who did this? This sucks, we should always call it here -PJM */

	for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ ) {
		if ( PFMW_PEVT_PFPPC_REG_NUM( evt, i ) ) {
			pfm_ita2_pmc_reg_t value;
			SUBDBG( "slot %d, register %lud active, config value %#lx\n",
					i, ( unsigned long ) ( PFMW_PEVT_PFPPC_REG_NUM( evt, i ) ),
					PFMW_PEVT_PFPPC_REG_VAL( evt, i ) );

			PFMW_ARCH_REG_PMCVAL( value ) = PFMW_PEVT_PFPPC_REG_VAL( evt, i );
			value.pmc_ita2_counter_reg.pmc_plm = mode;
			PFMW_PEVT_PFPPC_REG_VAL( evt, i ) = PFMW_ARCH_REG_PMCVAL( value );

			SUBDBG( "new config value %#lx\n",
					PFMW_PEVT_PFPPC_REG_VAL( evt, i ) );
		}
	}

	return ( PAPI_OK );
}

int
_ia64_mont_set_domain( hwd_control_state_t * this_state, int domain )
{
	int mode = 0, did = 0, i;
	pfmw_param_t *evt = &( ( ia64_control_state_t * ) this_state )->evt;

	if ( domain & PAPI_DOM_USER ) {
		did = 1;
		mode |= PFM_PLM3;
	}

	if ( domain & PAPI_DOM_KERNEL ) {
		did = 1;
		mode |= PFM_PLM0;
	}

	if ( !did )
		return ( PAPI_EINVAL );

	PFMW_PEVT_DFLPLM( evt ) = mode;

	/* Bug fix in case we don't call pfmw_dispatch_events after this code */
	/* Who did this? This sucks, we should always call it here -PJM */

	for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ ) {
		if ( PFMW_PEVT_PFPPC_REG_NUM( evt, i ) ) {
			pfm_mont_pmc_reg_t value;
			SUBDBG( "slot %d, register %lud active, config value %#lx\n",
					i, ( unsigned long ) ( PFMW_PEVT_PFPPC_REG_NUM( evt, i ) ),
					PFMW_PEVT_PFPPC_REG_VAL( evt, i ) );

			PFMW_ARCH_REG_PMCVAL( value ) = PFMW_PEVT_PFPPC_REG_VAL( evt, i );
			value.pmc_mont_counter_reg.pmc_plm = mode;
			PFMW_PEVT_PFPPC_REG_VAL( evt, i ) = PFMW_ARCH_REG_PMCVAL( value );

			SUBDBG( "new config value %#lx\n",
					PFMW_PEVT_PFPPC_REG_VAL( evt, i ) );
		}
	}

	return ( PAPI_OK );
}

int
_ia64_set_domain( hwd_control_state_t * this_state, int domain )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( _ia64_ita_set_domain( this_state, domain ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( _ia64_ita2_set_domain( this_state, domain ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( _ia64_mont_set_domain( this_state, domain ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

static  int
set_granularity( hwd_control_state_t * this_state, int domain )
{
	( void ) this_state;	 /*unused */
	switch ( domain ) {
	case PAPI_GRN_PROCG:
	case PAPI_GRN_SYS:
	case PAPI_GRN_SYS_CPU:
	case PAPI_GRN_PROC:
		return PAPI_ECMP;
	case PAPI_GRN_THR:
		break;
	default:
		return PAPI_EINVAL;
	}
	return PAPI_OK;
}

int
_ia64_ita_read( hwd_context_t * ctx, hwd_control_state_t * machdep,
				long long **events, int flags )
{
	( void ) flags;			 /*unused */
	unsigned int i;
	pfarg_reg_t readem[_ia64_vector.cmp_info.num_cntrs];

	pfmw_stop( ( ia64_context_t * ) ctx );
	memset( readem, 0x0, sizeof readem );

/* read the 4 counters, the high level function will process the 
   mapping for papi event to hardware counter 
*/
	for ( i = 0; i < ( unsigned int ) _ia64_vector.cmp_info.num_cntrs; i++ ) {
		readem[i].reg_num = PMU_FIRST_COUNTER + i;
	}

	if ( pfmw_perfmonctl
		 ( ( ( ia64_context_t * ) ctx )->tid, ( ( ia64_context_t * ) ctx )->fd,
		   PFM_READ_PMDS, readem, _ia64_vector.cmp_info.num_cntrs ) == -1 ) {
		SUBDBG( "perfmonctl error READ_PMDS errno %d\n", errno );
		pfmw_start( ( ia64_context_t * ) ctx );
		return PAPI_ESYS;
	}

	for ( i = 0; i < ( unsigned int ) _ia64_vector.cmp_info.num_cntrs; i++ ) {
		( ( ia64_control_state_t * ) machdep )->counters[i] =
			readem[i].reg_value;
		SUBDBG( "read counters is %ld\n", readem[i].reg_value );
	}

	pfmw_param_t *pevt = &( ( ( ia64_control_state_t * ) machdep )->evt );
	pfm_ita_pmc_reg_t flop_hack;
	/* special case, We need to scale FP_OPS_HI */
	for ( i = 0; i < PFMW_PEVT_EVTCOUNT( pevt ); i++ ) {
		PFMW_ARCH_REG_PMCVAL( flop_hack ) = PFMW_PEVT_PFPPC_REG_VAL( pevt, i );
		if ( flop_hack.pmc_ita_count_reg.pmc_es == 0xa )
			( ( ia64_control_state_t * ) machdep )->counters[i] *= 4;
	}

	*events = ( ( ia64_control_state_t * ) machdep )->counters;
	pfmw_start( ( ia64_context_t * ) ctx );
	return PAPI_OK;
}


int
_ia64_ita23_read( hwd_context_t * ctx, hwd_control_state_t * machdep,
				  long long **events, int flags )
{
	( void ) flags;			 /*unused */
	int i;
	pfarg_reg_t readem[_ia64_vector.cmp_info.num_cntrs];

	pfmw_stop( ( ia64_context_t * ) ctx );
	memset( readem, 0x0, sizeof readem );

/* read the 4 counters, the high level function will process the 
   mapping for papi event to hardware counter 
*/
	for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ ) {
		readem[i].reg_num = PMU_FIRST_COUNTER + i;
	}

	if ( pfmw_perfmonctl
		 ( ( ( ia64_context_t * ) ctx )->tid, ( ( ia64_context_t * ) ctx )->fd,
		   PFM_READ_PMDS, readem, _ia64_vector.cmp_info.num_cntrs ) == -1 ) {
		SUBDBG( "perfmonctl error READ_PMDS errno %d\n", errno );
		pfmw_start( ( ia64_context_t * ) ctx );
		return PAPI_ESYS;
	}

	for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ ) {
		( ( ia64_control_state_t * ) machdep )->counters[i] =
			readem[i].reg_value;
		SUBDBG( "read counters is %ld\n", readem[i].reg_value );
	}

	*events = ( ( ia64_control_state_t * ) machdep )->counters;
	pfmw_start( ( ia64_context_t * ) ctx );
	return PAPI_OK;
}

int
_ia64_read( hwd_context_t * ctx, hwd_control_state_t * machdep,
			long long **events, int flags )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( _ia64_ita_read( ctx, machdep, events, flags ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( _ia64_ita23_read( ctx, machdep, events, flags ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( _ia64_ita23_read( ctx, machdep, events, flags ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static  int
set_inherit( int arg )
{
	( void ) arg;			 /*unused */
	return PAPI_ECMP;
}

static  int
set_default_domain( hwd_control_state_t * this_state, int domain )
{
	return ( _ia64_set_domain( this_state, domain ) );
}

static  int
set_default_granularity( hwd_control_state_t * this_state, int granularity )
{
	return ( set_granularity( this_state, granularity ) );
}




int
_ia64_init_component( int cidx )
{
	( void ) cidx;			 /*unused */
	int i, retval, type;
	unsigned int version;
	pfmlib_options_t pfmlib_options;
	itanium_preset_search_t *ia_preset_search_map = NULL;

	/* Always initialize globals dynamically to handle forks properly. */

	preset_search_map = NULL;

	/* Opened once for all threads. */
	if ( pfm_initialize(  ) != PFMLIB_SUCCESS )
		return ( PAPI_ESYS );

	if ( pfm_get_version( &version ) != PFMLIB_SUCCESS )
		return PAPI_ECMP;

	if ( PFM_VERSION_MAJOR( version ) != PFM_VERSION_MAJOR( PFMLIB_VERSION ) ) {
		PAPIERROR( "Version mismatch of libpfm: compiled %#x vs. installed %#x",
				   PFM_VERSION_MAJOR( PFMLIB_VERSION ),
				   PFM_VERSION_MAJOR( version ) );
		return PAPI_ECMP;
	}

	memset( &pfmlib_options, 0, sizeof ( pfmlib_options ) );
#ifdef DEBUG
	if ( ISLEVEL( DEBUG_SUBSTRATE ) ) {
		pfmlib_options.pfm_debug = 1;
		pfmlib_options.pfm_verbose = 1;
	}
#endif

	if ( pfm_set_options( &pfmlib_options ) )
		return ( PAPI_ESYS );

	if ( pfm_get_pmu_type( &type ) != PFMLIB_SUCCESS )
		return ( PAPI_ESYS );

	_perfmon2_pfm_pmu_type = type;

	/* Setup presets */

	switch ( type ) {
	case PFMLIB_ITANIUM_PMU:
		ia_preset_search_map = ia1_preset_search_map;
		break;
	case PFMLIB_ITANIUM2_PMU:
		ia_preset_search_map = ia2_preset_search_map;
		break;
	case PFMLIB_MONTECITO_PMU:
		ia_preset_search_map = ia3_preset_search_map;
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component", type );
		return ( PAPI_EBUG );
	}

	int ncnt, nnev;

	retval = pfmw_get_num_events( &nnev );
	if ( retval != PAPI_OK )
		return ( retval );

	retval = pfmw_get_num_counters( &ncnt );
	if ( retval != PAPI_OK )
		return ( retval );

	sprintf( _ia64_vector.cmp_info.support_version, 
		 "%08x", PFMLIB_VERSION );
	sprintf( _ia64_vector.cmp_info.kernel_version, 
		 "%08x", 2 << 16 );	/* 2.0 */

	_ia64_vector.cmp_info.num_native_events = nnev;
	_ia64_vector.cmp_info.num_cntrs = ncnt;
	_ia64_vector.cmp_info.num_mpx_cntrs = ncnt;

	_ia64_vector.cmp_info.clock_ticks = sysconf( _SC_CLK_TCK );
	/* Put the signal handler in use to consume PFM_END_MSG's */
	_papi_hwi_start_signal( _ia64_vector.cmp_info.hardware_intr_sig, 1,
							_ia64_vector.cmp_info.CmpIdx );

	retval = mmtimer_setup();
	if ( retval )
		return ( retval );

	retval =
		generate_preset_search_map( &preset_search_map, ia_preset_search_map,
									_ia64_vector.cmp_info.num_cntrs );
	if ( retval )
		return ( retval );

	retval = _papi_hwi_setup_all_presets( preset_search_map, NULL );
	if ( retval )
		return ( retval );

	/* get_memory_info has a CPU model argument that is not used,
	 * faking it here with hw_info.model which is not set by this
	 * component 
	 */
	retval = _linux_get_memory_info( &_papi_hwi_system_info.hw_info,
				     _papi_hwi_system_info.hw_info.model );
	if ( retval )
		return ( retval );

	return ( PAPI_OK );
}

int
_ia64_init( hwd_context_t * zero )
{
#if defined(USE_PROC_PTTIMER)
	{
		char buf[LINE_MAX];
		int fd;
		sprintf( buf, "/proc/%d/task/%d/stat", getpid(  ), mygettid(  ) );
		fd = open( buf, O_RDONLY );
		if ( fd == -1 ) {
			PAPIERROR( "open(%s)", buf );
			return ( PAPI_ESYS );
		}
		zero->stat_fd = fd;
	}
#endif
	return ( pfmw_create_context( zero ) );
}

/* reset the hardware counters */
int
_ia64_reset( hwd_context_t * ctx, hwd_control_state_t * machdep )
{
	pfmw_param_t *pevt = &( machdep->evt );
	pfarg_reg_t writeem[MAX_COUNTERS];
	int i;

	pfmw_stop( ctx );
	memset( writeem, 0, sizeof writeem );
	for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ ) {
		/* Writing doesn't matter, we're just zeroing the counter. */
		writeem[i].reg_num = PMU_FIRST_COUNTER + i;
		if ( PFMW_PEVT_PFPPC_REG_FLG( pevt, i ) & PFM_REGFL_OVFL_NOTIFY )
			writeem[i].reg_value = machdep->pd[i].reg_long_reset;
	}
	if ( pfmw_perfmonctl
		 ( ctx->tid, ctx->fd, PFM_WRITE_PMDS, writeem,
		   _ia64_vector.cmp_info.num_cntrs ) == -1 ) {
		PAPIERROR( "perfmonctl(PFM_WRITE_PMDS) errno %d", errno );
		return PAPI_ESYS;
	}
	pfmw_start( ctx );
	return ( PAPI_OK );
}

int
_ia64_start( hwd_context_t * ctx, hwd_control_state_t * current_state )
{
	int i;
	pfmw_param_t *pevt = &( current_state->evt );

	pfmw_stop( ctx );

/* write PMCS */
	if ( pfmw_perfmonctl( ctx->tid, ctx->fd, PFM_WRITE_PMCS,
						  PFMW_PEVT_PFPPC( pevt ),
						  PFMW_PEVT_PFPPC_COUNT( pevt ) ) == -1 ) {
		PAPIERROR( "perfmonctl(PFM_WRITE_PMCS) errno %d", errno );
		return ( PAPI_ESYS );
	}
	if ( pfmw_perfmonctl
		 ( ctx->tid, ctx->fd, PFM_WRITE_PMDS, PFMW_PEVT_PFPPD( pevt ),
		   PFMW_PEVT_EVTCOUNT( pevt ) ) == -1 ) {
		PAPIERROR( "perfmonctl(PFM_WRITE_PMDS) errno %d", errno );
		return ( PAPI_ESYS );
	}

/* set the initial value of the hardware counter , if PAPI_overflow or
  PAPI_profil are called, then the initial value is the threshold
*/
	for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ )
		current_state->pd[i].reg_num = PMU_FIRST_COUNTER + i;

	if ( pfmw_perfmonctl( ctx->tid, ctx->fd,
						  PFM_WRITE_PMDS, current_state->pd,
						  _ia64_vector.cmp_info.num_cntrs ) == -1 ) {
		PAPIERROR( "perfmonctl(WRITE_PMDS) errno %d", errno );
		return ( PAPI_ESYS );
	}

	pfmw_start( ctx );

	return PAPI_OK;
}

int
_ia64_stop( hwd_context_t * ctx, hwd_control_state_t * zero )
{
	( void ) zero;			 /*unused */
	pfmw_stop( ctx );
	return PAPI_OK;
}

static  int
round_requested_ns( int ns )
{
	if ( ns < _papi_os_info.itimer_res_ns ) {
		return _papi_os_info.itimer_res_ns;
	} else {
		int leftover_ns = ns % _papi_os_info.itimer_res_ns;
		return ns + leftover_ns;
	}
}

int
_ia64_ctl( hwd_context_t * zero, int code, _papi_int_option_t * option )
{
	int ret;
	switch ( code ) {
	case PAPI_DEFDOM:
		return ( set_default_domain( option->domain.ESI->ctl_state,
									 option->domain.domain ) );
	case PAPI_DOMAIN:
		return ( _ia64_set_domain
				 ( option->domain.ESI->ctl_state, option->domain.domain ) );
	case PAPI_DEFGRN:
		return ( set_default_granularity
				 ( option->granularity.ESI->ctl_state,
				   option->granularity.granularity ) );
	case PAPI_GRANUL:
		return ( set_granularity( option->granularity.ESI->ctl_state,
								  option->granularity.granularity ) );
#if 0
	case PAPI_INHERIT:
		return ( set_inherit( option->inherit.inherit ) );
#endif
	case PAPI_DATA_ADDRESS:
		ret =
			set_default_domain( option->address_range.ESI->ctl_state,
								option->address_range.domain );
		if ( ret != PAPI_OK )
			return ( ret );
		set_drange( zero, option->address_range.ESI->ctl_state, option );
		return ( PAPI_OK );
	case PAPI_INSTR_ADDRESS:
		ret =
			set_default_domain( option->address_range.ESI->ctl_state,
								option->address_range.domain );
		if ( ret != PAPI_OK )
			return ( ret );
		set_irange( zero, option->address_range.ESI->ctl_state, option );
		return ( PAPI_OK );
	case PAPI_DEF_ITIMER:{
		/* flags are currently ignored, eventually the flags will be able
		   to specify whether or not we use POSIX itimers (clock_gettimer) */
		if ( ( option->itimer.itimer_num == ITIMER_REAL ) &&
			 ( option->itimer.itimer_sig != SIGALRM ) )
			return PAPI_EINVAL;
		if ( ( option->itimer.itimer_num == ITIMER_VIRTUAL ) &&
			 ( option->itimer.itimer_sig != SIGVTALRM ) )
			return PAPI_EINVAL;
		if ( ( option->itimer.itimer_num == ITIMER_PROF ) &&
			 ( option->itimer.itimer_sig != SIGPROF ) )
			return PAPI_EINVAL;
		if ( option->itimer.ns > 0 )
			option->itimer.ns = round_requested_ns( option->itimer.ns );
		/* At this point, we assume the user knows what he or
		   she is doing, they maybe doing something arch specific */
		return PAPI_OK;
	}
	case PAPI_DEF_MPX_NS:{
		option->multiplex.ns = round_requested_ns( option->multiplex.ns );
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER_NS:{
		option->itimer.ns = round_requested_ns( option->itimer.ns );
		return ( PAPI_OK );
	}
	default:
		return ( PAPI_EINVAL );
	}
}

int
_ia64_shutdown( hwd_context_t * ctx )
{
#if defined(USE_PROC_PTTIMER)
	close( ctx->stat_fd );
#endif

	return ( pfmw_destroy_context( ctx ) );
}

static int
ia64_ita_process_profile_buffer( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	( void ) thread;		 /*unused */
	pfmw_smpl_hdr_t *hdr;
	pfmw_smpl_entry_t *ent;
	unsigned long buf_pos;
	unsigned long entry_size;
	int ret, reg_num, count, pos;
	unsigned int i, EventCode = 0, eventindex, native_index = 0;
	ia64_control_state_t *this_state;
	pfm_ita_pmd_reg_t *reg;
	unsigned long overflow_vector, pc;


	if ( ( ESI->state & PAPI_PROFILING ) == 0 )
		return ( PAPI_EBUG );

	this_state = ( ia64_control_state_t * ) ( ESI->ctl_state );
	hdr = ( pfmw_smpl_hdr_t * ) this_state->smpl_vaddr;

	entry_size = sizeof ( pfmw_smpl_entry_t );

	/*
	 * walk through all the entries recorded in the buffer
	 */
	buf_pos = ( unsigned long ) ( hdr + 1 );
	for ( i = 0; i < hdr->hdr_count; i++ ) {
		ret = 0;
		ent = ( pfmw_smpl_entry_t * ) buf_pos;
		/* PFM30 only one PMD overflows in each sample */
		overflow_vector = 1 << ent->ovfl_pmd;

		SUBDBG( "Entry %d PID:%d CPU:%d ovfl_vector:%#lx IIP:%#016lx\n",
				i, ent->pid, ent->cpu, overflow_vector, ent->ip );

		while ( overflow_vector ) {
			reg_num = ffs( overflow_vector ) - 1;
			/* find the event code */
			for ( count = 0; count < ESI->profile.event_counter; count++ ) {
				eventindex = ESI->profile.EventIndex[count];
				pos = ESI->EventInfoArray[eventindex].pos[0];
				if ( pos + PMU_FIRST_COUNTER == reg_num ) {
					EventCode = ESI->profile.EventCode[count];
					native_index =
						ESI->NativeInfoArray[pos].
						ni_event & PAPI_NATIVE_AND_MASK;
					break;
				}
			}
			/* something is wrong */
			if ( count == ESI->profile.event_counter ) {
				PAPIERROR
					( "wrong count: %d vs. ESI->profile.event_counter %d\n",
					  count, ESI->profile.event_counter );
				return ( PAPI_EBUG );
			}

			/* print entry header */
			pc = ent->ip;
			if ( pfm_ita_is_dear( native_index ) ) {
				reg = ( pfm_ita_pmd_reg_t * ) ( ent + 1 );
				reg++;
				reg++;
				pc = ( reg->pmd17_ita_reg.dear_iaddr << 4 ) | ( reg->
																pmd17_ita_reg.
																dear_slot );
				/* adjust pointer position */
				buf_pos += ( hweight64( DEAR_REGS_MASK ) << 3 );
			}

			_papi_hwi_dispatch_profile( ESI, ( vptr_t ) pc, ( long long ) 0,
										count );
			overflow_vector ^= ( unsigned long ) 1 << reg_num;
		}
		/*  move to next entry */
		buf_pos += entry_size;
	}						 /* end of if */
	return ( PAPI_OK );
}

static int
ia64_ita2_process_profile_buffer( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	( void ) thread;		 /*unused */
	pfmw_smpl_hdr_t *hdr;
	pfmw_smpl_entry_t *ent;
	unsigned long buf_pos;
	unsigned long entry_size;
	int ret, reg_num, count, pos;
	unsigned int i, EventCode = 0, eventindex, native_index = 0;
	ia64_control_state_t *this_state;
	pfm_ita2_pmd_reg_t *reg;
	unsigned long overflow_vector, pc;


	if ( ( ESI->state & PAPI_PROFILING ) == 0 )
		return ( PAPI_EBUG );

	this_state = ( ia64_control_state_t * ) ( ESI->ctl_state );
	hdr = ( pfmw_smpl_hdr_t * ) ( this_state->smpl_vaddr );

	entry_size = sizeof ( pfmw_smpl_entry_t );

	/*
	 * walk through all the entries recorded in the buffer
	 */
	buf_pos = ( unsigned long ) ( hdr + 1 );
	for ( i = 0; i < hdr->hdr_count; i++ ) {
		ret = 0;
		ent = ( pfmw_smpl_entry_t * ) buf_pos;
		/* PFM30 only one PMD overflows in each sample */
		overflow_vector = 1 << ent->ovfl_pmd;

		SUBDBG( "Entry %d PID:%d CPU:%d ovfl_vector:%#lx IIP:%#016lx\n",
				i, ent->pid, ent->cpu, overflow_vector, ent->ip );

		while ( overflow_vector ) {
			reg_num = ffs( overflow_vector ) - 1;
			/* find the event code */
			for ( count = 0; count < ESI->profile.event_counter; count++ ) {
				eventindex = ESI->profile.EventIndex[count];
				pos = ESI->EventInfoArray[eventindex].pos[0];
				if ( pos + PMU_FIRST_COUNTER == reg_num ) {
					EventCode = ESI->profile.EventCode[count];
					native_index =
						ESI->NativeInfoArray[pos].
						ni_event & PAPI_NATIVE_AND_MASK;
					break;
				}
			}
			/* something is wrong */
			if ( count == ESI->profile.event_counter ) {
				PAPIERROR
					( "wrong count: %d vs. ESI->profile.event_counter %d\n",
					  count, ESI->profile.event_counter );
				return ( PAPI_EBUG );
			}

			/* print entry header */
			pc = ent->ip;
			if ( pfm_ita2_is_dear( native_index ) ) {
				reg = ( pfm_ita2_pmd_reg_t * ) ( ent + 1 );
				reg++;
				reg++;
				pc = ( ( reg->pmd17_ita2_reg.dear_iaddr +
						 reg->pmd17_ita2_reg.dear_bn ) << 4 )
					| reg->pmd17_ita2_reg.dear_slot;

				/* adjust pointer position */
				buf_pos += ( hweight64( DEAR_REGS_MASK ) << 3 );
			}

			_papi_hwi_dispatch_profile( ESI, ( vptr_t ) pc, ( long long ) 0,
										count );
			overflow_vector ^= ( unsigned long ) 1 << reg_num;
		}
		/*  move to next entry */
		buf_pos += entry_size;
	}						 /* end of if */
	return ( PAPI_OK );
}

static int
ia64_mont_process_profile_buffer( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	( void ) thread;		 /*unused */
	pfmw_smpl_hdr_t *hdr;
	pfmw_smpl_entry_t *ent;
	unsigned long buf_pos;
	unsigned long entry_size;
	int ret, reg_num, count, pos;
	unsigned int i, EventCode = 0, eventindex, native_index = 0;
	ia64_control_state_t *this_state;
	pfm_mont_pmd_reg_t *reg;
	unsigned long overflow_vector, pc;
	unsigned int umask;


	if ( ( ESI->state & PAPI_PROFILING ) == 0 )
		return ( PAPI_EBUG );

	this_state = ( ia64_control_state_t * ) ESI->ctl_state;
	hdr = ( pfmw_smpl_hdr_t * ) this_state->smpl_vaddr;

	entry_size = sizeof ( pfmw_smpl_entry_t );

	/*
	 * walk through all the entries recorded in the buffer
	 */
	buf_pos = ( unsigned long ) ( hdr + 1 );
	for ( i = 0; i < hdr->hdr_count; i++ ) {
		ret = 0;
		ent = ( pfmw_smpl_entry_t * ) buf_pos;
		/* PFM30 only one PMD overflows in each sample */
		overflow_vector = 1 << ent->ovfl_pmd;

		SUBDBG( "Entry %d PID:%d CPU:%d ovfl_vector:%#lx IIP:%#016lx\n",
				i, ent->pid, ent->cpu, overflow_vector, ent->ip );

		while ( overflow_vector ) {
			reg_num = ffs( overflow_vector ) - 1;
			/* find the event code */
			for ( count = 0; count < ESI->profile.event_counter; count++ ) {
				eventindex = ESI->profile.EventIndex[count];
				pos = ESI->EventInfoArray[eventindex].pos[0];
				if ( pos + PMU_FIRST_COUNTER == reg_num ) {
					EventCode = ESI->profile.EventCode[count];
					if ( _pfm_decode_native_event
						 ( ESI->NativeInfoArray[pos].ni_event, &native_index,
						   &umask ) != PAPI_OK )
						return ( PAPI_ENOEVNT );
					break;
				}
			}
			/* something is wrong */
			if ( count == ESI->profile.event_counter ) {
				PAPIERROR
					( "wrong count: %d vs. ESI->profile.event_counter %d\n",
					  count, ESI->profile.event_counter );
				return ( PAPI_EBUG );
			}

			/* print entry header */
			pc = ent->ip;
			if ( pfm_mont_is_dear( native_index ) ) {
				reg = ( pfm_mont_pmd_reg_t * ) ( ent + 1 );
				reg++;
				reg++;
				pc = ( ( reg->pmd36_mont_reg.dear_iaddr +
						 reg->pmd36_mont_reg.dear_bn ) << 4 )
					| reg->pmd36_mont_reg.dear_slot;
				/* adjust pointer position */
				buf_pos += ( hweight64( DEAR_REGS_MASK ) << 3 );
			}

			_papi_hwi_dispatch_profile( ESI, ( vptr_t ) pc, ( long long ) 0,
										count );
			overflow_vector ^= ( unsigned long ) 1 << reg_num;
		}
		/*  move to next entry */
		buf_pos += entry_size;
	}						 /* end of if */
	return ( PAPI_OK );
}

static int
ia64_process_profile_buffer( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( ia64_ita_process_profile_buffer( thread, ESI ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( ia64_ita2_process_profile_buffer( thread, ESI ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( ia64_mont_process_profile_buffer( thread, ESI ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

static  void
ia64_dispatch_sigprof( int n, hwd_siginfo_t * info, hwd_ucontext_t *sc )
{
	( void ) n;				 /*unused */
	_papi_hwi_context_t ctx;
	ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );
	vptr_t address;
	int cidx = _ia64_vector.cmp_info.CmpIdx;

#if defined(DEBUG)
	if ( thread == NULL ) {
		PAPIERROR( "thread == NULL in _papi_hwd_dispatch_timer!" );
		return;
	}
#endif

	ctx.si = info;
	ctx.ucontext = sc;
	address = GET_OVERFLOW_ADDRESS( ( ctx ) );

	if ( ( thread == NULL ) || ( thread->running_eventset[cidx] == NULL ) ) {
		SUBDBG( "%p, %p\n", thread, thread->running_eventset[cidx] );
		return;
	}

	if ( thread->running_eventset[cidx]->overflow.
		 flags & PAPI_OVERFLOW_FORCE_SW ) {
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, NULL, 0,
											0, &thread, cidx );
		return;
	}

	pfm_msg_t msg;
	int ret, fd;
	fd = info->si_fd;
  retry:
	ret = read( fd, &msg, sizeof ( msg ) );
	if ( ret == -1 ) {
		if ( errno == EINTR ) {
			SUBDBG( "read(%d) interrupted, retrying\n", fd );
			goto retry;
		} else {
			PAPIERROR( "read(%d): errno %d", fd, errno );
		}
	} else if ( ret != sizeof ( msg ) ) {
		PAPIERROR( "read(%d): short %d vs. %d bytes", fd, ret, sizeof ( msg ) );
		ret = -1;
	}
#if defined(HAVE_PFM_MSG_TYPE)
	if ( msg.type == PFM_MSG_END ) {
		SUBDBG( "PFM_MSG_END\n" );
		return;
	}
	if ( msg.type != PFM_MSG_OVFL ) {
		PAPIERROR( "unexpected msg type %d", msg.type );
		return;
	}
#else
	if ( msg.pfm_gen_msg.msg_type == PFM_MSG_END ) {
		SUBDBG( "PFM_MSG_END\n" );
		return;
	}
	if ( msg.pfm_gen_msg.msg_type != PFM_MSG_OVFL ) {
		PAPIERROR( "unexpected msg type %d", msg.pfm_gen_msg.msg_type );
		return;
	}
#endif
	if ( ret != -1 ) {
		if ( ( thread->running_eventset[cidx]->state & PAPI_PROFILING ) &&
			 !( thread->running_eventset[cidx]->profile.
				flags & PAPI_PROFIL_FORCE_SW ) )
			ia64_process_profile_buffer( thread,
										 thread->running_eventset[cidx] );
		else
			_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address,
												NULL,
												msg.pfm_ovfl_msg.
												msg_ovfl_pmds[0] >>
												PMU_FIRST_COUNTER, 0,
												&thread, cidx );
	}
	if ( pfmw_perfmonctl( 0, fd, PFM_RESTART, 0, 0 ) == -1 ) {
		PAPIERROR( "perfmonctl(PFM_RESTART) errno %d, %s", errno,
				   strerror( errno ) );
		return;
	}
}

void
_ia64_dispatch_timer( int signal, hwd_siginfo_t * info, void *context )
{
	ia64_dispatch_sigprof( signal, info, context );
}

static int
set_notify( EventSetInfo_t * ESI, int index, int value )
{
	int *pos, count, i;
	unsigned int hwcntr;
	pfmw_param_t *pevt =
		&( ( ( ia64_control_state_t * ) ESI->ctl_state )->evt );

	pos = ESI->EventInfoArray[index].pos;
	count = 0;
	while ( pos[count] != -1 && count < _ia64_vector.cmp_info.num_cntrs ) {
		hwcntr = pos[count] + PMU_FIRST_COUNTER;
		for ( i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++ ) {
			if ( PFMW_PEVT_PFPPC_REG_NUM( pevt, i ) == hwcntr ) {
				SUBDBG( "Found hw counter %d in %d, flags %d\n", hwcntr, i,
						value );
				PFMW_PEVT_PFPPC_REG_FLG( pevt, i ) = value;
/*
         #ifdef PFM30
            if (value)
               pevt->pc[i].reg_reset_pmds[0] = 1UL << pevt->pc[i].reg_num;
            else 
               pevt->pc[i].reg_reset_pmds[0] = 0;
         #endif
*/
				break;
			}
		}
		count++;
	}
	return ( PAPI_OK );
}

int
_ia64_stop_profiling( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	int cidx = _ia64_vector.cmp_info.CmpIdx;

	pfmw_stop( thread->context[cidx] );
	return ( ia64_process_profile_buffer( thread, ESI ) );
}


int
_ia64_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	int cidx = _ia64_vector.cmp_info.CmpIdx;
	hwd_control_state_t *this_state = ESI->ctl_state;
	hwd_context_t *ctx = ESI->master->context[cidx];
	int ret;

	ret = _ia64_vector.set_overflow( ESI, EventIndex, threshold );
	if ( ret != PAPI_OK )
		return ret;
	ret = pfmw_destroy_context( ctx );
	if ( ret != PAPI_OK )
		return ret;
	if ( threshold == 0 )
		ret = pfmw_create_context( ctx );
	else
		ret =
			pfmw_recreate_context( ESI, ctx, &this_state->smpl_vaddr,
								   EventIndex );

//#warning "This should be handled in the high level layers"
	ESI->state ^= PAPI_OVERFLOWING;
	ESI->overflow.flags ^= PAPI_OVERFLOW_HARDWARE;

	return ( ret );
}

int
_ia64_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	hwd_control_state_t *this_state = ESI->ctl_state;
	int j, retval = PAPI_OK, *pos;
	int cidx = _ia64_vector.cmp_info.CmpIdx;

	pos = ESI->EventInfoArray[EventIndex].pos;
	j = pos[0];
	SUBDBG( "Hardware counter %d used in overflow, threshold %d\n", j,
			threshold );

	if ( threshold == 0 ) {
		/* Remove the signal handler */

		retval = _papi_hwi_stop_signal( _ia64_vector.cmp_info.hardware_intr_sig );
		if ( retval != PAPI_OK )
			return ( retval );

		/* Remove the overflow notifier on the proper event. */

		set_notify( ESI, EventIndex, 0 );

		this_state->pd[j].reg_value = 0;
		this_state->pd[j].reg_long_reset = 0;
		this_state->pd[j].reg_short_reset = 0;
	} else {
		retval =
			_papi_hwi_start_signal( _ia64_vector.cmp_info.hardware_intr_sig, 1,
									cidx );
		if ( retval != PAPI_OK )
			return ( retval );

		/* Set the overflow notifier on the proper event. Remember that selector */

		set_notify( ESI, EventIndex, PFM_REGFL_OVFL_NOTIFY );

		this_state->pd[j].reg_value =
			( ~0UL ) - ( unsigned long ) threshold + 1;
		this_state->pd[j].reg_short_reset =
			( ~0UL ) - ( unsigned long ) threshold + 1;
		this_state->pd[j].reg_long_reset =
			( ~0UL ) - ( unsigned long ) threshold + 1;

	}
	return ( retval );
}

int
_ia64_ntv_code_to_name( unsigned int EventCode, char *ntv_name, int len )
{
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU )
		return ( _papi_pfm_ntv_code_to_name( EventCode, ntv_name, len ) );
	else {
		char name[PAPI_MAX_STR_LEN];
		int ret = 0;

		pfmw_get_event_name( name, EventCode ^ PAPI_NATIVE_MASK );

		if ( ret != PAPI_OK )
			return ( PAPI_ENOEVNT );

		strncpy( ntv_name, name, len );
		return ( PAPI_OK );
	}
}

int
_ia64_ntv_code_to_descr( unsigned int EventCode, char *ntv_descr, int len )
{
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU )
		return ( _papi_pfm_ntv_code_to_descr( EventCode, ntv_descr, len ) );
	else {
#if defined(HAVE_PFM_GET_EVENT_DESCRIPTION)
		pfmw_get_event_description( EventCode ^ PAPI_NATIVE_MASK, ntv_descr,
									len );
		return ( PAPI_OK );
#else
		return ( _ia64_ntv_code_to_name( EventCode, ntv_descr, len ) );
#endif
	}
}

static  int
_ia64_modify_event( unsigned int event, int modifier )
{
	switch ( modifier ) {
	case PAPI_NTV_ENUM_IARR:
		return ( pfmw_support_iarr( event ) );
	case PAPI_NTV_ENUM_DARR:
		return ( pfmw_support_darr( event ) );
	case PAPI_NTV_ENUM_OPCM:
		return ( pfmw_support_opcm( event ) );
	case PAPI_NTV_ENUM_DEAR:
		return ( pfmw_is_dear( event ) );
	case PAPI_NTV_ENUM_IEAR:
		return ( pfmw_is_iear( event ) );
	default:
		return ( 1 );
	}
}

int
_ia64_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU )
		return ( _papi_pfm_ntv_enum_events( EventCode, modifier ) );
	else {
		int index = *EventCode & PAPI_NATIVE_AND_MASK;

		if ( modifier == PAPI_ENUM_FIRST ) {
			*EventCode = PAPI_NATIVE_MASK;
			return ( PAPI_OK );
		}

		while ( index++ < _ia64_vector.cmp_info.num_native_events - 1 ) {
			*EventCode += 1;
			if ( _ia64_modify_event
				 ( ( *EventCode ^ PAPI_NATIVE_MASK ), modifier ) )
				return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	}
}

int
_ia64_ita_init_control_state( hwd_control_state_t * this_state )
{
	pfmw_param_t *evt;
	pfmw_ita1_param_t *param;
	ia64_control_state_t *ptr;

	ptr = ( ia64_control_state_t * ) this_state;
	evt = &( ptr->evt );

	param = &( ptr->ita_lib_param.ita_param );
	memset( evt, 0, sizeof ( pfmw_param_t ) );
	memset( param, 0, sizeof ( pfmw_ita1_param_t ) );

	_ia64_ita_set_domain( this_state, _ia64_vector.cmp_info.default_domain );
/* set library parameter pointer */

	return ( PAPI_OK );
}

int
_ia64_ita2_init_control_state( hwd_control_state_t * this_state )
{
	pfmw_param_t *evt;
	pfmw_ita2_param_t *param;
	ia64_control_state_t *ptr;

	ptr = ( ia64_control_state_t * ) this_state;
	evt = &( ptr->evt );

	param = &( ptr->ita_lib_param.ita2_param );
	memset( evt, 0, sizeof ( pfmw_param_t ) );
	memset( param, 0, sizeof ( pfmw_ita2_param_t ) );

	_ia64_ita2_set_domain( this_state, _ia64_vector.cmp_info.default_domain );
/* set library parameter pointer */
	evt->mod_inp = &( param->ita2_input_param );
	evt->mod_outp = &( param->ita2_output_param );

	return ( PAPI_OK );
}

int
_ia64_mont_init_control_state( hwd_control_state_t * this_state )
{
	pfmw_param_t *evt;
	pfmw_mont_param_t *param;
	ia64_control_state_t *ptr;

	ptr = ( ia64_control_state_t * ) this_state;
	evt = &( ptr->evt );

	param = &( ptr->ita_lib_param.mont_param );
	memset( evt, 0, sizeof ( pfmw_param_t ) );
	memset( param, 0, sizeof ( pfmw_mont_param_t ) );

	_ia64_mont_set_domain( this_state, _ia64_vector.cmp_info.default_domain );
/* set library parameter pointer */
	evt->mod_inp = &( param->mont_input_param );
	evt->mod_outp = &( param->mont_output_param );

	return ( PAPI_OK );
}

int
_ia64_init_control_state( hwd_control_state_t * this_state )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( _ia64_ita_init_control_state( this_state ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( _ia64_ita2_init_control_state( this_state ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( _ia64_mont_init_control_state( this_state ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

void
_ia64_remove_native( hwd_control_state_t * this_state,
					 NativeInfo_t * nativeInfo )
{
	( void ) this_state;	 /*unused */
	( void ) nativeInfo;	 /*unused */
	return;
}

int
_ia64_mont_update_control_state( hwd_control_state_t * this_state,
								 NativeInfo_t * native, int count,
								 hwd_context_t * zero )
{
	( void ) zero;			 /*unused */
	int org_cnt;
	pfmw_param_t *evt = &this_state->evt;
	pfmw_param_t copy_evt;
	unsigned int i, j, event, umask, EventCode;
	pfmlib_event_t gete;
	char name[128];

	if ( count == 0 ) {
		for ( i = 0; i < ( unsigned int ) _ia64_vector.cmp_info.num_cntrs; i++ )
			PFMW_PEVT_EVENT( evt, i ) = 0;
		PFMW_PEVT_EVTCOUNT( evt ) = 0;
		memset( PFMW_PEVT_PFPPC( evt ), 0, sizeof ( PFMW_PEVT_PFPPC( evt ) ) );
		memset( &evt->inp.pfp_unavail_pmcs, 0, sizeof ( pfmlib_regmask_t ) );
		return ( PAPI_OK );
	}

/* save the old data */
	org_cnt = PFMW_PEVT_EVTCOUNT( evt );

	memcpy( &copy_evt, evt, sizeof ( pfmw_param_t ) );

	for ( i = 0; i < ( unsigned int ) _ia64_vector.cmp_info.num_cntrs; i++ )
		PFMW_PEVT_EVENT( evt, i ) = 0;
	PFMW_PEVT_EVTCOUNT( evt ) = 0;
	memset( PFMW_PEVT_PFPPC( evt ), 0, sizeof ( PFMW_PEVT_PFPPC( evt ) ) );
	memset( &evt->inp.pfp_unavail_pmcs, 0, sizeof ( pfmlib_regmask_t ) );

	SUBDBG( " original count is %d\n", org_cnt );

/* add new native events to the evt structure */
	for ( i = 0; i < ( unsigned int ) count; i++ ) {
		memset( &gete, 0, sizeof ( gete ) );
		EventCode = native[i].ni_event;
		_papi_pfm_ntv_code_to_name( EventCode, name, 128 );
		if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
			return ( PAPI_ENOEVNT );

		SUBDBG( " evtcode=%#x evtindex=%d name: %s\n", EventCode, event,
				name );

		PFMW_PEVT_EVENT( evt, i ) = event;
		evt->inp.pfp_events[i].num_masks = 0;
		gete.event = event;
		gete.num_masks = prepare_umask( umask, gete.unit_masks );
		if ( gete.num_masks ) {
			evt->inp.pfp_events[i].num_masks = gete.num_masks;
			for ( j = 0; j < gete.num_masks; j++ )
				evt->inp.pfp_events[i].unit_masks[j] = gete.unit_masks[j];
		}
	}
	PFMW_PEVT_EVTCOUNT( evt ) = count;
	/* Recalcuate the pfmlib_param_t structure, may also signal conflict */
	if ( pfmw_dispatch_events( evt ) ) {
		SUBDBG( "pfmw_dispatch_events fail\n" );
		/* recover the old data */
		PFMW_PEVT_EVTCOUNT( evt ) = org_cnt;
		/*for (i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++)
		   PFMW_PEVT_EVENT(evt,i) = events[i];
		 */
		memcpy( evt, &copy_evt, sizeof ( pfmw_param_t ) );
		return ( PAPI_ECNFLCT );
	}
	SUBDBG( "event_count=%d\n", PFMW_PEVT_EVTCOUNT( evt ) );

	for ( i = 0; i < PFMW_PEVT_EVTCOUNT( evt ); i++ ) {
		native[i].ni_position = PFMW_PEVT_PFPPC_REG_NUM( evt, i )
			- PMU_FIRST_COUNTER;
		SUBDBG( "event_code is %d, reg_num is %d\n",
				native[i].ni_event & PAPI_NATIVE_AND_MASK,
				native[i].ni_position );
	}

	return ( PAPI_OK );
}

int
_ia64_ita_update_control_state( hwd_control_state_t * this_state,
								NativeInfo_t * native, int count,
								hwd_context_t * zero )
{
	( void ) zero;			 /*unused */
	int index, org_cnt;
	unsigned int i;
	pfmw_param_t *evt = &this_state->evt;
	pfmw_param_t copy_evt;

	if ( count == 0 ) {
		for ( i = 0; i < ( unsigned int ) _ia64_vector.cmp_info.num_cntrs; i++ )
			PFMW_PEVT_EVENT( evt, i ) = 0;
		PFMW_PEVT_EVTCOUNT( evt ) = 0;
		memset( PFMW_PEVT_PFPPC( evt ), 0, sizeof ( PFMW_PEVT_PFPPC( evt ) ) );
		memset( &evt->inp.pfp_unavail_pmcs, 0, sizeof ( pfmlib_regmask_t ) );
		return ( PAPI_OK );
	}

/* save the old data */
	org_cnt = PFMW_PEVT_EVTCOUNT( evt );

	memcpy( &copy_evt, evt, sizeof ( pfmw_param_t ) );
	for ( i = 0; i < ( unsigned int ) _ia64_vector.cmp_info.num_cntrs; i++ )
		PFMW_PEVT_EVENT( evt, i ) = 0;
	PFMW_PEVT_EVTCOUNT( evt ) = 0;
	memset( PFMW_PEVT_PFPPC( evt ), 0, sizeof ( PFMW_PEVT_PFPPC( evt ) ) );
	memset( &evt->inp.pfp_unavail_pmcs, 0, sizeof ( pfmlib_regmask_t ) );

	SUBDBG( " original count is %d\n", org_cnt );

/* add new native events to the evt structure */
	for ( i = 0; i < ( unsigned int ) count; i++ ) {
		index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
		PFMW_PEVT_EVENT( evt, i ) = index;
	}
	PFMW_PEVT_EVTCOUNT( evt ) = count;
	/* Recalcuate the pfmlib_param_t structure, may also signal conflict */
	if ( pfmw_dispatch_events( evt ) ) {
		SUBDBG( "pfmw_dispatch_events fail\n" );
		/* recover the old data */
		PFMW_PEVT_EVTCOUNT( evt ) = org_cnt;
		/*for (i = 0; i < _ia64_vector.cmp_info.num_cntrs; i++)
		   PFMW_PEVT_EVENT(evt,i) = events[i];
		 */
		memcpy( evt, &copy_evt, sizeof ( pfmw_param_t ) );
		return ( PAPI_ECNFLCT );
	}
	SUBDBG( "event_count=%d\n", PFMW_PEVT_EVTCOUNT( evt ) );

	for ( i = 0; i < PFMW_PEVT_EVTCOUNT( evt ); i++ ) {
		native[i].ni_position = PFMW_PEVT_PFPPC_REG_NUM( evt, i )
			- PMU_FIRST_COUNTER;
		SUBDBG( "event_code is %d, reg_num is %d\n",
				native[i].ni_event & PAPI_NATIVE_AND_MASK,
				native[i].ni_position );
	}

	return ( PAPI_OK );
}

int
_ia64_update_control_state( hwd_control_state_t * this_state,
							NativeInfo_t * native, int count,
							hwd_context_t * zero )
{
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_ITANIUM_PMU:
		return ( _ia64_ita_update_control_state
				 ( this_state, native, count, zero ) );
		break;
	case PFMLIB_ITANIUM2_PMU:
		return ( _ia64_ita_update_control_state
				 ( this_state, native, count, zero ) );
		break;
	case PFMLIB_MONTECITO_PMU:
		return ( _ia64_mont_update_control_state
				 ( this_state, native, count, zero ) );
		break;
	default:
		PAPIERROR( "PMU type %d is not supported by this component",
				   _perfmon2_pfm_pmu_type );
		return ( PAPI_EBUG );
	}
}

papi_vector_t _ia64_vector = {
   .cmp_info = {
      .name = "perfmon-ia64.c",
      .version = "5.0",

      /* default component information (unspecified values initialized to 0) */
      .default_domain = PAPI_DOM_USER,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
      .default_granularity = PAPI_GRN_THR,
      .available_granularities = PAPI_GRN_THR,
      .hardware_intr_sig = PAPI_INT_SIGNAL,
      .hardware_intr = 1,
      /* component specific cmp_info initializations */
      .fast_real_timer = 1,
      .fast_virtual_timer = 0,
      .attach = 0,
      .attach_must_ptrace = 0,
      .kernel_profile = 1,
      .cntr_umasks = 1;	
  },

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( ia64_context_t ),
			 .control_state = sizeof ( ia64_control_state_t ),
			 .reg_value = sizeof ( ia64_register_t ),
			 .reg_alloc = sizeof ( ia64_reg_alloc_t ),
			 }
	,

	/* function pointers in this component */
	.init_control_state = _ia64_init_control_state,
	.start = _ia64_start,
	.stop = _ia64_stop,
	.read = _ia64_read,
	.shutdown_thread = _ia64_shutdown,
	.ctl = _ia64_ctl,
	.update_control_state = _ia64_update_control_state,
	.set_domain = _ia64_set_domain,
	.reset = _ia64_reset,
	.set_overflow = _ia64_set_overflow,
	.set_profile = _ia64_set_profile,
	.stop_profiling = _ia64_stop_profiling,
	.init_component = _ia64_init_component,
	.dispatch_timer = _ia64_dispatch_timer,
	.init_thread = _ia64_init,

	.ntv_enum_events = _ia64_ntv_enum_events,
	.ntv_code_to_name = _ia64_ntv_code_to_name,
	.ntv_code_to_descr = _ia64_ntv_code_to_descr,

};
