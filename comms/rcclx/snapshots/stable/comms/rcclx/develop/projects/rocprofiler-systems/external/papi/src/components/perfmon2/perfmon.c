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
   - Enum event uses native_map not pfm()
   - Hook up globals to be freed to sub_info
   - Better feature bit support for IEAR
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_libpfm_events.h"
#include "extras.h"

#include "perfmon.h"

#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"

#ifdef __ia64__
#include "perfmon/pfmlib_itanium2.h"
#include "perfmon/pfmlib_montecito.h"
#endif

typedef unsigned uint;

/* Advance declarations */
static int _papi_pfm_set_overflow( EventSetInfo_t * ESI, int EventIndex,
							int threshold );
papi_vector_t _perfmon2_vector;


/* Static locals */

static int _perfmon2_pfm_pmu_type = -1;
static pfmlib_regmask_t _perfmon2_pfm_unavailable_pmcs;
static pfmlib_regmask_t _perfmon2_pfm_unavailable_pmds;

/* Debug functions */

#ifdef DEBUG
static void
dump_smpl_arg( pfm_dfl_smpl_arg_t * arg )
{
	SUBDBG( "SMPL_ARG.buf_size = %llu\n",
			( unsigned long long ) arg->buf_size );
	SUBDBG( "SMPL_ARG.buf_flags = %d\n", arg->buf_flags );
}

static void
dump_sets( pfarg_setdesc_t * set, int num_sets )
{
	int i;

	for ( i = 0; i < num_sets; i++ ) {
		SUBDBG( "SET[%d]\n", i );
		SUBDBG( "SET[%d].set_id = %d\n", i, set[i].set_id );
		// SUBDBG("SET[%d].set_id_next = %d\n",i,set[i].set_id_next);
		SUBDBG( "SET[%d].set_flags = %d\n", i, set[i].set_flags );
		SUBDBG( "SET[%d].set_timeout = %llu\n", i,
				( unsigned long long ) set[i].set_timeout );
		//  SUBDBG("SET[%d].set_mmap_offset = %#016llx\n",i,(unsigned long long)set[i].set_mmap_offset);
	}
}

static void
dump_setinfo( pfarg_setinfo_t * setinfo, int num_sets )
{
	int i;

	for ( i = 0; i < num_sets; i++ ) {
		SUBDBG( "SETINFO[%d]\n", i );
		SUBDBG( "SETINFO[%d].set_id = %d\n", i, setinfo[i].set_id );
		// SUBDBG("SETINFO[%d].set_id_next = %d\n",i,setinfo[i].set_id_next);
		SUBDBG( "SETINFO[%d].set_flags = %d\n", i, setinfo[i].set_flags );
		SUBDBG( "SETINFO[%d].set_ovfl_pmds[0] = %#016llx\n", i,
				( unsigned long long ) setinfo[i].set_ovfl_pmds[0] );
		SUBDBG( "SETINFO[%d].set_runs = %llu\n", i,
				( unsigned long long ) setinfo[i].set_runs );
		SUBDBG( "SETINFO[%d].set_timeout = %llu\n", i,
				( unsigned long long ) setinfo[i].set_timeout );
		SUBDBG( "SETINFO[%d].set_act_duration = %llu\n", i,
				( unsigned long long ) setinfo[i].set_act_duration );
		// SUBDBG("SETINFO[%d].set_mmap_offset = %#016llx\n",i,(unsigned long long)setinfo[i].set_mmap_offset);
		SUBDBG( "SETINFO[%d].set_avail_pmcs[0] = %#016llx\n", i,
				( unsigned long long ) setinfo[i].set_avail_pmcs[0] );
		SUBDBG( "SETINFO[%d].set_avail_pmds[0] = %#016llx\n", i,
				( unsigned long long ) setinfo[i].set_avail_pmds[0] );
	}
}

static void
dump_pmc( pfm_control_state_t * ctl )
{
	unsigned int i;
	pfarg_pmc_t *pc = ctl->pc;

	for ( i = 0; i < ctl->out.pfp_pmc_count; i++ ) {
		SUBDBG( "PC[%d]\n", i );
		SUBDBG( "PC[%d].reg_num = %d\n", i, pc[i].reg_num );
		SUBDBG( "PC[%d].reg_set = %d\n", i, pc[i].reg_set );
		SUBDBG( "PC[%d].reg_flags = %#08x\n", i, pc[i].reg_flags );
		SUBDBG( "PC[%d].reg_value = %#016llx\n", i,
				( unsigned long long ) pc[i].reg_value );
	}
}

static void
dump_pmd( pfm_control_state_t * ctl )
{
	unsigned int i;
	pfarg_pmd_t *pd = ctl->pd;

	for ( i = 0; i < ctl->in.pfp_event_count; i++ ) {
		SUBDBG( "PD[%d]\n", i );
		SUBDBG( "PD[%d].reg_num = %d\n", i, pd[i].reg_num );
		SUBDBG( "PD[%d].reg_set = %d\n", i, pd[i].reg_set );
		SUBDBG( "PD[%d].reg_flags = %#08x\n", i, pd[i].reg_flags );
		SUBDBG( "PD[%d].reg_value = %#016llx\n", i,
				( unsigned long long ) pd[i].reg_value );
		SUBDBG( "PD[%d].reg_long_reset = %llu\n", i,
				( unsigned long long ) pd[i].reg_long_reset );
		SUBDBG( "PD[%d].reg_short_reset = %llu\n", i,
				( unsigned long long ) pd[i].reg_short_reset );
		SUBDBG( "PD[%d].reg_last_reset_val = %llu\n", i,
				( unsigned long long ) pd[i].reg_last_reset_val );
		SUBDBG( "PD[%d].reg_ovfl_switch_cnt = %llu\n", i,
				( unsigned long long ) pd[i].reg_ovfl_switch_cnt );
		SUBDBG( "PD[%d].reg_reset_pmds[0] = %#016llx\n", i,
				( unsigned long long ) pd[i].reg_reset_pmds[0] );
		SUBDBG( "PD[%d].reg_smpl_pmds[0] = %#016llx\n", i,
				( unsigned long long ) pd[i].reg_smpl_pmds[0] );
		SUBDBG( "PD[%d].reg_smpl_eventid = %llu\n", i,
				( unsigned long long ) pd[i].reg_smpl_eventid );
		SUBDBG( "PD[%d].reg_random_mask = %llu\n", i,
				( unsigned long long ) pd[i].reg_random_mask );
		SUBDBG( "PD[%d].reg_random_seed = %d\n", i, pd[i].reg_random_seed );
	}
}

static void
dump_smpl_hdr( pfm_dfl_smpl_hdr_t * hdr )
{
	SUBDBG( "SMPL_HDR.hdr_count = %llu\n",
			( unsigned long long ) hdr->hdr_count );
	SUBDBG( "SMPL_HDR.hdr_cur_offs = %llu\n",
			( unsigned long long ) hdr->hdr_cur_offs );
	SUBDBG( "SMPL_HDR.hdr_overflows = %llu\n",
			( unsigned long long ) hdr->hdr_overflows );
	SUBDBG( "SMPL_HDR.hdr_buf_size = %llu\n",
			( unsigned long long ) hdr->hdr_buf_size );
	SUBDBG( "SMPL_HDR.hdr_min_buf_space = %llu\n",
			( unsigned long long ) hdr->hdr_min_buf_space );
	SUBDBG( "SMPL_HDR.hdr_version = %d\n", hdr->hdr_version );
	SUBDBG( "SMPL_HDR.hdr_buf_flags = %d\n", hdr->hdr_buf_flags );
}

static void
dump_smpl( pfm_dfl_smpl_entry_t * entry )
{
	SUBDBG( "SMPL.pid = %d\n", entry->pid );
	SUBDBG( "SMPL.ovfl_pmd = %d\n", entry->ovfl_pmd );
	SUBDBG( "SMPL.last_reset_val = %llu\n",
			( unsigned long long ) entry->last_reset_val );
	SUBDBG( "SMPL.ip = %#llx\n", ( unsigned long long ) entry->ip );
	SUBDBG( "SMPL.tstamp = %llu\n", ( unsigned long long ) entry->tstamp );
	SUBDBG( "SMPL.cpu = %d\n", entry->cpu );
	SUBDBG( "SMPL.set = %d\n", entry->set );
	SUBDBG( "SMPL.tgid = %d\n", entry->tgid );
}
#endif

#define PFM_MAX_PMCDS 20

int
_papi_pfm_write_pmcs( pfm_context_t * ctx, pfm_control_state_t * ctl )
{
	( void ) ctx;			 /*unused */
	unsigned int i = 0;
	int ret;

	SUBDBG( "PFM_WRITE_PMCS(%d,%p,%d)\n", ctl->ctx_fd, ctl->pc,
			ctl->out.pfp_pmc_count );
	if ( ctl->out.pfp_pmc_count > PFM_MAX_PMCDS ) {
		for ( i = 0; i < ctl->out.pfp_pmc_count - PFM_MAX_PMCDS;
			  i += PFM_MAX_PMCDS ) {
			if ( ( ret =
				   pfm_write_pmcs( ctl->ctx_fd, ctl->pc + i,
								   PFM_MAX_PMCDS ) ) ) {
				DEBUGCALL( DEBUG_SUBSTRATE, dump_pmc( ctl ) );
				PAPIERROR( "pfm_write_pmcs(%d,%p,%d): %s", ctl->ctx_fd, ctl->pc,
						   ctl->out.pfp_pmc_count, strerror( ret ) );
				return ( PAPI_ESYS );
			}
		}
		DEBUGCALL( DEBUG_SUBSTRATE, dump_pmc( ctl ) );
	}
	if ( ( ret =
		   pfm_write_pmcs( ctl->ctx_fd, ctl->pc + i,
						   ctl->out.pfp_pmc_count - i ) ) ) {
		DEBUGCALL( DEBUG_SUBSTRATE, dump_pmc( ctl ) );
		PAPIERROR( "pfm_write_pmcs(%d,%p,%d): %s", ctl->ctx_fd, ctl->pc,
				   ctl->out.pfp_pmc_count, strerror( ret ) );
		return ( PAPI_ESYS );
	}
	DEBUGCALL( DEBUG_SUBSTRATE, dump_pmc( ctl ) );

	return PAPI_OK;
}

int
_papi_pfm_write_pmds( pfm_context_t * ctx, pfm_control_state_t * ctl )
{
	( void ) ctx;			 /*unused */
	unsigned int i = 0;
	int ret;

	SUBDBG( "PFM_WRITE_PMDS(%d,%p,%d)\n", ctl->ctx_fd, ctl->pd,
			ctl->in.pfp_event_count );
	if ( ctl->in.pfp_event_count > PFM_MAX_PMCDS ) {
		for ( i = 0; i < ctl->in.pfp_event_count - PFM_MAX_PMCDS;
			  i += PFM_MAX_PMCDS ) {
			if ( ( ret =
				   pfm_write_pmds( ctl->ctx_fd, ctl->pd + i,
								   PFM_MAX_PMCDS ) ) ) {
				DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );
				PAPIERROR( "pfm_write_pmds(%d,%p,%d): errno=%d %s", ctl->ctx_fd,
						   ctl->pd, ctl->in.pfp_event_count, errno,
						   strerror( ret ) );
				perror( "pfm_write_pmds" );
				return ( PAPI_ESYS );
			}
		}
		DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );
	}
	if ( ( ret =
		   pfm_write_pmds( ctl->ctx_fd, ctl->pd + i,
						   ctl->in.pfp_event_count - i ) ) ) {
		DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );
		PAPIERROR( "pfm_write_pmds(%d,%p,%d): errno=%d %s", ctl->ctx_fd,
				   ctl->pd, ctl->in.pfp_event_count, errno, strerror( ret ) );
		perror( "pfm_write_pmds" );
		return ( PAPI_ESYS );
	}
	DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );

	return PAPI_OK;
}

int
_papi_pfm_read_pmds( pfm_context_t * ctx, pfm_control_state_t * ctl )
{
	( void ) ctx;			 /*unused */
	unsigned int i = 0;
	int ret;

	SUBDBG( "PFM_READ_PMDS(%d,%p,%d)\n", ctl->ctx_fd, ctl->pd,
			ctl->in.pfp_event_count );
	if ( ctl->in.pfp_event_count > PFM_MAX_PMCDS ) {
		for ( i = 0; i < ctl->in.pfp_event_count - PFM_MAX_PMCDS;
			  i += PFM_MAX_PMCDS ) {
			if ( ( ret =
				   pfm_read_pmds( ctl->ctx_fd, ctl->pd + i,
								  PFM_MAX_PMCDS ) ) ) {
				DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );
				PAPIERROR( "pfm_read_pmds(%d,%p,%d): %s", ctl->ctx_fd, ctl->pd,
						   ctl->in.pfp_event_count, strerror( ret ) );
				return ( ( errno == EBADF ) ? PAPI_ECLOST : PAPI_ESYS );
			}
		}
		DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );
	}
	if ( ( ret =
		   pfm_read_pmds( ctl->ctx_fd, ctl->pd + i,
						  ctl->in.pfp_event_count - i ) ) ) {
		DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );
		PAPIERROR( "pfm_read_pmds(%d,%p,%d): %s", ctl->ctx_fd, ctl->pd,
				   ctl->in.pfp_event_count, strerror( ret ) );
		return ( ( errno == EBADF ) ? PAPI_ECLOST : PAPI_ESYS );
	}
	DEBUGCALL( DEBUG_SUBSTRATE, dump_pmd( ctl ) );

	return PAPI_OK;
}


/* This routine effectively does argument checking as the real magic will happen
   in compute_kernel_args. This just gets the value back from the kernel. */

static int
check_multiplex_timeout( int ctx_fd, unsigned long *timeout_ns )
{
	int ret;
	pfarg_setdesc_t set[2];

	memset( set, 0, sizeof ( pfarg_setdesc_t ) * 2 );
	set[1].set_id = 1;
	set[1].set_flags = PFM_SETFL_TIME_SWITCH;
	set[1].set_timeout = *timeout_ns;
	SUBDBG( "Multiplexing interval requested is %llu ns.\n",
			( unsigned long long ) set[1].set_timeout );

	/* Create a test eventset */

	SUBDBG( "PFM_CREATE_EVTSETS(%d,%p,1)\n", ctx_fd, &set[1] );
	if ( ( ret = pfm_create_evtsets( ctx_fd, &set[1], 1 ) ) != PFMLIB_SUCCESS ) {
		DEBUGCALL( DEBUG_SUBSTRATE, dump_sets( &set[1], 1 ) );
		PAPIERROR( "pfm_create_evtsets(%d,%p,%d): %s", ctx_fd, &set[1], 1,
				   strerror( ret ) );
		return ( PAPI_ESYS );
	}

	SUBDBG( "Multiplexing interval returned is %llu ns.\n",
			( unsigned long long ) set[1].set_timeout );
	*timeout_ns = set[1].set_timeout;

	/* Delete the second eventset */

	pfm_delete_evtsets( ctx_fd, &set[1], 1 );

	return ( PAPI_OK );
}

/* The below function is stolen from libpfm from Stephane Eranian */
static int
detect_timeout_and_unavail_pmu_regs( pfmlib_regmask_t * r_pmcs,
									 pfmlib_regmask_t * r_pmds,
									 unsigned long *timeout_ns )
{
	pfarg_ctx_t ctx;
	pfarg_setinfo_t setf;
	unsigned int i;
	int ret, j, myfd;

	memset( r_pmcs, 0, sizeof ( *r_pmcs ) );
	memset( r_pmds, 0, sizeof ( *r_pmds ) );

	memset( &ctx, 0, sizeof ( ctx ) );
	memset( &setf, 0, sizeof ( setf ) );
	/*
	 * if no context descriptor is passed, then create
	 * a temporary context
	 */
	SUBDBG( "PFM_CREATE_CONTEXT(%p,%p,%p,%d)\n", &ctx, NULL, NULL, 0 );
	myfd = pfm_create_context( &ctx, NULL, NULL, 0 );
	if ( myfd == -1 ) {
		PAPIERROR( "detect_unavail_pmu_regs:pfm_create_context(): %s",
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}
	SUBDBG( "PFM_CREATE_CONTEXT returned fd %d\n", myfd );
	/*
	 * retrieve available register bitmasks from set0
	 * which is guaranteed to exist for every context
	 */
	ret = pfm_getinfo_evtsets( myfd, &setf, 1 );
	if ( ret != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_getinfo_evtsets(): %s", strerror( ret ) );
		return ( PAPI_ESYS );
	}
	DEBUGCALL( DEBUG_SUBSTRATE, dump_setinfo( &setf, 1 ) );
	if ( r_pmcs )
		for ( i = 0; i < PFM_PMC_BV; i++ ) {
			for ( j = 0; j < 64; j++ ) {
				if ( ( setf.set_avail_pmcs[i] & ( 1ULL << j ) ) == 0 )
					pfm_regmask_set( r_pmcs, ( i << 6 ) + j );
			}
		}
	if ( r_pmds )
		for ( i = 0; i < PFM_PMD_BV; i++ ) {
			for ( j = 0; j < 64; j++ ) {
				if ( ( setf.set_avail_pmds[i] & ( 1ULL << j ) ) == 0 )
					pfm_regmask_set( r_pmds, ( i << 6 ) + j );
			}
		}
	check_multiplex_timeout( myfd, timeout_ns );
	i = close( myfd );
	SUBDBG( "CLOSE fd %d returned %d\n", myfd, i );
	return PAPI_OK;
}

/* BEGIN COMMON CODE */

static inline int
compute_kernel_args( hwd_control_state_t * ctl0 )
{
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ctl0;
	pfmlib_input_param_t *inp = &ctl->in;
	pfmlib_output_param_t *outp = &ctl->out;
	pfmlib_input_param_t tmpin;
	pfmlib_output_param_t tmpout;
#if 0
	/* This will be used to fixup the overflow and sample args after re-allocation */
	pfarg_pmd_t oldpd;
#endif
	pfarg_pmd_t *pd = ctl->pd;
	pfarg_pmc_t *pc = ctl->pc;
	pfarg_setdesc_t *sets = ctl->set;
	pfarg_setinfo_t *setinfos = ctl->setinfo;
	int *num_sets = &ctl->num_sets;
	unsigned int set = 0;
	int donepc = 0, donepd = 0, ret, j;
	unsigned int i, dispatch_count = inp->pfp_event_count;
	int togo = inp->pfp_event_count, done = 0;

	/* Save old PD array so we can reconstruct certain flags.  */
        /* This can be removed when we have higher level code call */
        /* set_profile,set_overflow etc when there is hardware     */
        /* (component) support, but this change won't happen for PAPI 3.5 */

	SUBDBG
		( "entry multiplexed %d, pfp_event_count %d, num_cntrs %d, num_sets %d\n",
		  ctl->multiplexed, inp->pfp_event_count, _perfmon2_vector.cmp_info.num_cntrs,
		  *num_sets );
	if ( ( ctl->multiplexed ) &&
		 ( inp->pfp_event_count >
		   ( unsigned int ) _perfmon2_vector.cmp_info.num_cntrs ) ) {
		dispatch_count = _perfmon2_vector.cmp_info.num_cntrs;
	}

	while ( togo ) {
	  again:
		memset( &tmpin, 0x0, sizeof ( tmpin ) );
		memset( &tmpout, 0x0, sizeof ( tmpout ) );

		SUBDBG( "togo %d, done %d, dispatch_count %d, num_cntrs %d\n", togo,
				done, dispatch_count, _perfmon2_vector.cmp_info.num_cntrs );
		tmpin.pfp_event_count = dispatch_count;
		tmpin.pfp_dfl_plm = inp->pfp_dfl_plm;

		/* Make sure we tell dispatch that these PMC's are not available */
		memcpy( &tmpin.pfp_unavail_pmcs, &_perfmon2_pfm_unavailable_pmcs,
				sizeof ( _perfmon2_pfm_unavailable_pmcs ) );

		for ( i = 0, j = done; i < dispatch_count; i++, j++ ) {
			memcpy( tmpin.pfp_events + i, inp->pfp_events + j,
					sizeof ( pfmlib_event_t ) );
		}

		if ( ( ret =
			   pfm_dispatch_events( &tmpin, NULL, &tmpout,
									NULL ) ) != PFMLIB_SUCCESS ) {
			if ( ctl->multiplexed ) {
				dispatch_count--;
				if ( dispatch_count == 0 ) {
					PAPIERROR( "pfm_dispatch_events(): %s",
							   pfm_strerror( ret ) );
					return ( _papi_libpfm_error( ret ) );
				}
				SUBDBG
					( "Dispatch failed because of counter conflict, trying again with %d counters.\n",
					  dispatch_count );
				goto again;
			}
			PAPIERROR( "pfm_dispatch_events(): %s", pfm_strerror( ret ) );
			return ( _papi_libpfm_error( ret ) );
		}

		/*
		 * Now prepare the argument to initialize the PMDs and PMCS.
		 * We must pfp_pmc_count to determine the number of PMC to intialize.
		 * We must use pfp_event_count to determine the number of PMD to initialize.
		 * Some events causes extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
		 *
		 * This step is new compared to libpfm-2.x. It is necessary because the library no
		 * longer knows about the kernel data structures.
		 */

		for ( i = 0; i < tmpout.pfp_pmc_count; i++, donepc++ ) {
			pc[donepc].reg_num = tmpout.pfp_pmcs[i].reg_num;
			pc[donepc].reg_value = tmpout.pfp_pmcs[i].reg_value;
			pc[donepc].reg_set = set;
			SUBDBG( "PC%d (i%d) is reg num %d, value %llx, set %d\n", donepc, i,
					pc[donepc].reg_num,
					( unsigned long long ) pc[donepc].reg_value,
					pc[donepc].reg_set );
		}

		/* figure out pmd mapping from output pmc */

#if defined(HAVE_PFM_REG_EVT_IDX)
		for ( i = 0, j = 0; i < tmpin.pfp_event_count; i++, donepd++ ) {
			pd[donepd].reg_num = tmpout.pfp_pmcs[j].reg_pmd_num;
			pd[donepd].reg_set = set;
			SUBDBG( "PD%d (i%d,j%d) is reg num %d, set %d\n", donepd, i, j,
					pd[donepd].reg_num, pd[donepd].reg_set );

			/* Skip over entries that map to the same PMD, 
			   PIV has 2 PMCS for every PMD */

			for ( ; j < tmpout.pfp_pmc_count; j++ )
				if ( tmpout.pfp_pmcs[j].reg_evt_idx != i )
					break;
		}
#else
		for ( i = 0; i < tmpout.pfp_pmd_count; i++, donepd++ ) {
			pd[donepd].reg_num = tmpout.pfp_pmds[i].reg_num;
			pd[donepd].reg_set = set;
			SUBDBG( "PD%d (i%d) is reg num %d, set %d\n", donepd, i,
					pd[donepd].reg_num, pd[donepd].reg_set );
		}
#endif

		togo -= dispatch_count;
		done += dispatch_count;
		if ( togo > _perfmon2_vector.cmp_info.num_cntrs )
			dispatch_count = _perfmon2_vector.cmp_info.num_cntrs;
		else
			dispatch_count = togo;

		setinfos[set].set_id = set;
		sets[set].set_id = set;
		set++;
	}

	*num_sets = set;
	outp->pfp_pmc_count = donepc;

	if ( ctl->multiplexed && ( set > 1 ) ) {
		for ( i = 0; i < set; i++ ) {
			sets[i].set_flags = PFM_SETFL_TIME_SWITCH;
			sets[i].set_timeout = ctl->multiplexed;
		}
	}
	SUBDBG
		( "exit multiplexed %d (ns switch time), pfp_pmc_count %d, num_sets %d\n",
		  ctl->multiplexed, outp->pfp_pmc_count, *num_sets );
	return ( PAPI_OK );
}

int
tune_up_fd( int ctx_fd )
{
	int ret;

	/* set close-on-exec to ensure we will be getting the PFM_END_MSG, i.e.,
	 * fd not visible to child. */
	ret = fcntl( ctx_fd, F_SETFD, FD_CLOEXEC );
	if ( ret == -1 ) {
		PAPIERROR( "cannot fcntl(FD_CLOEXEC) on %d: %s", ctx_fd,
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}
	/* setup asynchronous notification on the file descriptor */
	ret = fcntl( ctx_fd, F_SETFL, fcntl( ctx_fd, F_GETFL, 0 ) | O_ASYNC );
	if ( ret == -1 ) {
		PAPIERROR( "cannot fcntl(O_ASYNC) on %d: %s", ctx_fd,
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}
	/* get ownership of the descriptor */
	ret = fcntl( ctx_fd, F_SETOWN, mygettid(  ) );
	if ( ret == -1 ) {
		PAPIERROR( "cannot fcntl(F_SETOWN) on %d: %s", ctx_fd,
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}
	/*
	 * when you explicitely declare that you want a particular signal,
	 * even with you use the default signal, the kernel will send more
	 * information concerning the event to the signal handler.
	 *
	 * In particular, it will send the file descriptor from which the
	 * event is originating which can be quite useful when monitoring
	 * multiple tasks from a single thread.
	 */
	ret = fcntl( ctx_fd, F_SETSIG, _perfmon2_vector.cmp_info.hardware_intr_sig );
	if ( ret == -1 ) {
		PAPIERROR( "cannot fcntl(F_SETSIG,%d) on %d: %s",
				   _perfmon2_vector.cmp_info.hardware_intr_sig, ctx_fd,
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}
	return ( PAPI_OK );
}

static int
attach( hwd_control_state_t * ctl, unsigned long tid )
{
	pfarg_ctx_t *newctx = ( pfarg_ctx_t * ) malloc( sizeof ( pfarg_ctx_t ) );
	pfarg_load_t *load_args =
		( pfarg_load_t * ) malloc( sizeof ( pfarg_load_t ) );
	int ret;

	if ( ( newctx == NULL ) || ( load_args == NULL ) )
		return ( PAPI_ENOMEM );
	memset( newctx, 0x0, sizeof ( *newctx ) );
	memset( load_args, 0, sizeof ( *load_args ) );

	/* Make sure the process exists and is being ptraced() */

	ret = ptrace( PTRACE_ATTACH, tid, NULL, NULL );
	if ( ret == 0 ) {
		ptrace( PTRACE_DETACH, tid, NULL, NULL );
		PAPIERROR( "Process/thread %d is not being ptraced", tid );
		free( newctx );
		free( load_args );
		return ( PAPI_EINVAL );
	}
	/* If we get here, then we should hope that the process is being
	   ptraced, if not, then we probably can't attach to it. */

	if ( ( ret == -1 ) && ( errno != EPERM ) ) {
		PAPIERROR( "Process/thread %d cannot be ptraced: %s", tid,
				   strerror( errno ) );
		free( newctx );
		free( load_args );
		return ( PAPI_EINVAL );
	}

	SUBDBG( "PFM_CREATE_CONTEXT(%p,%p,%p,%d)\n", newctx, NULL, NULL, 0 );
	if ( ( ret = pfm_create_context( newctx, NULL, NULL, 0 ) ) == -1 ) {
		PAPIERROR( "attach:pfm_create_context(): %s", strerror( errno ) );
		free( newctx );
		free( load_args );
		return ( PAPI_ESYS );
	}
	SUBDBG( "PFM_CREATE_CONTEXT returned fd %d\n", ret );
	tune_up_fd( ret );

	( ( pfm_control_state_t * ) ctl )->ctx_fd = ret;
	( ( pfm_control_state_t * ) ctl )->ctx = newctx;
	load_args->load_pid = tid;
	( ( pfm_control_state_t * ) ctl )->load = load_args;

	return ( PAPI_OK );
}

static int
detach( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	int i;

	i = close( ( ( pfm_control_state_t * ) ctl )->ctx_fd );
	SUBDBG( "CLOSE fd %d returned %d\n",
			( ( pfm_control_state_t * ) ctl )->ctx_fd, i );
	(void) i;

	/* Restore to main threads context */
	free( ( ( pfm_control_state_t * ) ctl )->ctx );
	( ( pfm_control_state_t * ) ctl )->ctx = &( ( pfm_context_t * ) ctx )->ctx;
	( ( pfm_control_state_t * ) ctl )->ctx_fd =
		( ( pfm_context_t * ) ctx )->ctx_fd;
	free( ( ( pfm_control_state_t * ) ctl )->load );
	( ( pfm_control_state_t * ) ctl )->load =
		&( ( pfm_context_t * ) ctx )->load;

	return ( PAPI_OK );
}

static inline int
set_domain( hwd_control_state_t * ctl0, int domain )
{
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ctl0;
	int mode = 0, did = 0;
	pfmlib_input_param_t *inp = &ctl->in;

	if ( domain & PAPI_DOM_USER ) {
		did = 1;
		mode |= PFM_PLM3;
	}

	if ( domain & PAPI_DOM_KERNEL ) {
		did = 1;
		mode |= PFM_PLM0;
	}

	if ( domain & PAPI_DOM_SUPERVISOR ) {
		did = 1;
		mode |= PFM_PLM1;
	}

	if ( domain & PAPI_DOM_OTHER ) {
		did = 1;
		mode |= PFM_PLM2;
	}

	if ( !did )
		return ( PAPI_EINVAL );

	inp->pfp_dfl_plm = mode;

	return ( compute_kernel_args( ctl ) );
}

static inline int
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

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static inline int
set_inherit( int arg )
{
	( void ) arg;			 /*unused */
	return PAPI_ECMP;
}

static int
get_string_from_file( char *file, char *str, int len )
{
	FILE *f = fopen( file, "r" );
	char buf[PAPI_HUGE_STR_LEN];
	if ( f == NULL ) {
		PAPIERROR( "fopen(%s): %s", file, strerror( errno ) );
		return ( PAPI_ESYS );
	}
	if ( fscanf( f, "%s\n", buf ) != 1 ) {
		PAPIERROR( "fscanf(%s, %%s\\n): Unable to scan 1 token", file );
		fclose( f );
		return PAPI_ESYS;
	}
	strncpy( str, buf, ( len > PAPI_HUGE_STR_LEN ? PAPI_HUGE_STR_LEN : len ) );
	fclose( f );
	return ( PAPI_OK );
}

int
_papi_pfm_init_component( int cidx )
{
   int retval;
   char buf[PAPI_HUGE_STR_LEN];

   /* The following checks the PFMLIB version 
      against the perfmon2 kernel version... */
   strncpy( _perfmon2_vector.cmp_info.support_version, buf,
	    sizeof ( _perfmon2_vector.cmp_info.support_version ) );

   retval = get_string_from_file( "/sys/kernel/perfmon/version",
			          _perfmon2_vector.cmp_info.kernel_version,
			sizeof ( _perfmon2_vector.cmp_info.kernel_version ) );
   if ( retval != PAPI_OK ) {
      strncpy(_perfmon2_vector.cmp_info.disabled_reason,
	     "/sys/kernel/perfmon/version not found",PAPI_MAX_STR_LEN);
      goto fn_fail;
   }

#ifdef PFM_VERSION
   sprintf( buf, "%d.%d", PFM_VERSION_MAJOR( PFM_VERSION ),
			  PFM_VERSION_MINOR( PFM_VERSION ) );
   SUBDBG( "Perfmon2 library versions...kernel: %s library: %s\n",
			_perfmon2_vector.cmp_info.kernel_version, buf );
   if ( strcmp( _perfmon2_vector.cmp_info.kernel_version, buf ) != 0 ) {
      /* do a little exception processing; 81 is compatible with 80 */
      if ( !( ( PFM_VERSION_MINOR( PFM_VERSION ) == 81 ) && 
            ( strncmp( _perfmon2_vector.cmp_info.kernel_version, "2.8", 3 ) ==
					 0 ) ) ) {
	 PAPIERROR( "Version mismatch of libpfm: compiled %s "
                    "vs. installed %s\n",
		    buf, _perfmon2_vector.cmp_info.kernel_version );
     retval = PAPI_ESYS;
     goto fn_fail;
      }
   }
#endif

   _perfmon2_vector.cmp_info.hardware_intr_sig = SIGRTMIN + 2,


   /* Run the libpfm-specific setup */
   retval=_papi_libpfm_init(&_perfmon2_vector, cidx);
   if (retval) goto fn_fail;

   /* Load the module, find out if any PMC's/PMD's are off limits */

   /* Perfmon2 timeouts are based on the clock tick, we need to check
      them otherwise it will complain at us when we multiplex */

   unsigned long min_timeout_ns;

   struct timespec ts;

   if ( syscall( __NR_clock_getres, CLOCK_REALTIME, &ts ) == -1 ) {
      PAPIERROR( "Could not detect proper HZ rate, multiplexing may fail\n" );
      min_timeout_ns = 10000000;
   } else {
      min_timeout_ns = ts.tv_nsec;
   }

   /* This will fail if we've done timeout detection wrong */
   retval=detect_timeout_and_unavail_pmu_regs( &_perfmon2_pfm_unavailable_pmcs,
					       &_perfmon2_pfm_unavailable_pmds,
					       &min_timeout_ns );
   if ( retval != PAPI_OK ) {
       goto fn_fail;
   }	

   if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_IBM ) {
      /* powerpc */
      _perfmon2_vector.cmp_info.available_domains |= PAPI_DOM_KERNEL | 
                                                     PAPI_DOM_SUPERVISOR;
      if (strcmp(_papi_hwi_system_info.hw_info.model_string, "POWER6" ) == 0) {
	 _perfmon2_vector.cmp_info.default_domain = PAPI_DOM_USER | 
	                                            PAPI_DOM_KERNEL | 
	                                            PAPI_DOM_SUPERVISOR;
      }
   } else {
      _perfmon2_vector.cmp_info.available_domains |= PAPI_DOM_KERNEL;
   }

   if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_SUN ) {
      switch ( _perfmon2_pfm_pmu_type ) {
#ifdef PFMLIB_SPARC_ULTRA12_PMU
	     case PFMLIB_SPARC_ULTRA12_PMU:
	     case PFMLIB_SPARC_ULTRA3_PMU:
	     case PFMLIB_SPARC_ULTRA3I_PMU:
	     case PFMLIB_SPARC_ULTRA3PLUS_PMU:
	     case PFMLIB_SPARC_ULTRA4PLUS_PMU:
		  break;
#endif
	     default:
		   _perfmon2_vector.cmp_info.available_domains |= 
                                                      PAPI_DOM_SUPERVISOR;
		   break;
      }
   }

   if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_CRAY ) {
      _perfmon2_vector.cmp_info.available_domains |= PAPI_DOM_OTHER;
   }

   if ( ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL ) ||
	( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD ) ) {
      _perfmon2_vector.cmp_info.fast_counter_read = 1;
      _perfmon2_vector.cmp_info.fast_real_timer = 1;
      _perfmon2_vector.cmp_info.cntr_umasks = 1;
   }

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}

int
_papi_pfm_shutdown_component(  )
{
	return PAPI_OK;
}

static int
_papi_pfm_init_thread( hwd_context_t * thr_ctx )
{
	pfarg_load_t load_args;
	pfarg_ctx_t newctx;
	int ret, ctx_fd;

#if defined(USE_PROC_PTTIMER)
	ret = init_proc_thread_timer( thr_ctx );
	if ( ret != PAPI_OK )
		return ( ret );
#endif

	memset( &newctx, 0, sizeof ( newctx ) );
	memset( &load_args, 0, sizeof ( load_args ) );

	if ( ( ret = pfm_create_context( &newctx, NULL, NULL, 0 ) ) == -1 ) {
		PAPIERROR( "pfm_create_context(): %s",
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}
	SUBDBG( "PFM_CREATE_CONTEXT returned fd %d\n", ret );
	tune_up_fd( ret );
	ctx_fd = ret;

	memcpy( &( ( pfm_context_t * ) thr_ctx )->ctx, &newctx, sizeof ( newctx ) );
	( ( pfm_context_t * ) thr_ctx )->ctx_fd = ctx_fd;
	load_args.load_pid = mygettid(  );
	memcpy( &( ( pfm_context_t * ) thr_ctx )->load, &load_args,
			sizeof ( load_args ) );

	return ( PAPI_OK );
}

/* reset the hardware counters */
int
_papi_pfm_reset( hwd_context_t * ctx, hwd_control_state_t * ctl )
{
	unsigned int i;
	int ret;

	/* Read could have clobbered the values */
	for ( i = 0; i < ( ( pfm_control_state_t * ) ctl )->in.pfp_event_count;
		  i++ ) {
		if ( ( ( pfm_control_state_t * ) ctl )->pd[i].
			 reg_flags & PFM_REGFL_OVFL_NOTIFY )
			( ( pfm_control_state_t * ) ctl )->pd[i].reg_value =
				( ( pfm_control_state_t * ) ctl )->pd[i].reg_long_reset;
		else
			( ( pfm_control_state_t * ) ctl )->pd[i].reg_value = 0ULL;
	}

	ret =
		_papi_pfm_write_pmds( ( pfm_context_t * ) ctx,
							  ( pfm_control_state_t * ) ctl );
	if ( ret != PAPI_OK )
		return PAPI_ESYS;

	return ( PAPI_OK );
}

/* write(set) the hardware counters */
int
_papi_pfm_write( hwd_context_t * ctx, hwd_control_state_t * ctl,
				 long long *from )
{
	unsigned int i;
	int ret;

	/* Read could have clobbered the values */
	for ( i = 0; i < ( ( pfm_control_state_t * ) ctl )->in.pfp_event_count;
		  i++ ) {
		if ( ( ( pfm_control_state_t * ) ctl )->pd[i].
			 reg_flags & PFM_REGFL_OVFL_NOTIFY )
			( ( pfm_control_state_t * ) ctl )->pd[i].reg_value =
				from[i] +
				( ( pfm_control_state_t * ) ctl )->pd[i].reg_long_reset;
		else
			( ( pfm_control_state_t * ) ctl )->pd[i].reg_value = from[i];
	}

	ret =
		_papi_pfm_write_pmds( ( pfm_context_t * ) ctx,
							  ( pfm_control_state_t * ) ctl );
	if ( ret != PAPI_OK )
		return PAPI_ESYS;


	return ( PAPI_OK );
}

int
_papi_pfm_read( hwd_context_t * ctx0, hwd_control_state_t * ctl0,
				long long **events, int flags )
{
	( void ) flags;			 /*unused */
	unsigned int i;
	int ret;
	long long tot_runs = 0LL;
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ctl0;
	pfm_context_t *ctx = ( pfm_context_t * ) ctx0;

	ret = _papi_pfm_read_pmds( ctx, ctl );
	if ( ret != PAPI_OK )
		return PAPI_ESYS;

	/* Copy the values over */

	for ( i = 0; i < ctl->in.pfp_event_count; i++ ) {
		if ( ctl->pd[i].reg_flags & PFM_REGFL_OVFL_NOTIFY )
			ctl->counts[i] = ctl->pd[i].reg_value - ctl->pd[i].reg_long_reset;
		else
			ctl->counts[i] = ctl->pd[i].reg_value;
		SUBDBG( "PMD[%d] = %lld (LLD),%llu (LLU)\n", i,
				( unsigned long long ) ctl->counts[i],
				( unsigned long long ) ctl->pd[i].reg_value );
	}
	*events = ctl->counts;

	/* If we're not multiplexing, bail now */

	if ( ctl->num_sets == 1 )
		return ( PAPI_OK );

	/* If we're multiplexing, get the scaling information */

	SUBDBG( "PFM_GETINFO_EVTSETS(%d,%p,%d)\n", ctl->ctx_fd, ctl->setinfo,
			ctl->num_sets );
	if ( ( ret =
		   pfm_getinfo_evtsets( ctl->ctx_fd, ctl->setinfo, ctl->num_sets ) ) ) {
		DEBUGCALL( DEBUG_SUBSTRATE,
				   dump_setinfo( ctl->setinfo, ctl->num_sets ) );
		PAPIERROR( "pfm_getinfo_evtsets(%d,%p,%d): %s", ctl->ctx_fd,
				   ctl->setinfo, ctl->num_sets, strerror( ret ) );
		*events = NULL;
		return ( PAPI_ESYS );
	}
	DEBUGCALL( DEBUG_SUBSTRATE, dump_setinfo( ctl->setinfo, ctl->num_sets ) );

	/* Add up the number of total runs */

	for ( i = 0; i < ( unsigned int ) ctl->num_sets; i++ )
		tot_runs += ctl->setinfo[i].set_runs;

	/* Now scale the values */

	for ( i = 0; i < ctl->in.pfp_event_count; i++ ) {
		SUBDBG
			( "Counter %d is in set %d ran %llu of %llu times, old count %lld.\n",
			  i, ctl->pd[i].reg_set,
			  ( unsigned long long ) ctl->setinfo[ctl->pd[i].reg_set].set_runs,
			  ( unsigned long long ) tot_runs, ctl->counts[i] );
		if ( ctl->setinfo[ctl->pd[i].reg_set].set_runs )
			ctl->counts[i] =
				( ctl->counts[i] * tot_runs ) /
				ctl->setinfo[ctl->pd[i].reg_set].set_runs;
		else {
			ctl->counts[i] = 0;
			SUBDBG( "Set %lld didn't run!!!!\n",
					( unsigned long long ) ctl->pd[i].reg_set );
		}
		SUBDBG( "Counter %d, new count %lld.\n", i, ctl->counts[i] );
	}

	return PAPI_OK;
}

#if defined(__crayxt)
int _papi_hwd_start_create_context = 0;	/* CrayPat checkpoint support */
#endif /* XT */

int
_papi_pfm_start( hwd_context_t * ctx0, hwd_control_state_t * ctl0 )
{
	unsigned int i;
	int ret;
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ctl0;
	pfm_context_t *ctx = ( pfm_context_t * ) ctx0;

#if defined(__crayxt)
	if ( _papi_hwd_start_create_context ) {
		pfarg_ctx_t tmp;

		memset( &tmp, 0, sizeof ( tmp ) );
		if ( ( ret = pfm_create_context( &tmp, NULL, NULL, 0 ) ) == -1 ) {
			PAPIERROR( "_papi_hwd_init:pfm_create_context(): %s",
					   strerror( errno ) );
			return ( PAPI_ESYS );
		}
		tune_up_fd( ret );
		ctl->ctx_fd = ctx->ctx_fd = ret;
	}
#endif /* XT */

	if ( ctl->num_sets > 1 ) {
		SUBDBG( "PFM_CREATE_EVTSETS(%d,%p,%d)\n", ctl->ctx_fd, ctl->set,
				ctl->num_sets );
		if ( ( ret =
			   pfm_create_evtsets( ctl->ctx_fd, ctl->set,
								   ctl->num_sets ) ) != PFMLIB_SUCCESS ) {
			DEBUGCALL( DEBUG_SUBSTRATE, dump_sets( ctl->set, ctl->num_sets ) );
			PAPIERROR( "pfm_create_evtsets(%d,%p,%d): errno=%d  %s",
					   ctl->ctx_fd, ctl->set, ctl->num_sets, errno,
					   strerror( ret ) );
			perror( "pfm_create_evtsets" );
			return ( PAPI_ESYS );
		}
		DEBUGCALL( DEBUG_SUBSTRATE, dump_sets( ctl->set, ctl->num_sets ) );
	}

	/*
	 * Now program the registers
	 *
	 * We don't use the same variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events (pmd) we specified, i.e., contains more than counting
	 * monitors.
	 */

	ret = _papi_pfm_write_pmcs( ctx, ctl );
	if ( ret != PAPI_OK )
		return PAPI_ESYS;

	/* Set counters to zero as per PAPI_start man page, unless it is set to overflow */

	for ( i = 0; i < ctl->in.pfp_event_count; i++ )
		if ( !( ctl->pd[i].reg_flags & PFM_REGFL_OVFL_NOTIFY ) )
			ctl->pd[i].reg_value = 0ULL;

	/*
	 * To be read, each PMD must be either written or declared
	 * as being part of a sample (reg_smpl_pmds)
	 */

	ret = _papi_pfm_write_pmds( ctx, ctl );
	if ( ret != PAPI_OK )
		return PAPI_ESYS;

	SUBDBG( "PFM_LOAD_CONTEXT(%d,%p(%u))\n", ctl->ctx_fd, ctl->load,
			ctl->load->load_pid );
	if ( ( ret = pfm_load_context( ctl->ctx_fd, ctl->load ) ) ) {
		PAPIERROR( "pfm_load_context(%d,%p(%u)): %s", ctl->ctx_fd, ctl->load,
				   ctl->load->load_pid, strerror( ret ) );
		return PAPI_ESYS;
	}

	SUBDBG( "PFM_START(%d,%p)\n", ctl->ctx_fd, NULL );
	if ( ( ret = pfm_start( ctl->ctx_fd, NULL ) ) ) {
		PAPIERROR( "pfm_start(%d): %s", ctl->ctx_fd, strerror( ret ) );
		return ( PAPI_ESYS );
	}
	return PAPI_OK;
}

int
_papi_pfm_stop( hwd_context_t * ctx0, hwd_control_state_t * ctl0 )
{
	( void ) ctx0;			 /*unused */
	int ret;
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ctl0;
//  pfm_context_t *ctx = (pfm_context_t *)ctx0;

	SUBDBG( "PFM_STOP(%d)\n", ctl->ctx_fd );
	if ( ( ret = pfm_stop( ctl->ctx_fd ) ) ) {
		/* If this thread is attached to another thread, and that thread
		   has exited, we can safely discard the error here. */

		if ( ( ret == PFMLIB_ERR_NOTSUPP ) &&
			 ( ctl->load->load_pid != ( unsigned int ) mygettid(  ) ) )
			return ( PAPI_OK );

		PAPIERROR( "pfm_stop(%d): %s", ctl->ctx_fd, strerror( ret ) );
		return ( PAPI_ESYS );
	}

	SUBDBG( "PFM_UNLOAD_CONTEXT(%d) (tid %u)\n", ctl->ctx_fd,
			ctl->load->load_pid );
	if ( ( ret = pfm_unload_context( ctl->ctx_fd ) ) ) {
		PAPIERROR( "pfm_unload_context(%d): %s", ctl->ctx_fd, strerror( ret ) );
		return PAPI_ESYS;
	}

	if ( ctl->num_sets > 1 ) {
		static pfarg_setdesc_t set = { 0, 0, 0, 0, {0, 0, 0, 0, 0, 0} };
		/* Delete the high sets */
		SUBDBG( "PFM_DELETE_EVTSETS(%d,%p,%d)\n", ctl->ctx_fd, &ctl->set[1],
				ctl->num_sets - 1 );
		if ( ( ret =
			   pfm_delete_evtsets( ctl->ctx_fd, &ctl->set[1],
								   ctl->num_sets - 1 ) ) != PFMLIB_SUCCESS ) {
			DEBUGCALL( DEBUG_SUBSTRATE,
					   dump_sets( &ctl->set[1], ctl->num_sets - 1 ) );
			PAPIERROR( "pfm_delete_evtsets(%d,%p,%d): %s", ctl->ctx_fd,
					   &ctl->set[1], ctl->num_sets - 1, strerror( ret ) );
			return ( PAPI_ESYS );
		}
		DEBUGCALL( DEBUG_SUBSTRATE,
				   dump_sets( &ctl->set[1], ctl->num_sets - 1 ) );
		/* Reprogram the 0 set */
		SUBDBG( "PFM_CREATE_EVTSETS(%d,%p,%d)\n", ctl->ctx_fd, &set, 1 );
		if ( ( ret =
			   pfm_create_evtsets( ctl->ctx_fd, &set,
								   1 ) ) != PFMLIB_SUCCESS ) {
			DEBUGCALL( DEBUG_SUBSTRATE, dump_sets( &set, 1 ) );
			PAPIERROR( "pfm_create_evtsets(%d,%p,%d): %s", ctl->ctx_fd, &set,
					   ctl->num_sets, strerror( ret ) );
			return ( PAPI_ESYS );
		}
		DEBUGCALL( DEBUG_SUBSTRATE, dump_sets( &set, 1 ) );
	}

	return PAPI_OK;
}

static inline int
round_requested_ns( int ns )
{
	if ( ns <= _papi_os_info.itimer_res_ns ) {
		return _papi_os_info.itimer_res_ns;
	} else {
		int leftover_ns = ns % _papi_os_info.itimer_res_ns;
		return ( ns - leftover_ns + _papi_os_info.itimer_res_ns );
	}
}

int
_papi_pfm_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	switch ( code ) {
	case PAPI_MULTIPLEX:
	{
		option->multiplex.ns = round_requested_ns( option->multiplex.ns );
		( ( pfm_control_state_t * ) ( option->multiplex.ESI->ctl_state ) )->
			multiplexed = option->multiplex.ns;
		return ( PAPI_OK );
	}

	case PAPI_ATTACH:
		return ( attach
				 ( ( pfm_control_state_t * ) ( option->attach.ESI->ctl_state ),
				   option->attach.tid ) );
	case PAPI_DETACH:
		return ( detach
				 ( ctx,
				   ( pfm_control_state_t * ) ( option->attach.ESI->
											   ctl_state ) ) );

	case PAPI_DOMAIN:
		return ( set_domain
				 ( ( pfm_control_state_t * ) ( option->domain.ESI->ctl_state ),
				   option->domain.domain ) );
	case PAPI_GRANUL:
		return ( set_granularity
				 ( ( pfm_control_state_t * ) ( option->granularity.ESI->
											   ctl_state ),
				   option->granularity.granularity ) );
#if 0
	case PAPI_DATA_ADDRESS:
		ret =
			set_default_domain( ( pfm_control_state_t * ) ( option->
															address_range.ESI->
															ctl_state ),
								option->address_range.domain );
		if ( ret != PAPI_OK )
			return ( ret );
		set_drange( ctx,
					( pfm_control_state_t * ) ( option->address_range.ESI->
												ctl_state ), option );
		return ( PAPI_OK );
	case PAPI_INSTR_ADDRESS:
		ret =
			set_default_domain( ( pfm_control_state_t * ) ( option->
															address_range.ESI->
															ctl_state ),
								option->address_range.domain );
		if ( ret != PAPI_OK )
			return ( ret );
		set_irange( ctx,
					( pfm_control_state_t * ) ( option->address_range.ESI->
												ctl_state ), option );
		return ( PAPI_OK );
#endif


	case PAPI_DEF_ITIMER:
	{
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

	case PAPI_DEF_MPX_NS:
	{
		option->multiplex.ns = round_requested_ns( option->multiplex.ns );
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER_NS:
	{
		option->itimer.ns = round_requested_ns( option->itimer.ns );
		return ( PAPI_OK );
	}
	default:
		return ( PAPI_ENOSUPP );
	}
}

int
_papi_pfm_shutdown( hwd_context_t * ctx0 )
{
	pfm_context_t *ctx = ( pfm_context_t * ) ctx0;
	int ret;
#if defined(USE_PROC_PTTIMER)
	close( ctx->stat_fd );
#endif


	ret = close( ctx->ctx_fd );
	SUBDBG( "CLOSE fd %d returned %d\n", ctx->ctx_fd, ret );
	(void) ret;

	return ( PAPI_OK );
}

/* This will need to be modified for the Pentium IV */

static inline int
find_profile_index( EventSetInfo_t * ESI, int pmd, int *flags,
					unsigned int *native_index, int *profile_index )
{
	int pos, esi_index, count;
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ESI->ctl_state;
	pfarg_pmd_t *pd;
	unsigned int i;

	pd = ctl->pd;

	/* Find virtual PMD index, the one we actually read from the physical PMD number that
	   overflowed. This index is the one related to the profile buffer. */

	for ( i = 0; i < ctl->in.pfp_event_count; i++ ) {
		if ( pd[i].reg_num == pmd ) {
			SUBDBG( "Physical PMD %d is Virtual PMD %d\n", pmd, i );
			pmd = i;
			break;
		}
	}


	SUBDBG( "(%p,%d,%p)\n", ESI, pmd, index );

	for ( count = 0; count < ESI->profile.event_counter; count++ ) {
		/* Find offset of PMD that gets read from the kernel */
		esi_index = ESI->profile.EventIndex[count];
		pos = ESI->EventInfoArray[esi_index].pos[0];
		SUBDBG( "Examining event at ESI index %d, PMD position %d\n", esi_index,
				pos );
		// PMU_FIRST_COUNTER
		if ( pos == pmd ) {
			*profile_index = count;
			*native_index =
				ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;
			*flags = ESI->profile.flags;
			SUBDBG( "Native event %d is at profile index %d, flags %d\n",
					*native_index, *profile_index, *flags );
			return ( PAPI_OK );
		}
	}

	PAPIERROR( "wrong count: %d vs. ESI->profile.event_counter %d", count,
			   ESI->profile.event_counter );
	return ( PAPI_EBUG );
}

#if defined(__ia64__)
static inline int
is_montecito_and_dear( unsigned int native_index )
{
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU ) {
		if ( pfm_mont_is_dear( native_index ) )
			return ( 1 );
	}
	return ( 0 );
}
static inline int
is_montecito_and_iear( unsigned int native_index )
{
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU ) {
		if ( pfm_mont_is_iear( native_index ) )
			return ( 1 );
	}
	return ( 0 );
}
static inline int
is_itanium2_and_dear( unsigned int native_index )
{
	if ( _perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU ) {
		if ( pfm_ita2_is_dear( native_index ) )
			return ( 1 );
	}
	return ( 0 );
}
static inline int
is_itanium2_and_iear( unsigned int native_index )
{
	if ( _perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU ) {
		if ( pfm_ita2_is_iear( native_index ) )
			return ( 1 );
	}
	return ( 0 );
}
#endif

#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6
static inline void
pfm_bv_set( uint64_t * bv, uint16_t rnum )
{
	bv[rnum >> LBPL] |= 1UL << ( rnum & ( BPL - 1 ) );
}

static inline int
setup_ear_event( unsigned int native_index, pfarg_pmd_t * pd, int flags )
{
	( void ) flags;			 /*unused */
#if defined(__ia64__)
	if ( _perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU ) {
		if ( pfm_mont_is_dear( native_index ) ) {	/* 2,3,17 */
			pfm_bv_set( pd[0].reg_smpl_pmds, 32 );
			pfm_bv_set( pd[0].reg_smpl_pmds, 33 );
			pfm_bv_set( pd[0].reg_smpl_pmds, 36 );
			pfm_bv_set( pd[0].reg_reset_pmds, 36 );
			return ( 1 );
		} else if ( pfm_mont_is_iear( native_index ) ) {	/* O,1 MK */
			pfm_bv_set( pd[0].reg_smpl_pmds, 34 );
			pfm_bv_set( pd[0].reg_smpl_pmds, 35 );
			pfm_bv_set( pd[0].reg_reset_pmds, 34 );
			return ( 1 );
		}
		return ( 0 );
	} else if ( _perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU ) {
		if ( pfm_mont_is_dear( native_index ) ) {	/* 2,3,17 */
			pfm_bv_set( pd[0].reg_smpl_pmds, 2 );
			pfm_bv_set( pd[0].reg_smpl_pmds, 3 );
			pfm_bv_set( pd[0].reg_smpl_pmds, 17 );
			pfm_bv_set( pd[0].reg_reset_pmds, 17 );
			return ( 1 );
		} else if ( pfm_mont_is_iear( native_index ) ) {	/* O,1 MK */
			pfm_bv_set( pd[0].reg_smpl_pmds, 0 );
			pfm_bv_set( pd[0].reg_smpl_pmds, 1 );
			pfm_bv_set( pd[0].reg_reset_pmds, 0 );
			return ( 1 );
		}
		return ( 0 );
	}
#else
	( void ) native_index;	 /*unused */
	( void ) pd;			 /*unused */
#endif
	return ( 0 );
}

static inline int
process_smpl_entry( unsigned int native_pfm_index, int flags,
					pfm_dfl_smpl_entry_t ** ent, vptr_t * pc )
{
#ifndef __ia64__
	( void ) native_pfm_index;	/*unused */
	( void ) flags;			 /*unused */
#endif
	SUBDBG( "process_smpl_entry(%d,%d,%p,%p)\n", native_pfm_index, flags, ent,
			pc );

#ifdef __ia64__
	/* Fixup EAR stuff here */
	if ( is_montecito_and_dear( native_pfm_index ) ) {
		pfm_mont_pmd_reg_t data_addr;
		pfm_mont_pmd_reg_t latency;
		pfm_mont_pmd_reg_t load_addr;
		unsigned long newent;

		if ( ( flags & ( PAPI_PROFIL_DATA_EAR | PAPI_PROFIL_INST_EAR ) ) == 0 )
			goto safety;

		/* Skip the header */
		++( *ent );

		// PMD32 has data address on Montecito
		// PMD33 has latency on Montecito
		// PMD36 has instruction address on Montecito
		data_addr = *( pfm_mont_pmd_reg_t * ) * ent;
		latency =
			*( pfm_mont_pmd_reg_t * ) ( ( unsigned long ) *ent +
										sizeof ( data_addr ) );
		load_addr =
			*( pfm_mont_pmd_reg_t * ) ( ( unsigned long ) *ent +
										sizeof ( data_addr ) +
										sizeof ( latency ) );

		SUBDBG( "PMD[32]: %#016llx\n",
				( unsigned long long ) data_addr.pmd_val );
		SUBDBG( "PMD[33]: %#016llx\n",
				( unsigned long long ) latency.pmd_val );
		SUBDBG( "PMD[36]: %#016llx\n",
				( unsigned long long ) load_addr.pmd_val );

		if ( ( !load_addr.pmd36_mont_reg.dear_vl ) ||
			 ( !load_addr.pmd33_mont_reg.dear_stat ) ) {
			SUBDBG
				( "Invalid DEAR sample found, dear_vl = %d, dear_stat = %#x\n",
				  load_addr.pmd36_mont_reg.dear_vl,
				  load_addr.pmd33_mont_reg.dear_stat );
		  bail1:
			newent = ( unsigned long ) *ent;
			newent += 3 * sizeof ( pfm_mont_pmd_reg_t );
			*ent = ( pfm_dfl_smpl_entry_t * ) newent;
			return 0;
		}

		if ( flags & PAPI_PROFIL_DATA_EAR )
			*pc = ( vptr_t ) data_addr.pmd_val;
		else if ( flags & PAPI_PROFIL_INST_EAR ) {
			unsigned long tmp =
				( ( load_addr.pmd36_mont_reg.dear_iaddr +
					( unsigned long ) load_addr.pmd36_mont_reg.
					dear_bn ) << 4 ) | ( unsigned long ) load_addr.
				pmd36_mont_reg.dear_slot;
			*pc = ( vptr_t ) tmp;
		} else {
			PAPIERROR( "BUG!" );
			goto bail1;
		}

		newent = ( unsigned long ) *ent;
		newent += 3 * sizeof ( pfm_mont_pmd_reg_t );
		*ent = ( pfm_dfl_smpl_entry_t * ) newent;
		return 0;
	} else if ( is_montecito_and_iear( native_pfm_index ) ) {
		pfm_mont_pmd_reg_t latency;
		pfm_mont_pmd_reg_t icache_line_addr;
		unsigned long newent;

		if ( ( flags & PAPI_PROFIL_INST_EAR ) == 0 )
			goto safety;

		/* Skip the header */
		++( *ent );

		// PMD34 has data address on Montecito
		// PMD35 has latency on Montecito
		icache_line_addr = *( pfm_mont_pmd_reg_t * ) * ent;
		latency =
			*( pfm_mont_pmd_reg_t * ) ( ( unsigned long ) *ent +
										sizeof ( icache_line_addr ) );

		SUBDBG( "PMD[34]: %#016llx\n",
				( unsigned long long ) icache_line_addr.pmd_val );
		SUBDBG( "PMD[35]: %#016llx\n",
				( unsigned long long ) latency.pmd_val );

		if ( ( icache_line_addr.pmd34_mont_reg.iear_stat & 0x1 ) == 0 ) {
			SUBDBG( "Invalid IEAR sample found, iear_stat = %#x\n",
					icache_line_addr.pmd34_mont_reg.iear_stat );
		  bail2:
			newent = ( unsigned long ) *ent;
			newent += 2 * sizeof ( pfm_mont_pmd_reg_t );
			*ent = ( pfm_dfl_smpl_entry_t * ) newent;
			return ( 0 );
		}

		if ( flags & PAPI_PROFIL_INST_EAR ) {
			unsigned long tmp = icache_line_addr.pmd34_mont_reg.iear_iaddr << 5;
			*pc = ( vptr_t ) tmp;
		} else {
			PAPIERROR( "BUG!" );
			goto bail2;
		}

		newent = ( unsigned long ) *ent;
		newent += 2 * sizeof ( pfm_mont_pmd_reg_t );
		*ent = ( pfm_dfl_smpl_entry_t * ) newent;
		return 0;
	} else if ( is_itanium2_and_dear( native_pfm_index ) ) {
		pfm_ita2_pmd_reg_t data_addr;
		pfm_ita2_pmd_reg_t latency;
		pfm_ita2_pmd_reg_t load_addr;
		unsigned long newent;

		if ( ( flags & ( PAPI_PROFIL_DATA_EAR | PAPI_PROFIL_INST_EAR ) ) == 0 )
			goto safety;

		/* Skip the header */
		++( *ent );

		// PMD2 has data address on Itanium 2
		// PMD3 has latency on Itanium 2
		// PMD17 has instruction address on Itanium 2
		data_addr = *( pfm_ita2_pmd_reg_t * ) * ent;
		latency =
			*( pfm_ita2_pmd_reg_t * ) ( ( unsigned long ) *ent +
										sizeof ( data_addr ) );
		load_addr =
			*( pfm_ita2_pmd_reg_t * ) ( ( unsigned long ) *ent +
										sizeof ( data_addr ) +
										sizeof ( latency ) );

		SUBDBG( "PMD[2]: %#016llx\n",
				( unsigned long long ) data_addr.pmd_val );
		SUBDBG( "PMD[3]: %#016llx\n", ( unsigned long long ) latency.pmd_val );
		SUBDBG( "PMD[17]: %#016llx\n",
				( unsigned long long ) load_addr.pmd_val );

		if ( ( !load_addr.pmd17_ita2_reg.dear_vl ) ||
			 ( !load_addr.pmd3_ita2_reg.dear_stat ) ) {
			SUBDBG
				( "Invalid DEAR sample found, dear_vl = %d, dear_stat = %#x\n",
				  load_addr.pmd17_ita2_reg.dear_vl,
				  load_addr.pmd3_ita2_reg.dear_stat );
		  bail3:
			newent = ( unsigned long ) *ent;
			newent += 3 * sizeof ( pfm_mont_pmd_reg_t );
			*ent = ( pfm_dfl_smpl_entry_t * ) newent;
			return 0;
		}

		if ( flags & PAPI_PROFIL_DATA_EAR )
			*pc = ( vptr_t ) data_addr.pmd_val;
		else if ( flags & PAPI_PROFIL_INST_EAR ) {
			unsigned long tmp =
				( ( load_addr.pmd17_ita2_reg.dear_iaddr +
					( unsigned long ) load_addr.pmd17_ita2_reg.
					dear_bn ) << 4 ) | ( unsigned long ) load_addr.
				pmd17_ita2_reg.dear_slot;
			*pc = ( vptr_t ) tmp;
		} else {
			PAPIERROR( "BUG!" );
			goto bail3;
		}

		newent = ( unsigned long ) *ent;
		newent += 3 * sizeof ( pfm_ita2_pmd_reg_t );
		*ent = ( pfm_dfl_smpl_entry_t * ) newent;
		return 0;
	} else if ( is_itanium2_and_iear( native_pfm_index ) ) {
		pfm_ita2_pmd_reg_t latency;
		pfm_ita2_pmd_reg_t icache_line_addr;
		unsigned long newent;

		if ( ( flags & PAPI_PROFIL_INST_EAR ) == 0 )
			goto safety;

		/* Skip the header */
		++( *ent );

		// PMD0 has address on Itanium 2
		// PMD1 has latency on Itanium 2
		icache_line_addr = *( pfm_ita2_pmd_reg_t * ) * ent;
		latency =
			*( pfm_ita2_pmd_reg_t * ) ( ( unsigned long ) *ent +
										sizeof ( icache_line_addr ) );

		SUBDBG( "PMD[0]: %#016llx\n",
				( unsigned long long ) icache_line_addr.pmd_val );
		SUBDBG( "PMD[1]: %#016llx\n", ( unsigned long long ) latency.pmd_val );

		if ( ( icache_line_addr.pmd0_ita2_reg.iear_stat & 0x1 ) == 0 ) {
			SUBDBG( "Invalid IEAR sample found, iear_stat = %#x\n",
					icache_line_addr.pmd0_ita2_reg.iear_stat );
		  bail4:
			newent = ( unsigned long ) *ent;
			newent += 2 * sizeof ( pfm_mont_pmd_reg_t );
			*ent = ( pfm_dfl_smpl_entry_t * ) newent;
			return ( 0 );
		}

		if ( flags & PAPI_PROFIL_INST_EAR ) {
			unsigned long tmp = icache_line_addr.pmd0_ita2_reg.iear_iaddr << 5;
			*pc = ( vptr_t ) tmp;
		} else {
			PAPIERROR( "BUG!" );
			goto bail4;
		}

		newent = ( unsigned long ) *ent;
		newent += 2 * sizeof ( pfm_ita2_pmd_reg_t );
		*ent = ( pfm_dfl_smpl_entry_t * ) newent;
		return 0;
	}
#if 0
	( is_btb( native_pfm_index ) ) {
		// PMD48-63,39 on Montecito
		// PMD8-15,16 on Itanium 2
	}
#endif
	else
  safety:
#endif
	{
		*pc = ( vptr_t ) ( ( size_t ) ( ( *ent )->ip ) );
		++( *ent );
		return ( 0 );
	}
}

static inline int
process_smpl_buf( int num_smpl_pmds, int entry_size, ThreadInfo_t ** thr )
{
	( void ) num_smpl_pmds;	 /*unused */
	( void ) entry_size;	 /*unused */
	int cidx = _perfmon2_vector.cmp_info.CmpIdx;
	pfm_dfl_smpl_entry_t *ent;
	uint64_t entry, count;
	pfm_dfl_smpl_hdr_t *hdr =
		( ( pfm_context_t * ) ( *thr )->context[cidx] )->smpl_buf;
	int ret, profile_index, flags;
	unsigned int native_pfm_index;
	vptr_t pc = NULL;
	long long weight;

	DEBUGCALL( DEBUG_SUBSTRATE, dump_smpl_hdr( hdr ) );
	count = hdr->hdr_count;
	ent = ( pfm_dfl_smpl_entry_t * ) ( hdr + 1 );
	entry = 0;

	SUBDBG( "This buffer has %llu samples in it.\n",
			( unsigned long long ) count );
	while ( count-- ) {
		SUBDBG( "Processing sample entry %llu\n",
				( unsigned long long ) entry );
		DEBUGCALL( DEBUG_SUBSTRATE, dump_smpl( ent ) );

		/* Find the index of the profile buffers if we are profiling on many events */

		ret =
			find_profile_index( ( *thr )->running_eventset[cidx], ent->ovfl_pmd,
								&flags, &native_pfm_index, &profile_index );
		if ( ret != PAPI_OK )
			return ( ret );

		weight = process_smpl_entry( native_pfm_index, flags, &ent, &pc );

		_papi_hwi_dispatch_profile( ( *thr )->running_eventset[cidx], pc,
									weight, profile_index );

		entry++;
	}
	return ( PAPI_OK );
}


/* This function  used when hardware overflows ARE working 
    or when software overflows are forced					*/

static void
_papi_pfm_dispatch_timer( int n, hwd_siginfo_t * info, void *uc )
{
	_papi_hwi_context_t ctx;
#ifdef HAVE_PFM_MSG_TYPE
	pfm_msg_t msg;
#else
	pfarg_msg_t msg;
#endif
	int ret, wanted_fd, fd = info->si_fd;
	vptr_t address;
	ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );
	int cidx = _perfmon2_vector.cmp_info.CmpIdx;

	if ( thread == NULL ) {
		PAPIERROR( "thread == NULL in _papi_pfm_dispatch_timer!" );
		if ( n == _perfmon2_vector.cmp_info.hardware_intr_sig ) {
			ret = read( fd, &msg, sizeof ( msg ) );
			pfm_restart( fd );
		}
		return;
	}

	if ( thread->running_eventset[cidx] == NULL ) {
		PAPIERROR
			( "thread->running_eventset == NULL in _papi_pfm_dispatch_timer!" );
		if ( n == _perfmon2_vector.cmp_info.hardware_intr_sig ) {
			ret = read( fd, &msg, sizeof ( msg ) );
			pfm_restart( fd );
		}
		return;
	}

	if ( thread->running_eventset[cidx]->overflow.flags == 0 ) {
		PAPIERROR
			( "thread->running_eventset->overflow.flags == 0 in _papi_pfm_dispatch_timer!" );
		if ( n == _perfmon2_vector.cmp_info.hardware_intr_sig ) {
			ret = read( fd, &msg, sizeof ( msg ) );
			pfm_restart( fd );
		}
		return;
	}

	ctx.si = info;
	ctx.ucontext = ( hwd_ucontext_t * ) uc;

	if ( thread->running_eventset[cidx]->overflow.
		 flags & PAPI_OVERFLOW_FORCE_SW ) {
		address = GET_OVERFLOW_ADDRESS( ctx );
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, NULL,
											0, 0, &thread, cidx );
	} else {
		if ( thread->running_eventset[cidx]->overflow.flags ==
			 PAPI_OVERFLOW_HARDWARE ) {
			wanted_fd =
				( ( pfm_control_state_t * ) ( thread->running_eventset[cidx]->
											  ctl_state ) )->ctx_fd;
		} else {
			wanted_fd = ( ( pfm_context_t * ) thread->context[cidx] )->ctx_fd;
		}
		if ( wanted_fd != fd ) {
			SUBDBG( "expected fd %d, got %d in _papi_hwi_dispatch_timer!",
					wanted_fd, fd );
			if ( n == _perfmon2_vector.cmp_info.hardware_intr_sig ) {
				ret = read( fd, &msg, sizeof ( msg ) );
				pfm_restart( fd );
			}
			return;
		}
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
			PAPIERROR( "read(%d): short %d vs. %d bytes", fd, ret,
					   sizeof ( msg ) );
			ret = -1;
		}

		if ( msg.type != PFM_MSG_OVFL ) {
			PAPIERROR( "unexpected msg type %d", msg.type );
			ret = -1;
		}
#if 0
		if ( msg.pfm_ovfl_msg.msg_ovfl_tid != mygettid(  ) ) {
			PAPIERROR( "unmatched thread id %lx vs. %lx",
					   msg.pfm_ovfl_msg.msg_ovfl_tid, mygettid(  ) );
			ret = -1;
		}
#endif

		if ( ret != -1 ) {
			if ( ( thread->running_eventset[cidx]->state & PAPI_PROFILING ) &&
				 !( thread->running_eventset[cidx]->profile.
					flags & PAPI_PROFIL_FORCE_SW ) )
				process_smpl_buf( 0, sizeof ( pfm_dfl_smpl_entry_t ), &thread );
			else {
				/* PAPI assumes that the overflow vector contains the register index of the
				   overflowing native event. That is generally true, but Stephane used some
				   tricks to offset the fixed counters on Core2 (Core? i7?) by 16. This hack
				   corrects for that hack in a (hopefully) transparent manner */
				unsigned long i, vector = msg.pfm_ovfl_msg.msg_ovfl_pmds[0];
				pfm_control_state_t *ctl =
					( pfm_control_state_t * ) thread->running_eventset[cidx]->
					ctl_state;
				for ( i = 0; i < ctl->in.pfp_event_count; i++ ) {
					/* We're only comparing to pmds[0]. A more robust implementation would
					   compare to pmds[0-3]. The bit mask must be converted to an index
					   for the comparison to work */
					if ( ctl->pd[i].reg_num ==
						 ffsl( msg.pfm_ovfl_msg.msg_ovfl_pmds[0] ) - 1 ) {
						/* if a match is found, convert the index back to a bitmask */
						vector = 1 << i;
						break;
					}
				}
				_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx,
													( vptr_t ) ( ( size_t )
																  msg.
																  pfm_ovfl_msg.
																  msg_ovfl_ip ),
													NULL, vector, 0, &thread,
													cidx );
			}
		}

		if ( ( ret = pfm_restart( fd ) ) ) {
			PAPIERROR( "pfm_restart(%d): %s", fd, strerror( ret ) );
		}
	}
}

static int
_papi_pfm_stop_profiling( ThreadInfo_t * thread, EventSetInfo_t * ESI )
{
	( void ) ESI;			 /*unused */
	/* Process any remaining samples in the sample buffer */
	return ( process_smpl_buf( 0, sizeof ( pfm_dfl_smpl_entry_t ), &thread ) );
}

static int
_papi_pfm_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	int cidx = _perfmon2_vector.cmp_info.CmpIdx;
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ( ESI->ctl_state );
	pfm_context_t *ctx = ( pfm_context_t * ) ( ESI->master->context[cidx] );
	pfarg_ctx_t newctx;
	void *buf_addr = NULL;
	pfm_dfl_smpl_arg_t buf_arg;
	pfm_dfl_smpl_hdr_t *hdr;
	int i, ret, ctx_fd;

	memset( &newctx, 0, sizeof ( newctx ) );

	if ( threshold == 0 ) {
		SUBDBG( "MUNMAP(%p,%lld)\n", ctx->smpl_buf,
				( unsigned long long ) ctx->smpl.buf_size );
		munmap( ctx->smpl_buf, ctx->smpl.buf_size );

		i = close( ctl->ctx_fd );
		SUBDBG( "CLOSE fd %d returned %d\n", ctl->ctx_fd, i );
		(void) i;

		/* Thread has master context */

		ctl->ctx_fd = ctx->ctx_fd;
		ctl->ctx = &ctx->ctx;
		memset( &ctx->smpl, 0, sizeof ( buf_arg ) );
		ctx->smpl_buf = NULL;
		ret = _papi_pfm_set_overflow( ESI, EventIndex, threshold );
//#warning "This should be handled somewhere else"
		ESI->state &= ~( PAPI_OVERFLOWING );
		ESI->overflow.flags &= ~( PAPI_OVERFLOW_HARDWARE );

		return ( ret );
	}

	memset( &buf_arg, 0, sizeof ( buf_arg ) );
	buf_arg.buf_size = 2 * getpagesize(  );

	SUBDBG( "PFM_CREATE_CONTEXT(%p,%s,%p,%d)\n", &newctx, PFM_DFL_SMPL_NAME,
			&buf_arg, ( int ) sizeof ( buf_arg ) );
	if ( ( ret =
		   pfm_create_context( &newctx, PFM_DFL_SMPL_NAME, &buf_arg,
							   sizeof ( buf_arg ) ) ) == -1 ) {
		DEBUGCALL( DEBUG_SUBSTRATE, dump_smpl_arg( &buf_arg ) );
		PAPIERROR( "_papi_hwd_set_profile:pfm_create_context(): %s",
				   strerror( errno ) );
		return ( PAPI_ESYS );
	}
	ctx_fd = ret;
	SUBDBG( "PFM_CREATE_CONTEXT returned fd %d\n", ctx_fd );
	tune_up_fd( ret );

	SUBDBG( "MMAP(NULL,%lld,%d,%d,%d,0)\n",
			( unsigned long long ) buf_arg.buf_size, PROT_READ, MAP_PRIVATE,
			ctx_fd );
	buf_addr =
		mmap( NULL, ( size_t ) buf_arg.buf_size, PROT_READ, MAP_PRIVATE, ctx_fd,
			  0 );
	if ( buf_addr == MAP_FAILED ) {
		PAPIERROR( "mmap(NULL,%d,%d,%d,%d,0): %s", buf_arg.buf_size, PROT_READ,
				   MAP_PRIVATE, ctx_fd, strerror( errno ) );
		close( ctx_fd );
		return ( PAPI_ESYS );
	}
	SUBDBG( "Sample buffer is located at %p\n", buf_addr );

	hdr = ( pfm_dfl_smpl_hdr_t * ) buf_addr;
	SUBDBG( "hdr_cur_offs=%llu version=%u.%u\n",
			( unsigned long long ) hdr->hdr_cur_offs,
			PFM_VERSION_MAJOR( hdr->hdr_version ),
			PFM_VERSION_MINOR( hdr->hdr_version ) );

	if ( PFM_VERSION_MAJOR( hdr->hdr_version ) < 1 ) {
		PAPIERROR( "invalid buffer format version %d",
				   PFM_VERSION_MAJOR( hdr->hdr_version ) );
		munmap( buf_addr, buf_arg.buf_size );
		close( ctx_fd );
		return PAPI_ESYS;
	}

	ret = _papi_pfm_set_overflow( ESI, EventIndex, threshold );
	if ( ret != PAPI_OK ) {
		munmap( buf_addr, buf_arg.buf_size );
		close( ctx_fd );
		return ( ret );
	}

	/* Look up the native event code */

	if ( ESI->profile.flags & ( PAPI_PROFIL_DATA_EAR | PAPI_PROFIL_INST_EAR ) ) {
		pfarg_pmd_t *pd;
		int pos, native_index;
		pd = ctl->pd;
		pos = ESI->EventInfoArray[EventIndex].pos[0];
		native_index =
			( ( pfm_register_t * ) ( ESI->NativeInfoArray[pos].ni_bits ) )->
			event;
		setup_ear_event( native_index, &pd[pos], ESI->profile.flags );
	}

	if ( ESI->profile.flags & PAPI_PROFIL_RANDOM ) {
		pfarg_pmd_t *pd;
		int pos;
		pd = ctl->pd;
		pos = ESI->EventInfoArray[EventIndex].pos[0];
		pd[pos].reg_random_seed = 5;
		pd[pos].reg_random_mask = 0xff;
	}

	/* Now close our context it is safe */

	// close(ctx->ctx_fd);

	/* Copy the new data to the threads context control block */

	ctl->ctx_fd = ctx_fd;
	memcpy( &ctx->smpl, &buf_arg, sizeof ( buf_arg ) );
	ctx->smpl_buf = buf_addr;

	return ( PAPI_OK );
}



static int
_papi_pfm_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	pfm_control_state_t *this_state =
		( pfm_control_state_t * ) ( ESI->ctl_state );
	int j, retval = PAPI_OK, *pos;

	/* Which counter are we on, this looks suspicious because of the pos[0],
	   but this could be because of derived events. We should do more here
	   to figure out exactly what the position is, because the event may
	   actually have more than one position. */

	pos = ESI->EventInfoArray[EventIndex].pos;
	j = pos[0];
	SUBDBG( "Hardware counter %d used in overflow, threshold %d\n", j,
			threshold );

	if ( threshold == 0 ) {
		/* If this counter isn't set to overflow */

		if ( ( this_state->pd[j].reg_flags & PFM_REGFL_OVFL_NOTIFY ) == 0 )
			return ( PAPI_EINVAL );

		/* Remove the signal handler */

		retval = _papi_hwi_stop_signal( _perfmon2_vector.cmp_info.hardware_intr_sig );
		if ( retval != PAPI_OK )
			return ( retval );

		/* Disable overflow */

		this_state->pd[j].reg_flags ^= PFM_REGFL_OVFL_NOTIFY;

		/*
		 * we may want to reset the other PMDs on
		 * every overflow. If we do not set
		 * this, the non-overflowed counters
		 * will be untouched.

		 if (inp.pfp_event_count > 1)
		 this_state->pd[j].reg_reset_pmds[0] ^= 1UL << counter_to_reset */

		/* Clear the overflow period */

		this_state->pd[j].reg_value = 0;
		this_state->pd[j].reg_long_reset = 0;
		this_state->pd[j].reg_short_reset = 0;
		this_state->pd[j].reg_random_seed = 0;
		this_state->pd[j].reg_random_mask = 0;
	} else {
		/* Enable the signal handler */

		retval =
			_papi_hwi_start_signal( _perfmon2_vector.cmp_info.hardware_intr_sig, 1,
									_perfmon2_vector.cmp_info.CmpIdx );
		if ( retval != PAPI_OK )
			return ( retval );

		/* Set it to overflow */

		this_state->pd[j].reg_flags |= PFM_REGFL_OVFL_NOTIFY;

		/*
		 * we may want to reset the other PMDs on
		 * every overflow. If we do not set
		 * this, the non-overflowed counters
		 * will be untouched.

		 if (inp.pfp_event_count > 1)
		 this_state->pd[j].reg_reset_pmds[0] |= 1UL << counter_to_reset */

		/* Set the overflow period */

		this_state->pd[j].reg_value = -( unsigned long long ) threshold + 1;
		this_state->pd[j].reg_short_reset =
			-( unsigned long long ) threshold + 1;
		this_state->pd[j].reg_long_reset =
			-( unsigned long long ) threshold + 1;
	}
	return ( retval );
}

static int
_papi_pfm_init_control_state( hwd_control_state_t * ctl0 )
{
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ctl0;
	pfmlib_input_param_t *inp = &ctl->in;
	pfmlib_output_param_t *outp = &ctl->out;
	pfarg_pmd_t *pd = ctl->pd;
	pfarg_pmc_t *pc = ctl->pc;
	pfarg_setdesc_t *set = ctl->set;
	pfarg_setinfo_t *setinfo = ctl->setinfo;

	memset( inp, 0, sizeof ( *inp ) );
	memset( outp, 0, sizeof ( *inp ) );
	memset( pc, 0, sizeof ( ctl->pc ) );
	memset( pd, 0, sizeof ( ctl->pd ) );
	memset( set, 0, sizeof ( ctl->set ) );
	memset( setinfo, 0, sizeof ( ctl->setinfo ) );
	/* Will be filled by update now...until this gets another arg */
	ctl->ctx = NULL;
	ctl->ctx_fd = -1;
	ctl->load = NULL;
	set_domain( ctl, _perfmon2_vector.cmp_info.default_domain );
	return ( PAPI_OK );
}

static int
_papi_pfm_allocate_registers( EventSetInfo_t * ESI )
{
	int i, j;
	for ( i = 0; i < ESI->NativeCount; i++ ) {
		if ( _papi_libpfm_ntv_code_to_bits
			 ( ESI->NativeInfoArray[i].ni_event,
			   ESI->NativeInfoArray[i].ni_bits ) != PAPI_OK )
			goto bail;
	}
	return PAPI_OK;
  bail:
	for ( j = 0; j < i; j++ )
		memset( ESI->NativeInfoArray[j].ni_bits, 0x0,
				sizeof ( pfm_register_t ) );
	return PAPI_ECNFLCT;
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

static int
_papi_pfm_update_control_state( hwd_control_state_t * ctl0,
								NativeInfo_t * native, int count,
								hwd_context_t * ctx0 )
{
	pfm_control_state_t *ctl = ( pfm_control_state_t * ) ctl0;
	pfm_context_t *ctx = ( pfm_context_t * ) ctx0;
	int i = 0, ret;
	int last_reg_set = 0, reg_set_done = 0, offset = 0;
	pfmlib_input_param_t tmpin, *inp = &ctl->in;
	pfmlib_output_param_t tmpout, *outp = &ctl->out;
	pfarg_pmd_t *pd = ctl->pd;

	if ( count == 0 ) {
		SUBDBG( "Called with count == 0\n" );
		inp->pfp_event_count = 0;
		outp->pfp_pmc_count = 0;
		memset( inp->pfp_events, 0x0, sizeof ( inp->pfp_events ) );
		return ( PAPI_OK );
	}

	memcpy( &tmpin, inp, sizeof ( tmpin ) );
	memcpy( &tmpout, outp, sizeof ( tmpout ) );

	for ( i = 0; i < count; i++ ) {
		SUBDBG
			( "Stuffing native event index %d (code %#x) into input structure.\n",
			  i, ( ( pfm_register_t * ) native[i].ni_bits )->event );
		memcpy( inp->pfp_events + i, native[i].ni_bits,
				sizeof ( pfmlib_event_t ) );
	}
	inp->pfp_event_count = count;

	/* let the library figure out the values for the PMCS */

	ret = compute_kernel_args( ctl );
	if ( ret != PAPI_OK ) {
		/* Restore values */
		memcpy( inp, &tmpin, sizeof ( tmpin ) );
		memcpy( outp, &tmpout, sizeof ( tmpout ) );
		return ( ret );
	}

	/* Update the native structure, because the allocation is done here. */

	last_reg_set = pd[0].reg_set;
	for ( i = 0; i < count; i++ ) {
		if ( pd[i].reg_set != last_reg_set ) {
			offset += reg_set_done;
			reg_set_done = 0;
		}
		reg_set_done++;

		native[i].ni_position = i;
		SUBDBG( "native event index %d (code %#x) is at PMD offset %d\n", i,
				( ( pfm_register_t * ) native[i].ni_bits )->event,
				native[i].ni_position );
	}

	/* If structure has not yet been filled with a context, fill it
	   from the thread's context. This should happen in init_control_state
	   when we give that a *ctx argument */

	if ( ctl->ctx == NULL ) {
		ctl->ctx = &ctx->ctx;
		ctl->ctx_fd = ctx->ctx_fd;
		ctl->load = &ctx->load;
	}

	return ( PAPI_OK );
}


papi_vector_t _perfmon2_vector = {
   .cmp_info = {
      /* default component information (unspecified values initialized to 0) */
      .name = "perfmon",
      .description =  "Linux perfmon2 CPU counters",
      .version = "3.8",

      .default_domain = PAPI_DOM_USER,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
      .default_granularity = PAPI_GRN_THR,
      .available_granularities = PAPI_GRN_THR,

      .hardware_intr = 1,
      .kernel_multiplex = 1,
      .kernel_profile = 1,
      .num_mpx_cntrs = PFMLIB_MAX_PMDS,

      /* component specific cmp_info initializations */
      .fast_real_timer = 1,
      .fast_virtual_timer = 0,
      .attach = 1,
      .attach_must_ptrace = 1,
  },

	/* sizes of framework-opaque component-private structures */
  .size = {
       .context = sizeof ( pfm_context_t ),
       .control_state = sizeof ( pfm_control_state_t ),
       .reg_value = sizeof ( pfm_register_t ),
       .reg_alloc = sizeof ( pfm_reg_alloc_t ),
  },
	/* function pointers in this component */
  .init_control_state =   _papi_pfm_init_control_state,
  .start =                _papi_pfm_start,
  .stop =                 _papi_pfm_stop,
  .read =                 _papi_pfm_read,
  .shutdown_thread =      _papi_pfm_shutdown,
  .shutdown_component =   _papi_pfm_shutdown_component,
  .ctl =                  _papi_pfm_ctl,
  .update_control_state = _papi_pfm_update_control_state,	
  .set_domain =           set_domain,
  .reset =                _papi_pfm_reset,
  .set_overflow =         _papi_pfm_set_overflow,
  .set_profile =          _papi_pfm_set_profile,
  .stop_profiling =       _papi_pfm_stop_profiling,
  .init_component =       _papi_pfm_init_component,
  .dispatch_timer =       _papi_pfm_dispatch_timer,
  .init_thread =          _papi_pfm_init_thread,
  .allocate_registers =   _papi_pfm_allocate_registers,
  .write =                _papi_pfm_write,

	/* from the counter name library */
  .ntv_enum_events =      _papi_libpfm_ntv_enum_events,
  .ntv_name_to_code =     _papi_libpfm_ntv_name_to_code,
  .ntv_code_to_name =     _papi_libpfm_ntv_code_to_name,
  .ntv_code_to_descr =    _papi_libpfm_ntv_code_to_descr,
  .ntv_code_to_bits =     _papi_libpfm_ntv_code_to_bits,

};
