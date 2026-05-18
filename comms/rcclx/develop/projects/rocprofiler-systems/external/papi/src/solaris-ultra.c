/* 
* File:    solaris-ultra.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Larry Meadows(helped us to build the native table dynamically) 
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu 
* Mods:    Vince Weaver
*          vweaver1@eecs.utk.edu 
*/

/* to understand this program, first you should read the user's manual
   about UltraSparc II and UltraSparc III, then the man pages
   about cpc_take_sample(cpc_event_t *event)
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include <sys/utsname.h>

#include "solaris-common.h"
#include "solaris-memory.h"

#ifdef CPC_ULTRA3_I
#define LASTULTRA3 CPC_ULTRA3_I
#else
#define LASTULTRA3 CPC_ULTRA3_PLUS
#endif

#define MAX_ENAME 40

static void action( void *arg, int regno, const char *name, uint8_t bits );

/* Probably could dispense with this and just use native_table */
typedef struct ctr_info
{
	char *name;						   /* Counter name */
	int bits[2];					   /* bits for register */
	int bitmask;					   /* 1 = pic0; 2 = pic1; 3 = both */
} ctr_info_t;

typedef struct einfo
{
	unsigned int papi_event;
	char *event_str;
} einfo_t;
static einfo_t us3info[] = {
	{PAPI_FP_INS, "FA_pipe_completion+FM_pipe_completion"},
	{PAPI_FAD_INS, "FA_pipe_completion"},
	{PAPI_FML_INS, "FM_pipe_completion"},
	{PAPI_TLB_IM, "ITLB_miss"},
	{PAPI_TLB_DM, "DTLB_miss"},
	{PAPI_TOT_CYC, "Cycle_cnt"},
	{PAPI_TOT_IIS, "Instr_cnt"},
	{PAPI_TOT_INS, "Instr_cnt"},
	{PAPI_L2_TCM, "EC_misses"},
	{PAPI_L2_ICM, "EC_ic_miss"},
	{PAPI_L1_ICM, "IC_miss"},
	{PAPI_L1_LDM, "DC_rd_miss"},
	{PAPI_L1_STM, "DC_wr_miss"},
	{PAPI_L2_LDM, "EC_rd_miss"},
	{PAPI_BR_MSP, "IU_Stat_Br_miss_taken+IU_Stat_Br_miss_untaken"},
	{PAPI_L1_DCR, "DC_rd"},
	{PAPI_L1_DCW, "DC_wr"},
	{PAPI_L1_ICH, "IC_ref-IC_miss"},	/* Is this really hits only? */
	{PAPI_L1_ICA, "IC_ref"}, /* Ditto? */
	{PAPI_L2_TCH, "EC_ref-EC_misses"},
	{PAPI_L2_TCA, "EC_ref"},
};

static einfo_t us2info[] = {
	{PAPI_L1_ICM, "IC_ref-IC_hit"},
	{PAPI_L2_TCM, "EC_ref-EC_hit"},
	{PAPI_CA_SNP, "EC_snoop_cb"},
	{PAPI_CA_INV, "EC_snoop_inv"},
	{PAPI_L1_LDM, "DC_rd-DC_rd_hit"},
	{PAPI_L1_STM, "DC_wr-DC_wr_hit"},
	{PAPI_L2_LDM, "EC_rd_miss"},
	{PAPI_BR_MSP, "Dispatch0_mispred"},
	{PAPI_TOT_IIS, "Instr_cnt"},
	{PAPI_TOT_INS, "Instr_cnt"},
	{PAPI_LD_INS, "DC_rd"},
	{PAPI_SR_INS, "DC_wr"},
	{PAPI_TOT_CYC, "Cycle_cnt"},
	{PAPI_L1_DCR, "DC_rd"},
	{PAPI_L1_DCW, "DC_wr"},
	{PAPI_L1_ICH, "IC_hit"},
	{PAPI_L2_ICH, "EC_ic_hit"},
	{PAPI_L1_ICA, "IC_ref"},
	{PAPI_L2_TCH, "EC_hit"},
	{PAPI_L2_TCA, "EC_ref"},
};

papi_vector_t _solaris_vector;

static native_info_t *native_table;
static hwi_search_t *preset_table;

static struct ctr_info *ctrs;
static int nctrs;

static int build_tables( void );
static void add_preset( hwi_search_t * tab, int *np, einfo_t e );

/* Globals used to access the counter registers. */

static int cpuver;
static int pcr_shift[2];

hwi_search_t *preset_search_map;

#ifdef DEBUG
static void
dump_cmd( papi_cpc_event_t * t )
{
	SUBDBG( "cpc_event_t.ce_cpuver %d\n", t->cmd.ce_cpuver );
	SUBDBG( "ce_tick %llu\n", t->cmd.ce_tick );
	SUBDBG( "ce_pic[0] %llu ce_pic[1] %llu\n", t->cmd.ce_pic[0],
			t->cmd.ce_pic[1] );
	SUBDBG( "ce_pcr %#llx\n", t->cmd.ce_pcr );
	SUBDBG( "flags %#x\n", t->flags );
}
#endif

static void
dispatch_emt( int signal, siginfo_t * sip, void *arg )
{
	int event_counter;
	_papi_hwi_context_t ctx;
	vptr_t address;
	ctx.si = sip;
	ctx.ucontext = arg;
	SUBDBG( "%d, %p, %p\n", signal, sip, arg );

	if ( sip->si_code == EMT_CPCOVF ) {
		papi_cpc_event_t *sample;
		EventSetInfo_t *ESI;
		ThreadInfo_t *thread = NULL;
		int t, overflow_vector, readvalue;

		thread = _papi_hwi_lookup_thread( 0 );
		ESI = ( EventSetInfo_t * ) thread->running_eventset;
		int cidx = ESI->CmpIdx;

		if ( ( ESI == NULL ) || ( ( ESI->state & PAPI_OVERFLOWING ) == 0 ) ) {
			OVFDBG( "Either no eventset or eventset not set to overflow.\n" );
			return;
		}

		if ( ESI->master != thread ) {
			PAPIERROR
				( "eventset->thread %%lx vs. current thread %#lx mismatch",
				  ESI->master, thread );
			return;
		}

		event_counter = ESI->overflow.event_counter;
		sample = &( ESI->ctl_state->counter_cmd );

		/* GROSS! This is a hack to 'push' the correct values 
		   back into the hardware, such that when PAPI handles
		   the overflow and reads the values, it gets the correct ones.
		 */

		/* Find which HW counter overflowed */

		if ( ESI->EventInfoArray[ESI->overflow.EventIndex[0]].pos[0] == 0 )
			t = 0;
		else
			t = 1;

		if ( cpc_take_sample( &sample->cmd ) == -1 )
			return;
		if ( event_counter == 1 ) {
			/* only one event is set to be the overflow monitor */

			/* generate the overflow vector */
			overflow_vector = 1 << t;
			/* reset the threshold */
			sample->cmd.ce_pic[t] = UINT64_MAX - ESI->overflow.threshold[0];
		} else {
			/* two events are set to be the overflow monitors */
			overflow_vector = 0;
			readvalue = sample->cmd.ce_pic[0];
			if ( readvalue >= 0 ) {
				/* the first counter overflowed */

				/* generate the overflow vector */
				overflow_vector = 1;
				/* reset the threshold */
				if ( t == 0 )
					sample->cmd.ce_pic[0] =
						UINT64_MAX - ESI->overflow.threshold[0];
				else
					sample->cmd.ce_pic[0] =
						UINT64_MAX - ESI->overflow.threshold[1];
			}
			readvalue = sample->cmd.ce_pic[1];
			if ( readvalue >= 0 ) {
				/* the second counter overflowed */

				/* generate the overflow vector */
				overflow_vector ^= 1 << 1;
				/* reset the threshold */
				if ( t == 0 )
					sample->cmd.ce_pic[1] =
						UINT64_MAX - ESI->overflow.threshold[1];
				else
					sample->cmd.ce_pic[1] =
						UINT64_MAX - ESI->overflow.threshold[0];
			}
			SUBDBG( "overflow_vector, = %d\n", overflow_vector );
			/* something is wrong here */
			if ( overflow_vector == 0 ) {
				PAPIERROR( "BUG! overflow_vector is 0, dropping interrupt" );
				return;
			}
		}

		/* Call the regular overflow function in extras.c */
		if ( thread->running_eventset[cidx]->overflow.
			 flags & PAPI_OVERFLOW_FORCE_SW ) {
			address = GET_OVERFLOW_ADDRESS(ctx);
			_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, NULL,
												overflow_vector, 0, &thread,
												cidx );
		} else {
			PAPIERROR( "Additional implementation needed in dispatch_emt!" );
		}

#if DEBUG
		dump_cmd( sample );
#endif
		/* push back the correct values and start counting again */
		if ( cpc_bind_event( &sample->cmd, sample->flags ) == -1 )
			return;
	} else {
		SUBDBG( "dispatch_emt() dropped, si_code = %d\n", sip->si_code );
		return;
	}
}

static int
scan_prtconf( char *cpuname, int len_cpuname, int *hz, int *ver )
{
	/* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
	/* See the PCL home page for the German version of PAPI. */
	/* Modified by Nils Smeds, all new bugs are my fault */
	/*    The routine now looks for the first "Node" with the following: */
	/*           "device_type"     = 'cpu'                    */
	/*           "name"            = (Any value)              */
	/*           "sparc-version"   = (Any value)              */
	/*           "clock-frequency" = (Any value)              */
	int ihz, version;
	char line[256], cmd[80], name[256];
	FILE *f = NULL;
	char cmd_line[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], fname[L_tmpnam];
	unsigned int matched;

	/*??? system call takes very long */
	/* get system configuration and put output into file */

	tmpnam( fname );
	SUBDBG( "Temporary name %s\n", fname );

	sprintf( cmd_line, "/usr/sbin/prtconf -vp > %s", fname );
	SUBDBG( "Executing %s\n", cmd_line );
	if ( system( cmd_line ) == -1 ) {
		remove( fname );
		return -1;
	}

	f = fopen( fname, "r" );
	/* open output file */
	if ( f == NULL ) {
		remove( fname );
		return -1;
	}

	/* ignore all lines until we reach something with a sparc line */
	matched = 0x0;
	ihz = -1;
	while ( fgets( line, 256, f ) != NULL ) {
		/*SUBDBG(">>> %s",line); */
		if ( ( sscanf( line, "%s", cmd ) == 1 )
			 && strstr( line, "Node 0x" ) ) {
			matched = 0x0;
			/*SUBDBG("Found 'Node' -- search reset. (%#2.2x)\n",matched); */
		} else {
			if ( strstr( cmd, "device_type:" ) && strstr( line, "'cpu'" ) ) {
				matched |= 0x1;
				SUBDBG( "Found 'cpu'. (%#2.2x)\n", matched );
			} else if ( !strcmp( cmd, "sparc-version:" ) &&
						( sscanf( line, "%s %#x", cmd, &version ) == 2 ) ) {
				matched |= 0x2;
				SUBDBG( "Found version=%d. (%#2.2x)\n", version, matched );
			} else if ( !strcmp( cmd, "clock-frequency:" ) &&
						( sscanf( line, "%s %#x", cmd, &ihz ) == 2 ) ) {
				matched |= 0x4;
				SUBDBG( "Found ihz=%d. (%#2.2x)\n", ihz, matched );
			} else if ( !strcmp( cmd, "name:" ) &&
						( sscanf( line, "%s %s", cmd, name ) == 2 ) ) {
				matched |= 0x8;
				SUBDBG( "Found name: %s. (%#2.2x)\n", name, matched );
			}
		}
		if ( ( matched & 0xF ) == 0xF )
			break;
	}
	SUBDBG( "Parsing found name=%s, speed=%dHz, version=%d\n", name, ihz,
			version );

	if ( matched ^ 0x0F )
		ihz = -1;
	else {
		*hz = ( float ) ihz;
		*ver = version;
		strncpy( cpuname, name, len_cpuname );
	}

	return ihz;

	/* End stolen code */
}

int
_ultra_set_domain( hwd_control_state_t * this_state, int domain )
{
	papi_cpc_event_t *command = &this_state->counter_cmd;
	cpc_event_t *event = &command->cmd;
	uint64_t pcr = event->ce_pcr;
	int did = 0;

	pcr = pcr | 0x7;
	pcr = pcr ^ 0x7;

	if ( domain & PAPI_DOM_USER ) {
		pcr = pcr | 1 << CPC_ULTRA_PCR_USR;
		did = 1;
	}
	if ( domain & PAPI_DOM_KERNEL ) {
		pcr = pcr | 1 << CPC_ULTRA_PCR_SYS;
		did = 1;
	}
	/* DOMAIN ERROR */
	if ( !did ) {
		return ( PAPI_EINVAL );
	}

	event->ce_pcr = pcr;

	return ( PAPI_OK );
}

static int
set_granularity( hwd_control_state_t * this_state, int domain )
{
	switch ( domain ) {
	case PAPI_GRN_PROCG:
	case PAPI_GRN_SYS:
	case PAPI_GRN_SYS_CPU:
	case PAPI_GRN_PROC:
		return PAPI_ECMP;
	case PAPI_GRN_THR:
		break;
	default:
		return ( PAPI_EINVAL );
	}
	return ( PAPI_OK );
}

/* Utility functions */

/* This is a wrapper arount fprintf(stderr,...) for cpc_walk_events() */
void
print_walk_names( void *arg, int regno, const char *name, uint8_t bits )
{
	SUBDBG( arg, regno, name, bits );
}


static int
build_tables( void )
{
	int i;
	int regno;
	int npic;
	einfo_t *ep;
	int n;
	int npresets;
	npic = cpc_getnpic( cpuver );
	nctrs = 0;
	for ( regno = 0; regno < npic; ++regno ) {
		cpc_walk_names( cpuver, regno, 0, action );
	}
	SUBDBG( "%d counters\n", nctrs );
	if ( ( ctrs = papi_malloc( nctrs * sizeof ( struct ctr_info ) ) ) == 0 ) {
		return PAPI_ENOMEM;
	}
	nctrs = 0;
	for ( regno = 0; regno < npic; ++regno ) {
		cpc_walk_names( cpuver, regno, ( void * ) 1, action );
	}
	SUBDBG( "%d counters\n", nctrs );
#if DEBUG
	if ( ISLEVEL( DEBUG_SUBSTRATE ) ) {
		for ( i = 0; i < nctrs; ++i ) {
			SUBDBG( "%s: bits (%#x,%#x) pics %#x\n", ctrs[i].name, ctrs[i].bits[0],
					ctrs[i].bits[1], ctrs[i].bitmask );
		}
	}
#endif
	/* Build the native event table */
	if ( ( native_table =
		   papi_malloc( nctrs * sizeof ( native_info_t ) ) ) == 0 ) {
		papi_free( ctrs );
		return PAPI_ENOMEM;
	}
	for ( i = 0; i < nctrs; ++i ) {
		native_table[i].name[39] = 0;
		strncpy( native_table[i].name, ctrs[i].name, 39 );
		if ( ctrs[i].bitmask & 1 )
			native_table[i].encoding[0] = ctrs[i].bits[0];
		else
			native_table[i].encoding[0] = -1;
		if ( ctrs[i].bitmask & 2 )
			native_table[i].encoding[1] = ctrs[i].bits[1];
		else
			native_table[i].encoding[1] = -1;
	}
	papi_free( ctrs );

	/* Build the preset table */
	if ( cpuver <= CPC_ULTRA2 ) {
		n = sizeof ( us2info ) / sizeof ( einfo_t );
		ep = us2info;
	} else if ( cpuver <= LASTULTRA3 ) {
		n = sizeof ( us3info ) / sizeof ( einfo_t );
		ep = us3info;
	} else
		return PAPI_ECMP;
	preset_table = papi_malloc( ( n + 1 ) * sizeof ( hwi_search_t ) );
	npresets = 0;
	for ( i = 0; i < n; ++i ) {
		add_preset( preset_table, &npresets, ep[i] );
	}
	memset( &preset_table[npresets], 0, sizeof ( hwi_search_t ) );

#ifdef DEBUG
	if ( ISLEVEL( DEBUG_SUBSTRATE ) ) {
		SUBDBG( "Native table: %d\n", nctrs );
		for ( i = 0; i < nctrs; ++i ) {
			SUBDBG( "%40s: %8x %8x\n", native_table[i].name,
					native_table[i].encoding[0], native_table[i].encoding[1] );
		}
		SUBDBG( "\nPreset table: %d\n", npresets );
		for ( i = 0; preset_table[i].event_code != 0; ++i ) {
			SUBDBG( "%8x: op %2d e0 %8x e1 %8x\n",
					preset_table[i].event_code,
					preset_table[i].data.derived,
					preset_table[i].data.native[0],
					preset_table[i].data.native[1] );
		}
	}
#endif

	_solaris_vector.cmp_info.num_native_events = nctrs;

	return PAPI_OK;
}

static int
srch_event( char *e1 )
{
	int i;

	for ( i = 0; i < nctrs; ++i ) {
		if ( strcmp( e1, native_table[i].name ) == 0 )
			break;
	}
	if ( i >= nctrs )
		return -1;
	return i;
}

/* we should read from the CSV file and make this all unnecessary */

static void
add_preset( hwi_search_t * tab, int *np, einfo_t e )
{
	/* Parse the event info string and build the PAPI preset.
	 * If parse fails, just return, otherwise increment the table
	 * size. We assume that the table is big enough.
	 */
	char *p;
	char *q;
	char op;
	char e1[MAX_ENAME], e2[MAX_ENAME];
	int i;
	int ne;
	int ne2;

	p = e.event_str;
	/* Assume p is the name of a native event, the sum of two
	 * native events, or the difference of two native events.
	 * This could be extended with a real parser (hint).
	 */
	while ( isspace( *p ) )
		++p;
	q = p;
	i = 0;
	while ( isalnum( *p ) || ( *p == '_' ) ) {
		if ( i >= MAX_ENAME - 1 )
			break;
		e1[i] = *p++;
		++i;
	}
	e1[i] = 0;
	if ( *p == '+' || *p == '-' )
		op = *p++;
	else
		op = 0;
	while ( isspace( *p ) )
		++p;
	q = p;
	i = 0;
	while ( isalnum( *p ) || ( *p == '_' ) ) {
		if ( i >= MAX_ENAME - 1 )
			break;
		e2[i] = *p++;
		++i;
	}
	e2[i] = 0;

	if ( e2[0] == 0 && e1[0] == 0 ) {
		return;
	}
	if ( e2[0] == 0 || op == 0 ) {
		ne = srch_event( e1 );
		if ( ne == -1 )
			return;
		tab[*np].event_code = e.papi_event;
		tab[*np].data.derived = 0;
		tab[*np].data.native[0] = PAPI_NATIVE_MASK | ne;
		tab[*np].data.native[1] = PAPI_NULL;
		memset( tab[*np].data.operation, 0,
				sizeof ( tab[*np].data.operation ) );
		++*np;
		return;
	}
	ne = srch_event( e1 );
	ne2 = srch_event( e2 );
	if ( ne == -1 || ne2 == -1 )
		return;
	tab[*np].event_code = e.papi_event;
	tab[*np].data.derived = ( op == '-' ) ? DERIVED_SUB : DERIVED_ADD;
	tab[*np].data.native[0] = PAPI_NATIVE_MASK | ne;
	tab[*np].data.native[1] = PAPI_NATIVE_MASK | ne2;
	tab[*np].data.native[2] = PAPI_NULL;
	memset( tab[*np].data.operation, 0, sizeof ( tab[*np].data.operation ) );
	++*np;
}

void
action( void *arg, int regno, const char *name, uint8_t bits )
{
	int i;

	if ( arg == 0 ) {
		++nctrs;
		return;
	}
	assert( regno == 0 || regno == 1 );
	for ( i = 0; i < nctrs; ++i ) {
		if ( strcmp( ctrs[i].name, name ) == 0 ) {
			ctrs[i].bits[regno] = bits;
			ctrs[i].bitmask |= ( 1 << regno );
			return;
		}
	}
	memset( &ctrs[i], 0, sizeof ( ctrs[i] ) );
	ctrs[i].name = papi_strdup( name );
	ctrs[i].bits[regno] = bits;
	ctrs[i].bitmask = ( 1 << regno );
	++nctrs;
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static int
set_inherit( EventSetInfo_t * global, int arg )
{
	return PAPI_ECMP;

/*
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  papi_cpc_event_t *command= &machdep->counter_cmd;

  return(PAPI_EINVAL);
*/

#if 0
	if ( arg == 0 ) {
		if ( command->flags & CPC_BIND_LWP_INHERIT )
			command->flags = command->flags ^ CPC_BIND_LWP_INHERIT;
	} else if ( arg == 1 ) {
		command->flags = command->flags | CPC_BIND_LWP_INHERIT;
	} else
		return ( PAPI_EINVAL );

	return ( PAPI_OK );
#endif
}

static int
set_default_domain( hwd_control_state_t * ctrl_state, int domain )
{
	/* This doesn't exist on this platform */

	if ( domain == PAPI_DOM_OTHER )
		return ( PAPI_EINVAL );

	return ( _ultra_set_domain( ctrl_state, domain ) );
}

static int
set_default_granularity( hwd_control_state_t * current_state, int granularity )
{
	return ( set_granularity( current_state, granularity ) );
}

rwlock_t lock[PAPI_MAX_LOCK];

static void
lock_init( void )
{
	memset( lock, 0x0, sizeof ( rwlock_t ) * PAPI_MAX_LOCK );
}

int
_ultra_hwd_shutdown_component( void )
{
	( void ) cpc_rele(  );
	return ( PAPI_OK );
}

int
_ultra_hwd_init_component( int cidx )
{
	int retval;
 /* retval = _papi_hwi_setup_vector_table(vtable, _solaris_ultra_table);
	if ( retval != PAPI_OK ) return(retval); */

	/* Fill in what we can of the papi_system_info. */
	retval = _solaris_get_system_info( &_papi_hwi_system_info );
	if ( retval )
		return ( retval );

	/* Setup memory info */
        retval = _papi_os_vector.get_memory_info( &_papi_hwi_system_info.hw_info,
				     0 );
        if ( retval )
	  return ( retval );


	lock_init(  );

	SUBDBG( "Found %d %s %s CPUs at %d Mhz.\n",
			_papi_hwi_system_info.hw_info.totalcpus,
			_papi_hwi_system_info.hw_info.vendor_string,
			_papi_hwi_system_info.hw_info.model_string,
			_papi_hwi_system_info.hw_info.cpu_max_mhz );

	return ( PAPI_OK );
}

/* reset the hardware counter */
int
_ultra_hwd_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	int retval;

	/* reset the hardware counter */
	ctrl->counter_cmd.cmd.ce_pic[0] = 0;
	ctrl->counter_cmd.cmd.ce_pic[1] = 0;
	/* let's rock and roll */
	retval = cpc_bind_event( &ctrl->counter_cmd.cmd, ctrl->counter_cmd.flags );
	if ( retval == -1 )
		return ( PAPI_ESYS );

	return ( PAPI_OK );
}


int
_ultra_hwd_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
				long long **events, int flags )
{
	int retval;

	retval = cpc_take_sample( &ctrl->counter_cmd.cmd );
	if ( retval == -1 )
		return ( PAPI_ESYS );

	*events = ( long long * ) ctrl->counter_cmd.cmd.ce_pic;

	return PAPI_OK;
}

int
_ultra_hwd_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{

	switch ( code ) {
	case PAPI_DEFDOM:
		return ( set_default_domain
				 ( option->domain.ESI->ctl_state, option->domain.domain ) );
	case PAPI_DOMAIN:
		return ( _ultra_set_domain
				 ( option->domain.ESI->ctl_state, option->domain.domain ) );
	case PAPI_DEFGRN:
		return ( set_default_granularity
				 ( option->domain.ESI->ctl_state,
				   option->granularity.granularity ) );
	case PAPI_GRANUL:
		return ( set_granularity
				 ( option->granularity.ESI->ctl_state,
				   option->granularity.granularity ) );
	default:
		return ( PAPI_EINVAL );
	}
}

void
_ultra_hwd_dispatch_timer( int signal, siginfo_t * si, void *context )
{

  _papi_hwi_context_t ctx;
  ThreadInfo_t *master = NULL;
  int isHardware = 0;
  vptr_t address;
  int cidx = _solaris_vector.cmp_info.CmpIdx;

  ctx.si = si;
  ctx.ucontext = ( ucontext_t * ) context;

  address = GET_OVERFLOW_ADDRESS( ctx );
  _papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, &isHardware,
				      0, 0, &master, _solaris_vector.cmp_info.CmpIdx );

  /* We are done, resume interrupting counters */
  if ( isHardware ) {
    //    errno = vperfctr_iresume( master->context[cidx]->perfctr );
    //if ( errno < 0 ) {
    //  PAPIERROR( "vperfctr_iresume errno %d", errno );
    //}
  }


#if 0
        EventSetInfo_t *ESI = NULL;
        ThreadInfo_t *thread = NULL;
        int overflow_vector = 0;
        hwd_control_state_t *ctrl = NULL;
        long_long results[MAX_COUNTERS];
        int i;
        _papi_hwi_context_t ctx;
	vptr_t address;
	int cidx = _solaris_vector.cmp_info.CmpIdx;

        ctx.si = si;
        ctx.ucontext = ( hwd_ucontext_t * ) info;

	thread = _papi_hwi_lookup_thread( 0 );

	if ( thread == NULL ) {
		PAPIERROR( "thread == NULL in _papi_hwd_dispatch_timer");
		return;
	}

        ESI = ( EventSetInfo_t * ) thread->running_eventset[cidx];


	if ( ESI == NULL || ESI->master != thread || ESI->ctl_state == NULL ||
	     ( ( ESI->state & PAPI_OVERFLOWING ) == 0 ) ) {

	  if ( ESI == NULL )
	     PAPIERROR( "ESI is NULL\n");

	  if ( ESI->master != thread )
	     PAPIERROR( "Thread mismatch, ESI->master=%#x thread=%#x\n",
		        ESI->master, thread );

	  if ( ESI->ctl_state == NULL )
	     PAPIERROR( "Counter state invalid\n");

	  if ( ( ( ESI->state & PAPI_OVERFLOWING ) == 0 ) )
	    PAPIERROR( "Overflow flag missing");
	}

	ctrl = ESI->ctl_state;

	if ( thread->running_eventset[cidx]->overflow.flags & PAPI_OVERFLOW_FORCE_SW ) {
		address = GET_OVERFLOW_ADDRESS( ctx );
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, address, NULL, 0,
											0, &thread, cidx );
       } else {
	     PAPIERROR ( "Need to implement additional code in _papi_hwd_dispatch_timer!" );
       }
#endif
}

int
_ultra_hwd_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	hwd_control_state_t *this_state = ESI->ctl_state;
	papi_cpc_event_t *arg = &this_state->counter_cmd;
	int hwcntr;

	if ( threshold == 0 ) {
		if ( this_state->overflow_num == 1 ) {
			arg->flags ^= CPC_BIND_EMT_OVF;
			if ( sigaction
				 ( _solaris_vector.cmp_info.hardware_intr_sig, NULL,
				   NULL ) == -1 )
				return ( PAPI_ESYS );
			this_state->overflow_num = 0;
		} else
			this_state->overflow_num--;

	} else {
		struct sigaction act;
		/* increase the counter for overflow events */
		this_state->overflow_num++;

		act.sa_sigaction = dispatch_emt;
		memset( &act.sa_mask, 0x0, sizeof ( act.sa_mask ) );
		act.sa_flags = SA_RESTART | SA_SIGINFO;
		if ( sigaction
			 ( _solaris_vector.cmp_info.hardware_intr_sig, &act,
			   NULL ) == -1 )
			return ( PAPI_ESYS );

		arg->flags |= CPC_BIND_EMT_OVF;
		hwcntr = ESI->EventInfoArray[EventIndex].pos[0];
		if ( hwcntr == 0 )
			arg->cmd.ce_pic[0] = UINT64_MAX - ( uint64_t ) threshold;
		else if ( hwcntr == 1 )
			arg->cmd.ce_pic[1] = UINT64_MAX - ( uint64_t ) threshold;
	}

	return ( PAPI_OK );
}


_ultra_shutdown( hwd_context_t * ctx )
{

  return PAPI_OK;
}


/*
int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI)
{
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}
*/

void *
_ultra_hwd_get_overflow_address( void *context )
{
	void *location;
	ucontext_t *info = ( ucontext_t * ) context;
	location = ( void * ) info->uc_mcontext.gregs[REG_PC];

	return ( location );
}

int
_ultra_hwd_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	int retval;

	/* reset the hardware counter */
	if ( ctrl->overflow_num == 0 ) {
		ctrl->counter_cmd.cmd.ce_pic[0] = 0;
		ctrl->counter_cmd.cmd.ce_pic[1] = 0;
	}
	/* let's rock and roll */
	retval = cpc_bind_event( &ctrl->counter_cmd.cmd, ctrl->counter_cmd.flags );
	if ( retval == -1 )
		return ( PAPI_ESYS );

	return ( PAPI_OK );
}

int
_ultra_hwd_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	cpc_bind_event( NULL, 0 );
	return PAPI_OK;
}

int
_ultra_hwd_remove_event( hwd_register_map_t * chosen,
						unsigned int hardware_index, hwd_control_state_t * out )
{
	return PAPI_OK;
}

int
_ultra_hwd_encode_native( char *name, int *code )
{
	return ( PAPI_OK );
}

int
_ultra_hwd_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int index = *EventCode & PAPI_NATIVE_AND_MASK;

	if ( modifier == PAPI_ENUM_FIRST ) {
	   *EventCode = PAPI_NATIVE_MASK + 1;

	   return PAPI_OK;
	}

	if ( cpuver <= CPC_ULTRA2 ) {
		if ( index < MAX_NATIVE_EVENT_USII - 1 ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );
	} else if ( cpuver <= LASTULTRA3 ) {
		if ( index < MAX_NATIVE_EVENT - 1 ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );
	};
	return ( PAPI_ENOEVNT );
}

int
_ultra_hwd_ntv_code_to_name( unsigned int EventCode, char *ntv_name, int len )
{

        int event_code = EventCode & PAPI_NATIVE_AND_MASK;

	if ( event_code >= 0 && event_code < nctrs ) {
	  strlcpy( ntv_name, native_table[event_code].name, len );
	  return PAPI_OK;
	}
	return PAPI_ENOEVNT;
}


int
_ultra_hwd_ntv_code_to_descr( unsigned int EventCode, char *hwd_descr, int len )
{
	return ( _ultra_hwd_ntv_code_to_name( EventCode, hwd_descr, len ) );
}

static void
copy_value( unsigned int val, char *nam, char *names, unsigned int *values,
			int len )
{
	*values = val;
	strncpy( names, nam, len );
	names[len - 1] = 0;
}

int
_ultra_hwd_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK;

	if ( cpuver <= CPC_ULTRA2 ) {
		if ( index >= MAX_NATIVE_EVENT_USII ) {
			return ( PAPI_ENOEVNT );
		}
	} else if ( cpuver <= LASTULTRA3 ) {
		if ( index >= MAX_NATIVE_EVENT ) {
			return ( PAPI_ENOEVNT );
		}
	} else
		return ( PAPI_ENOEVNT );

	bits->event[0] = native_table[index].encoding[0];
	bits->event[1] = native_table[index].encoding[1];
	return ( PAPI_OK );
}

int
_ultra_hwd_init_control_state( hwd_control_state_t * ptr )
{
	ptr->counter_cmd.flags = 0x0;
	ptr->counter_cmd.cmd.ce_cpuver = cpuver;
	ptr->counter_cmd.cmd.ce_pcr = 0x0;
	ptr->counter_cmd.cmd.ce_pic[0] = 0;
	ptr->counter_cmd.cmd.ce_pic[1] = 0;

	_ultra_set_domain( ptr, _solaris_vector.cmp_info.default_domain );
	set_granularity( ptr, _solaris_vector.cmp_info.default_granularity );

	return PAPI_OK;
}

int
_ultra_hwd_update_control_state( hwd_control_state_t * this_state,
								NativeInfo_t * native, int count,
								hwd_context_t * zero )
{
	int nidx1, nidx2, hwcntr;
	uint64_t tmp = 0;
	uint64_t pcr;
	int64_t cmd0, cmd1;

/* save the last three bits */
	pcr = this_state->counter_cmd.cmd.ce_pcr & 0x7;

/* clear the control register */
	this_state->counter_cmd.cmd.ce_pcr = pcr;

/* no native events left */
	if ( count == 0 )
		return ( PAPI_OK );

	cmd0 = -1;
	cmd1 = -1;
/* one native event */
	if ( count == 1 ) {
		nidx1 = native[0].ni_event & PAPI_NATIVE_AND_MASK;
		hwcntr = 0;
		cmd0 = native_table[nidx1].encoding[0];
		native[0].ni_position = 0;
		if ( cmd0 == -1 ) {
			cmd1 = native_table[nidx1].encoding[1];
			native[0].ni_position = 1;
		}
	}

/* two native events */
	if ( count == 2 ) {
		int avail1, avail2;

		avail1 = 0;
		avail2 = 0;
		nidx1 = native[0].ni_event & PAPI_NATIVE_AND_MASK;
		nidx2 = native[1].ni_event & PAPI_NATIVE_AND_MASK;
		if ( native_table[nidx1].encoding[0] != -1 )
			avail1 = 0x1;
		if ( native_table[nidx1].encoding[1] != -1 )
			avail1 += 0x2;
		if ( native_table[nidx2].encoding[0] != -1 )
			avail2 = 0x1;
		if ( native_table[nidx2].encoding[1] != -1 )
			avail2 += 0x2;
		if ( ( avail1 | avail2 ) != 0x3 )
			return ( PAPI_ECNFLCT );
		if ( avail1 == 0x3 ) {
			if ( avail2 == 0x1 ) {
				cmd0 = native_table[nidx2].encoding[0];
				cmd1 = native_table[nidx1].encoding[1];
				native[0].ni_position = 1;
				native[1].ni_position = 0;
			} else {
				cmd1 = native_table[nidx2].encoding[1];
				cmd0 = native_table[nidx1].encoding[0];
				native[0].ni_position = 0;
				native[1].ni_position = 1;
			}
		} else {
			if ( avail1 == 0x1 ) {
				cmd0 = native_table[nidx1].encoding[0];
				cmd1 = native_table[nidx2].encoding[1];
				native[0].ni_position = 0;
				native[1].ni_position = 1;
			} else {
				cmd0 = native_table[nidx2].encoding[0];
				cmd1 = native_table[nidx1].encoding[1];
				native[0].ni_position = 1;
				native[1].ni_position = 0;
			}
		}
	}

/* set the control register */
	if ( cmd0 != -1 ) {
		tmp = ( ( uint64_t ) cmd0 << pcr_shift[0] );
	}
	if ( cmd1 != -1 ) {
		tmp = tmp | ( ( uint64_t ) cmd1 << pcr_shift[1] );
	}
	this_state->counter_cmd.cmd.ce_pcr = tmp | pcr;
#if DEBUG
	dump_cmd( &this_state->counter_cmd );
#endif

	return ( PAPI_OK );
}




papi_vector_t _solaris_vector = {
	.cmp_info = {
                                 .name = "solaris.ultra",
                                 .description = "Solaris CPU counters",
				 .num_cntrs = MAX_COUNTERS,
				 .num_mpx_cntrs = MAX_COUNTERS,
				 .default_domain = PAPI_DOM_USER,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .fast_real_timer = 1,
				 .fast_virtual_timer = 1,
				 .attach = 1,
				 .attach_must_ptrace = 1,
				 .hardware_intr = 0,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,
				 .precise_intr = 0,
				 }
	,

	/* component data structure sizes */
	.size = {
			 .context = sizeof ( hwd_context_t ),
			 .control_state = sizeof ( hwd_control_state_t ),
			 .reg_value = sizeof ( hwd_register_t ),
			 .reg_alloc = sizeof ( hwd_reg_alloc_t ),
			 }
	,

	/* component interface functions */
	.init_control_state = _ultra_hwd_init_control_state,
	.start = _ultra_hwd_start,
	.stop = _ultra_hwd_stop,
	.read = _ultra_hwd_read,
        .shutdown = _ultra_shutdown,
	.shutdown_component = _ultra_hwd_shutdown_component,
	.ctl = _ultra_hwd_ctl,
	.update_control_state = _ultra_hwd_update_control_state,
        .set_domain = _ultra_set_domain,
	.reset = _ultra_hwd_reset,
	.set_overflow = _ultra_hwd_set_overflow,
	/* .set_profile */
	/* .stop_profiling = _papi_hwd_stop_profiling, */
	.ntv_enum_events = _ultra_hwd_ntv_enum_events,
        /* .ntv_name_to_code */
	.ntv_code_to_name = _ultra_hwd_ntv_code_to_name,
	.ntv_code_to_descr = _ultra_hwd_ntv_code_to_descr,
	.ntv_code_to_bits = _ultra_hwd_ntv_code_to_bits,
	.init_component = _ultra_hwd_init_component,
	.dispatch_timer = _ultra_hwd_dispatch_timer,

};

papi_os_vector_t _papi_os_vector = {

	/* OS dependent local routines */
        .get_memory_info =    _solaris_get_memory_info,
        .get_dmem_info =      _solaris_get_dmem_info,
	.update_shlib_info =  _solaris_update_shlib_info,
	.get_system_info =    _solaris_get_system_info,
	.get_real_usec =      _solaris_get_real_usec,
	.get_real_cycles =    _solaris_get_real_cycles,
	.get_virt_usec =      _solaris_get_virt_usec,
};
