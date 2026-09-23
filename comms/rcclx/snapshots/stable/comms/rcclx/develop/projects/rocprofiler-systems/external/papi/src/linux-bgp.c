/*
 * File:    linux-bgp.c
 * Author:  Dave Hermsmeier
 *          dlherms@us.ibm.com
 */

/*
 * PAPI stuff
 */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"

#include "linux-bgp.h"
/*
 * BG/P specific 'stuff'
 */

/* BG/P includes */
#include <common/bgp_personality_inlines.h>
#include <spi/bgp_SPI.h>
#include <ucontext.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>

/* BG/P macros */
#define get_cycles _bgp_GetTimeBase

/* BG/P external structures/functions */

papi_vector_t _bgp_vectors;

/* Defined in linux-bgp-memory.c */
extern int _bgp_get_memory_info( PAPI_hw_info_t * pHwInfo, int pCPU_Type );
extern int _bgp_get_dmem_info( PAPI_dmem_info_t * pDmemInfo );

/* BG/P globals */
hwi_search_t *preset_search_map;
volatile unsigned int lock[PAPI_MAX_LOCK];
const char *BGP_NATIVE_RESERVED_EVENTID = "Reserved";
PAPI_os_info_t _papi_os_info;


/*
 * Get BGP Native Event Id from PAPI Event Id
 */
inline BGP_UPC_Event_Id_t
get_bgp_native_event_id( int pEventId )
{
	return ( BGP_UPC_Event_Id_t ) ( pEventId & PAPI_NATIVE_AND_MASK );
}

/*
 * Lock initialization
 */
void
_papi_hwd_lock_init( void )
{
	/* PAPI on BG/P does not need locks. */

	return;
}

/*
 * Lock
 */
void
_papi_hwd_lock( int lock )
{
	/* PAPI on BG/P does not need locks. */

	return;
}

/*
 * Unlock
 */
void
_papi_hwd_unlock( int lock )
{
	/* PAPI on BG/P does not need locks. */

	return;
}



/*
 * Get System Information
 *
 * Initialize system information structure
 */
int
_bgp_get_system_info( papi_mdi_t *mdi )
{
	_BGP_Personality_t bgp;
	int tmp;
	unsigned utmp;
	char chipID[64];

	/* Hardware info */
	if ( ( tmp = Kernel_GetPersonality( &bgp, sizeof bgp ) ) ) {

#include "error.h"

		fprintf( stdout, "Kernel_GetPersonality returned %d (sys error=%d).\n"
				 "\t%s\n", tmp, errno, strerror( errno ) );
		return PAPI_ESYS;
	}

	_papi_hwi_system_info.hw_info.ncpu = Kernel_ProcessorCount(  );
	_papi_hwi_system_info.hw_info.nnodes =
		( int ) BGP_Personality_numComputeNodes( &bgp );
	_papi_hwi_system_info.hw_info.totalcpus =
		_papi_hwi_system_info.hw_info.ncpu *
		_papi_hwi_system_info.hw_info.nnodes;

	utmp = Kernel_GetProcessorVersion(  );
	_papi_hwi_system_info.hw_info.model = ( int ) utmp;

	_papi_hwi_system_info.hw_info.vendor = ( utmp >> ( 31 - 11 ) ) & 0xFFF;

	_papi_hwi_system_info.hw_info.revision =
		( ( float ) ( ( utmp >> ( 31 - 15 ) ) & 0xFFFF ) ) +
		0.00001 * ( ( float ) ( utmp & 0xFFFF ) );

	strcpy( _papi_hwi_system_info.hw_info.vendor_string, "IBM" );
	tmp = snprintf( _papi_hwi_system_info.hw_info.model_string,
					sizeof _papi_hwi_system_info.hw_info.model_string,
					"PVR=%#4.4x:%#4.4x",
					( utmp >> ( 31 - 15 ) ) & 0xFFFF, ( utmp & 0xFFFF ) );

	BGP_Personality_getLocationString( &bgp, chipID );
	tmp += 12 + sizeof ( chipID );
	if ( sizeof ( _papi_hwi_system_info.hw_info.model_string ) > tmp ) {
		strcat( _papi_hwi_system_info.hw_info.model_string, "  Serial=" );
		strncat( _papi_hwi_system_info.hw_info.model_string,
				 chipID, sizeof ( chipID ) );
	}

	_papi_hwi_system_info.hw_info.mhz =
		( float ) BGP_Personality_clockMHz( &bgp );
	SUBDBG( "_bgp_get_system_info:  Detected MHZ is %f\n",
			_papi_hwi_system_info.hw_info.mhz );

	_papi_hwi_system_info.hw_info.cpu_max_mhz=_papi_hwi_system_info.hw_info.mhz;
	_papi_hwi_system_info.hw_info.cpu_min_mhz=_papi_hwi_system_info.hw_info.mhz;

	// Memory information structure not filled in - same as BG/L
	// _papi_hwi_system_info.hw_info.mem_hierarchy = ???;
	// The mpx_info structure disappeared in PAPI-C
	//_papi_hwi_system_info.mpx_info.timer_sig = PAPI_NULL;

	return PAPI_OK;
}


/*
 * Initialize Control State
 *
 * All state is kept in BG/P UPC structures
 */
int
_bgp_init_control_state( hwd_control_state_t *ctl )
{
	int i;

	//bgp_control_state_t *bgp_ctl = (bgp_control_state_t *)ctl;

	for ( i = 1; i < BGP_UPC_MAX_MONITORED_EVENTS; i++ )
		ctl->counters[i] = 0;

	return PAPI_OK;
}

/*
 * Set Domain
 *
 * All state is kept in BG/P UPC structures
 */
int
_bgp_set_domain( hwd_control_state_t * cntrl, int domain )
{

	return ( PAPI_OK );
}

/*
 * PAPI Initialization
 *
 * All state is kept in BG/P UPC structures
 */
int
_bgp_init_thread( hwd_context_t * ctx )
{

	return PAPI_OK;
}

/*
 * PAPI Global Initialization
 *
 * Global initialization - does initial PAPI setup and
 *                         calls BGP_UPC_Initialize()
 */
int
_bgp_init_global( void )
{
	int retval;
	int cidx = _bgp_vectors.cmp_info.CmpIdx;
	
	/*
	 * Fill in what we can of the papi_system_info
	 */
	SUBDBG( "Before _bgp_get_system_info()...\n" );
	retval = _bgp_get_system_info( &_papi_hwi_system_info );
	SUBDBG( "After _bgp_get_system_info(), retval=%d...\n", retval );
	if ( retval != PAPI_OK )
		return ( retval );

	/*
	 * Setup presets
	 */
	SUBDBG( "Before setup_bgp_presets, _papi_hwi_system_info.hw_info.model=%d...\n",
		  _papi_hwi_system_info.hw_info.model );
	retval = _papi_load_preset_table( "BGP", 0, cidx );
	SUBDBG( "After setup_bgp_presets, retval=%d...\n", retval );
	if ( retval )
		return ( retval );

	/*
	 * Setup memory info
	 */
	SUBDBG( "Before _bgp_get_memory_info...\n" );
	retval = _bgp_get_memory_info( &_papi_hwi_system_info.hw_info,
								   ( int ) _papi_hwi_system_info.hw_info.
								   model );
	SUBDBG( "After _bgp_get_memory_info, retval=%d...\n", retval );
	if ( retval )
		return ( retval );

	/*
	 * Initialize BG/P global variables...
	 * NOTE:  If the BG/P SPI interface is to be used, then this
	 *        initialize routine must be called from each process for the
	 *        application.  It does not matter if this routine is called more
	 *        than once per process, but must be called by each process at
	 *        least once, preferably at the beginning of the application.
	 */
	SUBDBG( "Before BGP_UPC_Initialize()...\n" );
	BGP_UPC_Initialize(  );
	SUBDBG( "After BGP_UPC_Initialize()...\n" );

	return PAPI_OK;
}

/*
 * PAPI Shutdown Global
 *
 * Called once per process - nothing to do
 */
int
_bgp_shutdown_global( void )
{

	return PAPI_OK;
}

/*
 * Register Allocation
 *
 * Sets up the UPC configuration to monitor those events
 * as identified in the event set.
 */
int
_bgp_allocate_registers( EventSetInfo_t * ESI )
{
	int i, natNum;
	BGP_UPC_Event_Id_t xEventId;

	/*
	 * If an active UPC unit, return error
	 */
	if ( BGP_UPC_Check_Active(  ) ) {
		SUBDBG( "_bgp_allocate_registers:  UPC is active...\n" );
		return PAPI_ESYS;
	}

	/*
	 * If a counter mode of 1, return error
	 */
	if ( BGP_UPC_Get_Counter_Mode(  ) ) {
		SUBDBG( "_bgp_allocate_registers:  Inconsistent counter mode...\n" );
		return PAPI_ESYS;
	}

	/*
	 * Start monitoring the events...
	 */
	natNum = ESI->NativeCount;
//  printf("_bgp_allocate_registers:  natNum=%d\n", natNum);
	for ( i = 0; i < natNum; i++ ) {
		xEventId = get_bgp_native_event_id( ESI->NativeInfoArray[i].ni_event );
//    printf("_bgp_allocate_registers:  xEventId = %d\n", xEventId);
		if ( !BGP_UPC_Check_Active_Event( xEventId ) ) {
			// NOTE:  We do not have to start monitoring for elapsed time...  It is always being
			//        monitored at location 255...
			if ( ( xEventId % BGP_UPC_MAX_MONITORED_EVENTS ) != 255 ) {
				/*
				 * The event is not already being monitored by the UPC, start monitoring
				 * for the event.  This will automatically zero the counter and turn off any
				 * threshold value...
				 */
//        printf("_bgp_allocate_registers:  Event id %d not being monitored...\n", xEventId);
				if ( BGP_UPC_Monitor_Event( xEventId, BGP_UPC_CFG_EDGE_DEFAULT )
					 < 0 ) {
//          printf("_bgp_allocate_registers:  Monitor_Event failed...\n");
					return PAPI_ECMP;
				}
			}
                        /* here is if we are event 255 */ 
			else {

			}

		} else {
			/*
			 * The event is already being monitored by the UPC.  This is a normal
			 * case where the UPC is monitoring all events for a particular user
			 * mode.  We are in this leg because the PAPI event set has not yet
			 * started monitoring the event.  So, simply zero the counter and turn
			 * off any threshold value...
			 */
//      printf("_bgp_allocate_registers:  Event id %d is already being monitored...\n", xEventId);
			// NOTE:  Can't zero the counter or reset the threshold for the timestamp counter...
			if ( ESI->NativeInfoArray[i].ni_event != PNE_BGP_IC_TIMESTAMP ) {
				if ( BGP_UPC_Zero_Counter_Value( xEventId ) < 0 ) {
//          printf("_bgp_allocate_registers:  Zero_Counter failed...\n");
					return PAPI_ECMP;
				}
				if ( BGP_UPC_Set_Counter_Threshold_Value( xEventId, 0 ) < 0 ) {
//          printf("_bgp_allocate_registers:  Set_Counter_Threshold_Value failed...\n");
					return PAPI_ECMP;
				}
			}
		}
		ESI->NativeInfoArray[i].ni_position =
			xEventId % BGP_UPC_MAX_MONITORED_EVENTS;
//    printf("_bgp_allocate_registers:  ESI->NativeInfoArray[i].ni_position=%d\n", ESI->NativeInfoArray[i].ni_position);
	}

//  printf("_bgp_allocate_registers:  Exiting normally...\n");

	return PAPI_OK;
}

/*
 * Update Control State
 *
 * This function clears the current contents of the control
 * structure and updates it with whatever resources are allocated
 * for all the native events in the native info structure array.
 *
 * Since no BGP specific state is kept at the PAPI level, there is
 * nothing to update and we simply return.
 */
int
_bgp_update_control_state( hwd_control_state_t *ctl,
			   NativeInfo_t *native, int count,
			   hwd_context_t *ctx )
{

	return PAPI_OK;
}


/* Hack to get cycle count */
static long_long begin_cycles;

/*
 * PAPI Start
 *
 * Start UPC unit(s)
 */
int
_bgp_start( hwd_context_t * ctx, hwd_control_state_t * ctrlstate )
{
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );
        begin_cycles=_bgp_GetTimeBase();
	BGP_UPC_Start( BGP_UPC_NO_RESET_COUNTERS );
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
	return ( PAPI_OK );
}

/*
 * PAPI Stop
 *
 * Stop UPC unit(s)
 */
int
_bgp_stop( hwd_context_t * ctx, hwd_control_state_t * state )
{
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );
	BGP_UPC_Stop(  );
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
	return PAPI_OK;
}

/*
 * PAPI Read Counters
 *
 * Read the counters into local storage
 */
int
_bgp_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
		   long_long ** dp, int flags )
{
//  printf("_bgp_read:  this_state* = %p\n", this_state);
//  printf("_bgp_read:  (long_long*)&this_state->counters[0] = %p\n", (long_long*)&this_state->counters[0]);
//  printf("_bgp_read:  (long_long*)&this_state->counters[1] = %p\n", (long_long*)&this_state->counters[1]);

	
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );

	if ( BGP_UPC_Read_Counters
		 ( ( long_long * ) & ctl->counters[0],
		   BGP_UPC_MAXIMUM_LENGTH_READ_COUNTERS_ONLY,
		   BGP_UPC_READ_EXCLUSIVE ) < 0 ) {
		sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
		return PAPI_ECMP;
	}
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
        /* hack to emulate BGP_MISC_ELAPSED_TIME counter */
        ctl->counters[255]=_bgp_GetTimeBase()-begin_cycles;
	*dp = ( long_long * ) & ctl->counters[0];

//  printf("_bgp_read:  dp = %p\n", dp);
//  printf("_bgp_read:  *dp = %p\n", *dp);
//  printf("_bgp_read:  (*dp)[0]* = %p\n", &((*dp)[0]));
//  printf("_bgp_read:  (*dp)[1]* = %p\n", &((*dp)[1]));
//  printf("_bgp_read:  (*dp)[2]* = %p\n", &((*dp)[2]));
//  int i;
//  for (i=0; i<256; i++)
//    if ((*dp)[i])
//      printf("_bgp_read: i=%d, (*dp)[i]=%lld\n", i, (*dp)[i]);

	return PAPI_OK;
}

/*
 * PAPI Reset
 *
 * Zero the counter values
 */
int
_bgp_reset( hwd_context_t * ctx, hwd_control_state_t * ctrlstate )
{
// NOTE:  PAPI can reset the counters with the UPC running.  One way it happens
//        is with PAPI_accum.  In that case, stop and restart the UPC, resetting
//        the counters.
	sigset_t mask_set;
	sigset_t old_set;
	sigemptyset( &mask_set );
	sigaddset( &mask_set, SIGXCPU );
	sigprocmask( SIG_BLOCK, &mask_set, &old_set );
	if ( BGP_UPC_Check_Active(  ) ) {
		// printf("_bgp_reset:  BGP_UPC_Stop()\n");
		BGP_UPC_Stop(  );
		// printf("_bgp_reset:  BGP_UPC_Start(BGP_UPC_RESET_COUNTERS)\n");
		BGP_UPC_Start( BGP_UPC_RESET_COUNTERS );
	} else {
		// printf("_bgp_reset:  BGP_UPC_Zero_Counter_Values()\n");
		BGP_UPC_Zero_Counter_Values(  );
	}
	sigprocmask( SIG_UNBLOCK, &mask_set, NULL );
	return ( PAPI_OK );
}

/*
 * PAPI Shutdown
 *
 * This routine is for shutting down threads,
 * including the master thread.
 * Effectively a no-op, same as BG/L...
 */
int
_bgp_shutdown( hwd_context_t * ctx )
{

	return ( PAPI_OK );
}

/*
 * PAPI Write
 *
 * Write counter values
 * NOTE:  Could possible support, but signal error as BG/L does...
 */
int
_bgp_write( hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long * from )
{

	return PAPI_ECMP;
}

/*
 * Dispatch Timer
 *
 * Same as BG/L - simple return
 */
void
_bgp_dispatch_timer( int signal, hwd_siginfo_t * si, void *context )
{

	return;
}




void
user_signal_handler( int signum, hwd_siginfo_t * siginfo, void *mycontext )
{

	EventSetInfo_t *ESI;
	ThreadInfo_t *thread = NULL;
	int isHardware = 1;
	vptr_t pc;
	_papi_hwi_context_t ctx;
	BGP_UPC_Event_Id_t xEventId = 0;
//  int thresh;
	int event_index, i;
	long_long overflow_bit = 0;
	int64_t threshold;

	ctx.si = siginfo;
	ctx.ucontext = ( ucontext_t * ) mycontext;

	ucontext_t *context = ( ucontext_t * ) mycontext;
	pc = ( vptr_t ) context->uc_mcontext.regs->nip;
	thread = _papi_hwi_lookup_thread( 0 );
	//int cidx = (int) &thread;
	ESI = thread->running_eventset[0];
	//ESI = (EventSetInfo_t *) thread->running_eventset;

	if ( ESI == NULL ) {
		//printf("ESI is null\n");
		return;
	} else {
		BGP_UPC_Stop(  );
		//xEventId = get_bgp_native_event_id(ESI->NativeInfoArray[0].ni_event); //*ESI->overflow.EventIndex].ni_event);
		event_index = *ESI->overflow.EventIndex;
		//printf("event index %d\n", event_index);

		for ( i = 0; i <= event_index; i++ ) {
			xEventId =
				get_bgp_native_event_id( ESI->NativeInfoArray[i].ni_event );
			if ( BGP_UPC_Read_Counter( xEventId, 1 ) >=
				 BGP_UPC_Get_Counter_Threshold_Value( xEventId ) &&
				 BGP_UPC_Get_Counter_Threshold_Value( xEventId ) != 0 ) {
				break;
			}
		}
		overflow_bit ^= 1 << xEventId;
		//ESI->overflow.handler(ESI->EventSetIndex, pc, 0, (void *) &ctx); 
		_papi_hwi_dispatch_overflow_signal( ( void * ) &ctx, pc, &isHardware,
											overflow_bit, 0, &thread, 0 );
		//thresh = (int)(*ESI->overflow.threshold + BGP_UPC_Read_Counter_Value(xEventId, 1)); //(int)BGP_UPC_Get_Counter_Threshold_Value(xEventId));
		//printf("thresh %llu val %llu\n", (int64_t)*ESI->overflow.threshold, BGP_UPC_Read_Counter_Value(xEventId, 1));
		threshold =
			( int64_t ) * ESI->overflow.threshold +
			BGP_UPC_Read_Counter_Value( xEventId, 1 );
		//printf("threshold %llu\n", threshold);
		BGP_UPC_Set_Counter_Threshold_Value( xEventId, threshold );
		BGP_UPC_Start( 0 );
	}
}

/*
 * Set Overflow
 *
 * This is commented out in BG/L - need to explore and complete...
 * However, with true 64-bit counters in BG/P and all counters for PAPI
 * always starting from a true zero (we don't allow write...), the possibility
 * for overflow is remote at best...
 *
 * Commented out code is carry-over from BG/L...
 */
int
_bgp_set_overflow( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	int rc = 0;
	BGP_UPC_Event_Id_t xEventId;	   // = get_bgp_native_event_id(EventCode);
	xEventId =
		get_bgp_native_event_id( ESI->NativeInfoArray[EventIndex].ni_event );
	//rc = BGP_UPC_Monitor_Event(xEventId, BGP_UPC_CFG_LEVEL_HIGH);
	rc = BGP_UPC_Set_Counter_Threshold_Value( xEventId, threshold );

	//printf("setting up sigactioni %d\n", xEventId); //ESI->NativeInfoArray[EventIndex].ni_event);
	/*struct sigaction act;
	   act.sa_sigaction = user_signal_handler;
	   memset(&act.sa_mask, 0x0, sizeof(act.sa_mask));
	   act.sa_flags = SA_RESTART | SA_SIGINFO;
	   if (sigaction(SIGXCPU, &act, NULL) == -1) {
	   return (PAPI_ESYS);
	   } */

	struct sigaction new_action;
	sigemptyset( &new_action.sa_mask );
	new_action.sa_sigaction = ( void * ) user_signal_handler;
	new_action.sa_flags = SA_RESTART | SA_SIGINFO;
	sigaction( SIGXCPU, &new_action, NULL );


	return PAPI_OK;
}


/*
 * Set Profile
 *
 * Same as for BG/L, routine not used and returns error
 */
int
_bgp_set_profile( EventSetInfo_t * ESI, int EventIndex, int threshold )
{
	/* This function is not used and shouldn't be called. */

	return PAPI_ECMP;
}

/*
 * Stop Profiling
 *
 * Same as for BG/L...
 */
int
_bgp_stop_profiling( ThreadInfo_t * master, EventSetInfo_t * ESI )
{
	return PAPI_OK;
}

/*
 * PAPI Control
 *
 * Same as for BG/L - initialize the domain
 */
int
_bgp_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
//  extern int _bgp_set_domain(hwd_control_state_t * cntrl, int domain);

	switch ( code ) {
	case PAPI_DOMAIN:
	case PAPI_DEFDOM:
//    Simply return PAPI_OK, as no state is kept.
		return PAPI_OK;
	case PAPI_GRANUL:
	case PAPI_DEFGRN:
		return PAPI_ECMP;
	default:
		return PAPI_EINVAL;
	}
}

/*
 * Get Real Micro-seconds
 */
long long
_bgp_get_real_usec( void )
{
	/*
	 * NOTE:  _papi_hwi_system_info.hw_info.mhz is really a representation of unit of time per cycle.
	 *        On BG/P, it's value is 8.5e-4.  Therefore, to get cycles per sec, we have to multiply
	 *        by 1.0e12.  To then convert to usec, we have to divide by 1.0e-3.
	 */

//  SUBDBG("_bgp_get_real_usec:  _papi_hwi_system_info.hw_info.mhz=%e\n",(_papi_hwi_system_info.hw_info.mhz));
//  float x = (float)get_cycles();
//  float y = (_papi_hwi_system_info.hw_info.mhz)*(1.0e9);
//  SUBDBG("_bgp_get_real_usec: _papi_hwi_system_info.hw_info.mhz=%e, x=%e, y=%e, x/y=%e, (long long)(x/y) = %lld\n",
//         (_papi_hwi_system_info.hw_info.mhz), x, y, x/y, (long long)(x/y));
//  return (long long)(x/y);

	return ( ( long long ) ( ( ( float ) get_cycles(  ) ) /
	       ( ( _papi_hwi_system_info.hw_info.cpu_max_mhz ) ) ) );
}

/*
 * Get Real Cycles
 *
 * Same for BG/L, using native function...
 */
long long
_bgp_get_real_cycles( void )
{

	return ( get_cycles(  ) );
}

/*
 * Get Virtual Micro-seconds
 *
 * Same calc as for BG/L, returns real usec...
 */
long long
_bgp_get_virt_usec( void )
{

	return _bgp_get_real_usec(  );
}

/*
 * Get Virtual Cycles
 *
 * Same calc as for BG/L, returns real cycles...
 */
long long
_bgp_get_virt_cycles( void )
{

	return _bgp_get_real_cycles(  );
}

/*
 * Component setup and shutdown
 *
 * Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_bgp_init_component( int cidx )
{
	int retval;

	_bgp_vectors.cmp_info.CmpIdx = cidx;
	retval = _bgp_init_global(  );
	
	return ( retval );
}


/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/
/*
 * Native Code to Event Name
 *
 * Given a native event code, returns the short text label
 */
int
_bgp_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{

	char xNativeEventName[BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME];
	BGP_UPC_Event_Id_t xEventId = get_bgp_native_event_id( EventCode );
	/*
	 * NOTE:  We do not return the event name for a user mode 2 or 3 event...
	 */
	if ( ( int ) xEventId < 0 || ( int ) xEventId > 511 )
		return ( PAPI_ENOEVNT );

	if ( BGP_UPC_Get_Event_Name
		 ( xEventId, BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME,
		   xNativeEventName ) != BGP_UPC_SUCCESS )
		return ( PAPI_ENOEVNT );

	SUBDBG( "_bgp_ntv_code_to_name:  EventCode = %d\n, xEventName = %s\n",
			EventCode, xEventName );
	strncpy( name, "PNE_", len );
	strncat( name, xNativeEventName, len - strlen( name ) );
	return ( PAPI_OK );
}

/*
 * Native Code to Event Description
 *
 * Given a native event code, returns the longer native event description
 */
int
_bgp_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{

	char xNativeEventDesc[BGP_UPC_MAXIMUM_LENGTH_EVENT_DESCRIPTION];

	BGP_UPC_Event_Id_t xEventId = get_bgp_native_event_id( EventCode );
	/*
	 * NOTE:  We do not return the event name for a user mode 2 or 3 event...
	 */
	if ( ( int ) xEventId < 0 || ( int ) xEventId > 511 )
		return ( PAPI_ENOEVNT );
	else if ( BGP_UPC_Get_Event_Description
			  ( xEventId, BGP_UPC_MAXIMUM_LENGTH_EVENT_DESCRIPTION,
				xNativeEventDesc ) != BGP_UPC_SUCCESS )
		return ( PAPI_ENOEVNT );

	strncpy( name, xNativeEventDesc, len );
	return ( PAPI_OK );
}

/*
 * Native Code to Bit Configuration
 *
 * Given a native event code, assigns the native event's
 * information to a given pointer.
 * NOTE: The info must be COPIED to location addressed by
 *       the provided pointer, not just referenced!
 * NOTE: For BG/P, the bit configuration is not needed,
 *       as the native SPI is used to configure events.
 */
int
_bgp_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{

	return ( PAPI_OK );
}

/*
 * Native ENUM Events
 *
 * Given a native event code, looks for next MOESI bit if applicable.
 * If not, looks for the next event in the table if the next one exists.
 * If not, returns the proper error code.
 *
 * For BG/P, we simply we simply return the native event id to the
 * to the next logical non-reserved event id.
 *
 * We only support enumerating all or available events.
 */
int
_bgp_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	/*
	 * Check for a valid EventCode and we only process a modifier of 'all events'...
	 */
//  printf("_bgp_ntv_enum_events:  EventCode=%8.8x\n", *EventCode);
	if ( *EventCode < 0x40000000 || *EventCode > 0x400001FF ||
		 ( modifier != PAPI_ENUM_ALL && modifier != PAPI_PRESET_ENUM_AVAIL ) )
		return PAPI_ECMP;

	char xNativeEventName[BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME];
	BGP_UPC_RC_t xRC;

	// NOTE:  We turn off the PAPI_NATIVE bit here...
	int32_t xNativeEventId =
		( ( *EventCode ) & PAPI_NATIVE_AND_MASK ) + 0x00000001;
	while ( xNativeEventId <= 0x000001FF ) {
		xRC =
			BGP_UPC_Get_Event_Name( xNativeEventId,
									BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME,
									xNativeEventName );
//    printf("_bgp_ntv_enum_events:  xNativeEventId = %8.8x, xRC=%d\n", xNativeEventId, xRC);
		if ( ( xRC == BGP_UPC_SUCCESS ) && ( strlen( xNativeEventName ) > 0 ) ) {
//      printf("_bgp_ntv_enum_events:  len(xNativeEventName)=%d, xNativeEventName=%s\n", strlen(xNativeEventName), xNativeEventName);
			break;
		}
		xNativeEventId++;
	}

	if ( xNativeEventId > 0x000001FF )
		return ( PAPI_ENOEVNT );
	else {
		// NOTE:  We turn the PAPI_NATIVE bit back on here...
		*EventCode = xNativeEventId | PAPI_NATIVE_MASK;
		return ( PAPI_OK );
	}
}

int 
_papi_hwi_init_os(void) {

    struct utsname uname_buffer;

    uname(&uname_buffer);

    strncpy(_papi_os_info.name,uname_buffer.sysname,PAPI_MAX_STR_LEN);

    strncpy(_papi_os_info.version,uname_buffer.release,PAPI_MAX_STR_LEN);

    _papi_os_info.itimer_sig = PAPI_INT_MPX_SIGNAL;
    _papi_os_info.itimer_num = PAPI_INT_ITIMER;
    _papi_os_info.itimer_res_ns = 1;

    return PAPI_OK;
}

/*
 * PAPI Vector Table for BG/P
 */
papi_vector_t _bgp_vectors = {
	.cmp_info = {
             .name = "linux-bgp",
	     .short_name = "bgp",
	     .description = "BlueGene/P component",
	     .num_cntrs = BGP_UPC_MAX_MONITORED_EVENTS,
	     .num_mpx_cntrs = BGP_UPC_MAX_MONITORED_EVENTS,
	     .default_domain = PAPI_DOM_USER,
	     .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
	     .default_granularity = PAPI_GRN_THR,
	     .available_granularities = PAPI_GRN_THR,
	     .hardware_intr_sig = PAPI_INT_SIGNAL,
	     .hardware_intr = 1,
	     .fast_real_timer = 1,
	     .fast_virtual_timer = 0,
  },

  /* Sizes of framework-opaque component-private structures */
  .size = {
	     .context = sizeof ( hwd_context_t ),
	     .control_state = sizeof ( hwd_control_state_t ),
	     .reg_value = sizeof ( hwd_register_t ),
	     .reg_alloc = sizeof ( hwd_reg_alloc_t ),
  },
  /* Function pointers in this component */
  .dispatch_timer = _bgp_dispatch_timer,
  .start = _bgp_start,
  .stop = _bgp_stop,
  .read = _bgp_read,
  .reset = _bgp_reset,
  .write = _bgp_write,
  .stop_profiling = _bgp_stop_profiling,
  .init_component = _bgp_init_component,
  .init_thread = _bgp_init_thread,
  .init_control_state = _bgp_init_control_state,
  .update_control_state = _bgp_update_control_state,
  .ctl = _bgp_ctl,
  .set_overflow = _bgp_set_overflow,
  .set_profile = _bgp_set_profile,
  .set_domain = _bgp_set_domain,
  .ntv_enum_events = _bgp_ntv_enum_events,
  .ntv_code_to_name = _bgp_ntv_code_to_name,
  .ntv_code_to_descr = _bgp_ntv_code_to_descr,
  .ntv_code_to_bits = _bgp_ntv_code_to_bits,
  .allocate_registers = _bgp_allocate_registers,
  .shutdown_thread = _bgp_shutdown
};

papi_os_vector_t _papi_os_vector = {
	.get_memory_info = _bgp_get_memory_info,
	.get_dmem_info = _bgp_get_dmem_info,
	.get_real_cycles = _bgp_get_real_cycles,
	.get_real_usec = _bgp_get_real_usec,
	.get_virt_cycles = _bgp_get_virt_cycles,
	.get_virt_usec = _bgp_get_virt_usec,
	.get_system_info = _bgp_get_system_info,
};
