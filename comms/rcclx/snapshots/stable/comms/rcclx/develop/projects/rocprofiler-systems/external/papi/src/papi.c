/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
* @file:    papi.c
*
* @author:  Philip Mucci
*          mucci@cs.utk.edu
* @author    dan terpstra
*          terpstra@cs.utk.edu
* @author    Min Zhou
*          min@cs.utk.edu
* @author  Kevin London
*	   london@cs.utk.edu
* @author  Per Ekman
*          pek@pdc.kth.se
* @author  Frank Winkler
*          frank.winkler@icl.utk.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
*
* @brief Most of the low-level API is here.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_preset.h"

#include "cpus.h"
#include "extras.h"
#include "sw_multiplex.h"


/* simplified papi functions for event rates */

/* For dynamic linking to libpapi */
/* Weak symbol for pthread_once to avoid additional linking
 * against libpthread when not used. */
#pragma weak pthread_once

#define STOP      0
#define FLIP      1
#define FLOP      2
#define IPC       3
#define EPC       4

/** \internal 
 * This is stored per thread
 */
typedef struct _RateInfo
{
   int EventSet;                 /**< EventSet of the thread */
   int event_0;                  /**< first event of the eventset */
   short int running;            /**< STOP, FLIP, FLOP, IPC or EPC */
   long long last_real_time;     /**< Previous value of real time */
   long long last_proc_time;     /**< Previous value of processor time */
} RateInfo;

THREAD_LOCAL_STORAGE_KEYWORD RateInfo *_rate_state = NULL;
bool _papi_rate_initiated = false;

static void _internal_papi_init(void);
static void _internal_onetime_papi_init(void);
static int _start_new_rate_call(float *real_time, float *proc_time, int *events,
                         int num_events, long long *ins, float *rate);
static int _rate_calls( float *real_time, float *proc_time, int *events, 
                 long long *values, long long *ins, float *rate, int mode );
static int _internal_check_rate_state();


static void _internal_papi_init(void)
{
   /* This function is only called by the first thread! */
   int retval;

   /* check if user has already initialzed PAPI with thread support */
   if ( init_level != ( PAPI_LOW_LEVEL_INITED | PAPI_THREAD_LEVEL_INITED ) ) {
      if ( ( retval = PAPI_library_init(PAPI_VER_CURRENT) ) != PAPI_VER_CURRENT ) {
         fprintf( stderr, "PAPI Error: PAPI_library_init failed with return value %d.\n", retval);
      } else {
      
         if ((retval = PAPI_thread_init(_papi_gettid)) != PAPI_OK) {
            fprintf( stderr, "PAPI Error: PAPI_thread_init failed with return value %d.\n", retval);
            fprintf( stderr, "PAPI Error: PAPI could not be initiated!\n");
         } else {
            _papi_rate_initiated = true;
         }
      }
   } else {
      _papi_rate_initiated = true;
   }
}

static void _internal_onetime_papi_init(void)
{
   static pthread_once_t library_is_initialized = PTHREAD_ONCE_INIT;
   if ( pthread_once ) {
      /* we assume that this function was called from a parallel region */
      pthread_once(&library_is_initialized, _internal_papi_init);
      /* wait until first thread has finished */
      int i = 0;
      /* give it 5 seconds in case PAPI_thread_init crashes */
      while ( !_papi_rate_initiated && (i++) < 500000 )
         usleep(10);
   } else {
      /* we assume that this function was called from a serial application
       * that was not linked against libpthread */
      _internal_papi_init();
   }
}

static int
_internal_check_rate_state()
{
   /* check if PAPI is initialized for rate functions */
   if ( _papi_rate_initiated == false ) {
      _internal_onetime_papi_init();

      if ( _papi_rate_initiated == false )
         return ( PAPI_EINVAL );
   }

   if ( _rate_state== NULL ) {
      _rate_state= ( RateInfo* ) papi_malloc( sizeof ( RateInfo ) );
      if ( _rate_state== NULL )
         return ( PAPI_ENOMEM );
      
      memset( _rate_state, 0, sizeof ( RateInfo ) );
      _rate_state->running = STOP;
   }
   return ( PAPI_OK );
}

/** @class PAPI_flips_rate
  *	@brief Simplified call to get Mflips/s (floating point instruction rate), real and processor time. 
  *
  *	@par C Interface: 
  *	\#include <papi.h> @n
  *	int PAPI_flips_rate( int event, float *rtime, float *ptime, long long *flpins, float *mflips );
  *
  *   @param event
  *     one of the three presets PAPI_FP_INS, PAPI_VEC_SP or PAPI_VEC_DP 
  *   @param *rtime
  *		realtime since the latest call
  *	@param *ptime
  *		process time since the latest call
  *	@param *flpins
  *		floating point instructions since the latest call
  *	@param *mflips
  *		incremental (Mega) floating point instructions per seconds since the latest call
  *  
  *	@retval PAPI_EINVAL 
  *		The counters were already started by something other than PAPI_flips_rate().
  *	@retval PAPI_ENOEVNT 
  *		The floating point instructions event does not exist.
  *	@retval PAPI_ENOMEM 
  *		Insufficient memory to complete the operation. 
  *
  * The first call to PAPI_flips_rate() will initialize the PAPI interface, 
  * set up the counters to monitor the floating point instructions event and start the counters.
  *
  * Subsequent calls will read the counters and return real time, process time,
  * floating point instructions and the Mflip/s rate since the latest call to PAPI_flips_rate(). 
  *
  * PAPI_flips_rate() returns information related to floating point instructions using 
  * the floating point instructions event. This is intended to measure instruction rate through the 
  * floating point pipe with no massaging. Note that PAPI_flips_rate() is thread-safe and can
  * therefore be called by multiple threads.
  *
  * @see PAPI_flops_rate()
  * @see PAPI_ipc()
  * @see PAPI_epc()
 */
int
PAPI_flips_rate( int event, float *rtime, float *ptime, long long *flpins, float *mflips )
{
   int retval;

   /* check event first */
   if ( event == PAPI_FP_INS || event == PAPI_VEC_DP || event == PAPI_VEC_SP ) {

      int events[1] = {event};
      long long values = 0;

      if ( rtime == NULL || ptime == NULL ||
            flpins == NULL || mflips == NULL ) {
         return PAPI_EINVAL;
      }

      retval = _rate_calls( rtime, ptime, events,
            &values, flpins, mflips, FLIP );

      return ( retval );
   }
   return ( PAPI_ENOEVNT );
}

/** @class PAPI_flops_rate
  *	@brief Simplified call to get Mflops/s (floating point operation rate), real and processor time. 
  *
  *	@par C Interface: 
  *	\#include <papi.h> @n
  *	int PAPI_flops_rate ( int event, float *rtime, float *ptime, long long *flpops, float *mflops );
  *
  *   @param event
  *     one of the three presets PAPI_FP_OPS, PAPI_SP_OPS or PAPI_DP_OPS
  *   @param *rtime
  *		realtime since the latest call
  *	@param *ptime
  *		process time since the latest call
  *	@param *flpops
  *		floating point operations since the latest call
  *	@param *mflops
  *		incremental (Mega) floating point operations per seconds since the latest call
  * 
  *	@retval PAPI_EINVAL 
  *		The counters were already started by something other than PAPI_flops_rate().
  *	@retval PAPI_ENOEVNT 
  *		The floating point operations event does not exist.
  *	@retval PAPI_ENOMEM 
  *		Insufficient memory to complete the operation. 
  *
  * The first call to PAPI_flops_rate() will initialize the PAPI interface, 
  * set up the counters to monitor the floating point operations event and start the counters. 
  *
  * Subsequent calls will read the counters and return real time, process time,
  * floating point operations and the Mflop/s rate since the latest call to PAPI_flops_rate(). 
  *
  * PAPI_flops_rate() returns information related to theoretical floating point operations
  * rather than simple instructions. It uses the floating point operations event which attempts to 
  * 'correctly' account for, e.g., FMA undercounts and FP Store overcounts. Note that
  * PAPI_flops_rate() is thread-safe and can therefore be called by multiple threads.
  *
  * @see PAPI_flips_rate()
  * @see PAPI_ipc()
  * @see PAPI_epc()
  * @see PAPI_rate_stop()
 */
int
PAPI_flops_rate( int event, float *rtime, float *ptime, long long *flpops, float *mflops )
{
   int retval;

   /* check event first */
   if ( event == PAPI_FP_OPS || event == PAPI_SP_OPS || event == PAPI_DP_OPS ) {

      int events[1] = {event};
      long long values = 0;

      if ( rtime == NULL || ptime == NULL ||
            flpops == NULL || mflops == NULL ) {
         return PAPI_EINVAL;
      }

      retval = _rate_calls( rtime, ptime, events,
            &values, flpops, mflops, FLOP );

      return ( retval );
   }
   return ( PAPI_ENOEVNT );
}

/** @class PAPI_ipc
  *	@brief Simplified call to get instructions per cycle, real and processor time. 
  *
  *	@par C Interface: 
  *	\#include <papi.h> @n
  *	int PAPI_ipc( float *rtime, float *ptime, long long *ins, float *ipc );
  *
  * @param *rtime
  *		realtime since the latest call
  *	@param *ptime
  *		process time since the latest call
  *	@param *ins
  *		instructions since the latest call
  *	@param *ipc
  *		incremental instructions per cycle since the latest call
  * 
  *	@retval PAPI_EINVAL 
  *		The counters were already started by something other than PAPI_ipc().
  *	@retval PAPI_ENOEVNT 
  *		The events PAPI_TOT_INS and PAPI_TOT_CYC are not supported.
  *	@retval PAPI_ENOMEM 
  *		Insufficient memory to complete the operation. 
  *
  * The first call to PAPI_ipc() will initialize the PAPI interface, 
  * set up the counters to monitor PAPI_TOT_INS and PAPI_TOT_CYC events 
  * and start the counters. 
  *
  * Subsequent calls will read the counters and return real time, 
  * process time, instructions and the IPC rate since the latest call to PAPI_ipc().
  *
  * PAPI_ipc() should return a ratio greater than 1.0, indicating instruction level
  * parallelism within the chip. The larger this ratio the more effeciently the program
  * is running. Note that PAPI_ipc() is thread-safe and can therefore be called by multiple threads.
  *
  * @see PAPI_flips_rate()
  * @see PAPI_flops_rate()
  * @see PAPI_epc()
  * @see PAPI_rate_stop()
 */
int
PAPI_ipc( float *rtime, float *ptime, long long *ins, float *ipc )
{
   long long values[2] = { 0, 0 };
   int events[2] = {PAPI_TOT_INS, PAPI_TOT_CYC};
      int retval = 0;

   if ( rtime == NULL || ptime == NULL || ins == NULL || ipc == NULL )
      return PAPI_EINVAL;

   retval = _rate_calls( rtime, ptime, events, values, ins, ipc, IPC );
   return ( retval );
}

/** @class PAPI_epc
  *	@brief Simplified call to get arbitrary events per cycle, real and processor time. 
  *
  *	@par C Interface: 
  *	\#include <papi.h> @n
  *	int PAPI_epc( int event, float *rtime, float *ptime, long long *ref, long long *core, long long *evt, float *epc );
  *
  * @param event
  *		event code to be measured (0 defaults to PAPI_TOT_INS)
  * @param *rtime
  *		realtime since the latest call
  *	@param *ptime
  *		process time since the latest call
  *	@param *ref
  *		incremental reference clock cycles since the latest call
  *	@param *core
  *		incremental core clock cycles since the latest call
  *	@param *evt
  *		events since the latest call
  *	@param *epc
  *		incremental events per cycle since the latest call
  * 
  *	@retval PAPI_EINVAL 
  *		The counters were already started by something other than PAPI_epc().
  *	@retval PAPI_ENOEVNT 
  *		One of the requested events does not exist.
  *	@retval PAPI_ENOMEM 
  *		Insufficient memory to complete the operation. 
  *
  * The first call to PAPI_epc() will initialize the PAPI interface, 
  * set up the counters to monitor the user specified event, PAPI_TOT_CYC, 
  * and PAPI_REF_CYC (if it exists) and start the counters. 
  *
  * Subsequent calls will read the counters and return real time, 
  * process time, event counts, the core and reference cycle count and EPC rate
  * since the latest call to PAPI_epc(). 
  *
  * PAPI_epc() can provide a more detailed look at algorithm efficiency in light of clock
  * variability in modern cpus. MFLOPS is no longer an adequate description of peak
  * performance if clock rates can arbitrarily speed up or slow down. By allowing a
  * user specified event and reporting reference cycles, core cycles and real time,
  * PAPI_epc provides the information to compute an accurate effective clock rate, and
  * an accurate measure of computational throughput. Note that PAPI_epc() is thread-safe and can
  * therefore be called by multiple threads.
  *
  * @see PAPI_flips_rate()
  * @see PAPI_flops_rate()
  * @see PAPI_ipc()
  * @see PAPI_rate_stop()
 */
int
PAPI_epc( int event, float *rtime, float *ptime, long long *ref, long long *core, long long *evt, float *epc )
{
   long long values[3] = { 0, 0, 0 };
   int events[3] = {PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_REF_CYC};
   int retval = 0;

   if ( rtime == NULL || ptime == NULL || ref == NULL ||core == NULL || evt == NULL || epc == NULL )
      return PAPI_EINVAL;

   // if an event is provided, use it; otherwise use TOT_INS
   if (event != 0 ) events[0] = event;

   retval = _rate_calls( rtime, ptime, events, values, evt, epc, EPC );
   *ref = values[2];
   *core = values[1];
   return ( retval );
}

/** @class PAPI_rate_stop
  * @brief Stop a running event set of a rate function.
  *
  * @par C Interface: 
  * \#include <papi.h> @n
  * int PAPI_rate_stop();
  * 
  * @retval PAPI_ENOEVNT 
  * -- The EventSet is not started yet.
  * @retval PAPI_ENOMEM 
  * -- Insufficient memory to complete the operation. 
  *
  * PAPI_rate_stop stops a running event set of a rate function.
  *
  * @see PAPI_flips_rate()
  * @see PAPI_flops_rate()
  * @see PAPI_ipc()
  * @see PAPI_epc()
 */
int
PAPI_rate_stop()
{
   int retval;
   long long tmp_values[3];

   if ( _papi_rate_events_running == 1 ) {
      if ( _rate_state!= NULL ) {
         if ( _rate_state->running > STOP ) {
            retval = PAPI_stop( _rate_state->EventSet, tmp_values );
            if ( retval == PAPI_OK ) {
               PAPI_cleanup_eventset( _rate_state->EventSet );
               _rate_state->running = STOP;
            }
            _papi_rate_events_running = 0;
            return retval;
         }
      }
   }
   return ( PAPI_ENOEVNT );
}

static int
_start_new_rate_call(float *real_time, float *proc_time, int *events,
                     int num_events, long long *ins, float *rate)
{
   int retval;
   _rate_state->EventSet = -1;

   if ( ( retval = PAPI_create_eventset( &_rate_state->EventSet ) ) != PAPI_OK )
      return ( retval );
   
   if (( retval = PAPI_add_events( _rate_state->EventSet, events, num_events )) != PAPI_OK )
      return retval;

   /* remember the event for subsequent calls of PAPI_flips_rate and PAPI_flops_rate */
   _rate_state->event_0 = events[0];
   *real_time  = 0.0;
   *proc_time  = 0.0;
   *rate       = 0.0;
   *ins        = 0;

   _rate_state->last_real_time = PAPI_get_real_usec( );
   _rate_state->last_proc_time = PAPI_get_virt_usec( );

   if ( ( retval = PAPI_start( _rate_state->EventSet ) ) != PAPI_OK ) {
      return retval;
   }

   return ( PAPI_OK );
}

static int
_rate_calls( float *real_time, float *proc_time, int *events,
             long long *values, long long *ins, float *rate, int mode )
{

   // printf("_rate_calls event %d, mode %d\n", events[0], mode);

   long long rt, pt; // current elapsed real and process times in usec
   int num_events = 2;
   int retval = 0;

   /* if a high-level event set is running stop it */
   if ( _papi_hl_events_running == 1 ) {
      if ( ( retval = PAPI_hl_stop() ) != PAPI_OK )
         return ( retval );
   }

   if ( ( retval = _internal_check_rate_state() ) != PAPI_OK ) {
      return ( retval );
   }


   switch (mode) {
      case FLOP:
      case FLIP:
         if ( (retval = PAPI_query_event(events[0])) != PAPI_OK)
            return retval;
         num_events = 1;
         break;
      case IPC:
         break;
      case EPC:
         if ( (retval = PAPI_query_event(events[0])) != PAPI_OK)
            return retval;
         if ( (retval = PAPI_query_event(events[2])) == PAPI_OK)
            num_events = 3;
         break;
      default:
         return PAPI_EINVAL;
   }

   /* STOP means the first call of a rate function */
   if ( _rate_state->running == STOP ) {

      if ( ( retval = _start_new_rate_call(real_time, proc_time, events, num_events, ins, rate)) != PAPI_OK )
         return retval;
      _rate_state->running = mode;

   } else {
      // check last mode
      // printf("current mode: %d, last mode: %d\n", mode, _rate_state->running);
      // printf("current event: %d, last event: %d\n", events[0], _rate_state->event_0);

      if ( mode != _rate_state->running || events[0] != _rate_state->event_0 ) {
              
         long long tmp_values[3];
         retval = PAPI_stop( _rate_state->EventSet, tmp_values );
         if ( retval == PAPI_OK ) {
            PAPI_cleanup_eventset( _rate_state->EventSet );
         } else {
            return retval;
         }

         if ( ( retval = _start_new_rate_call(real_time, proc_time, events, num_events, ins, rate)) != PAPI_OK )
            return retval;
         _rate_state->running = mode;
         _papi_rate_events_running = 1;
         return ( PAPI_OK );
      }

      if ( ( retval = PAPI_stop( _rate_state->EventSet, values ) ) != PAPI_OK ) {
         _rate_state->running = STOP;
         return retval;
      }

      /* Read elapsed real and process times  */
      rt = PAPI_get_real_usec();
      pt = PAPI_get_virt_usec();

      /* Convert to seconds with multiplication because it is much faster */
      *real_time = ((float)( rt - _rate_state->last_real_time )) * .000001;
      *proc_time = ((float)( pt - _rate_state->last_proc_time )) * .000001;

      *ins = values[0];

      switch (mode) {
         case FLOP:
         case FLIP:
            /* Calculate MFLOP and MFLIP rates */
            if ( pt > 0 ) {
                  *rate = (float)values[0] / (pt - _rate_state->last_proc_time);
            } else *rate = 0;
            break;
         case IPC:
         case EPC:
            /* Calculate IPC */
            if (values[1]!=0) {
               *rate = (float) ((float)values[0] / (float) ( values[1]));
            }
            break;
         default:
            return PAPI_EINVAL;
      }
      _rate_state->last_real_time = rt;
      _rate_state->last_proc_time = pt;

      if ( ( retval = PAPI_start( _rate_state->EventSet ) ) != PAPI_OK ) {
         _rate_state->running = STOP;
         return retval;
      }
   }
   _papi_rate_events_running = 1;
   return PAPI_OK;
}


/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/


extern hwi_presets_t user_defined_events[PAPI_MAX_USER_EVENTS];
extern int user_defined_events_count;
extern int num_all_presets;
extern int _papi_hwi_start_idx[PAPI_NUM_COMP];
extern int first_comp_with_presets;
extern int first_comp_preset_idx;


#ifdef DEBUG
#define papi_return(a) do { \
	int b = a; \
	if (b != PAPI_OK) {\
		_papi_hwi_errno = b;\
	} \
	APIDBG("EXIT: return: %d\n", b);\
	return((_papi_hwi_debug_handler ? _papi_hwi_debug_handler(b) : b)); \
} while (0)
#else
#define papi_return(a) do { \
	int b = a; \
	if (b != PAPI_OK) {\
		_papi_hwi_errno = b;\
	} \
	APIDBG("EXIT: return: %d\n", b);\
	return(b);\
} while(0)
#endif


/*
#ifdef DEBUG
#define papi_return(a) return((_papi_hwi_debug_handler ? _papi_hwi_debug_handler(a) : a))
#else
#define papi_return(a) return(a)
#endif
*/

#ifdef DEBUG
int _papi_hwi_debug;
#endif


static int init_retval = DEADBEEF;

inline_static int
valid_component( int cidx )
{
	if ( _papi_hwi_invalid_cmp( cidx ) )
		return ( PAPI_ENOCMP );
	return ( cidx );
}

inline_static int
valid_ESI_component( EventSetInfo_t * ESI )
{
	return ( valid_component( ESI->CmpIdx ) );
}

/** @class	PAPI_thread_init
 *  @brief Initialize thread support in the PAPI library.
 *
 *	@param *id_fn 
 *		Pointer to a function that returns current thread ID. 
 *
 *	PAPI_thread_init initializes thread support in the PAPI library. 
 *	Applications that make no use of threads do not need to call this routine. 
 *	This function MUST return a UNIQUE thread ID for every new thread/LWP created. 
 *	The OpenMP call omp_get_thread_num() violates this rule, as the underlying 
 *	LWPs may have been killed off by the run-time system or by a call to omp_set_num_threads() . 
 *	In that case, it may still possible to use omp_get_thread_num() in 
 *	conjunction with PAPI_unregister_thread() when the OpenMP thread has finished. 
 *	However it is much better to use the underlying thread subsystem's call, 
 *	which is pthread_self() on Linux platforms. 
 *
 *	@code
if ( PAPI_thread_init(pthread_self) != PAPI_OK )
	exit(1);
 *	@endcode
 *
 *	@see PAPI_register_thread PAPI_unregister_thread PAPI_get_thr_specific PAPI_set_thr_specific PAPI_thread_id PAPI_list_threads
 */
int
PAPI_thread_init( unsigned long int ( *id_fn ) ( void ) )
{
	/* Thread support not implemented on Alpha/OSF because the OSF pfm
	 * counter device driver does not support per-thread counters.
	 * When this is updated, we can remove this if statement
	 */
	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );

	if ( ( init_level & PAPI_THREAD_LEVEL_INITED ) )
		papi_return( PAPI_OK );

	init_level |= PAPI_THREAD_LEVEL_INITED;
	papi_return( _papi_hwi_set_thread_id_fn( id_fn ) );
}

/** @class PAPI_thread_id
 *  @brief Get the thread identifier of the current thread.
 *
 *	@retval PAPI_EMISC 
 *		is returned if there are no threads registered.
 *	@retval -1 
 *		is returned if the thread id function returns an error. 
 *
 *	This function returns a valid thread identifier. 
 *	It calls the function registered with PAPI through a call to 
 *	PAPI_thread_init().
 *
 *	@code
unsigned long tid;

if ((tid = PAPI_thread_id()) == (unsigned long int)-1 )
	exit(1);

printf("Initial thread id is: %lu\n", tid );
 *	@endcode
 *	@see PAPI_thread_init
 */
unsigned long
PAPI_thread_id( void )
{
	if ( _papi_hwi_thread_id_fn != NULL )
		return ( ( *_papi_hwi_thread_id_fn ) (  ) );
	else
#ifdef DEBUG
	if ( _papi_hwi_debug_handler )
		return ( unsigned long ) _papi_hwi_debug_handler( PAPI_EMISC );
#endif
	return ( unsigned long ) PAPI_EMISC;
}

/* Thread Functions */

/* 
 * Notify PAPI that a thread has 'appeared'
 * We lookup the thread, if it does not exist we create it
 */

/** @class PAPI_register_thread
 *  @brief Notify PAPI that a thread has 'appeared'.
 *
 *  @par C Interface:
 *  \#include <papi.h> @n
 *  int PAPI_register_thread (void);
 *
 *  PAPI_register_thread() should be called when the user wants to force 
 *  PAPI to initialize a thread that PAPI has not seen before. 
 *
 *  Usually this is not necessary as PAPI implicitly detects the thread when 
 *  an eventset is created or other thread local PAPI functions are called. 
 *  However, it can be useful for debugging and performance enhancements 
 *  in the run-time systems of performance tools. 
 *
 *  @retval PAPI_ENOMEM 
 *	Space could not be allocated to store the new thread information.
 *  @retval PAPI_ESYS 
 *	A system or C library call failed inside PAPI, see the errno variable.
 *  @retval PAPI_ECMP 
 *	Hardware counters for this thread could not be initialized. 
 *
 *   @see PAPI_unregister_thread 
 *   @see PAPI_thread_id 
 *   @see PAPI_thread_init
 */
int
PAPI_register_thread( void )
{
	ThreadInfo_t *thread;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	papi_return( _papi_hwi_lookup_or_create_thread( &thread, 0 ) );
}

/* 
 * Notify PAPI that a thread has 'disappeared'
 * We lookup the thread, if it does not exist we return an error
 */
/** @class PAPI_unregister_thread
 *  @brief Notify PAPI that a thread has 'disappeared'.
 *
 *	@retval PAPI_ENOMEM 
 *		Space could not be allocated to store the new thread information.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ECMP 
 *		Hardware counters for this thread could not be initialized. 
 *
 *	PAPI_unregister_thread should be called when the user wants to shutdown 
 *	a particular thread and free the associated thread ID. 
 *	THIS IS IMPORTANT IF YOUR THREAD LIBRARY REUSES THE SAME THREAD ID FOR A NEW KERNEL LWP. 
 *	OpenMP does this. OpenMP parallel regions, if separated by a call to 
 *	omp_set_num_threads() will often kill off the underlying kernel LWPs and 
 *	then start new ones for the next region. 
 *	However, omp_get_thread_id() does not reflect this, as the thread IDs 
 *	for the new LWPs will be the same as the old LWPs. 
 *	PAPI needs to know that the underlying LWP has changed so it can set up 
 *	the counters for that new thread. 
 *	This is accomplished by calling this function. 
 */
int
PAPI_unregister_thread( void )
{
	ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );

	if ( thread )
		papi_return( _papi_hwi_shutdown_thread( thread, 0 ) );

	papi_return( PAPI_EMISC );
}

/** @class PAPI_list_threads
 *  @brief List the registered thread ids.
 *
 *  PAPI_list_threads() returns to the caller a list of all thread IDs 
 *  known to PAPI.
 *
 *  This call assumes an initialized PAPI library. 
 *
 * @par C Interface
 * \#include <papi.h> @n
 * int PAPI_list_threads(PAPI_thread_id_t *tids, int * number );
 *
 * @param[in,out] *tids
 *		-- A pointer to a preallocated array. 
 *		This may be NULL to only return a count of threads. 
 *		No more than *number codes will be stored in the array.
 * @param[in,out] *number
 *		-- An input and output parameter.  
 *              Input specifies the number of allocated elements in *tids 
 *              (if non-NULL) and output specifies the number of threads. 
 *
 * @retval PAPI_OK The call returned successfully.
 * @retval PAPI_EINVAL *number has an improper value
 *
 * @see PAPI_get_thr_specific 
 * @see PAPI_set_thr_specific 
 * @see PAPI_register_thread 
 * @see PAPI_unregister_thread 
 * @see PAPI_thread_init PAPI_thread_id
 *
 */
int
PAPI_list_threads( PAPI_thread_id_t *tids, int *number )
{
	PAPI_all_thr_spec_t tmp;
	int retval;

	/* If tids == NULL, then just count the threads, don't gather a list. */
	/* If tids != NULL, then we need the length of the tids array in num. */

	if ( ( number == NULL ) || ( tids && ( *number <= 0 ) ) )
		papi_return( PAPI_EINVAL );

	memset( &tmp, 0x0, sizeof ( tmp ) );

	/* data == NULL, since we don't want the thread specific pointers. */
	/* tids may be NULL, if the user doesn't want the thread IDs. */

	tmp.num = *number;
	tmp.id = tids;
	tmp.data = NULL;

	retval = _papi_hwi_gather_all_thrspec_data( 0, &tmp );
	if ( retval == PAPI_OK )
		*number = tmp.num;

	papi_return( retval );
}

/** @class PAPI_get_thr_specific
 *  @brief Retrieve a pointer to a thread specific data structure.
 *
 *	@par Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_get_thr_specific( int tag, void **ptr );
 *
 *	@param tag
 *		An identifier, the value of which is either PAPI_USR1_TLS or 
 *		PAPI_USR2_TLS. This identifier indicates which of several data 
 *		structures associated with this thread is to be accessed.
 *	@param ptr
 *		A pointer to the memory containing the data structure. 
 *
 *	@retval PAPI_OK
 *	@retval PAPI_EINVAL 
 *		The @em tag argument is out of range. 
 *
 *	In C, PAPI_get_thr_specific PAPI_get_thr_specific will retrieve the pointer from the array with index @em tag. 
 *	There are 2 user available locations and @em tag can be either 
 *	PAPI_USR1_TLS or PAPI_USR2_TLS. 
 *	The array mentioned above is managed by PAPI and allocated to each 
 *	thread which has called PAPI_thread_init. 
 *	There is no Fortran equivalent function. 
 *
 *	@par Example:
 *	@code
 int ret;
 RateInfo *state = NULL;
 ret = PAPI_thread_init(pthread_self);
 if (ret != PAPI_OK) handle_error(ret);
 
 // Do we have the thread specific data setup yet?

ret = PAPI_get_thr_specific(PAPI_USR1_TLS, (void *) &state);
if (ret != PAPI_OK || state == NULL) {
	state = (RateInfo *) malloc(sizeof(RateInfo));
	if (state == NULL) return (PAPI_ESYS);
	memset(state, 0, sizeof(RateInfo));
	state->EventSet = PAPI_NULL;
	ret = PAPI_create_eventset(&state->EventSet);
	if (ret != PAPI_OK) return (PAPI_ESYS);
	ret = PAPI_set_thr_specific(PAPI_USR1_TLS, state);
	if (ret != PAPI_OK) return (ret);
}
*	@endcode
*	@see PAPI_register_thread PAPI_thread_init PAPI_thread_id PAPI_set_thr_specific
*/
int
PAPI_get_thr_specific( int tag, void **ptr )
{
	ThreadInfo_t *thread;
	int doall = 0, retval = PAPI_OK;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	if ( tag & PAPI_TLS_ALL_THREADS ) {
		tag = tag ^ PAPI_TLS_ALL_THREADS;
		doall = 1;
	}
	if ( ( tag < 0 ) || ( tag > PAPI_TLS_NUM ) )
		papi_return( PAPI_EINVAL );

	if ( doall )
		papi_return( _papi_hwi_gather_all_thrspec_data
					 ( tag, ( PAPI_all_thr_spec_t * ) ptr ) );

	retval = _papi_hwi_lookup_or_create_thread( &thread, 0 );
	if ( retval == PAPI_OK )
		*ptr = thread->thread_storage[tag];
	else
		papi_return( retval );

	return ( PAPI_OK );
}

/** @class PAPI_set_thr_specific
 *  @brief Store a pointer to a thread specific data structure.
 *
 *	@par Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_set_thr_specific( int tag, void *ptr );
 *
 *	@param tag
 *		An identifier, the value of which is either PAPI_USR1_TLS or 
 *		PAPI_USR2_TLS. This identifier indicates which of several data 
 *		structures associated with this thread is to be accessed.
 *	@param ptr
 *		A pointer to the memory containing the data structure. 
 *
 *	@retval PAPI_OK
 *	@retval PAPI_EINVAL 
 *		The @em tag argument is out of range. 
 *
 *	In C, PAPI_set_thr_specific will save @em ptr into an array indexed by @em tag. 
 *	There are 2 user available locations and @em tag can be either 
 *	PAPI_USR1_TLS or PAPI_USR2_TLS. 
 *	The array mentioned above is managed by PAPI and allocated to each 
 *	thread which has called PAPI_thread_init. 
 *	There is no Fortran equivalent function. 
 *
 *	@par Example:
 *	@code
int ret;
RateInfo *state = NULL;
ret = PAPI_thread_init(pthread_self);
if (ret != PAPI_OK) handle_error(ret);
 
// Do we have the thread specific data setup yet?

ret = PAPI_get_thr_specific(PAPI_USR1_TLS, (void *) &state);
if (ret != PAPI_OK || state == NULL) {
	state = (RateInfo *) malloc(sizeof(RateInfo));
	if (state == NULL) return (PAPI_ESYS);
	memset(state, 0, sizeof(RateInfo));
	state->EventSet = PAPI_NULL;
	ret = PAPI_create_eventset(&state->EventSet);
	if (ret != PAPI_OK) return (PAPI_ESYS);
	ret = PAPI_set_thr_specific(PAPI_USR1_TLS, state);
	if (ret != PAPI_OK) return (ret);
}
 *	@endcode
 *	@see PAPI_register_thread PAPI_thread_init PAPI_thread_id PAPI_get_thr_specific
 */
int
PAPI_set_thr_specific( int tag, void *ptr )
{
	ThreadInfo_t *thread;
	int retval = PAPI_OK;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	if ( ( tag < 0 ) || ( tag > PAPI_NUM_TLS ) )
		papi_return( PAPI_EINVAL );

	retval = _papi_hwi_lookup_or_create_thread( &thread, 0 );
	if ( retval == PAPI_OK ) {
	   _papi_hwi_lock( THREADS_LOCK );
	   thread->thread_storage[tag] = ptr;
	   _papi_hwi_unlock( THREADS_LOCK );
	}
	else
		return ( retval );

	return ( PAPI_OK );
}


/** 	@class PAPI_library_init
 *	@brief initialize the PAPI library. 
 *	@param version 
 *		upon initialization, PAPI checks the argument against the internal 
 *		value of PAPI_VER_CURRENT when the library was compiled. 
 *		This guards against portability problems when updating the PAPI shared 
 *		libraries on your system. 
 *
 *	@retval PAPI_EINVAL 
 *		papi.h is different from the version used to compile the PAPI library.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ECMP 
 *		This component does not support the underlying hardware.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable. 
 *
 *	PAPI_library_init() initializes the PAPI library. 
 *	PAPI_is_initialized() check for initialization.
 *	It must be called before any low level PAPI functions can be used. 
 *	If your application is making use of threads PAPI_thread_init must also be 
 *	called prior to making any calls to the library other than PAPI_library_init() . 
 *	@par Examples:
 *	@code
 *		int retval;
 *		retval = PAPI_library_init(PAPI_VER_CURRENT);
 *		if (retval != PAPI_VER_CURRENT && retval > 0) {
 *			fprintf(stderr,"PAPI library version mismatch!\en");
 *			exit(1); }
 *		if (retval < 0)
 *			handle_error(retval);
 *		retval = PAPI_is_initialized();
 *		if (retval != PAPI_LOW_LEVEL_INITED)
 *			handle_error(retval)	
 *	@endcode
 *	@bug If you don't call this before using any of the low level PAPI calls, your application could core dump.
 *	@see PAPI_thread_init PAPI
 */
int
PAPI_library_init( int version )
{
    APIDBG( "Entry: version: %#x\n", version);

	int tmp = 0;

	/* This is a poor attempt at a lock. 
	   For 3.1 this should be replaced with a 
	   true UNIX semaphore. We cannot use PAPI
	   locks here because they are not initialized yet */
	static int _in_papi_library_init_cnt = 0;
#ifdef DEBUG
	char *var;
#endif
	_papi_hwi_init_errors();

	if ( version != PAPI_VER_CURRENT )
		papi_return( PAPI_EINVAL );

	++_in_papi_library_init_cnt;
	while ( _in_papi_library_init_cnt > 1 ) {
		PAPIERROR( "Multiple callers of PAPI_library_init" );
		sleep( 1 );
	}

	/* This checks to see if we have forked or called init more than once.
	   If we have forked, then we continue to init. If we have not forked, 
	   we check to see the status of initialization. */

	APIDBG( "Initializing library: current PID %d, old PID %d\n", 
                getpid(  ), _papi_hwi_system_info.pid );

	if ( _papi_hwi_system_info.pid == getpid(  ) ) {
		/* If the magic environment variable PAPI_ALLOW_STOLEN is set,
		   we call shutdown if PAPI has been initialized. This allows
		   tools that use LD_PRELOAD to run on applications that use PAPI.
		   In this circumstance, PAPI_ALLOW_STOLEN will be set to 'stolen'
		   so the tool can check for this case. */

		if ( getenv( "PAPI_ALLOW_STOLEN" ) ) {
			char buf[PAPI_HUGE_STR_LEN];
			if ( init_level != PAPI_NOT_INITED )
				PAPI_shutdown(  );
			sprintf( buf, "%s=%s", "PAPI_ALLOW_STOLEN", "stolen" );
			putenv( buf );
		}

		/* If the library has been successfully initialized *OR*
		   the library attempted initialization but failed. */

		else if ( ( init_level != PAPI_NOT_INITED ) ||
				  ( init_retval != DEADBEEF ) ) {
			_in_papi_library_init_cnt--;
			if ( init_retval < PAPI_OK )
				papi_return( init_retval );
			else
				return ( init_retval );
		}

		APIDBG( "system_info was initialized, but init did not succeed\n" );
	}
#ifdef DEBUG
	var = ( char * ) getenv( "PAPI_DEBUG" );
	_papi_hwi_debug = 0;

	if ( var != NULL ) {
		if ( strlen( var ) != 0 ) {
			if ( strstr( var, "SUBSTRATE" ) )
				_papi_hwi_debug |= DEBUG_SUBSTRATE;
			if ( strstr( var, "API" ) )
				_papi_hwi_debug |= DEBUG_API;
			if ( strstr( var, "INTERNAL" ) )
				_papi_hwi_debug |= DEBUG_INTERNAL;
			if ( strstr( var, "THREADS" ) )
				_papi_hwi_debug |= DEBUG_THREADS;
			if ( strstr( var, "MULTIPLEX" ) )
				_papi_hwi_debug |= DEBUG_MULTIPLEX;
			if ( strstr( var, "OVERFLOW" ) )
				_papi_hwi_debug |= DEBUG_OVERFLOW;
			if ( strstr( var, "PROFILE" ) )
				_papi_hwi_debug |= DEBUG_PROFILE;
			if ( strstr( var, "MEMORY" ) )
				_papi_hwi_debug |= DEBUG_MEMORY;
			if ( strstr( var, "LEAK" ) )
				_papi_hwi_debug |= DEBUG_LEAK;
			if ( strstr( var, "HIGHLEVEL" ) )
				_papi_hwi_debug |= DEBUG_HIGHLEVEL;
			if ( strstr( var, "ALL" ) )
				_papi_hwi_debug |= DEBUG_ALL;
		}

		if ( _papi_hwi_debug == 0 )
			_papi_hwi_debug |= DEBUG_API;
	}
#endif


	/* Initialize internal globals */
	if ( _papi_hwi_init_global_internal(  ) != PAPI_OK ) {
		_in_papi_library_init_cnt--;
		papi_return( PAPI_EINVAL );
	}

	/* Initialize OS */
	tmp = _papi_hwi_init_os();
	if ( tmp ) {
	   init_retval = tmp;
	   _papi_hwi_shutdown_global_internal(  );
	   _in_papi_library_init_cnt--;
	   papi_return( init_retval );
	}

	/* Initialize component globals EXCEPT for perf_event, perf_event_uncore.
    * To avoid race conditions, these components use the thread local storage
    * construct initialized by _papi_hwi_init_global_threads(), from within 
    * their init_component(). So these must have init_component() run AFTER
    * _papi_hwi_init_global_threads. Other components demand that init threads
    * run AFTER init_component(), which sets up globals they need.
    */

	tmp = _papi_hwi_init_global( 0 ); /* Selector 0 to skip perf_event, perf_event_uncore */
	if ( tmp ) {
		init_retval = tmp;
		_papi_hwi_shutdown_global_internal(  );
		_in_papi_library_init_cnt--;
		papi_return( init_retval );
	}
	
   /* Initialize thread globals, including the main threads  */

	tmp = _papi_hwi_init_global_threads(  );
	if ( tmp ) {
		init_retval = tmp;
		_papi_hwi_shutdown_global_internal(  );
   	_in_papi_library_init_cnt--;
		papi_return( init_retval );
	}

	/* Initialize perf_event, perf_event_uncore components */

	tmp = _papi_hwi_init_global( 1 ); /* Selector 1 for only perf_event, perf_event_uncore */
	if ( tmp ) {
		init_retval = tmp;
		_papi_hwi_shutdown_global_internal(  );
		_in_papi_library_init_cnt--;
		papi_return( init_retval );
	}


    /* Initialize component preset globals. */

    tmp = _papi_hwi_init_global_presets();
    if ( tmp ) {
        init_retval = tmp;
        _papi_hwi_shutdown_global_internal(  );
        _in_papi_library_init_cnt--;
        papi_return( init_retval );
    }
	
	init_level = PAPI_LOW_LEVEL_INITED;
	_in_papi_library_init_cnt--;

	return ( init_retval = PAPI_VER_CURRENT );
}

/** @class PAPI_query_event
 *  @brief Query if PAPI event exists.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_query_event(int EventCode);
 *
 * PAPI_query_event() asks the PAPI library if the PAPI Preset event can be 
 * counted on this architecture. 
 * If the event CAN be counted, the function returns PAPI_OK. 
 * If the event CANNOT be counted, the function returns an error code. 
 * This function also can be used to check the syntax of native and user events. 
 *
 * @param EventCode
 *    -- a defined event such as PAPI_TOT_INS. 
 *
 *  @retval PAPI_EINVAL 
 *	    One or more of the arguments is invalid.
 *  @retval PAPI_ENOEVNT 
 *	    The PAPI preset is not available on the underlying hardware. 
 *
 * @par Examples
 * @code
 * int retval;
 * // Initialize the library
 * retval = PAPI_library_init(PAPI_VER_CURRENT);
 * if (retval != PAPI_VER_CURRENT) {
 *   fprintf(stderr,\"PAPI library init error!\\n\");
 *   exit(1); 
 * }
 * if (PAPI_query_event(PAPI_TOT_INS) != PAPI_OK) {
 *   fprintf(stderr,\"No instruction counter? How lame.\\n\");
 *   exit(1);
 * }
 * @endcode
 *
 * @see PAPI_remove_event 
 * @see PAPI_remove_events 
 * @see PAPI_presets 
 * @see PAPI_native
 */
int
PAPI_query_event( int EventCode )
{
    APIDBG( "Entry: EventCode: %#x\n", EventCode);
	if ( IS_PRESET(EventCode) ) {
		EventCode &= PAPI_PRESET_AND_MASK;
		if ( EventCode < 0 || EventCode >= num_all_presets )
			papi_return( PAPI_ENOTPRESET );

        int preset_index = EventCode;
        int compIdx = get_preset_cmp(&preset_index);
        if( compIdx < 0 ) {
            return PAPI_ENOEVNT;
        }

		if ( _papi_hwi_comp_presets[compIdx][preset_index].count )
		        papi_return (PAPI_OK);
		else
			return PAPI_ENOEVNT;
	}

	if ( IS_NATIVE(EventCode) ) {
		papi_return( _papi_hwi_query_native_event
					 ( ( unsigned int ) EventCode ) );
	}

	if ( IS_USER_DEFINED(EventCode) ) {
	  EventCode &= PAPI_UE_AND_MASK;
	  if ( EventCode < 0 || EventCode >= PAPI_MAX_USER_EVENTS)
		  papi_return ( PAPI_ENOEVNT );

		if ( user_defined_events[EventCode].count )
			papi_return (PAPI_OK);
		else
			papi_return (PAPI_ENOEVNT);
	}

	papi_return( PAPI_ENOEVNT );
}

/** @class PAPI_query_named_event
 *  @brief Query if a named PAPI event exists.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_query_named_event(const char *EventName);
 *
 * PAPI_query_named_event() asks the PAPI library if the PAPI named event can be 
 * counted on this architecture. 
 * If the event CAN be counted, the function returns PAPI_OK. 
 * If the event CANNOT be counted, the function returns an error code. 
 * This function also can be used to check the syntax of native and user events. 
 *
 * @param EventName
 *    -- a defined event such as PAPI_TOT_INS. 
 *
 *  @retval PAPI_EINVAL 
 *	    One or more of the arguments is invalid.
 *  @retval PAPI_ENOEVNT 
 *	    The PAPI preset is not available on the underlying hardware. 
 *
 * @par Examples
 * @code
 * int retval;
 * // Initialize the library
 * retval = PAPI_library_init(PAPI_VER_CURRENT);
 * if (retval != PAPI_VER_CURRENT) {
 *   fprintf(stderr,\"PAPI library init error!\\n\");
 *   exit(1); 
 * }
 * if (PAPI_query_named_event("PAPI_TOT_INS") != PAPI_OK) {
 *   fprintf(stderr,\"No instruction counter? How lame.\\n\");
 *   exit(1);
 * }
 * @endcode
 *
 * @see PAPI_query_event 
 */
int
PAPI_query_named_event( const char *EventName )
{
	int ret, code;

	ret = PAPI_event_name_to_code( EventName, &code );
	if ( ret == PAPI_OK ) ret = PAPI_query_event( code );
	papi_return( ret);
}


/**	@class PAPI_get_component_info 
 *	@brief get information about a specific software component 
 *
 *	@param cidx
 *		Component index
 *
 *	This function returns a pointer to a structure containing detailed 
 *	information about a specific software component in the PAPI library. 
 *	This includes versioning information, preset and native event 
 *	information, and more. 
 *	For full details, see @ref PAPI_component_info_t. 
 *
 *	@par Examples:
 *	@code
 		const PAPI_component_info_t *cmpinfo = NULL;
 		if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
 		exit(1);
 		if ((cmpinfo = PAPI_get_component_info(0)) == NULL)
 		exit(1);
 		printf("This component supports %d Preset Events and %d Native events.\n",
		cmpinfo->num_preset_events, cmpinfo->num_native_events);
 *	@endcode
 *
 *	@see PAPI_get_executable_info
 *	@see PAPI_get_hardware_info
 *	@see PAPI_get_dmem_info
 *	@see PAPI_get_opt
 *	@see PAPI_component_info_t
 */
const PAPI_component_info_t *
PAPI_get_component_info( int cidx )
{
   APIDBG( "Entry: Component Index %d\n", cidx);
   if ( _papi_hwi_invalid_cmp( cidx ) )
      return ( NULL );
   else
      return ( &( _papi_hwd[cidx]->cmp_info ) );
}

/* PAPI_get_event_info:
   tests input EventCode and returns a filled in PAPI_event_info_t 
   structure containing descriptive strings and values for the 
   specified event. Handles both preset and native events by 
   calling either _papi_hwi_get_event_info or 
   _papi_hwi_get_native_event_info.
*/
/** @class PAPI_get_event_info
 *	@brief Get the event's name and description info.
 *
 *	@param EventCode
 *		event code (preset or native)
 *	@param info 
 *		structure with the event information @ref PAPI_event_info_t
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTPRESET 
 *		The PAPI preset mask was set, but the hardware event specified is 
 *		not a valid PAPI preset.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	This function fills the event information into a structure. 
 *	In Fortran, some fields of the structure are returned explicitly. 
 *	This function works with existing PAPI preset and native event codes. 
 *
 *	@see PAPI_event_name_to_code 
 */
int
PAPI_get_event_info( int EventCode, PAPI_event_info_t *info )
{
	APIDBG( "Entry: EventCode: 0x%x, info: %p\n", EventCode, info);
        int i;

	if ( info == NULL )
	   papi_return( PAPI_EINVAL );

	if ( IS_PRESET(EventCode) ) {
           i = EventCode & PAPI_PRESET_AND_MASK;
	   if ( i >= num_all_presets ) {
	      papi_return( PAPI_ENOTPRESET );
       }
	   papi_return( _papi_hwi_get_preset_event_info( EventCode, info ) );
	}

	if ( IS_NATIVE(EventCode) ) {
	   papi_return( _papi_hwi_get_native_event_info
			  ( ( unsigned int ) EventCode, info ) );
	}

	if ( IS_USER_DEFINED(EventCode) ) {
	   papi_return( _papi_hwi_get_user_event_info( EventCode, info ));
	}
	papi_return( PAPI_ENOTPRESET );
}


/** @class PAPI_event_code_to_name
 *	@brief Convert a numeric hardware event code to a name.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_event_code_to_name( int  EventCode, char * EventName );
 *
 *	PAPI_event_code_to_name is used to translate a 32-bit integer PAPI event 
 *	code into an ASCII PAPI event name. 
 *	Either Preset event codes or Native event codes can be passed to this routine. 
 *	Native event codes and names differ from platform to platform.
 *
 *	@param EventCode 
 *		The numeric code for the event. 
 *	@param *EventName
 *		A string containing the event name as listed in PAPI_presets or discussed in PAPI_native.
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTPRESET 
 *		The hardware event specified is not a valid PAPI preset.
 *	@retval PAPI_ENOEVNT 
 *		The hardware event is not available on the underlying hardware. 
 *
 *	@par Examples:
 *	@code
 *	int EventCode, EventSet = PAPI_NULL;
 *  int Event, number;
 *	char EventCodeStr[PAPI_MAX_STR_LEN];
 *	// Create the EventSet
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK )
 *	handle_error( 1 );
 *	number = 1;
 *	if ( PAPI_list_events( EventSet, &Event, &number ) != PAPI_OK )
 *	handle_error(1);
 *	// Convert integer code to name string
 *	if ( PAPI_event_code_to_name( Event, EventCodeStr ) != PAPI_OK )
 *	handle_error( 1 );
 *	printf( "Event Name: %s\n", EventCodeStr );
 *	@endcode
 *
 *	@see PAPI_event_name_to_code
 *	@see PAPI_remove_event
 *	@see PAPI_get_event_info
 *	@see PAPI_enum_event
 *	@see PAPI_add_event
 *	@see PAPI_presets
 *	@see PAPI_native
 */
int
PAPI_event_code_to_name( int EventCode, char *out )
{
	APIDBG( "Entry: EventCode: %#x, out: %p\n", EventCode, out);
	if ( out == NULL )
		papi_return( PAPI_EINVAL );

	if ( IS_PRESET(EventCode) ) {
		EventCode &= PAPI_PRESET_AND_MASK;
		if ( EventCode < 0 || EventCode >= num_all_presets )
			papi_return( PAPI_ENOTPRESET );

        int preset_index = EventCode;
        int compIdx = get_preset_cmp(&preset_index);
        if( compIdx < 0 ) {
            return PAPI_ENOEVNT;
        }
        if ( _papi_hwd[compIdx]->cmp_info.disabled == PAPI_EDELAY_INIT ) {
            int junk;
            _papi_hwd[compIdx]->ntv_enum_events(&junk, PAPI_ENUM_FIRST);
        }


		if (_papi_hwi_comp_presets[compIdx][preset_index].symbol == NULL )
			papi_return( PAPI_ENOTPRESET );

		strncpy( out, _papi_hwi_comp_presets[compIdx][preset_index].symbol, PAPI_MAX_STR_LEN-1 );
		out[PAPI_MAX_STR_LEN-1] = '\0';
		papi_return( PAPI_OK );
	}

	if ( IS_NATIVE(EventCode) ) {
		return ( _papi_hwi_native_code_to_name
				 ( ( unsigned int ) EventCode, out, PAPI_MAX_STR_LEN ) );
	}

	if ( IS_USER_DEFINED(EventCode) ) {
		EventCode &= PAPI_UE_AND_MASK;

		if ( EventCode < 0 || EventCode >= user_defined_events_count )
			papi_return( PAPI_ENOEVNT );

		if (user_defined_events[EventCode].symbol == NULL )
			papi_return( PAPI_ENOEVNT );

		strncpy( out, user_defined_events[EventCode].symbol, PAPI_MAX_STR_LEN-1);
		out[PAPI_MAX_STR_LEN-1] = '\0';
		papi_return( PAPI_OK );
	}

	papi_return( PAPI_ENOEVNT );
}

/** @class PAPI_event_name_to_code
 *	@brief Convert a name to a numeric hardware event code. 
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_event_name_to_code( const char * EventName, int * EventCode );
 *
 *	PAPI_event_name_to_code is used to translate an ASCII PAPI event name 
 *	into an integer PAPI event code. 
 *
 *	@param *EventCode 
 *		The numeric code for the event. 
 *	@param *EventName
 *		A string containing the event name as listed in PAPI_presets or discussed in PAPI_native.
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOTPRESET 
 *		The hardware event specified is not a valid PAPI preset.
 *	@retval PAPI_ENOINIT 
 *		The PAPI library has not been initialized.
 *	@retval PAPI_ENOEVNT 
 *		The hardware event is not available on the underlying hardware. 
 *
 *	@par Examples:
 *	@code
 *	int EventCode, EventSet = PAPI_NULL;
 *	// Convert to integer
 *	if ( PAPI_event_name_to_code( "PAPI_TOT_INS", &EventCode ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Create the EventSet
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_event( EventSet, EventCode ) != PAPI_OK )
 *	handle_error( 1 );
 *	@endcode
 *
 *	@see PAPI_event_code_to_name
 *	@see PAPI_remove_event
 *	@see PAPI_get_event_info
 *	@see PAPI_enum_event
 *	@see PAPI_add_event
 *	@see PAPI_add_named_event
 *	@see PAPI_presets
 *	@see PAPI_native
 */
int
PAPI_event_name_to_code( const char *in, int *out )
{
   APIDBG("Entry: in: %p, name: %s, out: %p\n", in, in, out);
	int i;

	if ( ( in == NULL ) || ( out == NULL ) )
		papi_return( PAPI_EINVAL );

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );

	/* All presets start with "PAPI_" so no need to */
	/* do an exhaustive search if that's not there  */
	if (strncmp(in, "PAPI_", 5) == 0) {

       /* Split event name into base name and qualifier. */
       int preset_idx = -1;
       char *evt_name_copy = strdup(in);
       if( NULL == evt_name_copy ) {
           PAPIERROR("Failed to allocate space for preset buffer.\n");
		   papi_return( PAPI_EINVAL );
       }

       char *evt_base_name = strtok(evt_name_copy, ":");
       if( NULL == evt_base_name ) {
           PAPIERROR("Failed to allocate space for base name of native event used in preset.\n");
		   papi_return( PAPI_EINVAL );
       }

       /* Since the preset could live inside of either the CPU or component preset list,
        * set the list pointer appropriately. */
       hwi_presets_t *_papi_hwi_list = NULL;

       /* Now check the component presets. */
       int cmpnt, breakFlag = 0;
       for(cmpnt = 0; cmpnt < PAPI_NUM_COMP; cmpnt++ ) {
           _papi_hwi_list = _papi_hwi_comp_presets[cmpnt];
           for(i = 0; i < _papi_hwi_max_presets[cmpnt]; i++ ) {
               if ( ( _papi_hwi_list[i].symbol )
                 && ( strcasecmp( _papi_hwi_list[i].symbol, evt_base_name ) == 0) ) {
                     *out = ( int ) ( (i + _papi_hwi_start_idx[cmpnt]) | PAPI_PRESET_MASK );

                     if ( _papi_hwd[cmpnt]->cmp_info.disabled == PAPI_EDELAY_INIT ) {
                         int junk;
                         _papi_hwd[cmpnt]->ntv_enum_events(&junk, PAPI_ENUM_FIRST);
                     }

                     preset_idx = i;
                     breakFlag = 1;
                     break;
               }
           }
           /* Checks whether preset was found. */
           if( breakFlag ) {
               break;
           }
       }

       free(evt_name_copy);

       /* User may have provided an invalid event name. */
       if( NULL != _papi_hwi_list ) {

           /* Keep track of all qualifiers provided by the user. */
           hwi_presets_t *prstPtr = &_papi_hwi_list[preset_idx];
           int status = overwrite_qualifiers(prstPtr, in, 1);
           if( status < 0 ) {
               papi_return( PAPI_ENOMEM );
           }

           status = construct_qualified_event(prstPtr);
           if( status < 0 ) {
               papi_return( status );
           }

           papi_return( PAPI_OK );
       }
    }

	// check to see if it is a user defined event
	for ( i=0; i < user_defined_events_count ; i++ ) {
		APIDBG("&user_defined_events[%d]: %p, user_defined_events[%d].symbol: %s, user_defined_events[%d].count: %d\n",
				i, &user_defined_events[i], i, user_defined_events[i].symbol, i, user_defined_events[i].count);
		if (user_defined_events[i].symbol == NULL)
			break;
		if (user_defined_events[i].count == 0)
			break;
		if ( strcasecmp( user_defined_events[i].symbol, in ) == 0 ) {
			*out = (int) ( i | PAPI_UE_MASK );
			papi_return( PAPI_OK );
		}
	}

	// go look for native events defined by one of the components
	papi_return( _papi_hwi_native_name_to_code( in, out ) );
}

/* Updates EventCode to next valid value, or returns error; 
  modifier can specify {all / available} for presets, or other values for native tables 
  and may be platform specific (Major groups / all mask bits; P / M / E chip, etc) */

/** @class PAPI_enum_event
 *	@brief Enumerate PAPI preset or native events.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_enum_event( int * EventCode, int  modifer );
 *
 *	Given a preset or native event code, PAPI_enum_event replaces the event 
 *	code with the next available event in either the preset or native table. 
 *	The modifier argument affects which events are returned. 
 *	For all platforms and event types, a value of PAPI_ENUM_ALL (zero) 
 *	directs the function to return all possible events. @n
 *
 *	For preset events, a TRUE (non-zero) value currently directs the function 
 *	to return event codes only for PAPI preset events available on this platform. 
 *	This may change in the future. 
 *	For native events, the effect of the modifier argument is different on each platform. 
 *	See the discussion below for platform-specific definitions.
 *
 *	@param *EventCode
 *		A defined preset or native event such as PAPI_TOT_INS.
 *	@param modifier 
 *		Modifies the search logic. See below for full list.
 *		For native events, each platform behaves differently. 
 *		See platform-specific documentation for details.
 *
 *	@retval PAPI_ENOEVNT 
 *		The next requested PAPI preset or native event is not available on 
 *		the underlying hardware.
 *
 *	@par Examples:
 *	@code
 *	// Scan for all supported native events on this platform
 *	printf( "Name\t\t\t       Code\t   Description\n" );
 *	do {
 *		retval = PAPI_get_event_info( i, &info );
 *		if ( retval == PAPI_OK ) {
 *		printf( "%-30s %#-10x\n%s\n", info.symbol, info.event_code, info.long_descr );
 *		}
 *	} while ( PAPI_enum_event( &i, PAPI_ENUM_ALL ) == PAPI_OK );
 *	@endcode
 *
 *      @par Generic Modifiers
 *	The following values are implemented for preset events
 *	<ul>
 *         <li> PAPI_ENUM_EVENTS -- Enumerate all (default)
 *         <li> PAPI_ENUM_FIRST -- Enumerate first event (preset or native)
 *                preset/native chosen based on type of EventCode
 *	</ul>
 *
 *      @par Native Modifiers
 *	The following values are implemented for native events
 *	<ul>
 *         <li>PAPI_NTV_ENUM_UMASKS -- Given an event, iterate through
 *                     possible umasks one at a time
 *         <li>PAPI_NTV_ENUM_UMASK_COMBOS -- Given an event, iterate
 *                     through all possible combinations of umasks.
 *                     This is not implemented on libpfm4.
 *	</ul>
 *
 *	@par Preset Modifiers
 *	The following values are implemented for preset events
 *	<ul>
 *         <li> PAPI_PRESET_ENUM_AVAIL -- enumerate only available presets
 *         <li> PAPI_PRESET_ENUM_CPU   -- enumerate CPU preset events
 *         <li> PAPI_PRESET_ENUM_CPU_AVAIL -- enumerate available CPU preset events
 *         <li> PAPI_PRESET_ENUM_FIRST_COMP -- enumerate first component preset event
 *	</ul>
 *
 *	@see PAPI @n
 *	PAPIF @n
 *      PAPI_enum_cmp_event @n
 *	PAPI_get_event_info @n
 *	PAPI_event_name_to_code @n
 *	PAPI_preset @n
 *	PAPI_native
 */
int
PAPI_enum_event( int *EventCode, int modifier )
{
	APIDBG( "Entry: EventCode: %#x, modifier: %d\n", *EventCode, modifier);
	int i = *EventCode;
	int retval;
	int cidx;
	int event_code;
	char *evt_name;

	cidx = _papi_hwi_component_index( *EventCode );
	if (cidx < 0) return PAPI_ENOCMP;

    /* check to see if a valid modifier is provided */
    if (modifier != PAPI_ENUM_EVENTS &&
        modifier != PAPI_ENUM_FIRST &&
        modifier != PAPI_ENUM_ALL &&
        modifier != PAPI_PRESET_ENUM_AVAIL && 
        modifier != PAPI_PRESET_ENUM_CPU && 
        modifier != PAPI_PRESET_ENUM_CPU_AVAIL && 
        modifier != PAPI_PRESET_ENUM_FIRST_COMP && 
        modifier != PAPI_NTV_ENUM_UMASKS && 
        modifier != PAPI_NTV_ENUM_UMASK_COMBOS)
        {
            return PAPI_EINVAL;
        }
		
    /* If it is a component preset, it will be in a separate array. */
    int preset_index;
    hwi_presets_t *_papi_hwi_list;

    if ( IS_PRESET(i) ) {

        /* Set to the first preset. */
        if ( modifier == PAPI_ENUM_FIRST ) {
            *EventCode = ( int ) PAPI_PRESET_MASK;
            APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
            return ( PAPI_OK );
        }

        i &= PAPI_PRESET_AND_MASK;

        /* Iterate over all or all available presets. */
        if ( modifier == PAPI_ENUM_EVENTS || modifier == PAPI_PRESET_ENUM_AVAIL ) {

            /* NULL pointer used to terminate the list. However, now we have
             * more presets that exist beyond the bounds of the original
             * array, so skip over the NULL entries. */
            do {
                cidx = _papi_hwi_component_index( (int)(i | PAPI_PRESET_MASK) );
                if (cidx < 0) return PAPI_ENOCMP;
                if ( _papi_hwd[cidx]->cmp_info.disabled == PAPI_EDELAY_INIT ) {
                    APIDBG("Triggered forced initialization of component ID=%d.\n", cidx);
                    int junk;
                    _papi_hwd[cidx]->ntv_enum_events(&junk, PAPI_ENUM_FIRST);
                }

                if ( ++i >= num_all_presets ) {
                    return ( PAPI_EINVAL );
                }

                /* Find the component to which the preset belongs and set the
                 * preset index relative to the component's presets' index range. */
                preset_index = i;
                int compIdx = get_preset_cmp(&preset_index);
                if( compIdx < 0 ) {
                    return ( PAPI_ENOEVNT );
                }

                _papi_hwi_list = _papi_hwi_comp_presets[compIdx];

            } while ( _papi_hwi_list[preset_index].symbol == NULL ||
                      (modifier == PAPI_PRESET_ENUM_AVAIL && _papi_hwi_list[preset_index].count == 0) );

            *EventCode = ( int ) ( i | PAPI_PRESET_MASK );
            APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
            return ( PAPI_OK );
        }

        /* Set to the first component preset. */
        if ( modifier == PAPI_PRESET_ENUM_FIRST_COMP ) {

            preset_index = get_first_cmp_preset_idx();
            if( preset_index < 0 ) {
                return ( PAPI_ENOEVNT );
            }

            if ( _papi_hwd[first_comp_with_presets]->cmp_info.disabled == PAPI_EDELAY_INIT ) {
                int junk;
                _papi_hwd[first_comp_with_presets]->ntv_enum_events(&junk, PAPI_ENUM_FIRST);
            }

            *EventCode = ( int ) ( preset_index | PAPI_PRESET_MASK );
            APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
            return ( PAPI_OK );
        }

        /* Iterate over CPU presets. */
        if ( modifier == PAPI_PRESET_ENUM_CPU || modifier == PAPI_PRESET_ENUM_CPU_AVAIL ) {

            while ( ++i < PAPI_MAX_PRESET_EVENTS ) {
                if ( _papi_hwi_presets[i].symbol == NULL ) {
                    APIDBG("EXIT: PAPI_ENOEVNT\n");
                    return ( PAPI_ENOEVNT );    /* NULL pointer terminates list */
                }
                if ( modifier == PAPI_PRESET_ENUM_CPU_AVAIL
                     && _papi_hwi_presets[i].count == 0 ) {
                    continue;
                }
                *EventCode = ( int ) ( i | PAPI_PRESET_MASK );
                APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
                return ( PAPI_OK );
            }
        }

        papi_return( PAPI_EINVAL );
    }

	if ( IS_NATIVE(i) ) {
	    // save event code so components can get it with call to: _papi_hwi_get_papi_event_code()
	    _papi_hwi_set_papi_event_code(*EventCode, 0);

		/* Should check against num native events here */

	    event_code=_papi_hwi_eventcode_to_native((int)*EventCode);
	    retval = _papi_hwd[cidx]->ntv_enum_events((unsigned int *)&event_code, modifier );

	    if (retval!=PAPI_OK) {
	       APIDBG("VMW: retval=%d\n",retval);
	       return PAPI_EINVAL;
	    }

	    evt_name = _papi_hwi_get_papi_event_string();
    	*EventCode = _papi_hwi_native_to_eventcode(cidx, event_code, -1, evt_name);
	    _papi_hwi_free_papi_event_string();

	    APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
	    return retval;
	}

	if ( IS_USER_DEFINED(i) ) {
		if (user_defined_events_count == 0) {
			APIDBG("EXIT: PAPI_ENOEVNT\n");
			return PAPI_ENOEVNT;
		}
		if ( modifier == PAPI_ENUM_FIRST ) {
			*EventCode = (int) (0 | PAPI_UE_MASK);
			APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
			return ( PAPI_OK );
		}

		i &= PAPI_UE_AND_MASK;
		++i;

		if ( i <= 0  ||  i >= user_defined_events_count ) {
			APIDBG("EXIT: PAPI_ENOEVNT\n");
			return ( PAPI_ENOEVNT );
		}

		// if next entry does not have an event name, we are done
		if (user_defined_events[i].symbol == NULL) {
			APIDBG("EXIT: PAPI_ENOEVNT\n");
			return ( PAPI_ENOEVNT );
		}

		// if next entry does not map to any other events, we are done
		if (user_defined_events[i].count == 0) {
			APIDBG("EXIT: PAPI_ENOEVNT\n");
			return ( PAPI_ENOEVNT );
		}

		*EventCode = (int) (i | PAPI_UE_MASK);
		APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
		return ( PAPI_OK );
	}

	papi_return( PAPI_EINVAL );
}


/** @class PAPI_enum_cmp_event
 *	@brief Enumerate PAPI preset or native events for a given component
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_enum_cmp_event( int *EventCode, int  modifer, int cidx );
 *
 *	Given an event code, PAPI_enum_event replaces the event 
 *	code with the next available event.
 * 
 *	The modifier argument affects which events are returned. 
 *	For all platforms and event types, a value of PAPI_ENUM_ALL (zero) 
 *	directs the function to return all possible events. @n
 *
 *	For native events, the effect of the modifier argument may be
 *      different on each platform. 
 *	See the discussion below for platform-specific definitions.
 *
 *	@param *EventCode
 *		A defined preset or native event such as PAPI_TOT_INS.
 *	@param modifier 
 *		Modifies the search logic. See below for full list.
 *		For native events, each platform behaves differently. 
 *		See platform-specific documentation for details.
 *
 *      @param cidx
 *              Specifies the component to search in 
 *
 *	@retval PAPI_ENOEVNT 
 *		The next requested PAPI preset or native event is not available on 
 *		the underlying hardware.
 *
 *	@par Examples:
 *	@code
 *	// Scan for all supported native events on the first component
 *	printf( "Name\t\t\t       Code\t   Description\n" );
 *	do {
 *		retval = PAPI_get_event_info( i, &info );
 *		if ( retval == PAPI_OK ) {
 *		printf( "%-30s %#-10x\n%s\n", info.symbol, info.event_code, info.long_descr );
 *		}
 *	} while ( PAPI_enum_cmp_event( &i, PAPI_ENUM_ALL, 0 ) == PAPI_OK );
 *	@endcode
 *
 *      @par Generic Modifiers
 *	The following values are implemented for preset events
 *	<ul>
 *         <li> PAPI_ENUM_EVENTS -- Enumerate all (default)
 *         <li> PAPI_ENUM_FIRST -- Enumerate first event (preset or native)
 *                preset/native chosen based on type of EventCode
 *	</ul>
 *
 *      @par Native Modifiers
 *	The following values are implemented for native events
 *	<ul>
 *         <li>PAPI_NTV_ENUM_UMASKS -- Given an event, iterate through
 *                     possible umasks one at a time
 *         <li>PAPI_NTV_ENUM_UMASK_COMBOS -- Given an event, iterate
 *                     through all possible combinations of umasks.
 *                     This is not implemented on libpfm4.
 *	</ul>
 *
 *	@par Preset Modifiers
 *	The following values are implemented for preset events
 *	<ul>
 *         <li> PAPI_PRESET_ENUM_AVAIL -- enumerate only available presets
 *         <li> PAPI_PRESET_ENUM_MSC   -- Miscellaneous preset events
 *         <li> PAPI_PRESET_ENUM_INS   -- Instruction related preset events
 *         <li> PAPI_PRESET_ENUM_IDL   -- Stalled or Idle preset events
 *         <li> PAPI_PRESET_ENUM_BR    -- Branch related preset events
 *         <li> PAPI_PRESET_ENUM_CND   -- Conditional preset events
 *         <li> PAPI_PRESET_ENUM_MEM   -- Memory related preset events
 *         <li> PAPI_PRESET_ENUM_CACH  -- Cache related preset events
 *         <li> PAPI_PRESET_ENUM_L1    -- L1 cache related preset events
 *         <li> PAPI_PRESET_ENUM_L2    -- L2 cache related preset events
 *         <li> PAPI_PRESET_ENUM_L3    -- L3 cache related preset events
 *         <li> PAPI_PRESET_ENUM_TLB   -- Translation Lookaside Buffer events
 *         <li> PAPI_PRESET_ENUM_FP    -- Floating Point related preset events
 *	</ul>
 *
 *	@par ITANIUM Modifiers
 *	The following values are implemented for modifier on Itanium: 
 *	<ul>
 *	   <li> PAPI_NTV_ENUM_IARR - Enumerate IAR (instruction address ranging) events 
 *	   <li> PAPI_NTV_ENUM_DARR - Enumerate DAR (data address ranging) events 
 *	   <li> PAPI_NTV_ENUM_OPCM - Enumerate OPC (opcode matching) events 
 *	   <li> PAPI_NTV_ENUM_IEAR - Enumerate IEAR (instr event address register) events 
 *	   <li> PAPI_NTV_ENUM_DEAR - Enumerate DEAR (data event address register) events
 *	</ul>
 *
 *	@par POWER Modifiers
 *	The following values are implemented for POWER
 *	<ul>
 *	   <li> PAPI_NTV_ENUM_GROUPS - Enumerate groups to which an event belongs
 *	</ul>
 *
 *	@see PAPI @n
 *	PAPIF @n
 *      PAPI_enum_event @n
 *	PAPI_get_event_info @n
 *	PAPI_event_name_to_code @n
 *	PAPI_preset @n
 *	PAPI_native
 */
int
PAPI_enum_cmp_event( int *EventCode, int modifier, int cidx )
{
	APIDBG( "Entry: EventCode: %#x, modifier: %d, cidx: %d\n", *EventCode, modifier, cidx);
	int i = *EventCode;
	int retval;
	int event_code;
	char *evt_name;

	if ( _papi_hwi_invalid_cmp(cidx) ) {
		return PAPI_ENOCMP;
	}

	if (_papi_hwd[cidx]->cmp_info.disabled &&
        _papi_hwd[cidx]->cmp_info.disabled != PAPI_EDELAY_INIT) {
	  return PAPI_ENOCMP;
	}

    if ( IS_PRESET(i) ) {

        if ( _papi_hwd[cidx]->cmp_info.disabled == PAPI_EDELAY_INIT ) {
            int junk;
            _papi_hwd[cidx]->ntv_enum_events(&junk, PAPI_ENUM_FIRST);
        }

        int preset_index;
        hwi_presets_t *_papi_hwi_list;

        /* Set to the first preset. */
        if ( modifier == PAPI_ENUM_FIRST ) {
            *EventCode = ( int ) ( _papi_hwi_start_idx[cidx] | PAPI_PRESET_MASK );
            APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
            return ( PAPI_OK );
        }

        i &= PAPI_PRESET_AND_MASK;

        /* Iterate over all or all available presets. */
        if ( modifier == PAPI_ENUM_EVENTS || modifier == PAPI_PRESET_ENUM_AVAIL ) {

            /* NULL pointer used to terminate the list. However, now we have
             * more presets that exist beyond the bounds of the original
             * array, so skip over the NULL entries. */
            do {
                if ( ++i >= _papi_hwi_start_idx[cidx] + _papi_hwi_max_presets[cidx] ) {
                    return ( PAPI_EINVAL );
                }

                /* Find the component to which the preset belongs. */
                _papi_hwi_list = _papi_hwi_comp_presets[cidx];
                preset_index = i - _papi_hwi_start_idx[cidx];

            } while ( _papi_hwi_list[preset_index].symbol == NULL ||
                      (modifier == PAPI_PRESET_ENUM_AVAIL && _papi_hwi_list[preset_index].count == 0) );

            *EventCode = ( int ) ( i | PAPI_PRESET_MASK );
            APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
            return ( PAPI_OK );
        }

        papi_return( PAPI_EINVAL );
    }

	if ( IS_NATIVE(i) ) {
	    // save event code so components can get it with call to: _papi_hwi_get_papi_event_code()
	    _papi_hwi_set_papi_event_code(*EventCode, 0);

		/* Should we check against num native events here? */
	    event_code=_papi_hwi_eventcode_to_native(*EventCode);
	    retval = _papi_hwd[cidx]->ntv_enum_events((unsigned int *)&event_code, modifier );

	    if (retval!=PAPI_OK) {
	       APIDBG("EXIT: PAPI_EINVAL retval=%d\n",retval);
	       return PAPI_EINVAL;
	    }

	    evt_name = _papi_hwi_get_papi_event_string();
	    *EventCode = _papi_hwi_native_to_eventcode(cidx, event_code, -1, evt_name);
	    _papi_hwi_free_papi_event_string();

	    APIDBG("EXIT: *EventCode: %#x\n", *EventCode);
	    return retval;
	} 

	papi_return( PAPI_EINVAL );
}

/** @class PAPI_create_eventset
 *	@brief Create a new empty PAPI EventSet.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	PAPI_create_eventset( int * EventSet );
 *
 *	PAPI_create_eventset creates a new EventSet pointed to by EventSet, 
 *	which must be initialized to PAPI_NULL before calling this routine. 
 *	The user may then add hardware events to the event set by calling 
 *	PAPI_add_event or similar routines.
 *
 *	@note PAPI-C uses a late binding model to bind EventSets to components. 
 *	When an EventSet is first created it is not bound to a component. 
 *	This will cause some API calls that modify EventSet options to fail. 
 *	An EventSet can be bound to a component explicitly by calling 
 *	PAPI_assign_eventset_component or implicitly by calling PAPI_add_event
 *	or similar routines. 
 *
 *	@param *EventSet
 *		Address of an integer location to store the new EventSet handle.
 *
 *	@exception PAPI_EINVAL 
 *		The argument handle has not been initialized to PAPI_NULL or the argument is a NULL pointer.
 *
 *	@exception PAPI_ENOMEM 
 *		Insufficient memory to complete the operation. 
 *
 *	@par Examples:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_event( EventSet, PAPI_TOT_INS)  != PAPI_OK )
 *	handle_error( 1 ); 
 *	@endcode
 *
 *	@see PAPI_add_event @n
 *	PAPI_assign_eventset_component @n
 *	PAPI_destroy_eventset @n
 *	PAPI_cleanup_eventset
  */
int
PAPI_create_eventset( int *EventSet )
{
   APIDBG("Entry: EventSet: %p\n", EventSet);

	ThreadInfo_t *master;
	int retval;

	if ( init_level == PAPI_NOT_INITED )
		papi_return( PAPI_ENOINIT );
	retval = _papi_hwi_lookup_or_create_thread( &master, 0 );
	if ( retval )
		papi_return( retval );

	papi_return( _papi_hwi_create_eventset( EventSet, master ) );
}

/** @class PAPI_assign_eventset_component
 *	@brief Assign a component index to an existing but empty EventSet.
 *	
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	PAPI_assign_eventset_component( int  EventSet, int  cidx );
 *
 *	@param EventSet 
 *		An integer identifier for an existing EventSet.
 *	@param cidx 
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *	@retval PAPI_ENOEVST 
 *		The EventSet doesn't exist.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation. 
 *
 *	PAPI_assign_eventset_component assigns a specific component index, 
 *	as specified by cidx, to a new EventSet identified by EventSet, as obtained 
 *	from PAPI_create_eventset. EventSets are ordinarily automatically bound 
 *	to components when the first event is added. This routine is useful to 
 *	explicitly bind an EventSet to a component before setting component related 
 *	options. 
 *
 *	@par Examples:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Bind our EventSet to the cpu component
 *	if ( PAPI_assign_eventset_component( EventSet, 0 ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Convert our EventSet to multiplexing
 *	if ( PAPI_set_multiplex( EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	@endcode
 *
 *	@see PAPI_set_opt @n
 *	PAPI_create_eventset @n
 *	PAPI_add_events @n
 *	PAPI_set_multiplex
 */
int
PAPI_assign_eventset_component( int EventSet, int cidx )
{
	EventSetInfo_t *ESI;
	int retval;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

/* validate cidx */
	retval = valid_component( cidx );
	if ( retval < 0 )
		papi_return( retval );

/* cowardly refuse to reassign eventsets */ 
	if ( ESI->CmpIdx >= 0 )
	  return PAPI_EINVAL;

	return ( _papi_hwi_assign_eventset( ESI, cidx ) );
}

/**	@class PAPI_get_eventset_component
 *	@brief return index for component an eventset is assigned to
 *
 *	@retval PAPI_ENOEVST
 *		eventset does not exist
 *	@retval PAPI_ENOCMP
 *		component is invalid or does not exist
 *	@retval positive value
 *		valid component index
 *	
 *	@param EventSet
 *              EventSet for which we want to know the component index
 *	@par Examples:
 *	@code
 		int cidx,eventcode;
 		cidx = PAPI_get_eventset_component(eventset);
 *	@endcode
 *	PAPI_get_eventset_component() returns the component an event
 *      belongs to.
 *	@see  PAPI_get_event_component
 */
int
PAPI_get_eventset_component( int EventSet)
{
	EventSetInfo_t *ESI;
	int retval;

/* validate eventset */
	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

/* check if a component has been assigned */ 
	if ( ESI->CmpIdx < 0 )
	  papi_return( PAPI_ENOCMP );

/* validate CmpIdx */
	retval = valid_component( ESI->CmpIdx );
	if ( retval < 0 )
		papi_return( retval );

/* return the index */
	return ( ESI->CmpIdx );
}


/**	@class PAPI_add_event
 *	@brief add PAPI preset or native hardware event to an event set
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_add_event( int  EventSet, int  EventCode );
 *
 *	PAPI_add_event adds one event to a PAPI Event Set. @n
 *	A hardware event can be either a PAPI preset or a native hardware event code.
 *	For a list of PAPI preset events, see PAPI_presets or run the avail test case
 *	in the PAPI distribution. PAPI presets can be passed to PAPI_query_event to see
 *	if they exist on the underlying architecture.
 *	For a list of native events available on current platform, run the papi_native_avail
 *	utility in the PAPI distribution. For the encoding of native events,
 *	see PAPI_event_name_to_code to learn how to generate native code for the
 *	supported native event on the underlying architecture.
 *
 *	@param EventSet
 *		An integer handle for a PAPI Event Set as created by PAPI_create_eventset.
 *	@param EventCode 
 *		A defined event such as PAPI_TOT_INS. 
 *
 *	@retval Positive-Integer
 *		The number of consecutive elements that succeeded before the error. 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other events 
 *		in the event set simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware.
 *	@retval PAPI_EBUG 
 *		Internal error, please send mail to the developers. 
 *  @retval PAPI_EMULPASS
 * 		Event exists, but cannot be counted due to multiple passes required by hardware.
 *
 *	@par Examples:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	unsigned int native = 0x0;
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add native event PM_CYC to EventSet
 *	if ( PAPI_event_name_to_code( "PM_CYC", &native ) != PAPI_OK )
 *	handle_error( 1 );
 *	if ( PAPI_add_event( EventSet, native ) != PAPI_OK )
 *	handle_error( 1 );
 *	@endcode
 *
 *	@bug
 *	The vector function should take a pointer to a length argument so a proper 
 *	return value can be set upon partial success.
 *
 *	@see PAPI_cleanup_eventset @n
 *	PAPI_destroy_eventset @n
 *	PAPI_event_code_to_name @n
 *	PAPI_remove_events @n
 *	PAPI_query_event @n
 *	PAPI_presets @n
 *	PAPI_native @n
 *	PAPI_remove_event
 */
int
PAPI_add_event( int EventSet, int EventCode )
{
   APIDBG("Entry: EventSet: %d, EventCode: %#x\n", EventSet, EventCode);
	EventSetInfo_t *ESI;

	/* Is the EventSet already in existence? */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* Check argument for validity */

	if ( ( ( EventCode & PAPI_PRESET_MASK ) == 0 ) &&
		 ( EventCode & PAPI_NATIVE_MASK ) == 0 )
		papi_return( PAPI_EINVAL );

	/* Of course, it must be stopped in order to modify it. */

	if ( ESI->state & PAPI_RUNNING )
		papi_return( PAPI_EISRUN );

	/* Now do the magic. */
	int retval = _papi_hwi_add_event( ESI, EventCode );
	papi_return( retval );
}

/**  @class PAPI_remove_event
 *   @brief removes a hardware event from a PAPI event set. 
 *
 *   A hardware event can be either a PAPI Preset or a native hardware 
 *   event code.  For a list of PAPI preset events, see PAPI_presets or 
 *   run the papi_avail utility in the PAPI distribution.  PAPI Presets 
 *   can be passed to PAPI_query_event to see if they exist on the 
 *   underlying architecture.  For a list of native events available on 
 *   the current platform, run papi_native_avail in the PAPI distribution. 
 *
 *   @par C Interface:
 *   \#include <papi.h> @n
 *   int PAPI_remove_event( int  EventSet, int  EventCode );
 *
 *   @param[in] EventSet
 *	   -- an integer handle for a PAPI event set as created 
 *            by PAPI_create_eventset
 *   @param[in] EventCode
 *	   -- a defined event such as PAPI_TOT_INS or a native event. 
 *
 *   @retval PAPI_OK 
 *		Everything worked.
 *   @retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *   @retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *   @retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *   @retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this 
 *              event and other events in the EventSet simultaneously.
 *   @retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *   @par Example:
 *   @code
 *   int EventSet = PAPI_NULL;
 *   int ret;
 *
 *   // Create an empty EventSet
 *   ret = PAPI_create_eventset(&EventSet);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Add Total Instructions Executed to our EventSet
 *   ret = PAPI_add_event(EventSet, PAPI_TOT_INS);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Start counting
 *   ret = PAPI_start(EventSet);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Stop counting, ignore values
 *   ret = PAPI_stop(EventSet, NULL);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Remove event
 *   ret = PAPI_remove_event(EventSet, PAPI_TOT_INS);
 *   if (ret != PAPI_OK) handle_error(ret);
 *   @endcode
 *
 *   @see PAPI_cleanup_eventset 
 *   @see PAPI_destroy_eventset
 *   @see PAPI_event_name_to_code 
 *   @see PAPI_presets 
 *   @see PAPI_add_event 
 *   @see PAPI_add_events
 */
int
PAPI_remove_event( int EventSet, int EventCode )
{
	APIDBG("Entry: EventSet: %d, EventCode: %#x\n", EventSet, EventCode);
	EventSetInfo_t *ESI;
	int i,retval;

	/* check for pre-existing ESI */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* Check argument for validity */

	if ( ( !IS_PRESET(EventCode) ) &&
		( !IS_NATIVE(EventCode) ) &&
		( !IS_USER_DEFINED(EventCode) ))
		papi_return( PAPI_EINVAL );

	/* Of course, it must be stopped in order to modify it. */

	if ( !( ESI->state & PAPI_STOPPED ) )
		papi_return( PAPI_EISRUN );

	/* if the state is PAPI_OVERFLOWING, you must first call
	   PAPI_overflow with threshold=0 to remove the overflow flag */

	/* Turn off the event that is overflowing */
	if ( ESI->state & PAPI_OVERFLOWING ) {
	   for ( i = 0; i < ESI->overflow.event_counter; i++ ) {
	       if ( ESI->overflow.EventCode[i] == EventCode ) {
		  retval = PAPI_overflow( EventSet, EventCode, 0, 0,
					  ESI->overflow.handler );
		  if (retval!=PAPI_OK) return retval;
		  break;
	       }
	   }
	}

	/* force the user to call PAPI_profil to clear the PAPI_PROFILING flag */
	if ( ESI->state & PAPI_PROFILING ) {
		for ( i = 0; i < ESI->profile.event_counter; i++ ) {
			if ( ESI->profile.EventCode[i] == EventCode ) {
				PAPI_sprofil( NULL, 0, EventSet, EventCode, 0, 0 );
				break;
			}
		}
	}

	/* Now do the magic. */

	papi_return( _papi_hwi_remove_event( ESI, EventCode ) );
}

/**	@class PAPI_add_named_event
 *	@brief add PAPI preset or native hardware event by name to an EventSet
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_add_named_event( int EventSet, const char *EventName );
 *
 *	PAPI_add_named_event adds one event to a PAPI EventSet. @n
 *	A hardware event can be either a PAPI preset or a native hardware event code.
 *	For a list of PAPI preset events, see PAPI_presets or run the avail test case
 *	in the PAPI distribution. PAPI presets can be passed to PAPI_query_event to see
 *	if they exist on the underlying architecture.
 *	For a list of native events available on current platform, run the papi_native_avail
 *	utility in the PAPI distribution.
 *
 *	@param EventSet
 *		An integer handle for a PAPI Event Set as created by PAPI_create_eventset.
 *	@param EventCode
 *		A defined event such as PAPI_TOT_INS.
 *
 *	@retval Positive-Integer
 *		The number of consecutive elements that succeeded before the error. 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOINIT 
 *		The PAPI library has not been initialized.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other events 
 *		in the event set simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware.
 *	@retval PAPI_EBUG 
 *		Internal error, please send mail to the developers. 
 *  @retval PAPI_EMULPASS
 * 		Event exists, but cannot be counted due to multiple passes required by hardware.
 *
 *	@par Examples:
 *	@code
 *  char EventName = "PAPI_TOT_INS";
 *	int EventSet = PAPI_NULL;
 *	unsigned int native = 0x0;
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_named_event( EventSet, EventName ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add native event PM_CYC to EventSet
 *	if ( PAPI_add_named_event( EventSet, "PM_CYC" ) != PAPI_OK )
 *	handle_error( 1 );
 *	@endcode
 *
 *	@bug
 *	The vector function should take a pointer to a length argument so a proper 
 *	return value can be set upon partial success.
 *
 *	@see PAPI_add_event @n
 *	PAPI_query_named_event @n
 *	PAPI_remove_named_event
 */
int
PAPI_add_named_event( int EventSet, const char *EventName )
{
	APIDBG("Entry: EventSet: %d, EventName: %s\n", EventSet, EventName);

	int ret, code;

	ret = PAPI_event_name_to_code( EventName, &code );
	if ( ret != PAPI_OK ) {
		APIDBG("EXIT: return: %d\n", ret);
		return ret;   // do not use papi_return here because if there was an error PAPI_event_name_to_code already reported it
	}

	ret = PAPI_add_event( EventSet, code );
	APIDBG("EXIT: return: %d\n", ret);
	return ret;   // do not use papi_return here because if there was an error PAPI_add_event already reported it
}

/**  @class PAPI_remove_named_event
 *   @brief removes a named hardware event from a PAPI event set. 
 *
 *   A hardware event can be either a PAPI Preset or a native hardware 
 *   event code.  For a list of PAPI preset events, see PAPI_presets or 
 *   run the papi_avail utility in the PAPI distribution.  PAPI Presets 
 *   can be passed to PAPI_query_event to see if they exist on the 
 *   underlying architecture.  For a list of native events available on 
 *   the current platform, run papi_native_avail in the PAPI distribution. 
 *
 *   @par C Interface:
 *   \#include <papi.h> @n
 *   int PAPI_remove_named_event( int  EventSet, const char *EventName );
 *
 *   @param[in] EventSet
 *	   -- an integer handle for a PAPI event set as created 
 *            by PAPI_create_eventset
 *   @param[in] EventName
 *	   -- a defined event such as PAPI_TOT_INS or a native event. 
 *
 *   @retval PAPI_OK 
 *		Everything worked.
 *   @retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOINIT 
 *		The PAPI library has not been initialized.
 *   @retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *   @retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *   @retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this 
 *              event and other events in the EventSet simultaneously.
 *   @retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *   @par Example:
 *   @code
 *   char EventName = "PAPI_TOT_INS";
 *   int EventSet = PAPI_NULL;
 *   int ret;
 *
 *   // Create an empty EventSet
 *   ret = PAPI_create_eventset(&EventSet);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Add Total Instructions Executed to our EventSet
 *   ret = PAPI_add_named_event(EventSet, EventName);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Start counting
 *   ret = PAPI_start(EventSet);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Stop counting, ignore values
 *   ret = PAPI_stop(EventSet, NULL);
 *   if (ret != PAPI_OK) handle_error(ret);
 *
 *   // Remove event
 *   ret = PAPI_remove_named_event(EventSet, EventName);
 *   if (ret != PAPI_OK) handle_error(ret);
 *   @endcode
 *
 *   @see PAPI_remove_event @n
 *	PAPI_query_named_event @n
 *	PAPI_add_named_event
 */
int
PAPI_remove_named_event( int EventSet, const char *EventName )
{
	APIDBG("Entry: EventSet: %d, EventName: %s\n", EventSet, EventName);
	int ret, code;

	ret = PAPI_event_name_to_code( EventName, &code );
	if ( ret == PAPI_OK ) ret = PAPI_remove_event( EventSet, code );
	papi_return( ret );

}

/** @class PAPI_destroy_eventset 
 *	@brief Empty and destroy an EventSet.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_destroy_eventset( int * EventSet );
 *
 * PAPI_destroy_eventset deallocates the memory associated with an empty PAPI EventSet.
 *
 *	@param *EventSet
 *		A pointer to the integer handle for a PAPI event set as created by PAPI_create_eventset.
 *		The value pointed to by EventSet is then set to PAPI_NULL on success. 
 *
 *	@retval PAPI_EINVAL
 *		One or more of the arguments is invalid.
 *		Attempting to destroy a non-empty event set or passing in a null pointer to be destroyed.
 *	@retval PAPI_ENOEVST
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN
 *		The EventSet is currently counting events.
 *	@retval PAPI_EBUG
 *		Internal error, send mail to <ptools-perfapi@icl.utk.edu> and complain.
 *
 *	@par Examples:
 *	@code
 *	// Free all memory and data structures, EventSet must be empty.
 *	if ( PAPI_destroy_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	@endcode
 *
 *	@bug
 *	If the user has set profile on an event with the call, then when destroying 
 *	the EventSet the memory allocated by will not be freed. 
 *	The user should turn off profiling on the Events before destroying the 
 *	EventSet to prevent this behavior.
 *
 *	@see PAPI_profil @n
 *	PAPI_create_eventset @n
 *	PAPI_add_event @n
 *	PAPI_stop
 */
int
PAPI_destroy_eventset( int *EventSet )
{
	APIDBG("Entry: EventSet: %p, *EventSet: %d\n", EventSet, *EventSet);

	EventSetInfo_t *ESI;

	/* check for pre-existing ESI */

	if ( EventSet == NULL )
		papi_return( PAPI_EINVAL );

	ESI = _papi_hwi_lookup_EventSet( *EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	if ( !( ESI->state & PAPI_STOPPED ) )
		papi_return( PAPI_EISRUN );

	if ( ESI->NumberOfEvents )
		papi_return( PAPI_EINVAL );

	_papi_hwi_remove_EventSet( ESI );
	*EventSet = PAPI_NULL;

	return PAPI_OK;
}

/* simply checks for valid EventSet, calls component start() call */
/** @class PAPI_start
 *	@brief Start counting hardware events in an event set.
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     int PAPI_start( int  EventSet );
 *
 *	@param EventSet
 *		-- an integer handle for a PAPI event set as created by PAPI_create_eventset
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL 
 *		-- One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		-- A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		-- The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		-- The EventSet is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		-- The underlying counter hardware can not count this event and other events 
 *		in the EventSet simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		-- The PAPI preset is not available on the underlying hardware. 
 *
 *	PAPI_start starts counting all of the hardware events contained in the previously defined EventSet. 
 *	All counters are implicitly set to zero before counting.
 *  Assumes an initialized PAPI library and a properly added event set. 
 *
 *  @par Example:
 *  @code
 *  int EventSet = PAPI_NULL;
 *  long long values[2];
 *  int ret;
 *  
 *  ret = PAPI_create_eventset(&EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  
 *  // Add Total Instructions Executed to our EventSet
 *  ret = PAPI_add_event(EventSet, PAPI_TOT_INS);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  
 *  // Start counting
 *  ret = PAPI_start(EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  poorly_tuned_function();
 *  ret = PAPI_stop(EventSet, values);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  printf("%lld\\n",values[0]);
 *  @endcode
 *
 *	@see  PAPI_create_eventset PAPI_add_event PAPI_stop
 */
int
PAPI_start( int EventSet )
{
	APIDBG("Entry: EventSet: %d\n", EventSet);

	int is_dirty=0;
	int i,retval;
	EventSetInfo_t *ESI;
	ThreadInfo_t *thread = NULL;
	CpuInfo_t *cpu = NULL;
	hwd_context_t *context;
	int cidx;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL ) {
	   papi_return( PAPI_ENOEVST );
	}

	APIDBG("EventSet: %p\n", ESI);

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 ) {
	   papi_return( cidx );
	}

	/* only one event set per thread  can be running at any time,    */
	/* so if another event set is running, the user must stop that   */
        /* event set explicitly */

	/* We used to check and not let multiple events be attached */
	/* to the same CPU, but this was unnecessary?               */

      	thread = ESI->master;
	cpu = ESI->CpuInfo;

	if ( thread->running_eventset[cidx] ) {
           APIDBG("Thread Running already (Only one active Eventset per component)\n");
	   papi_return( PAPI_EISRUN );
	}

	/* Check that there are added events */
	if ( ESI->NumberOfEvents < 1 ) {
	   papi_return( PAPI_EINVAL );
	}

	/* If multiplexing is enabled for this eventset,
	   call John May's code. */

	if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
	   retval = MPX_start( ESI->multiplex.mpx_evset );
	   if ( retval != PAPI_OK ) {
	      papi_return( retval );
	   }

	   /* Update the state of this EventSet */
	   ESI->state ^= PAPI_STOPPED;
	   ESI->state |= PAPI_RUNNING;

	   return PAPI_OK;
	}

	/* get the context we should use for this event set */
	context = _papi_hwi_get_context( ESI, &is_dirty );
	if (is_dirty) {
	   /* we need to reset the context state because it was last used   */
	   /* for some other event set and does not contain the information */
           /* for our events.                                               */
	   retval = _papi_hwd[ESI->CmpIdx]->update_control_state(
                                                        ESI->ctl_state,
							ESI->NativeInfoArray,
							ESI->NativeCount,
							context);
	   if ( retval != PAPI_OK ) {
	      papi_return( retval );
	   }

	   //update_control_state disturbs the overflow settings so set 
	   //it to initial values again
	   if ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) {
           	for( i = 0; i < ESI->overflow.event_counter; i++ ) {
	               	retval = _papi_hwd[ESI->CmpIdx]->set_overflow( ESI,
                                                                       ESI->overflow.EventIndex[i],
                                                                       ESI->overflow.threshold[i] );
                       if ( retval != PAPI_OK ) {
	                       	break;
             		}
          	}
          } 

	   /* now that the context contains this event sets information,    */
	   /* make sure the position array in the EventInfoArray is correct */

	   /* We have to do this because ->update_control_state() can */
	   /* in theory re-order the native events out from under us. */
	   _papi_hwi_map_events_to_native( ESI );

	}

	/* If overflowing is enabled, turn it on */
	if ( ( ESI->state & PAPI_OVERFLOWING ) &&
	     !( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) ) {
	   retval = _papi_hwi_start_signal( _papi_os_info.itimer_sig,
					    NEED_CONTEXT, cidx );
	   if ( retval != PAPI_OK ) {
	      papi_return( retval );
	   }

	   /* Update the state of this EventSet and thread */
	   /* before to avoid races                        */
	   ESI->state ^= PAPI_STOPPED;
	   ESI->state |= PAPI_RUNNING;
           /* can not be attached to thread or cpu if overflowing */
	   thread->running_eventset[cidx] = ESI;

	   retval = _papi_hwd[cidx]->start( context, ESI->ctl_state );
	   if ( retval != PAPI_OK ) {
	      _papi_hwi_stop_signal( _papi_os_info.itimer_sig );
	      ESI->state ^= PAPI_RUNNING;
	      ESI->state |= PAPI_STOPPED;
	      thread->running_eventset[cidx] = NULL;
	      papi_return( retval );
	   }

	   retval = _papi_hwi_start_timer( _papi_os_info.itimer_num,
					   _papi_os_info.itimer_sig,
					   _papi_os_info.itimer_ns );
	   if ( retval != PAPI_OK ) {
	      _papi_hwi_stop_signal( _papi_os_info.itimer_sig );
	      _papi_hwd[cidx]->stop( context, ESI->ctl_state );
	      ESI->state ^= PAPI_RUNNING;
	      ESI->state |= PAPI_STOPPED;
	      thread->running_eventset[cidx] = NULL;
	      papi_return( retval );
	   }
	} else {
	   /* Update the state of this EventSet and thread before */
	   /* to avoid races                                      */
	   ESI->state ^= PAPI_STOPPED;
	   ESI->state |= PAPI_RUNNING;

	   /* if not attached to cpu or another process */
	   if ( !(ESI->state & PAPI_CPU_ATTACHED) ) {
	      if ( !( ESI->state & PAPI_ATTACHED ) ) {
		 thread->running_eventset[cidx] = ESI;
	      }
	   } else {
	      cpu->running_eventset[cidx] = ESI;
	   }

	   retval = _papi_hwd[cidx]->start( context, ESI->ctl_state );
	   if ( retval != PAPI_OK ) {
	      _papi_hwd[cidx]->stop( context, ESI->ctl_state );
	      ESI->state ^= PAPI_RUNNING;
	      ESI->state |= PAPI_STOPPED;
	      if ( !(ESI->state & PAPI_CPU_ATTACHED) ) {
		 if ( !( ESI->state & PAPI_ATTACHED ) ) 
		    thread->running_eventset[cidx] = NULL;
	      } else {
		 cpu->running_eventset[cidx] = NULL;
	      }
	      papi_return( retval );
	   }
	}

	return retval;
}

/* checks for valid EventSet, calls component stop() function. */
/** @class PAPI_stop
 *	@brief Stop counting hardware events in an event set. 
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     int PAPI_stop( int  EventSet, long long * values );
 *
 *	@param EventSet
 *		-- an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param values
 *		-- an array to hold the counter values of the counting events 
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_ENOTRUN 
 *		The EventSet is currently not running.
 *
 *	PAPI_stop halts the counting of a previously defined event set and the 
 *	counter values contained in that EventSet are copied into the values array
 *	Assumes an initialized PAPI library and a properly added event set. 
 *
 *  @par Example:
 *  @code
 *  int EventSet = PAPI_NULL;
 *  long long values[2];
 *  int ret;
 *  
 *  ret = PAPI_create_eventset(&EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  
 *  // Add Total Instructions Executed to our EventSet
 *  ret = PAPI_add_event(EventSet, PAPI_TOT_INS);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  
 *  // Start counting
 *  ret = PAPI_start(EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  poorly_tuned_function();
 *  ret = PAPI_stop(EventSet, values);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  printf("%lld\\n",values[0]);
 *  @endcode
 *
 *	@see  PAPI_create_eventset PAPI_start
 */
int
PAPI_stop( int EventSet, long long *values )
{
   APIDBG("Entry: EventSet: %d, values: %p\n", EventSet, values);
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx, retval;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( !( ESI->state & PAPI_RUNNING ) )
		papi_return( PAPI_ENOTRUN );

	/* If multiplexing is enabled for this eventset, turn if off */

	if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		retval = MPX_stop( ESI->multiplex.mpx_evset, values );
		if ( retval != PAPI_OK )
			papi_return( retval );

		/* Update the state of this EventSet */

		ESI->state ^= PAPI_RUNNING;
		ESI->state |= PAPI_STOPPED;

		return ( PAPI_OK );
	}

	/* get the context we should use for this event set */
	context = _papi_hwi_get_context( ESI, NULL );
	/* Read the current counter values into the EventSet */
	retval = _papi_hwi_read( context, ESI, ESI->sw_stop );
	if ( retval != PAPI_OK )
		papi_return( retval );

	/* Remove the control bits from the active counter config. */
	retval = _papi_hwd[cidx]->stop( context, ESI->ctl_state );
	if ( retval != PAPI_OK )
		papi_return( retval );
	if ( values )
		memcpy( values, ESI->sw_stop,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );

	/* If kernel profiling is in use, flush and process the kernel buffer */

	if ( ESI->state & PAPI_PROFILING ) {
		if ( _papi_hwd[cidx]->cmp_info.kernel_profile &&
			 !( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
			retval = _papi_hwd[cidx]->stop_profiling( ESI->master, ESI );
			if ( retval < PAPI_OK )
				papi_return( retval );
		}
	}

	/* If overflowing is enabled, turn it off */

	if ( ESI->state & PAPI_OVERFLOWING ) {
		if ( !( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) ) {
			retval = _papi_hwi_stop_timer( _papi_os_info.itimer_num,
						       _papi_os_info.itimer_sig );
			if ( retval != PAPI_OK )
				papi_return( retval );
			_papi_hwi_stop_signal( _papi_os_info.itimer_sig );
		}
	}

	/* Update the state of this EventSet */

	ESI->state ^= PAPI_RUNNING;
	ESI->state |= PAPI_STOPPED;

	/* Update the running event set for this thread */
	if ( !(ESI->state & PAPI_CPU_ATTACHED) ) {
		if ( !( ESI->state & PAPI_ATTACHED ))
			ESI->master->running_eventset[cidx] = NULL;
	} else {
		ESI->CpuInfo->running_eventset[cidx] = NULL;
	}
	
#if defined(DEBUG)
	if ( _papi_hwi_debug & DEBUG_API ) {
		int i;
		for ( i = 0; i < ESI->NumberOfEvents; i++ ) {
			APIDBG( "PAPI_stop ESI->sw_stop[%d]:\t%llu\n", i, ESI->sw_stop[i] );
		}
	}
#endif

	return ( PAPI_OK );
}

/** @class PAPI_reset
 * @brief Reset the hardware event counts in an event set.
 *
 *	@par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_reset( int EventSet );
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset 
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist. 
 *  @details
 *	PAPI_reset() zeroes the values of the counters contained in EventSet. 
 *	This call assumes an initialized PAPI library and a properly added event set 
 *
 *	@par Example:
 *	@code
int EventSet = PAPI_NULL;
int Events[] = {PAPI_TOT_INS, PAPI_FP_OPS};
int ret;
 
// Create an empty EventSet
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Add two events to our EventSet
ret = PAPI_add_events(EventSet, Events, 2);
if (ret != PAPI_OK) handle_error(ret);

// Start counting
ret = PAPI_start(EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Stop counting, ignore values
ret = PAPI_stop(EventSet, NULL);
if (ret != PAPI_OK) handle_error(ret);

// reset the counters in this EventSet
ret = PAPI_reset(EventSet);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see PAPI_create_eventset
 */
int
PAPI_reset( int EventSet )
{
	APIDBG("Entry: EventSet: %d\n", EventSet);
	int retval = PAPI_OK;
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
			retval = MPX_reset( ESI->multiplex.mpx_evset );
		} else {
			/* If we're not the only one running, then just
			   read the current values into the ESI->start
			   array. This holds the starting value for counters
			   that are shared. */
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwd[cidx]->reset( context, ESI->ctl_state );
		}
	} else {
#ifdef __bgp__
		//  For BG/P, we always want to reset the 'real' hardware counters.  The counters
		//  can be controlled via multiple interfaces, and we need to ensure that the values
		//  are truly zero...
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( ESI, NULL );
		retval = _papi_hwd[cidx]->reset( context, ESI->ctl_state );
#endif
		memset( ESI->sw_stop, 0x00,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );
	}

	APIDBG( "EXIT: retval %d\n", retval );
	papi_return( retval );
}

/** @class PAPI_read
 *  @brief Read hardware counters from an event set.
 *	
 *  @par C Interface:
 *  \#include <papi.h> @n
 *  int PAPI_read(int  EventSet, long_long * values );
 *
 *  PAPI_read() copies the counters of the indicated event set into 
 *  the provided array. 
 *
 *  The counters continue counting after the read. 
 *
 *  Note the differences between PAPI_read() and PAPI_accum(), specifically
 *  that PAPI_accum() resets the values array to zero.
 *
 *  PAPI_read() assumes an initialized PAPI library and a properly added 
 *  event set. 
 *
 *  @param[in] EventSet
 *     -- an integer handle for a PAPI Event Set as created 
 *        by PAPI_create_eventset()
 *  @param[out] *values 
 *     -- an array to hold the counter values of the counting events 
 *
 *  @retval PAPI_EINVAL 
 *	    One or more of the arguments is invalid.
 *  @retval PAPI_ESYS 
 *	    A system or C library call failed inside PAPI, see the 
 *          errno variable.
 *  @retval PAPI_ENOEVST 
 *	    The event set specified does not exist. 
 *	
 * @par Examples
 * @code
 * do_100events();
 * if (PAPI_read(EventSet, values) != PAPI_OK)
 *    handle_error(1);
 * // values[0] now equals 100
 * do_100events();
 * if (PAPI_accum(EventSet, values) != PAPI_OK)
 *    handle_error(1);
 * // values[0] now equals 300
 * values[0] = -100;
 * do_100events();
 * if (PAPI_accum(EventSet, values) != PAPI_OK)
 *     handle_error(1);
 * // values[0] now equals 0 
 * @endcode
 *
 * @see PAPI_accum 
 * @see PAPI_start 
 * @see PAPI_stop 
 * @see PAPI_reset
 */
int
PAPI_read( int EventSet, long long *values )
{
	APIDBG( "Entry: EventSet: %d, values: %p\n", EventSet, values);
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx, retval = PAPI_OK;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		  retval = MPX_read( ESI->multiplex.mpx_evset, values, 0 );
		} else {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwi_read( context, ESI, values );
		}
		if ( retval != PAPI_OK )
			papi_return( retval );
	} else {
		memcpy( values, ESI->sw_stop,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );
	}

#if defined(DEBUG)
	if ( ISLEVEL( DEBUG_API ) ) {
		int i;
		for ( i = 0; i < ESI->NumberOfEvents; i++ ) {
			APIDBG( "PAPI_read values[%d]:\t%lld\n", i, values[i] );
		}
	}
#endif

	APIDBG( "PAPI_read returns %d\n", retval );
	return ( PAPI_OK );
}

/** @class PAPI_read_ts
 *  @brief Read hardware counters with a timestamp.
 *	
 *  @par C Interface:
 *  \#include <papi.h> @n
 *  int PAPI_read_ts(int EventSet, long long *values, long long *cycles );
 *
 *  PAPI_read_ts() copies the counters of the indicated event set into 
 *  the provided array.  It also places a real-time cycle timestamp 
 *  into the cycles array.
 *
 *  The counters continue counting after the read. 
 *
 *  PAPI_read_ts() assumes an initialized PAPI library and a properly added 
 *  event set. 
 *
 *  @param[in] EventSet
 *     -- an integer handle for a PAPI Event Set as created 
 *        by PAPI_create_eventset()
 *  @param[out] *values 
 *     -- an array to hold the counter values of the counting events 
 *  @param[out] *cycles
 *     -- an array to hold the timestamp values
 *
 *  @retval PAPI_EINVAL 
 *	    One or more of the arguments is invalid.
 *  @retval PAPI_ESYS 
 *	    A system or C library call failed inside PAPI, see the 
 *          errno variable.
 *  @retval PAPI_ENOEVST 
 *	    The event set specified does not exist. 
 *	
 * @par Examples
 * @code
 * @endcode
 *
 * @see PAPI_read 
 * @see PAPI_accum 
 * @see PAPI_start 
 * @see PAPI_stop 
 * @see PAPI_reset
 */
int
PAPI_read_ts( int EventSet, long long *values, long long *cycles )
{
	APIDBG( "Entry: EventSet: %d, values: %p, cycles: %p\n", EventSet, values, cycles);
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int cidx, retval = PAPI_OK;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		  retval = MPX_read( ESI->multiplex.mpx_evset, values, 0 );
		} else {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwi_read( context, ESI, values );
		}
		if ( retval != PAPI_OK )
			papi_return( retval );
	} else {
		memcpy( values, ESI->sw_stop,
				( size_t ) ESI->NumberOfEvents * sizeof ( long long ) );
	}

	*cycles = _papi_os_vector.get_real_cycles(  );

#if defined(DEBUG)
	if ( ISLEVEL( DEBUG_API ) ) {
		int i;
		for ( i = 0; i < ESI->NumberOfEvents; i++ ) {
			APIDBG( "PAPI_read values[%d]:\t%lld\n", i, values[i] );
		}
	}
#endif

	APIDBG( "PAPI_read_ts returns %d\n", retval );
	return PAPI_OK;
}

/**	@class PAPI_accum
 *	@brief Accumulate and reset counters in an EventSet.
 *	
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_accum( int  EventSet, long_long * values );
 *
 *	These calls assume an initialized PAPI library and a properly added event set. 
 *	PAPI_accum adds the counters of the indicated event set into the array values. 
 *	The counters are zeroed and continue counting after the operation.
 *	Note the differences between PAPI_read and PAPI_accum, specifically 
 *	that PAPI_accum resets the values array to zero. 
 *
 *	@param EventSet
 *		an integer handle for a PAPI Event Set 
 *		as created by PAPI_create_eventset
 *	@param *values 
 *		an array to hold the counter values of the counting events 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ESYS 
 *		A system or C library call failed inside PAPI, see the errno variable.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist. 
 *
 *	@par Examples:
 *	@code
 *	do_100events( );
 *	if ( PAPI_read( EventSet, values) != PAPI_OK )
 *	handle_error( 1 );
 *	// values[0] now equals 100
 *	do_100events( );
 *	if (PAPI_accum( EventSet, values ) != PAPI_OK )
 *	handle_error( 1 );
 *	// values[0] now equals 200
 *	values[0] = -100;
 *	do_100events( );
 *	if (PAPI_accum( EventSet, values ) != PAPI_OK )
 *	handle_error( 1 );
 *	// values[0] now equals 0
 *	@endcode
 *
 *	@see PAPIF_accum
 *	@see PAPI_start
 *	@see PAPI_set_opt
 *	@see PAPI_reset
 */
int
PAPI_accum( int EventSet, long long *values )
{
	APIDBG("Entry: EventSet: %d, values: %p\n", EventSet, values);
	EventSetInfo_t *ESI;
	hwd_context_t *context;
	int i, cidx, retval;
	long long a, b, c;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		  retval = MPX_read( ESI->multiplex.mpx_evset, ESI->sw_stop, 0 );
		} else {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwi_read( context, ESI, ESI->sw_stop );
		}
		if ( retval != PAPI_OK )
			papi_return( retval );
	}

	for ( i = 0; i < ESI->NumberOfEvents; i++ ) {
		a = ESI->sw_stop[i];
		b = values[i];
		c = a + b;
		values[i] = c;
	}

	papi_return( PAPI_reset( EventSet ) );
}

/** @class PAPI_write
 *	@brief Write counter values into counters.
 *
 *	@param EventSet 
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param *values
 *		an array to hold the counter values of the counting events 
 *
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_ECMP 
 *		PAPI_write() is not implemented for this architecture. 
 *      @retval PAPI_ESYS 
 *              The EventSet is currently counting events and 
 *		the component could not change the values of the 
 *              running counters.
 *
 *	PAPI_write() writes the counter values provided in the array values 
 *	into the event set EventSet. 
 *	The virtual counters managed by the PAPI library will be set to the values provided. 
 *	If the event set is running, an attempt will be made to write the values 
 *	to the running counters. 
 *	This operation is not permitted by all components and may result in a run-time error. 
 *
 *	@see PAPI_read
 */
int
PAPI_write( int EventSet, long long *values )
{
	APIDBG("Entry: EventSet: %d, values: %p\n", EventSet, values);

	int cidx, retval = PAPI_OK;
	EventSetInfo_t *ESI;
	hwd_context_t *context;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( values == NULL )
		papi_return( PAPI_EINVAL );

	if ( ESI->state & PAPI_RUNNING ) {
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( ESI, NULL );
		retval = _papi_hwd[cidx]->write( context, ESI->ctl_state, values );
		if ( retval != PAPI_OK )
			return ( retval );
	}

	memcpy( ESI->hw_start, values,
			( size_t ) _papi_hwd[cidx]->cmp_info.num_cntrs *
			sizeof ( long long ) );

	return ( retval );
}

/** @class PAPI_cleanup_eventset
 *	@brief Empty and destroy an EventSet.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_cleanup_eventset( int  EventSet );
 *
 * PAPI_cleanup_eventset removes all events from a PAPI event set and turns 
 * off profiling and overflow for all events in the EventSet.
 * This can not be called if the EventSet is not stopped.
 *
 *	@param EventSet
 *		An integer handle for a PAPI event set as created by PAPI_create_eventset.
 *
 *	@retval PAPI_EINVAL
 *		One or more of the arguments is invalid.
 *		Attempting to destroy a non-empty event set or passing in a null pointer to be destroyed.
 *	@retval PAPI_ENOEVST
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN
 *		The EventSet is currently counting events.
 *	@retval PAPI_EBUG
 *		Internal error, send mail to <ptools-perfapi@icl.utk.edu> and complain.
 *
 *	@par Examples:
 *	@code
 *	// Remove all events in the eventset
 *	if ( PAPI_cleanup_eventset( EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	@endcode
 *
 *	@bug
 *	If the user has set profile on an event with the call, then when destroying 
 *	the EventSet the memory allocated by will not be freed. 
 *	The user should turn off profiling on the Events before destroying the 
 *	EventSet to prevent this behavior.
 *
 *	@see PAPI_profil @n
 *	PAPI_create_eventset @n
 *	PAPI_add_event @n
 *	PAPI_stop
 */
int
PAPI_cleanup_eventset( int EventSet )
{
	APIDBG("Entry: EventSet: %d\n",EventSet);

	EventSetInfo_t *ESI;
	int i, cidx, total, retval;

	/* Is the EventSet already in existence? */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* if the eventset has no index and no events, return OK
	   otherwise return NOCMP */
	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 ) {
		if ( ESI->NumberOfEvents )
			papi_return( cidx );
		papi_return( PAPI_OK );
	}

	/* Of course, it must be stopped in order to modify it. */

	if ( ESI->state & PAPI_RUNNING )
		papi_return( PAPI_EISRUN );

	/* clear overflow flag and turn off hardware overflow handler */
	if ( ESI->state & PAPI_OVERFLOWING ) {
		total = ESI->overflow.event_counter;
		for ( i = 0; i < total; i++ ) {
			retval = PAPI_overflow( EventSet,
									ESI->overflow.EventCode[0], 0, 0, NULL );
			if ( retval != PAPI_OK )
				papi_return( retval );
		}
	}

	/* clear profile flag and turn off hardware profile handler */
	if ( ( ESI->state & PAPI_PROFILING ) &&
		 _papi_hwd[cidx]->cmp_info.hardware_intr &&
		 !( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
		total = ESI->profile.event_counter;
		for ( i = 0; i < total; i++ ) {
			retval =
				PAPI_sprofil( NULL, 0, EventSet, ESI->profile.EventCode[0], 0,
							  PAPI_PROFIL_POSIX );
			if ( retval != PAPI_OK )
				papi_return( retval );
		}
	}

	if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		retval = MPX_cleanup( &ESI->multiplex.mpx_evset );
		if ( retval != PAPI_OK )
			papi_return( retval );
	}

	retval = _papi_hwd[cidx]->cleanup_eventset( ESI->ctl_state );
	if ( retval != PAPI_OK ) 
		papi_return( retval );

	/* Now do the magic */
	papi_return( _papi_hwi_cleanup_eventset( ESI ) );
}

/**	@class PAPI_multiplex_init
 *	@brief Initialize multiplex support in the PAPI library.
 *
 *	PAPI_multiplex_init() enables and initializes multiplex support in 
 *      the PAPI library. 
 *	Multiplexing allows a user to count more events than total physical 
 *      counters by time sharing the existing counters at some loss in 
 *      precision. 
 *	Applications that make no use of multiplexing do not need to call 
 *      this routine. 
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_multiplex_init (void);
 *
 * @par Examples
 * @code
 * retval = PAPI_multiplex_init();
 * @endcode

 * @retval PAPI_OK This call always returns PAPI_OK
 *
 * @see PAPI_set_multiplex 
 * @see PAPI_get_multiplex
 */
int
PAPI_multiplex_init( void )
{
	APIDBG("Entry:\n");

	int retval;

	retval = mpx_init( _papi_os_info.itimer_ns );
	papi_return( retval );
}

/** @class PAPI_state
 * @brief Return the counting state of an EventSet.
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     int PAPI_state( int  EventSet, int * status );
 *
 *	@param EventSet -- an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param status -- an integer containing a boolean combination of one or more of the 
 *	following nonzero constants as defined in the PAPI header file papi.h:
 *	@arg PAPI_STOPPED	-- EventSet is stopped
 *	@arg PAPI_RUNNING	-- EventSet is running
 *	@arg PAPI_PAUSED	-- EventSet temporarily disabled by the library
 *	@arg PAPI_NOT_INIT	-- EventSet defined, but not initialized
 *	@arg PAPI_OVERFLOWING	-- EventSet has overflowing enabled
 *	@arg PAPI_PROFILING	-- EventSet has profiling enabled
 *	@arg PAPI_MULTIPLEXING	-- EventSet has multiplexing enabled
 *	@arg PAPI_ACCUMULATING	-- reserved for future use
 *	@arg PAPI_HWPROFILING	-- reserved for future use 
 *  @manonly
 *  @endmanonly
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist. 
 *  @manonly
 *  @endmanonly
 *
 *	PAPI_state() returns the counting state of the specified event set.
 *  @manonly
 *  @endmanonly
 *
 *  @par Example:
 *  @code
 *  int EventSet = PAPI_NULL;
 *  int status = 0;
 *  int ret;
 *  
 *  ret = PAPI_create_eventset(&EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  
 *  // Add Total Instructions Executed to our EventSet
 *  ret = PAPI_add_event(EventSet, PAPI_TOT_INS);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  
 *  // Start counting
 *  ret = PAPI_state(EventSet, &status);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  printf("State is now %d\n",status);
 *  ret = PAPI_start(EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  ret = PAPI_state(EventSet, &status);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  printf("State is now %d\n",status);
 *  @endcode
 *
 *	@see PAPI_stop PAPI_start
 */
int
PAPI_state( int EventSet, int *status )
{
	APIDBG("Entry: EventSet: %d, status: %p\n", EventSet, status);

	EventSetInfo_t *ESI;

	if ( status == NULL )
		papi_return( PAPI_EINVAL );

	/* check for good EventSetIndex value */

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/*read status FROM ESI->state */

	*status = ESI->state;

	return ( PAPI_OK );
}

/** @class PAPI_set_debug
 * @brief Set the current debug level for error output from PAPI.
 *
 * @par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_set_debug( int level );
 *
 * @param level
 *		one of the constants shown in the table below and defined in the papi.h 
 *		header file. @n
 *	The possible debug levels for debugging are shown below.
 *	@arg PAPI_QUIET			Do not print anything, just return the error code
 *	@arg PAPI_VERB_ECONT	Print error message and continue
 *	@arg PAPI_VERB_ESTOP	Print error message and exit 
 *  @n
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL
 *		The debug level is invalid.
 *  @n@n
 *
 *	The current debug level is used by both the internal error and debug message 
 *	handler subroutines. @n
 *	The debug handler is only used if the library was compiled with -DDEBUG. @n
 *	The debug handler is called when there is an error upon a call to the PAPI API.@n 
 *	The error handler is always active and its behavior cannot be modified except 
 *	for whether or not it prints anything.
 *	
 *	The default PAPI debug handler prints out messages in the following form: @n
 *		PAPI Error: Error Code code, symbol, description 
 *
 *	If the error was caused from a system call and the return code is PAPI_ESYS, 
 *	the message will have a colon space and the error string as reported by 
 *	strerror() appended to the end.
 *
 *	The PAPI error handler prints out messages in the following form: @n
 *				PAPI Error: message. 
 *  @n
 *	@note This is the ONLY function that may be called BEFORE PAPI_library_init(). 
 *  @n
 *	@par Example:
 *	@code
 int ret;
 ret = PAPI_set_debug(PAPI_VERB_ECONT);
 if ( ret != PAPI_OK ) handle_error();
 *	@endcode
 *
 *	@see  PAPI_library_init
 *	@see  PAPI_get_opt
 *	@see  PAPI_set_opt
 */
int
PAPI_set_debug( int level )
{
	APIDBG("Entry: level: %d\n", level);
	PAPI_option_t option;

	memset( &option, 0x0, sizeof ( option ) );
	option.debug.level = level;
	option.debug.handler = _papi_hwi_debug_handler;
	return ( PAPI_set_opt( PAPI_DEBUG, &option ) );
}

/* Attaches to or detaches from the specified thread id */
inline_static int
_papi_set_attach( int option, int EventSet, unsigned long tid )
{
	APIDBG("Entry: option: %d, EventSet: %d, tid: %lu\n", option, EventSet, tid);
	PAPI_option_t attach;

	memset( &attach, 0x0, sizeof ( attach ) );
	attach.attach.eventset = EventSet;
	attach.attach.tid = tid;
	return ( PAPI_set_opt( option, &attach ) );
}

/** @class PAPI_attach
 *	@brief Attach PAPI event set to the specified thread id.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_attach( int EventSet, unsigned long tid );
 *
 *	PAPI_attach is a wrapper function that calls PAPI_set_opt to allow PAPI to 
 *	monitor performance counts on a thread other than the one currently executing. 
 *	This is sometimes referred to as third party monitoring. 
 *	PAPI_attach connects the specified EventSet to the specified thread;
 *	PAPI_detach breaks that connection and restores the EventSet to the 
 *	original executing thread. 
 *
 *	@param EventSet 
 *		An integer handle for a PAPI EventSet as created by PAPI_create_eventset.
 *	@param tid 
 *		A thread id as obtained from, for example, PAPI_list_threads or PAPI_thread_id.
 *
 *	@retval PAPI_ECMP 
 *		This feature is unsupported on this component.
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 *	@par Examples:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	unsigned long pid;
 *	pid = fork( );
 *	if ( pid <= 0 )
 *	exit( 1 );
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	exit( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK )
 *	exit( 1 );
 *	// Attach this EventSet to the forked process
 *	if ( PAPI_attach( EventSet, pid ) != PAPI_OK )
 *	exit( 1 );
 *	@endcode
 *
 *	@see PAPI_set_opt
 *	@see PAPI_list_threads
 *	@see PAPI_thread_id
 *	@see PAPI_thread_init
 */
int
PAPI_attach( int EventSet, unsigned long tid )
{
	APIDBG( "Entry: EventSet: %d, tid: %lu\n", EventSet, tid);
	return ( _papi_set_attach( PAPI_ATTACH, EventSet, tid ) );
}

/** @class PAPI_detach
 *	@brief Detach PAPI event set from previously specified thread id and restore to executing thread.
 *	
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_detach( int  EventSet, unsigned long  tid );
 *
 *	PAPI_detach is a wrapper function that calls PAPI_set_opt to allow PAPI to 
 *	monitor performance counts on a thread other than the one currently executing. 
 *	This is sometimes referred to as third party monitoring. 
 *	PAPI_attach connects the specified EventSet to the specified thread;
 *	PAPI_detach breaks that connection and restores the EventSet to the 
 *	original executing thread. 
 *
 *	@param EventSet 
 *		An integer handle for a PAPI EventSet as created by PAPI_create_eventset.
 *	@param tid 
 *		A thread id as obtained from, for example, PAPI_list_threads or PAPI_thread_id.
 *
 *	@retval PAPI_ECMP
 *		This feature is unsupported on this component.
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events. 
 *
 *	@par Examples:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	unsigned long pid;
 *	pid = fork( );
 *	if ( pid <= 0 )
 *	exit( 1 );
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	exit( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK )
 *	exit( 1 );
 *	// Attach this EventSet to the forked process
 *	if ( PAPI_attach( EventSet, pid ) != PAPI_OK )
 *	exit( 1 );
 *	@endcode
 *
 *	@see PAPI_set_opt @n
 *	PAPI_list_threads @n
 *	PAPI_thread_id @n
 *	PAPI_thread_init
 */
int
PAPI_detach( int EventSet )
{
	APIDBG( "Entry: EventSet: %d\n", EventSet);
	return ( _papi_set_attach( PAPI_DETACH, EventSet, 0 ) );
}

/** @class PAPI_set_multiplex
 *	@brief Convert a standard event set to a multiplexed event set. 
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     int PAPI_set_multiplex( int  EventSet );
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *
 *	@retval PAPI_OK
 *	@retval PAPI_EINVAL 
 *		-- One or more of the arguments is invalid, or the EventSet is already multiplexed.
 *	@retval PAPI_ENOCMP
 *		-- The EventSet specified is not yet bound to a component.
 *	@retval PAPI_ENOEVST 
 *		-- The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		-- The EventSet is currently counting events.
 *	@retval PAPI_ENOMEM 
 *		-- Insufficient memory to complete the operation.
 *
 *	PAPI_set_multiplex converts a standard PAPI event set created by a call to 
 *	PAPI_create_eventset into an event set capable of handling multiplexed events. 
 *	This must be done after calling PAPI_multiplex_init, and either PAPI_add_event
 *	or PAPI_assign_eventset_component, but prior to calling PAPI_start(). 
 *	
 *	Events can be added to an event set either before or after converting it 
 *	into a multiplexed set, but the conversion must be done prior to using it 
 *	as a multiplexed set. 
 *
 *  @note Multiplexing can't be enabled until PAPI knows which component is targeted.
 *  Due to the late binding nature of PAPI event sets, this only happens after adding
 *  an event to an event set or explicitly binding the component with a call to
 *  PAPI_assign_eventset_component.
 *
 *	@par Example:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	int ret;
 *	 
 *	// Create an empty EventSet
 *	ret = PAPI_create_eventset(&EventSet);
 *	if (ret != PAPI_OK) handle_error(ret);
 *	
 *	// Bind it to the CPU component
 *	ret = PAPI_assign_eventset_component(EventSet, 0);
 *	if (ret != PAPI_OK) handle_error(ret);
 *	
 *	// Check  current multiplex status
 *	ret = PAPI_get_multiplex(EventSet);
 *	if (ret == TRUE) printf("This event set is ready for multiplexing\n.")
 *	if (ret == FALSE) printf("This event set is not enabled for multiplexing\n.")
 *	if (ret < 0) handle_error(ret);
 *	
 *	// Turn on multiplexing
 *	ret = PAPI_set_multiplex(EventSet);
 *	if ((ret == PAPI_EINVAL) && (PAPI_get_multiplex(EventSet) == TRUE))
 *	  printf("This event set already has multiplexing enabled\n");
 *	else if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see  PAPI_multiplex_init
 *	@see  PAPI_get_multiplex
 *	@see  PAPI_set_opt
 *	@see  PAPI_create_eventset
 */

int
PAPI_set_multiplex( int EventSet )
{
	APIDBG( "Entry: EventSet: %d\n", EventSet);

	PAPI_option_t mpx;
	EventSetInfo_t *ESI;
	int cidx;
	int ret;

	/* Is the EventSet already in existence? */

	ESI = _papi_hwi_lookup_EventSet( EventSet );

	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* if the eventset has no index return NOCMP */
	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 )
		papi_return( cidx );

	if ( ( ret = mpx_check( EventSet ) ) != PAPI_OK )
		papi_return( ret );

	memset( &mpx, 0x0, sizeof ( mpx ) );
	mpx.multiplex.eventset = EventSet;
	mpx.multiplex.flags = PAPI_MULTIPLEX_DEFAULT;
	mpx.multiplex.ns = _papi_os_info.itimer_ns;
	return ( PAPI_set_opt( PAPI_MULTIPLEX, &mpx ) );
}

/** @class PAPI_set_opt
 *	@brief Set PAPI library or event set options.
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     int PAPI_set_opt(  int option, PAPI_option_t * ptr );
 *
 *	@param[in]	option
 *		Defines the option to be set. 
 *		Possible values are briefly described in the table below. 
 *
 *	@param[in,out] ptr
 *		Pointer to a structure determined by the selected option. See PAPI_option_t
 *		for a description of possible structures.
 *
 *	@retval PAPI_OK
 *	@retval PAPI_EINVAL The specified option or parameter is invalid.
 *	@retval PAPI_ENOEVST The EventSet specified does not exist.
 *	@retval PAPI_EISRUN The EventSet is currently counting events.
 *	@retval PAPI_ECMP
 *              The option is not implemented for the current component.
 *	@retval PAPI_ENOINIT PAPI has not been initialized.
 *	@retval PAPI_EINVAL_DOM Invalid domain has been requested.
 *
 *	PAPI_set_opt() changes the options of the PAPI library or a specific EventSet created 
 *	by PAPI_create_eventset. Some options may require that the EventSet be bound to a 
 *	component before they can execute successfully. This can be done either by adding an 
 *	event or by explicitly calling PAPI_assign_eventset_component. 
 *	
 *	Ptr is a pointer to the PAPI_option_t structure, which is actually a union of different
 *	structures for different options. Not all options require or return information in these
 *	structures. Each requires different values to be set. Some options require a component 
 *	index to be provided. These options are handled implicitly through the option structures. 
 *
 *	@note Some options, such as PAPI_DOMAIN and PAPI_MULTIPLEX
 *	are also available as separate entry points in both C and Fortran.
 *
 *	The reader is encouraged to peruse the ctests code in the PAPI distribution for examples
 *  of usage of PAPI_set_opt. 
 *
 *	@par Possible values for the PAPI_set_opt option parameter
 *  @manonly
 * OPTION 			DEFINITION
 * PAPI_DEFDOM		Set default counting domain for newly created event sets. Requires a 
 *					component index.
 * PAPI_DEFGRN		Set default counting granularity. Requires a component index.
 * PAPI_DEBUG		Set the PAPI debug state and the debug handler. The debug state is 
 *					specified in ptr->debug.level. The debug handler is specified in 
 *					ptr->debug.handler. For further information regarding debug states and
 *					the behavior of the handler, see PAPI_set_debug.
 * PAPI_MULTIPLEX	Enable specified EventSet for multiplexing.
 * PAPI_DEF_ITIMER	Set the type of itimer used in software multiplexing, overflowing 
 *					and profiling.
 * PAPI_DEF_MPX_NS	Set the sampling time slice in nanoseconds for multiplexing and overflow.
 * PAPI_DEF_ITIMER_NS See PAPI_DEF_MPX_NS.
 * PAPI_ATTACH		Attach EventSet specified in ptr->attach.eventset to thread or process id
 *					specified in in ptr->attach.tid.
 * PAPI_CPU_ATTACH	Attach EventSet specified in ptr->cpu.eventset to cpu specified in in
 *					ptr->cpu.cpu_num.
 * PAPI_DETACH		Detach EventSet specified in ptr->attach.eventset from any thread
 *					or process id.
 * PAPI_DOMAIN		Set domain for EventSet specified in ptr->domain.eventset. 
 *					Will error if eventset is not bound to a component.
 * PAPI_GRANUL		Set granularity for EventSet specified in ptr->granularity.eventset. 
 *					Will error if eventset is not bound to a component.
 * PAPI_INHERIT		Enable or disable inheritance for specified EventSet.
 * PAPI_DATA_ADDRESS	Set data address range to restrict event counting for EventSet specified
 *					in ptr->addr.eventset. Starting and ending addresses are specified in
 *					ptr->addr.start and ptr->addr.end, respectively. If exact addresses
 *					cannot be instantiated, offsets are returned in ptr->addr.start_off and
 *					ptr->addr.end_off. Currently implemented on Itanium only.
 * PAPI_INSTR_ADDRESS	Set instruction address range as described above. Itanium only.
 * @endmanonly
 * @htmlonly
 * <table class="doxtable">
 * <tr><th>OPTION</th><th>DEFINITION</th></tr>
 * <tr><td>PAPI_DEFDOM</td><td>Set default counting domain for newly created event sets. Requires a component index.</td></tr>
 * <tr><td>PAPI_DEFGRN</td><td>Set default counting granularity. Requires a component index.</td></tr>
 * <tr><td>PAPI_DEBUG</td><td>Set the PAPI debug state and the debug handler. The debug state is specified in ptr->debug.level. The debug handler is specified in ptr->debug.handler. 
 *			For further information regarding debug states and the behavior of the handler, see PAPI_set_debug.</td></tr>
 * <tr><td>PAPI_MULTIPLEX</td><td>Enable specified EventSet for multiplexing.</td></tr>
 * <tr><td>xPAPI_DEF_ITIMER</td><td>Set the type of itimer used in software multiplexing, overflowing and profiling.</td></tr>
 * <tr><td>PAPI_DEF_MPX_NS</td><td>Set the sampling time slice in nanoseconds for multiplexing and overflow.</td></tr>
 * <tr><td>PAPI_DEF_ITIMER_NS</td><td>See PAPI_DEF_MPX_NS.</td></tr>
 * <tr><td>PAPI_ATTACH</td><td>Attach EventSet specified in ptr->attach.eventset to thread or process id specified in in ptr->attach.tid.</td></tr>
 * <tr><td>PAPI_CPU_ATTACH</td><td>Attach EventSet specified in ptr->cpu.eventset to cpu specified in in ptr->cpu.cpu_num.</td></tr>
 * <tr><td>PAPI_DETACH</td><td>Detach EventSet specified in ptr->attach.eventset from any thread or process id.</td></tr>
 * <tr><td>PAPI_DOMAIN</td><td>Set domain for EventSet specified in ptr->domain.eventset. Will error if eventset is not bound to a component.</td></tr>
 * <tr><td>PAPI_GRANUL</td><td>Set granularity for EventSet specified in ptr->granularity.eventset. Will error if eventset is not bound to a component.</td></tr>
 * <tr><td>PAPI_INHERIT</td><td>Enable or disable inheritance for specified EventSet.</td></tr>
 * <tr><td>PAPI_DATA_ADDRESS</td><td>Set data address range to restrict event counting for EventSet specified in ptr->addr.eventset. Starting and ending addresses are specified in ptr->addr.start and ptr->addr.end, respectively. If exact addresses cannot be instantiated, offsets are returned in ptr->addr.start_off and ptr->addr.end_off. Currently implemented on Itanium only.</td></tr>
 * <tr><td>PAPI_INSTR_ADDRESS</td><td>Set instruction address range as described above. Itanium only.</td></tr>
 * </table>
 * @endhtmlonly
 *
 *	@see PAPI_set_debug
 *	@see PAPI_set_multiplex
 *	@see PAPI_set_domain
 *	@see PAPI_option_t
 */
int
PAPI_set_opt( int option, PAPI_option_t * ptr )
{
	APIDBG("Entry:  option: %d, ptr: %p\n", option, ptr);

	_papi_int_option_t internal;
	int retval = PAPI_OK;
	hwd_context_t *context;
	int cidx;

	if ( ( option != PAPI_DEBUG ) && ( init_level == PAPI_NOT_INITED ) )
		papi_return( PAPI_ENOINIT );
	if ( ptr == NULL )
		papi_return( PAPI_EINVAL );

	memset( &internal, 0x0, sizeof ( _papi_int_option_t ) );

	switch ( option ) {
	case PAPI_DETACH:
	{
		internal.attach.ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( internal.attach.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.attach.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.attach == 0 )
			papi_return( PAPI_ECMP );

		/* if attached to a cpu, return an error */
		if (internal.attach.ESI->state & PAPI_CPU_ATTACHED)
			papi_return( PAPI_ECMP );

		if ( ( internal.attach.ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		if ( ( internal.attach.ESI->state & PAPI_ATTACHED ) == 0 )
			papi_return( PAPI_EINVAL );

		internal.attach.tid = internal.attach.ESI->attach.tid;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.attach.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_DETACH, &internal );
		if ( retval != PAPI_OK )
			papi_return( retval );

		internal.attach.ESI->state ^= PAPI_ATTACHED;
		internal.attach.ESI->attach.tid = 0;
		return ( PAPI_OK );
	}
	case PAPI_ATTACH:
	{
		internal.attach.ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( internal.attach.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.attach.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.attach == 0 )
			papi_return( PAPI_ECMP );

		if ( ( internal.attach.ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		if ( internal.attach.ESI->state & PAPI_ATTACHED )
			papi_return( PAPI_EINVAL );

		/* if attached to a cpu, return an error */
		if (internal.attach.ESI->state & PAPI_CPU_ATTACHED)
			papi_return( PAPI_ECMP );

		internal.attach.tid = ptr->attach.tid;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.attach.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_ATTACH, &internal );
		if ( retval != PAPI_OK )
			papi_return( retval );

		internal.attach.ESI->state |= PAPI_ATTACHED;
		internal.attach.ESI->attach.tid = ptr->attach.tid;

		papi_return (_papi_hwi_lookup_or_create_thread( 
				      &(internal.attach.ESI->master), ptr->attach.tid ));
	}
	case PAPI_CPU_ATTACH:
	{
		APIDBG("eventset: %d, cpu_num: %d\n", ptr->cpu.eventset, ptr->cpu.cpu_num);
		internal.cpu.ESI = _papi_hwi_lookup_EventSet( ptr->cpu.eventset );
		if ( internal.cpu.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		internal.cpu.cpu_num = ptr->cpu.cpu_num;
		APIDBG("internal: %p, ESI: %p, cpu_num: %d\n", &internal, internal.cpu.ESI, internal.cpu.cpu_num);

		cidx = valid_ESI_component( internal.cpu.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.cpu == 0 )
			papi_return( PAPI_ECMP );

		// can not attach to a cpu if already attached to a process or 
		// counters set to be inherited by child processes
		if ( internal.cpu.ESI->state & (PAPI_ATTACHED | PAPI_INHERIT) )
			papi_return( PAPI_EINVAL );

		if ( ( internal.cpu.ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		retval = _papi_hwi_lookup_or_create_cpu(&internal.cpu.ESI->CpuInfo, internal.cpu.cpu_num);
		if( retval != PAPI_OK) {
			papi_return( retval );
		}

		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.cpu.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_CPU_ATTACH, &internal );
		if ( retval != PAPI_OK )
			papi_return( retval );

		/* set to show this event set is attached to a cpu not a thread */
		internal.cpu.ESI->state |= PAPI_CPU_ATTACHED;
		return ( PAPI_OK );
	}
	case PAPI_DEF_MPX_NS:
	{
		cidx = 0;			 /* xxxx for now, assume we only check against cpu component */
		if ( ptr->multiplex.ns < 0 )
			papi_return( PAPI_EINVAL );
		/* We should check the resolution here with the system, either
		   component if kernel multiplexing or PAPI if SW multiplexing. */
		internal.multiplex.ns = ( unsigned long ) ptr->multiplex.ns;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.cpu.ESI, NULL );
		/* Low level just checks/adjusts the args for this component */
		retval = _papi_hwd[cidx]->ctl( context, PAPI_DEF_MPX_NS, &internal );
		if ( retval == PAPI_OK ) {
			_papi_os_info.itimer_ns = ( int ) internal.multiplex.ns;
			ptr->multiplex.ns = ( int ) internal.multiplex.ns;
		}
		papi_return( retval );
	}
	case PAPI_DEF_ITIMER_NS:
	{
		cidx = 0;			 /* xxxx for now, assume we only check against cpu component */
		if ( ptr->itimer.ns < 0 )
			papi_return( PAPI_EINVAL );
		internal.itimer.ns = ptr->itimer.ns;
		/* Low level just checks/adjusts the args for this component */
		retval = _papi_hwd[cidx]->ctl( NULL, PAPI_DEF_ITIMER_NS, &internal );
		if ( retval == PAPI_OK ) {
			_papi_os_info.itimer_ns = internal.itimer.ns;
			ptr->itimer.ns = internal.itimer.ns;
		}
		papi_return( retval );
	}
	case PAPI_DEF_ITIMER:
	{
		cidx = 0;			 /* xxxx for now, assume we only check against cpu component */
		if ( ptr->itimer.ns < 0 )
			papi_return( PAPI_EINVAL );
		memcpy( &internal.itimer, &ptr->itimer,
				sizeof ( PAPI_itimer_option_t ) );
		/* Low level just checks/adjusts the args for this component */
		retval = _papi_hwd[cidx]->ctl( NULL, PAPI_DEF_ITIMER, &internal );
		if ( retval == PAPI_OK ) {
			_papi_os_info.itimer_num = ptr->itimer.itimer_num;
			_papi_os_info.itimer_sig = ptr->itimer.itimer_sig;
			if ( ptr->itimer.ns > 0 )
				_papi_os_info.itimer_ns = ptr->itimer.ns;
			/* flags are currently ignored, eventually the flags will be able
			   to specify whether or not we use POSIX itimers (clock_gettimer) */
		}
		papi_return( retval );
	}
	case PAPI_MULTIPLEX:
	{
		EventSetInfo_t *ESI;
		ESI = _papi_hwi_lookup_EventSet( ptr->multiplex.eventset );
	   
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( ESI );
		if ( cidx < 0 )
			papi_return( cidx );
	   
		if ( !( ESI->state & PAPI_STOPPED ) )
			papi_return( PAPI_EISRUN );
		if ( ESI->state & PAPI_MULTIPLEXING )
			papi_return( PAPI_EINVAL );

		if ( ptr->multiplex.ns < 0 )
			papi_return( PAPI_EINVAL );
		internal.multiplex.ESI = ESI;
		internal.multiplex.ns = ( unsigned long ) ptr->multiplex.ns;
		internal.multiplex.flags = ptr->multiplex.flags;
		if ( ( _papi_hwd[cidx]->cmp_info.kernel_multiplex ) &&
			 ( ( ptr->multiplex.flags & PAPI_MULTIPLEX_FORCE_SW ) == 0 ) ) {
			/* get the context we should use for this event set */
			context = _papi_hwi_get_context( ESI, NULL );
			retval = _papi_hwd[cidx]->ctl( context, PAPI_MULTIPLEX, &internal );
		}
		/* Kernel or PAPI may have changed this value so send it back out to the user */
		ptr->multiplex.ns = ( int ) internal.multiplex.ns;
		if ( retval == PAPI_OK )
			papi_return( _papi_hwi_convert_eventset_to_multiplex
						 ( &internal.multiplex ) );
		return ( retval );
	}
	case PAPI_DEBUG:
	{
		int level = ptr->debug.level;
		switch ( level ) {
		case PAPI_QUIET:
		case PAPI_VERB_ESTOP:
		case PAPI_VERB_ECONT:
			_papi_hwi_error_level = level;
			break;
		default:
			papi_return( PAPI_EINVAL );
		}
		_papi_hwi_debug_handler = ptr->debug.handler;
		return ( PAPI_OK );
	}
	case PAPI_DEFDOM:
	{
		int dom = ptr->defdomain.domain;
		if ( ( dom < PAPI_DOM_MIN ) || ( dom > PAPI_DOM_MAX ) )
			papi_return( PAPI_EINVAL );

		/* Change the global structure. The _papi_hwd_init_control_state function 
		   in the components gets information from the global structure instead of
		   per-thread information. */
		cidx = valid_component( ptr->defdomain.def_cidx );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Check what the component supports */

		if ( dom == PAPI_DOM_ALL )
			dom = _papi_hwd[cidx]->cmp_info.available_domains;

		if ( dom & ~_papi_hwd[cidx]->cmp_info.available_domains )
			papi_return( PAPI_ENOSUPP );

		_papi_hwd[cidx]->cmp_info.default_domain = dom;

		return ( PAPI_OK );
	}
	case PAPI_DOMAIN:
	{
		int dom = ptr->domain.domain;
		if ( ( dom < PAPI_DOM_MIN ) || ( dom > PAPI_DOM_MAX ) )
			papi_return( PAPI_EINVAL_DOM );

		internal.domain.ESI = _papi_hwi_lookup_EventSet( ptr->domain.eventset );
		if ( internal.domain.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.domain.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Check what the component supports */

		if ( dom == PAPI_DOM_ALL )
			dom = _papi_hwd[cidx]->cmp_info.available_domains;

		if ( dom & ~_papi_hwd[cidx]->cmp_info.available_domains )
			papi_return( PAPI_EINVAL_DOM );

		if ( !( internal.domain.ESI->state & PAPI_STOPPED ) )
			papi_return( PAPI_EISRUN );

		/* Try to change the domain of the eventset in the hardware */
		internal.domain.domain = dom;
		internal.domain.eventset = ptr->domain.eventset;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.domain.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_DOMAIN, &internal );
		if ( retval < PAPI_OK )
			papi_return( retval );

		/* Change the domain of the eventset in the library */

		internal.domain.ESI->domain.domain = dom;

		return ( retval );
	}
	case PAPI_DEFGRN:
	{
		int grn = ptr->defgranularity.granularity;
		if ( ( grn < PAPI_GRN_MIN ) || ( grn > PAPI_GRN_MAX ) )
			papi_return( PAPI_EINVAL );

		cidx = valid_component( ptr->defgranularity.def_cidx );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Change the component structure. The _papi_hwd_init_control_state function 
		   in the components gets information from the global structure instead of
		   per-thread information. */

		/* Check what the component supports */

		if ( grn & ~_papi_hwd[cidx]->cmp_info.available_granularities )
			papi_return( PAPI_EINVAL );

		/* Make sure there is only 1 set. */
		if ( grn ^ ( 1 << ( ffs( grn ) - 1 ) ) )
			papi_return( PAPI_EINVAL );

		_papi_hwd[cidx]->cmp_info.default_granularity = grn;

		return ( PAPI_OK );
	}
	case PAPI_GRANUL:
	{
		int grn = ptr->granularity.granularity;

		if ( ( grn < PAPI_GRN_MIN ) || ( grn > PAPI_GRN_MAX ) )
			papi_return( PAPI_EINVAL );

		internal.granularity.ESI =
			_papi_hwi_lookup_EventSet( ptr->granularity.eventset );
		if ( internal.granularity.ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( internal.granularity.ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		/* Check what the component supports */

		if ( grn & ~_papi_hwd[cidx]->cmp_info.available_granularities )
			papi_return( PAPI_EINVAL );

		/* Make sure there is only 1 set. */
		if ( grn ^ ( 1 << ( ffs( grn ) - 1 ) ) )
			papi_return( PAPI_EINVAL );

		internal.granularity.granularity = grn;
		internal.granularity.eventset = ptr->granularity.eventset;
		retval = _papi_hwd[cidx]->ctl( NULL, PAPI_GRANUL, &internal );
		if ( retval < PAPI_OK )
			return ( retval );

		internal.granularity.ESI->granularity.granularity = grn;
		return ( retval );
	}
	case PAPI_INHERIT:
	{
		EventSetInfo_t *ESI;
		ESI = _papi_hwi_lookup_EventSet( ptr->inherit.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		if ( _papi_hwd[cidx]->cmp_info.inherit == 0 )
			papi_return( PAPI_ECMP );

		if ( ( ESI->state & PAPI_STOPPED ) == 0 )
			papi_return( PAPI_EISRUN );

		/* if attached to a cpu, return an error */
		if (ESI->state & PAPI_CPU_ATTACHED)
			papi_return( PAPI_ECMP );

		internal.inherit.ESI = ESI;
		internal.inherit.inherit = ptr->inherit.inherit;

		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.inherit.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, PAPI_INHERIT, &internal );
		if ( retval < PAPI_OK )
			return ( retval );

		ESI->inherit.inherit = ptr->inherit.inherit;
		return ( retval );
	}
	case PAPI_DATA_ADDRESS:
	case PAPI_INSTR_ADDRESS:
	{

		EventSetInfo_t *ESI;

		ESI = _papi_hwi_lookup_EventSet( ptr->addr.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );

		cidx = valid_ESI_component( ESI );
		if ( cidx < 0 )
			papi_return( cidx );

		internal.address_range.ESI = ESI;

		if ( !( internal.address_range.ESI->state & PAPI_STOPPED ) )
			papi_return( PAPI_EISRUN );

		/*set domain to be PAPI_DOM_USER */
		internal.address_range.domain = PAPI_DOM_USER;

		internal.address_range.start = ptr->addr.start;
		internal.address_range.end = ptr->addr.end;
		/* get the context we should use for this event set */
		context = _papi_hwi_get_context( internal.address_range.ESI, NULL );
		retval = _papi_hwd[cidx]->ctl( context, option, &internal );
		ptr->addr.start_off = internal.address_range.start_off;
		ptr->addr.end_off = internal.address_range.end_off;
		papi_return( retval );
	}
	case PAPI_USER_EVENTS_FILE:
	{
		APIDBG("User Events Filename is -%s-\n", ptr->events_file);

		// go load the user defined event definitions from the applications event definition file
		// do not know how to find a pmu name and type for this operation yet
//		retval = papi_load_derived_events(pmu_str, pmu_type, cidx, 0);

//		_papi_user_defined_events_setup(ptr->events_file);
		return( PAPI_OK );
	}
	default:
		papi_return( PAPI_EINVAL );
	}
}

/** @class PAPI_num_hwctrs
 *  @brief Return the number of hardware counters on the cpu.
 *
 * @deprecated
 *	This is included to preserve backwards compatibility.
 *      Use PAPI_num_cmp_hwctrs() instead.
 *
 * @see PAPI_num_cmp_hwctrs
 */
int
PAPI_num_hwctrs( void )
{
	APIDBG( "Entry:\n");
	return ( PAPI_num_cmp_hwctrs( 0 ) );
}

/** @class PAPI_num_cmp_hwctrs
 *  @brief Return the number of hardware counters for the specified component.
 *
 *  PAPI_num_cmp_hwctrs() returns the number of counters present in the 
 *  specified component. 
 *  By convention, component 0 is always the cpu. 
 *
 *  On some components, especially for CPUs, the value returned is
 *  a theoretical maximum for estimation purposes only.  It might not
 *  be possible to easily create an EventSet that contains the full
 *  number of events.  This can be due to a variety of reasons:
 *  1).  Some CPUs (especially Intel and POWER) have the notion
 *       of fixed counters that can only measure one thing, usually
 *       cycles.
 *  2).  Some CPUs have very explicit rules about which event can
 *       run in which counter.  In this case it might not be possible
 *       to add a wanted event even if counters are free.
 *  3).  Some CPUs halve the number of counters available when
 *       running with SMT (multiple CPU threads) enabled.
 *  4).  Some operating systems "steal" a counter to use for things
 *       such as NMI Watchdog timers.
 *  The only sure way to see if events will fit is to attempt
 *  adding events to an EventSet, and doing something sensible
 *  if an error is generated.
 *
 *  PAPI_library_init() must be called in order for this function to return 
 *  anything greater than 0. 
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_num_cmp_hwctrs(int  cidx );
 *
 * @param[in] cidx
 *         -- An integer identifier for a component. 
 *         By convention, component 0 is always the cpu component.
 *
 * @par Example
 * @code
 * // Query the cpu component for the number of counters.
 * printf(\"%d hardware counters found.\\n\", PAPI_num_cmp_hwctrs(0));
 * @endcode
 *
 * @returns 
 *  On success, this function returns a value greater than zero.@n
 *  A zero result usually means the library has not been initialized.
 *
 * @bug This count may include fixed-use counters in addition
 *      to the general purpose counters.
 */
int
PAPI_num_cmp_hwctrs( int cidx )
{
	APIDBG( "Entry: cidx: %d\n", cidx);
	return ( PAPI_get_cmp_opt( PAPI_MAX_HWCTRS, NULL, cidx ) );
}

/** @class PAPI_get_multiplex
 *	@brief Get the multiplexing status of specified event set.
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     int PAPI_get_multiplex( int  EventSet );
 *
 * @par Fortran Interface:
 *     \#include fpapi.h @n
 *     PAPIF_get_multiplex( C_INT  EventSet,  C_INT  check )
 *
 *	@param EventSet
 *	an integer handle for a PAPI event set as created by PAPI_create_eventset
 *
 *	@retval PAPI_OK
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid, or the EventSet 
 *		is already multiplexed.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation. 
 *
 *	PAPI_get_multiplex tests the state of the PAPI_MULTIPLEXING flag in the specified event set,
 *  returning @em TRUE if a PAPI event set is multiplexed, or FALSE if not.          
 *	@par Example:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	int ret;
 *	 
 *	// Create an empty EventSet
 *	ret = PAPI_create_eventset(&EventSet);
 *	if (ret != PAPI_OK) handle_error(ret);
 *	
 *	// Bind it to the CPU component
 *	ret = PAPI_assign_eventset_component(EventSet, 0);
 *	if (ret != PAPI_OK) handle_error(ret);
 *	
 *	// Check  current multiplex status
 *	ret = PAPI_get_multiplex(EventSet);
 *	if (ret == TRUE) printf("This event set is ready for multiplexing\n.")
 *	if (ret == FALSE) printf("This event set is not enabled for multiplexing\n.")
 *	if (ret < 0) handle_error(ret);
 *	
 *	// Turn on multiplexing
 *	ret = PAPI_set_multiplex(EventSet);
 *	if ((ret == PAPI_EINVAL) && (PAPI_get_multiplex(EventSet) == TRUE))
 *	  printf("This event set already has multiplexing enabled\n");
 *	else if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *	@see PAPI_multiplex_init 
 *	@see PAPI_set_opt 
 *	@see PAPI_create_eventset
 */
int
PAPI_get_multiplex( int EventSet )
{
	APIDBG( "Entry: EventSet: %d\n", EventSet);
	PAPI_option_t popt;
	int retval;

	popt.multiplex.eventset = EventSet;
	retval = PAPI_get_opt( PAPI_MULTIPLEX, &popt );
	if ( retval < 0 )
		retval = 0;
	return retval;
}

/** @class PAPI_get_opt
 *	@brief Get PAPI library or event set options.
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     int PAPI_get_opt(  int option, PAPI_option_t * ptr );
 *
 *	@param[in]	option
 *		Defines the option to get. 
 *		Possible values are briefly described in the table below. 
 *
 *	@param[in,out] ptr
 *		Pointer to a structure determined by the selected option. See PAPI_option_t
 *		for a description of possible structures.
 *
 *	@retval PAPI_OK
 *	@retval PAPI_EINVAL The specified option or parameter is invalid.
 *	@retval PAPI_ENOEVST The EventSet specified does not exist.
 *	@retval PAPI_ECMP 
 *              The option is not implemented for the current component.
 *	@retval PAPI_ENOINIT specified option requires PAPI to be initialized first.
 *
 *	PAPI_get_opt() queries the options of the PAPI library or a specific event set created by 
 *	PAPI_create_eventset. Some options may require that the eventset be bound to a component
 *	before they can execute successfully. This can be done either by adding an event or by
 *	explicitly calling PAPI_assign_eventset_component.
 *
 *	Ptr is a pointer to the PAPI_option_t structure, which is actually a union of different
 *	structures for different options. Not all options require or return information in these
 *	structures. Each returns different values in the structure. Some options require a component 
 *	index to be provided. These options are handled explicitly by the PAPI_get_cmp_opt() call. 
 *
 *	@note Some options, such as PAPI_DOMAIN and PAPI_MULTIPLEX
 *	are also available as separate entry points in both C and Fortran.
 *
 *	The reader is encouraged to peruse the ctests code in the PAPI distribution for examples
 *  of usage of PAPI_set_opt. 
 *
 *	@par Possible values for the PAPI_get_opt option parameter
 * @manonly
 * OPTION 			DEFINITION
 * PAPI_DEFDOM		Get default counting domain for newly created event sets. Requires a component index.
 * PAPI_DEFGRN		Get default counting granularity. Requires a component index.
 * PAPI_DEBUG		Get the PAPI debug state and the debug handler. The debug state is specified in ptr->debug.level. The debug handler is specified in ptr->debug.handler. 
 *					For further information regarding debug states and the behavior of the handler, see PAPI_set_debug.
 * PAPI_MULTIPLEX	Get current multiplexing state for specified EventSet.
 * PAPI_DEF_ITIMER	Get the type of itimer used in software multiplexing, overflowing and profiling.
 * PAPI_DEF_MPX_NS	Get the sampling time slice in nanoseconds for multiplexing and overflow.
 * PAPI_DEF_ITIMER_NS	See PAPI_DEF_MPX_NS.
 * PAPI_ATTACH		Get thread or process id to which event set is attached. Returns TRUE if currently attached.
 * PAPI_CPU_ATTACH	Get ptr->cpu.cpu_num and Attach state for EventSet specified in ptr->cpu.eventset.
 * PAPI_DETACH		Get thread or process id to which event set is attached. Returns TRUE if currently attached.
 * PAPI_DOMAIN		Get domain for EventSet specified in ptr->domain.eventset. Will error if eventset is not bound to a component.
 * PAPI_GRANUL		Get granularity for EventSet specified in ptr->granularity.eventset. Will error if eventset is not bound to a component.
 * PAPI_INHERIT		Get current inheritance state for specified EventSet.
 * PAPI_PRELOAD		Get LD_PRELOAD environment equivalent.
 * PAPI_CLOCKRATE	Get clockrate in MHz.
 * PAPI_MAX_CPUS	Get number of CPUs.
 * PAPI_EXEINFO		Get Executable addresses for text/data/bss.
 * PAPI_HWINFO		Get information about the hardware.
 * PAPI_LIB_VERSION	Get the full PAPI version of the library. This does not require PAPI to be initialized first.
 * PAPI_MAX_HWCTRS	Get number of counters. Requires a component index.
 * PAPI_MAX_MPX_CTRS	Get maximum number of multiplexing counters. Requires a component index.
 * PAPI_SHLIBINFO	Get shared library information used by the program.
 * PAPI_COMPONENTINFO	Get the PAPI features the specified component supports. Requires a component index.
 * @endmanonly
 * @htmlonly
 * <table class="doxtable">
 * <tr><th>OPTION</th><th>DEFINITION</th></tr>
 * <tr><td>PAPI_DEFDOM</td><td>Get default counting domain for newly created event sets. Requires a component index.</td></tr>
 * <tr><td>PAPI_DEFGRN</td><td>Get default counting granularity. Requires a component index.</td></tr>
 * <tr><td>PAPI_DEBUG</td><td>Get the PAPI debug state and the debug handler. The debug state is specified in ptr->debug.level. The debug handler is specified in ptr->debug.handler. 
 *			For further information regarding debug states and the behavior of the handler, see PAPI_set_debug.</td></tr>
 * <tr><td>PAPI_MULTIPLEX</td><td>Get current multiplexing state for specified EventSet.</td></tr>
 * <tr><td>PAPI_DEF_ITIMER</td><td>Get the type of itimer used in software multiplexing, overflowing and profiling.</td></tr>
 * <tr><td>PAPI_DEF_MPX_NS</td><td>Get the sampling time slice in nanoseconds for multiplexing and overflow.</td></tr>
 * <tr><td>PAPI_DEF_ITIMER_NS</td><td>See PAPI_DEF_MPX_NS.</td></tr>
 * <tr><td>PAPI_ATTACH</td><td>Get thread or process id to which event set is attached. Returns TRUE if currently attached.</td></tr>
 * <tr><td>PAPI_CPU_ATTACH</td><td>Get ptr->cpu.cpu_num and Attach state for EventSet specified in ptr->cpu.eventset.</td></tr>
 * <tr><td>PAPI_DETACH</td><td>Get thread or process id to which event set is attached. Returns TRUE if currently attached.</td></tr>
 * <tr><td>PAPI_DOMAIN</td><td>Get domain for EventSet specified in ptr->domain.eventset. Will error if eventset is not bound to a component.</td></tr>
 * <tr><td>PAPI_GRANUL</td><td>Get granularity for EventSet specified in ptr->granularity.eventset. Will error if eventset is not bound to a component.</td></tr>
 * <tr><td>PAPI_INHERIT</td><td>Get current inheritance state for specified EventSet.</td></tr>
 * <tr><td>PAPI_PRELOAD</td><td>Get LD_PRELOAD environment equivalent.</td></tr>
 * <tr><td>PAPI_CLOCKRATE</td><td>Get clockrate in MHz.</td></tr>
 * <tr><td>PAPI_MAX_CPUS</td><td>Get number of CPUs.</td></tr>
 * <tr><td>PAPI_EXEINFO</td><td>Get Executable addresses for text/data/bss.</td></tr>
 * <tr><td>PAPI_HWINFO</td><td>Get information about the hardware.</td></tr>
 * <tr><td>PAPI_LIB_VERSION</td><td>Get the full PAPI version of the library.</td></tr>
 * <tr><td>PAPI_MAX_HWCTRS</td><td>Get number of counters. Requires a component index.</td></tr>
 * <tr><td>PAPI_MAX_MPX_CTRS</td><td>Get maximum number of multiplexing counters. Requires a component index.</td></tr>
 * <tr><td>PAPI_SHLIBINFO</td><td>Get shared library information used by the program.</td></tr>
 * <tr><td>PAPI_COMPONENTINFO</td><td>Get the PAPI features the specified component supports. Requires a component index.</td></tr>
 * </table>
 * @endhtmlonly
 *
 *	@see PAPI_get_multiplex
 *	@see PAPI_get_cmp_opt
 *	@see PAPI_set_opt
 *	@see PAPI_option_t
 */
int
PAPI_get_opt( int option, PAPI_option_t * ptr )
{
	APIDBG( "Entry: option: %d, ptr: %p\n", option, ptr);
	EventSetInfo_t *ESI;

	if ( ( option != PAPI_DEBUG ) && ( init_level == PAPI_NOT_INITED ) &&
         ( option != PAPI_LIB_VERSION ) )
		papi_return( PAPI_ENOINIT );

	switch ( option ) {
	case PAPI_DETACH:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->attach.tid = ESI->attach.tid;
		return ( ( ESI->state & PAPI_ATTACHED ) == 0 );
	}
	case PAPI_ATTACH:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->attach.tid = ESI->attach.tid;
		return ( ( ESI->state & PAPI_ATTACHED ) != 0 );
	}
	case PAPI_CPU_ATTACH:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->attach.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->cpu.cpu_num = ESI->CpuInfo->cpu_num;
		return ( ( ESI->state & PAPI_CPU_ATTACHED ) != 0 );
	}
	case PAPI_DEF_MPX_NS:
	{
		/* xxxx for now, assume we only check against cpu component */
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->multiplex.ns = _papi_os_info.itimer_ns;
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER_NS:
	{
		/* xxxx for now, assume we only check against cpu component */
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->itimer.ns = _papi_os_info.itimer_ns;
		return ( PAPI_OK );
	}
	case PAPI_DEF_ITIMER:
	{
		/* xxxx for now, assume we only check against cpu component */
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->itimer.itimer_num = _papi_os_info.itimer_num;
		ptr->itimer.itimer_sig = _papi_os_info.itimer_sig;
		ptr->itimer.ns = _papi_os_info.itimer_ns;
		ptr->itimer.flags = 0;
		return ( PAPI_OK );
	}
	case PAPI_MULTIPLEX:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->multiplex.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->multiplex.ns = ESI->multiplex.ns;
		ptr->multiplex.flags = ESI->multiplex.flags;
		return ( ESI->state & PAPI_MULTIPLEXING ) != 0;
	}
	case PAPI_PRELOAD:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		memcpy( &ptr->preload, &_papi_hwi_system_info.preload_info,
				sizeof ( PAPI_preload_info_t ) );
		break;
	case PAPI_DEBUG:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->debug.level = _papi_hwi_error_level;
		ptr->debug.handler = _papi_hwi_debug_handler;
		break;
	case PAPI_CLOCKRATE:
		return ( ( int ) _papi_hwi_system_info.hw_info.cpu_max_mhz );
	case PAPI_MAX_CPUS:
		return ( _papi_hwi_system_info.hw_info.ncpu );
		/* For now, MAX_HWCTRS and MAX CTRS are identical.
		   At some future point, they may map onto different values.
		 */
	case PAPI_INHERIT:
	{
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->inherit.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->inherit.inherit = ESI->inherit.inherit;
		return ( PAPI_OK );
	}
	case PAPI_GRANUL:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->granularity.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->granularity.granularity = ESI->granularity.granularity;
		break;
	case PAPI_EXEINFO:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->exe_info = &_papi_hwi_system_info.exe_info;
		break;
	case PAPI_HWINFO:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->hw_info = &_papi_hwi_system_info.hw_info;
		break;

	case PAPI_DOMAIN:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ESI = _papi_hwi_lookup_EventSet( ptr->domain.eventset );
		if ( ESI == NULL )
			papi_return( PAPI_ENOEVST );
		ptr->domain.domain = ESI->domain.domain;
		return ( PAPI_OK );
	case PAPI_LIB_VERSION:
		return ( PAPI_VERSION );
/* The following cases all require a component index 
    and are handled by PAPI_get_cmp_opt() with cidx == 0*/
	case PAPI_MAX_HWCTRS:
	case PAPI_MAX_MPX_CTRS:
	case PAPI_DEFDOM:
	case PAPI_DEFGRN:
	case PAPI_SHLIBINFO:
	case PAPI_COMPONENTINFO:
		return ( PAPI_get_cmp_opt( option, ptr, 0 ) );
	default:
		papi_return( PAPI_EINVAL );
	}
	return ( PAPI_OK );
}

/** @class PAPI_get_cmp_opt
 *	@brief Get component specific PAPI options.
 *
 *	@param	option
 *		is an input parameter describing the course of action. 
 *		Possible values are defined in papi.h and briefly described in the table below. 
 *		The Fortran calls are implementations of specific options.
 *	@param ptr
 *		is a pointer to a structure that acts as both an input and output parameter.
 *	@param cidx
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *
 *	PAPI_get_opt() and PAPI_set_opt() query or change the options of the PAPI 
 *	library or a specific event set created by PAPI_create_eventset . 
 *	Some options may require that the eventset be bound to a component before 
 *	they can execute successfully. 
 *	This can be done either by adding an event or by explicitly calling 
 *	PAPI_assign_eventset_component . 
 *	
 *	The C interface for these functions passes a pointer to the PAPI_option_t structure. 
 *	Not all options require or return information in this structure, and not all 
 *	options are implemented for both get and set. 
 *	Some options require a component index to be provided. 
 *	These options are handled explicitly by the PAPI_get_cmp_opt() call for 'get' 
 *	and implicitly through the option structure for 'set'. 
 *	The Fortran interface is a series of calls implementing various subsets of 
 *	the C interface. Not all options in C are available in Fortran.
 *
 *	@note Some options, such as PAPI_DOMAIN and PAPI_MULTIPLEX, 
 *	are also available as separate entry points in both C and Fortran.
 *
 *	The reader is urged to see the example code in the PAPI distribution for usage of PAPI_get_opt. 
 *	The file papi.h contains definitions for the structures unioned in the PAPI_option_t structure. 
 *
 *	@see PAPI_set_debug PAPI_set_multiplex PAPI_set_domain PAPI_option_t
 */

int
PAPI_get_cmp_opt( int option, PAPI_option_t * ptr, int cidx )
{
	APIDBG( "Entry: option: %d, ptr: %p, cidx: %d\n", option, ptr, cidx);

  if (_papi_hwi_invalid_cmp(cidx)) {
     return PAPI_ECMP;
  }

	switch ( option ) {
		/* For now, MAX_HWCTRS and MAX CTRS are identical.
		   At some future point, they may map onto different values.
		 */
	case PAPI_MAX_HWCTRS:
		return ( _papi_hwd[cidx]->cmp_info.num_cntrs );
	case PAPI_MAX_MPX_CTRS:
		return ( _papi_hwd[cidx]->cmp_info.num_mpx_cntrs );
	case PAPI_DEFDOM:
		return ( _papi_hwd[cidx]->cmp_info.default_domain );
	case PAPI_DEFGRN:
		return ( _papi_hwd[cidx]->cmp_info.default_granularity );
	case PAPI_SHLIBINFO:
	{
		int retval;
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		retval = _papi_os_vector.update_shlib_info( &_papi_hwi_system_info );
		ptr->shlib_info = &_papi_hwi_system_info.shlib_info;
		papi_return( retval );
	}
	case PAPI_COMPONENTINFO:
		if ( ptr == NULL )
			papi_return( PAPI_EINVAL );
		ptr->cmp_info = &( _papi_hwd[cidx]->cmp_info );
		return PAPI_OK;
	default:
	  papi_return( PAPI_EINVAL );
	}
	return PAPI_OK;
}

/** @class PAPI_num_components
  *	@brief Get the number of components available on the system.
  *
  * @return 
  *		Number of components available on the system
  *
  *	@code
// Query the library for a component count. 
printf("%d components installed., PAPI_num_components() );
  * @endcode
  */
int
PAPI_num_components( void )
{
	APIDBG( "Entry:\n");
	return ( papi_num_components );
}

/** @class PAPI_num_events
  * @brief Return the number of events in an event set.
  * 
  * PAPI_num_events() returns the number of preset and/or native events 
  * contained in an event set. 
  * The event set should be created by @ref PAPI_create_eventset .
  *
  * @par C Interface:
  * \#include <papi.h> @n
  * int PAPI_num_events(int  EventSet );
  *
  * @param[in] EventSet -- 
  *   an integer handle for a PAPI event set created by PAPI_create_eventset.
  * @param[out] *count -- (Fortran only) 
  *   On output the variable contains the number of events in the event set
  *
  * @retval On success, this function returns the positive number of 
  *         events in the event set.
  * @retval PAPI_EINVAL The event count is zero; 
  *                     only if code is compiled with debug enabled.
  * @retval PAPI_ENOEVST The EventSet specified does not exist. 
  *
  * @par Example
  * @code
  * // Count the events in our EventSet 
  * printf(\"%d events found in EventSet.\\n\", PAPI_num_events(EventSet));
  * @endcode
  *
  * @see PAPI_add_event 
  * @see PAPI_create_eventset
  *
  */
int
PAPI_num_events( int EventSet )
{
	APIDBG( "Entry: EventSet: %d\n", EventSet);
	EventSetInfo_t *ESI;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( !ESI )
		papi_return( PAPI_ENOEVST );

	return ( ESI->NumberOfEvents );
}


/** @class PAPI_shutdown
  *	@brief Finish using PAPI and free all related resources. 
  *
  *	@par C Prototype:
  *		\#include <papi.h> @n
  *		void PAPI_shutdown( void );
  *
  * PAPI_shutdown() is an exit function used by the PAPI Library 
  * to free resources and shut down when certain error conditions arise. 
  * It is not necessary for the user to call this function, 
  * but doing so allows the user to have the capability to free memory 
  * and resources used by the PAPI Library.
  *
  *	@see PAPI_init_library
  */
void
PAPI_shutdown( void )
{
	APIDBG( "Entry:\n");

        EventSetInfo_t *ESI;
        ThreadInfo_t *master;
        DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
        int i, j = 0, k, retval;


	if ( init_retval == DEADBEEF ) {
		PAPIERROR( PAPI_SHUTDOWN_str );
		return;
	}

	MPX_shutdown(  );

	/* Free all EventSets for this thread */

   master = _papi_hwi_lookup_thread( 0 );

      /* Count number of running EventSets AND */
      /* Stop any running EventSets in this thread */

#ifdef DEBUG
again:
#endif
   for( i = 0; i < map->totalSlots; i++ ) {
      ESI = map->dataSlotArray[i];
      if ( ESI ) {
	 if ( ESI->master == master ) {
	    if ( ESI->state & PAPI_RUNNING ) {
	       if((retval = PAPI_stop( i, NULL )) != PAPI_OK) {
	    	   APIDBG("Call to PAPI_stop failed: %d\n", retval);
	       }
	    }
	    retval=PAPI_cleanup_eventset( i );
	    if (retval!=PAPI_OK) PAPIERROR("Error during cleanup.");
	    _papi_hwi_free_EventSet( ESI );
	 } 
         else {
            if ( ESI->state & PAPI_RUNNING ) {
	       j++;
	    }
	 }
      }
   }

	/* No locking required, we're just waiting for the others
	   to call shutdown or stop their eventsets. */

#ifdef DEBUG
	if ( j != 0 ) {
		PAPIERROR( PAPI_SHUTDOWN_SYNC_str );
		sleep( 1 );
		j = 0;
		goto again;
	}
#endif

	// if we have some user events defined, release the space they allocated
	// give back the strings which were allocated when each event was created
	for ( i=0 ; i<user_defined_events_count ; i++) {
		papi_free (user_defined_events[i].symbol);
		papi_free (user_defined_events[i].postfix);
		papi_free (user_defined_events[i].long_descr);
		papi_free (user_defined_events[i].short_descr);
		papi_free (user_defined_events[i].note);
		for ( k=0 ; k<(int)(user_defined_events[i].count) ; k++) {
			papi_free (user_defined_events[i].name[k]);
		}
	}
	// make sure the user events list is empty
	memset (user_defined_events, '\0' , sizeof(user_defined_events));
	user_defined_events_count = 0;

	/* Shutdown the entire component */
	//_papi_hwi_shutdown_highlevel(  );
	_papi_hwi_shutdown_global_internal(  );
	_papi_hwi_shutdown_global_threads(  );
	for( i = 0; i < papi_num_components; i++ ) {
	   if (!_papi_hwd[i]->cmp_info.disabled) {
              _papi_hwd[i]->shutdown_component(  );
	   }
	}

	/* Now it is safe to call re-init */

	init_retval = DEADBEEF;
	init_level = PAPI_NOT_INITED;
	_papi_mem_cleanup_all(  );
}

/** @class PAPI_strerror
 *	@brief Returns a string describing the PAPI error code. 
 *
 *  @par C Interface:
 *     \#include <papi.h> @n
 *     char * PAPI_strerror( int errorCode );
 *
 *  @param[in] code  
 *      -- the error code to interpret 
 *
 *	@retval *error 
 *		-- a pointer to the error string. 
 *	@retval NULL 
 *		-- the input error code to PAPI_strerror() is invalid. 
 *
 *	PAPI_strerror() returns a pointer to the error message corresponding to the 
 *	error code code. 
 *	If the call fails the function returns the NULL pointer. 
 *	This function is not implemented in Fortran.
 *
 *  @par Example:
 *  @code
 *  int ret;
 *  int EventSet = PAPI_NULL;
 *  int native = 0x0;
 *  char error_str[PAPI_MAX_STR_LEN];
 *
 *  ret = PAPI_create_eventset(&EventSet);
 *  if (ret != PAPI_OK)
 *  {
 *     fprintf(stderr, "PAPI error %d: %s\n", ret, PAPI_strerror(retval));
 *     exit(1);
 *  }
 *  // Add Total Instructions Executed to our EventSet
 *  ret = PAPI_add_event(EventSet, PAPI_TOT_INS);
 *  if (ret != PAPI_OK)
 *  {
 *     PAPI_perror( "PAPI_add_event");
 *     fprintf(stderr,"PAPI_error %d: %s\n", ret, error_str);
 *     exit(1);
 *  }
 *  // Start counting
 *  ret = PAPI_start(EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  @endcode
 *
 *	@see  PAPI_perror PAPI_set_opt PAPI_get_opt PAPI_shutdown PAPI_set_debug
 */
char *
PAPI_strerror( int errorCode )
{
	if ( ( errorCode > 0 ) || ( -errorCode > _papi_hwi_num_errors ) )
		return ( NULL );

	return ( _papi_errlist[-errorCode] );
}

/** @class PAPI_perror
 *  @brief Produces a string on standard error, describing the last library error.
 *
 * @par C Interface:
 *     \#include <papi.h> @n
 *     void PAPI_perror( const char *s );
 *
 *  @param[in] s
 *      -- Optional message to print before the string describing the last error message. 
 * 
 * 	The routine PAPI_perror() produces a message on the standard error output,
 * 	describing the last error encountered during a call to PAPI. 
 * 	If s is not NULL, s is printed, followed by a colon and a space. 
 * 	Then the error message and a new-line are printed. 
 *
 *  @par Example:
 *  @code
 *  int ret;
 *  int EventSet = PAPI_NULL;
 *  int native = 0x0;
 *
 *  ret = PAPI_create_eventset(&EventSet);
 *  if (ret != PAPI_OK)
 *  {
 *     fprintf(stderr, \"PAPI error %d: %s\\n\", ret, PAPI_strerror(retval));
 *     exit(1);
 *  }
 *  // Add Total Instructions Executed to our EventSet
 *  ret = PAPI_add_event(EventSet, PAPI_TOT_INS);
 *  if (ret != PAPI_OK)
 *  {
 *     PAPI_perror( "PAPI_add_event" );
 *     exit(1);
 *  }
 *  // Start counting
 *  ret = PAPI_start(EventSet);
 *  if (ret != PAPI_OK) handle_error(ret);
 *  @endcode
 *
 *  @see PAPI_strerror
 */
void
PAPI_perror( const char *msg )
{
	char *foo;

	foo = PAPI_strerror( _papi_hwi_errno );
	if ( foo == NULL )
		return;

	if ( msg )
		if ( *msg )
			fprintf( stderr, "%s: ", msg );

	fprintf( stderr, "%s\n", foo );
}

/** @class PAPI_overflow
 *  @brief Set up an event set to begin registering overflows.
 *
 * PAPI_overflow() marks a specific EventCode in an EventSet to generate an 
 * overflow signal after every threshold events are counted. 
 * More than one event in an event set can be used to trigger overflows. 
 * In such cases, the user must call this function once for each overflowing 
 * event. 
 * To turn off overflow on a specified event, call this function with a 
 * threshold value of 0.
 *
 * Overflows can be implemented in either software or hardware, but the scope 
 * is the entire event set. 
 * PAPI defaults to hardware overflow if it is available. 
 * In the case of software overflow, a periodic timer interrupt causes PAPI 
 * to compare the event counts against the threshold values and call the 
 * overflow handler if one or more events have exceeded their threshold. 
 * In the case of hardware overflow, the counters are typically set to the 
 * negative of the threshold value and count up to 0. 
 * This zero-crossing triggers a hardware interrupt that calls the overflow 
 * handler. 
 * Because of this counter interrupt, the counter values for overflowing 
 * counters 
 * may be very small or even negative numbers, and cannot be relied upon 
 * as accurate. 
 * In such cases the overflow handler can approximate the counts by supplying 
 * the threshold value whenever an overflow occurs. 
 *
 * _papi_overflow_handler()  is  a placeholder for a user-defined function
 * to process overflow events.  A pointer to this function  is  passed  to
 * the  PAPI_overflow  routine, where it is invoked whenever a software or
 * hardware overflow occurs.  This handler receives the  EventSet  of  the
 * overflowing  event,  the  Program  Counter  address  when the interrupt
 * occurred, an overflow_vector that can be processed to  determined  which
 * event(s)  caused  the  overflow,  and a pointer to the machine context,
 * which can be used in a  platform-specific  manor  to  extract  register
 * information about what was happening when the overflow occurred.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_overflow (int EventSet, int EventCode, int threshold, 
 * int flags, PAPI_overflow_handler_t handler ); @n@n
 * (*PAPI_overflow_handler_t) _papi_overflow_handler
 * (int  EventSet, void *address, long_long overflow_vector, 
 * void *context );
 *
 * @par Fortran Interface:
 * Not implemented
 *
 * @param[in] EventSet
 *	      -- an integer handle to a PAPI event set as created by 
 *            @ref PAPI_create_eventset
 * @param[in] EventCode
 *	      -- the preset or native event code to be set for overflow 
 *            detection. 
 *	      This event must have already been added to the EventSet.
 * @param[in] threshold
 *	      -- the overflow threshold value for this EventCode.
 * @param[in] flags
 *	      -- bitmap that controls the overflow mode of operation. 
 *	      Set to PAPI_OVERFLOW_FORCE_SW to force software 
 *            overflowing, even if hardware overflow support is available. 
 *	      If hardware overflow support is available on a given system, 
 *            it will be the default mode of operation. 
 *	      There are situations where it is advantageous to use software 
 *            overflow instead. 
 *	      Although software overflow is inherently less accurate, 
 *            with more latency and processing overhead, it does allow for 
 *            overflowing on derived events,  and for the accurate recording 
 *            of overflowing event counts. 
 *	      These two features are typically not available with hardware 
 *            overflow. 
 *	      Only one type of overflow is allowed per event set, so 
 *            setting one event to hardware overflow and another to forced 
 *            software overflow will result in an error being returned.
 *	@param[in] handler
 *	      -- pointer to the user supplied handler function to call upon 
 *            overflow 
 *      @param[in] address 
 *            -- the Program Counter address at the time of the overflow
 *      @param[in] overflow_vector  
 *            -- a long long word containing flag bits to indicate
 *               which hardware counter(s) caused the overflow
 *      @param[in] *context 
 *            -- pointer to a machine specific structure that defines the
 *               register context at the time of overflow. This parameter 
 *               is often unused and can be ignored in the user function.
 *
 * @retval PAPI_OK On success, PAPI_overflow returns PAPI_OK.  
 * @retval PAPI_EINVAL One or more of the arguments is invalid.   
 *            Most likely a bad threshold value.
 * @retval PAPI_ENOMEM Insufficient memory to complete the operation.
 * @retval PAPI_ENOEVST The EventSet specified does not exist.
 * @retval PAPI_EISRUN The EventSet is currently counting events.
 * @retval PAPI_ECNFLCT The underlying counter hardware cannot count 
 *             this event and other events in the EventSet simultaneously. 
 *             Also can happen if you are trying to overflow both by hardware
 *             and by forced software at the same time.
 * @retval PAPI_ENOEVNT The PAPI event is not available on 
 *             the underlying hardware.
 *
 * @par Example
 * @code
 * // Define a simple overflow handler:
 * void handler(int EventSet, void *address, long_long overflow_vector, void *context)
 * {
 *    fprintf(stderr,\"Overflow at %p! bit=%#llx \\n\",
 *             address,overflow_vector);
 * }
 *
 * // Call PAPI_overflow for an EventSet containing PAPI_TOT_INS,
 * // setting the threshold to 100000. Use the handler defined above.
 * retval = PAPI_overflow(EventSet, PAPI_TOT_INS, 100000, 0, handler);
 * @endcode
 *
 *
 * @see PAPI_get_overflow_event_index
 *
 */
int
PAPI_overflow( int EventSet, int EventCode, int threshold, int flags,
	       PAPI_overflow_handler_t handler )
{
	APIDBG( "Entry: EventSet: %d, EventCode: %#x, threshold: %d, flags: %#x, handler: %p\n", EventSet, EventCode, threshold, flags, handler);
	int retval, cidx, index, i;
	EventSetInfo_t *ESI;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL ) {
		OVFDBG("No EventSet\n");
		papi_return( PAPI_ENOEVST );
        }

	cidx = valid_ESI_component( ESI );
	if ( cidx < 0 ) {
		OVFDBG("Component Error\n");
		papi_return( cidx );
	}

	if ( ( ESI->state & PAPI_STOPPED ) != PAPI_STOPPED ) {
		OVFDBG("Already running\n");
		papi_return( PAPI_EISRUN );
	}

	if ( ESI->state & PAPI_ATTACHED ) {
		OVFDBG("Attached\n");
		papi_return( PAPI_EINVAL );
	}
	
	if ( ESI->state & PAPI_CPU_ATTACHED ) {
		OVFDBG("CPU attached\n");
		papi_return( PAPI_EINVAL );
	}
	
	if ( ( index = _papi_hwi_lookup_EventCodeIndex( ESI,
      					( unsigned int ) EventCode ) ) < 0 ) {
		papi_return( PAPI_ENOEVNT );
	}

	if ( threshold < 0 ) {
		OVFDBG("Threshold below zero\n");
		papi_return( PAPI_EINVAL );
	}

	/* We do not support derived events in overflow */
	/* Unless it's DERIVED_CMPD in which no calculations are done */

	if ( !( flags & PAPI_OVERFLOW_FORCE_SW ) && threshold != 0 &&
		 ( ESI->EventInfoArray[index].derived ) &&
		 ( ESI->EventInfoArray[index].derived != DERIVED_CMPD ) ) {
		OVFDBG("Derived event in overflow\n");
		papi_return( PAPI_EINVAL );
	}

	/* the first time to call PAPI_overflow function */

	if ( !( ESI->state & PAPI_OVERFLOWING ) ) {
		if ( handler == NULL ) {
			OVFDBG("NULL handler\n");
			papi_return( PAPI_EINVAL );
		}
		if ( threshold == 0 ) {
			OVFDBG("Zero threshold\n");
			papi_return( PAPI_EINVAL );
		}
	}
	if ( threshold > 0 &&
		 ESI->overflow.event_counter >= _papi_hwd[cidx]->cmp_info.num_cntrs )
		papi_return( PAPI_ECNFLCT );

	if ( threshold == 0 ) {
		for ( i = 0; i < ESI->overflow.event_counter; i++ ) {
			if ( ESI->overflow.EventCode[i] == EventCode )
				break;
		}
		/* EventCode not found */
		if ( i == ESI->overflow.event_counter )
			papi_return( PAPI_EINVAL );
		/* compact these arrays */
		while ( i < ESI->overflow.event_counter - 1 ) {
			ESI->overflow.deadline[i] = ESI->overflow.deadline[i + 1];
			ESI->overflow.threshold[i] = ESI->overflow.threshold[i + 1];
			ESI->overflow.EventIndex[i] = ESI->overflow.EventIndex[i + 1];
			ESI->overflow.EventCode[i] = ESI->overflow.EventCode[i + 1];
			i++;
		}
		ESI->overflow.deadline[i] = 0;
		ESI->overflow.threshold[i] = 0;
		ESI->overflow.EventIndex[i] = 0;
		ESI->overflow.EventCode[i] = 0;
		ESI->overflow.event_counter--;
	} else {
		if ( ESI->overflow.event_counter > 0 ) {
			if ( ( flags & PAPI_OVERFLOW_FORCE_SW ) &&
				 ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) )
				papi_return( PAPI_ECNFLCT );
			if ( !( flags & PAPI_OVERFLOW_FORCE_SW ) &&
				 ( ESI->overflow.flags & PAPI_OVERFLOW_FORCE_SW ) )
				papi_return( PAPI_ECNFLCT );
		}
		for ( i = 0; i < ESI->overflow.event_counter; i++ ) {
			if ( ESI->overflow.EventCode[i] == EventCode )
				break;
		}
		/* A new entry */
		if ( i == ESI->overflow.event_counter ) {
			ESI->overflow.EventCode[i] = EventCode;
			ESI->overflow.event_counter++;
		}
		/* New or existing entry */
		ESI->overflow.deadline[i] = threshold;
		ESI->overflow.threshold[i] = threshold;
		ESI->overflow.EventIndex[i] = index;
		ESI->overflow.flags = flags;

	}

	/* If overflowing is already active, we should check to
	   make sure that we don't specify a different handler
	   or different flags here. You can't mix them. */

	ESI->overflow.handler = handler;

	/* Set up the option structure for the low level.
	   If we have hardware interrupts and we are not using
	   forced software emulated interrupts */

	if ( _papi_hwd[cidx]->cmp_info.hardware_intr &&
		 !( ESI->overflow.flags & PAPI_OVERFLOW_FORCE_SW ) ) {
		retval = _papi_hwd[cidx]->set_overflow( ESI, index, threshold );
		if ( retval == PAPI_OK )
			ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
		else {
			papi_return( retval );	/* We should undo stuff here */
		}
	} else {
		/* Make sure hardware overflow is not set */
		ESI->overflow.flags &= ~( PAPI_OVERFLOW_HARDWARE );
	}

	APIDBG( "Overflow using: %s\n",
			( ESI->overflow.
			  flags & PAPI_OVERFLOW_HARDWARE ? "[Hardware]" : ESI->overflow.
			  flags & PAPI_OVERFLOW_FORCE_SW ? "[Forced Software]" :
			  "[Software]" ) );

	/* Toggle the overflow flags and ESI state */

	if ( ESI->overflow.event_counter >= 1 )
		ESI->state |= PAPI_OVERFLOWING;
	else {
		ESI->state ^= PAPI_OVERFLOWING;
		ESI->overflow.flags = 0;
		ESI->overflow.handler = NULL;
	}

	return PAPI_OK;
}

/** @class PAPI_sprofil
 *	@brief Generate PC histogram data from multiple code regions where hardware counter overflow occurs.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_sprofil( PAPI_sprofil_t * prof, int profcnt, int EventSet, int EventCode, int threshold, int flags );
 *
 *	@param *prof 
 *		pointer to an array of PAPI_sprofil_t structures. Each copy of the structure contains the following:
 *  @arg buf -- pointer to a buffer of bufsiz bytes in which the histogram counts are stored in an array of unsigned short, unsigned int, or unsigned long long values, or 'buckets'. The size of the buckets is determined by values in the flags argument.
 *  @arg bufsiz -- the size of the histogram buffer in bytes. It is computed from the length of the code region to be profiled, the size of the buckets, and the scale factor as discussed below.
 *  @arg offset -- the start address of the region to be profiled.
 *  @arg scale -- broadly and historically speaking, a contraction factor that indicates how much smaller the histogram buffer is than the region to be profiled. More precisely, scale is interpreted as an unsigned 16-bit fixed-point fraction with the decimal point implied on the left. Its value is the reciprocal of the number of addresses in a subdivision, per counter of histogram buffer.
 *
 *	@param profcnt 
 *		number of structures in the prof array for hardware profiling.
 *	@param EventSet 
 *		The PAPI EventSet to profile. This EventSet is marked as profiling-ready, 
 *		but profiling doesn't actually start until a PAPI_start() call is issued.
 *	@param EventCode
 *		Code of the Event in the EventSet to profile. 
 *		This event must already be a member of the EventSet.
 *	@param threshold 
 *		minimum number of events that must occur before the PC is sampled. 
 *		If hardware overflow is supported for your component, this threshold will 
 *		trigger an interrupt when reached. 
 *		Otherwise, the counters will be sampled periodically and the PC will be 
 *		recorded for the first sample that exceeds the threshold. 
 *		If the value of threshold is 0, profiling will be disabled for this event.
 *	@param flags 
 *		bit pattern to control profiling behavior. 
 *		Defined values are given in a table in the documentation for PAPI_pofil
 *	@manonly
 *
 *	@endmanonly
 *
 *  @retval
 *		Return values for PAPI_sprofil() are identical to those for PAPI_profil.
 *		Please refer to that page for further details.
 *	@manonly
 *
 *	@endmanonly
 *
 *	PAPI_sprofil() is a structure driven profiler that profiles one or more 
 *	disjoint regions of code in a single call. 
 *	It accepts a pointer to a preinitialized array of sprofil structures, and 
 *	initiates profiling based on the values contained in the array. 
 *	Each structure in the array defines the profiling parameters that are 
 *	normally passed to PAPI_profil(). 
 *	For more information on profiling, @ref PAPI_profil
 *	@manonly
 *
 *	@endmanonly
 *
 * @par Example:
 * @code
 * int retval;
 * unsigned long length;
 * PAPI_exe_info_t *prginfo;
 * unsigned short *profbuf1, *profbuf2, profbucket;
 * PAPI_sprofil_t sprof[3];
 *
 * prginfo = PAPI_get_executable_info();
 * if (prginfo == NULL) handle_error( NULL );
 * length = (unsigned long)(prginfo->text_end - prginfo->text_start);
 * // Allocate 2 buffers of equal length
 * profbuf1 = (unsigned short *)malloc(length);
 * profbuf2 = (unsigned short *)malloc(length);
 * if ((profbuf1 == NULL) || (profbuf2 == NULL))
 *   handle_error( NULL );
 * memset(profbuf1,0x00,length);
 * memset(profbuf2,0x00,length);
 * // First buffer
 * sprof[0].pr_base = profbuf1;
 * sprof[0].pr_size = length;
 * sprof[0].pr_off = (vptr_t) DO_FLOPS;
 * sprof[0].pr_scale = 0x10000;
 * // Second buffer
 * sprof[1].pr_base = profbuf2;
 * sprof[1].pr_size = length;
 * sprof[1].pr_off = (vptr_t) DO_READS;
 * sprof[1].pr_scale = 0x10000;
 * // Overflow bucket
 * sprof[2].pr_base = profbucket;
 * sprof[2].pr_size = 1;
 * sprof[2].pr_off = 0;
 * sprof[2].pr_scale = 0x0002;
 * retval = PAPI_sprofil(sprof, EventSet, PAPI_FP_INS, 1000000,
 * PAPI_PROFIL_POSIX | PAPI_PROFIL_BUCKET_16)) != PAPI_OK)
 * if ( retval != PAPI_OK ) handle_error( retval );
 * @endcode
 *
 *	@see PAPI_overflow
 *	@see PAPI_get_executable_info
 *	@see PAPI_profil
 */
int
PAPI_sprofil( PAPI_sprofil_t *prof, int profcnt, int EventSet,
			  int EventCode, int threshold, int flags )
{
	APIDBG( "Entry: prof: %p, profcnt: %d, EventSet: %d, EventCode: %#x, threshold: %d, flags: %#x\n", prof, profcnt, EventSet, EventCode, threshold, flags);
   EventSetInfo_t *ESI;
   int retval, index, i, buckets;
   int forceSW = 0;
   int cidx;

   /* Check to make sure EventSet exists */
   ESI = _papi_hwi_lookup_EventSet( EventSet );
   if ( ESI == NULL ) {
      papi_return( PAPI_ENOEVST );
   }

   /* Check to make sure EventSet is stopped */
   if ( ( ESI->state & PAPI_STOPPED ) != PAPI_STOPPED ) {
      papi_return( PAPI_EISRUN );
   }

   /* We cannot profile if attached */
   if ( ESI->state & PAPI_ATTACHED ) {
      papi_return( PAPI_EINVAL );
   }

   /* We cannot profile if cpu attached */
   if ( ESI->state & PAPI_CPU_ATTACHED ) {
      papi_return( PAPI_EINVAL );
   }

   /* Get component for EventSet */
   cidx = valid_ESI_component( ESI );
   if ( cidx < 0 ) {
      papi_return( cidx );
   }

   /* Get index of the Event we want to profile */
   if ( ( index = _papi_hwi_lookup_EventCodeIndex( ESI,
				      (unsigned int) EventCode ) ) < 0 ) {
      papi_return( PAPI_ENOEVNT );
   }

   /* We do not support derived events in overflow */
   /* Unless it's DERIVED_CMPD in which no calculations are done */
   if ( ( ESI->EventInfoArray[index].derived ) &&
	( ESI->EventInfoArray[index].derived != DERIVED_CMPD ) &&
	!( flags & PAPI_PROFIL_FORCE_SW ) ) {
      papi_return( PAPI_EINVAL );
   }

   /* If no prof structures, then make sure count is 0 */
   if ( prof == NULL ) {
      profcnt = 0;
   }

   /* check all profile regions for valid scale factors of:
      2 (131072/65536),
      1 (65536/65536),
      or < 1 (65535 -> 2) as defined in unix profil()
      2/65536 is reserved for single bucket profiling
      {0,1}/65536 are traditionally used to terminate profiling
      but are unused here since PAPI uses threshold instead
    */
   for( i = 0; i < profcnt; i++ ) {
      if ( !( ( prof[i].pr_scale == 131072 ) ||
	   ( ( prof[i].pr_scale <= 65536 && prof[i].pr_scale > 1 ) ) ) ) {
	 APIDBG( "Improper scale factor: %d\n", prof[i].pr_scale );
	 papi_return( PAPI_EINVAL );
      }
   }

   /* Make sure threshold is valid */
   if ( threshold < 0 ) {
      papi_return( PAPI_EINVAL );
   }

   /* the first time to call PAPI_sprofil */
   if ( !( ESI->state & PAPI_PROFILING ) ) {
      if ( threshold == 0 ) {
	 papi_return( PAPI_EINVAL );
      }
   }

   /* ??? */
   if ( (threshold > 0) &&
	(ESI->profile.event_counter >= _papi_hwd[cidx]->cmp_info.num_cntrs) ) {
      papi_return( PAPI_ECNFLCT );
   }

   if ( threshold == 0 ) {
      for( i = 0; i < ESI->profile.event_counter; i++ ) {
	 if ( ESI->profile.EventCode[i] == EventCode ) {
	    break;
	 }
      }
		
      /* EventCode not found */
      if ( i == ESI->profile.event_counter ) {
	 papi_return( PAPI_EINVAL );
      }

      /* compact these arrays */
      while ( i < ESI->profile.event_counter - 1 ) {
         ESI->profile.prof[i] = ESI->profile.prof[i + 1];
	 ESI->profile.count[i] = ESI->profile.count[i + 1];
	 ESI->profile.threshold[i] = ESI->profile.threshold[i + 1];
	 ESI->profile.EventIndex[i] = ESI->profile.EventIndex[i + 1];
	 ESI->profile.EventCode[i] = ESI->profile.EventCode[i + 1];
	 i++;
      }
      ESI->profile.prof[i] = NULL;
      ESI->profile.count[i] = 0;
      ESI->profile.threshold[i] = 0;
      ESI->profile.EventIndex[i] = 0;
      ESI->profile.EventCode[i] = 0;
      ESI->profile.event_counter--;
   } else {
      if ( ESI->profile.event_counter > 0 ) {
	 if ( ( flags & PAPI_PROFIL_FORCE_SW ) &&
	      !( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
	    papi_return( PAPI_ECNFLCT );
	 }
	 if ( !( flags & PAPI_PROFIL_FORCE_SW ) &&
	      ( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
	    papi_return( PAPI_ECNFLCT );
	 }
      }

      for( i = 0; i < ESI->profile.event_counter; i++ ) {
	 if ( ESI->profile.EventCode[i] == EventCode ) {
	    break;
	 }
      }

      if ( i == ESI->profile.event_counter ) {
	 i = ESI->profile.event_counter;
	 ESI->profile.event_counter++;
	 ESI->profile.EventCode[i] = EventCode;
      }
      ESI->profile.prof[i] = prof;
      ESI->profile.count[i] = profcnt;
      ESI->profile.threshold[i] = threshold;
      ESI->profile.EventIndex[i] = index;
   }

   APIDBG( "Profile event counter is %d\n", ESI->profile.event_counter );

   /* Clear out old flags */
   if ( threshold == 0 ) {
      flags |= ESI->profile.flags;
   }

   /* make sure no invalid flags are set */
   if ( flags &
	~( PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED |
	   PAPI_PROFIL_COMPRESS | PAPI_PROFIL_BUCKETS | PAPI_PROFIL_FORCE_SW |
	   PAPI_PROFIL_INST_EAR | PAPI_PROFIL_DATA_EAR ) ) {
      papi_return( PAPI_EINVAL );
   }

   /* if we have kernel-based profiling, then we're just asking for 
      signals on interrupt. */
   /* if we don't have kernel-based profiling, then we're asking for 
      emulated PMU interrupt */
   if ( ( flags & PAPI_PROFIL_FORCE_SW ) &&
	( _papi_hwd[cidx]->cmp_info.kernel_profile == 0 ) ) {
      forceSW = PAPI_OVERFLOW_FORCE_SW;
   }

   /* make sure one and only one bucket size is set */
   buckets = flags & PAPI_PROFIL_BUCKETS;
   if ( !buckets ) {
      flags |= PAPI_PROFIL_BUCKET_16;	/* default to 16 bit if nothing set */
   }
   else {
      /* return error if more than one set */
      if ( !( ( buckets == PAPI_PROFIL_BUCKET_16 ) ||
	      ( buckets == PAPI_PROFIL_BUCKET_32 ) ||
	      ( buckets == PAPI_PROFIL_BUCKET_64 ) ) ) {
	 papi_return( PAPI_EINVAL );
      }
   }

   /* Set up the option structure for the low level */
   ESI->profile.flags = flags;

   if ( _papi_hwd[cidx]->cmp_info.kernel_profile &&
	!( ESI->profile.flags & PAPI_PROFIL_FORCE_SW ) ) {
      retval = _papi_hwd[cidx]->set_profile( ESI, index, threshold );
      if ( ( retval == PAPI_OK ) && ( threshold > 0 ) ) {
	 /* We need overflowing because we use the overflow dispatch handler */
	 ESI->state |= PAPI_OVERFLOWING;
	 ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
      }
   } else {
      retval = PAPI_overflow( EventSet, EventCode, threshold, forceSW,
			      _papi_hwi_dummy_handler );
   }
	
   if ( retval < PAPI_OK ) {
      papi_return( retval );	/* We should undo stuff here */
   }

   /* Toggle the profiling flags and ESI state */

   if ( ESI->profile.event_counter >= 1 ) {
      ESI->state |= PAPI_PROFILING;
   }
   else {
      ESI->state ^= PAPI_PROFILING;
      ESI->profile.flags = 0;
   }

   return PAPI_OK;
}

/** @class PAPI_profil
 *  @brief Generate a histogram of hardware counter overflows vs. PC addresses.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_profil(void *buf, unsigned bufsiz, unsigned long offset,
 * unsigned scale, int EventSet, int EventCode, int threshold, int flags );
 *
 * @par Fortran Interface
 * The profiling routines have no Fortran interface.
 *
 * @param *buf
 *    -- pointer to a buffer of bufsiz bytes in which the histogram counts are 
 *	 stored in an array of unsigned short, unsigned int, or 
 *	 unsigned long long values, or 'buckets'. 
 *	 The size of the buckets is determined by values in the flags argument.
 * @param bufsiz
 *    -- the size of the histogram buffer in bytes. 
 *	 It is computed from the length of the code region to be profiled, 
 *	 the size of the buckets, and the scale factor as discussed above.
 * @param offset
 *    -- the start address of the region to be profiled.
 * @param scale
 *    -- broadly and historically speaking, a contraction factor that 
 *       indicates how much smaller the histogram buffer is than the 
 *       region to be profiled.  More precisely, scale is interpreted as an 
 *       unsigned 16-bit fixed-point fraction with the decimal point 
 *       implied on the left. 
 *	 Its value is the reciprocal of the number of addresses in a 
 *       subdivision, per counter of histogram buffer. 
 *	 Below is a table of representative values for scale.
 * @param EventSet
 *    -- The PAPI EventSet to profile. This EventSet is marked as 
 *       profiling-ready, but profiling doesn't actually start until a 
 *       PAPI_start() call is issued.
 * @param EventCode
 *    -- Code of the Event in the EventSet to profile. 
 *	 This event must already be a member of the EventSet.
 * @param threshold
 *    -- minimum number of events that must occur before the PC is sampled. 
 *	 If hardware overflow is supported for your component, this threshold 
 *	 will trigger an interrupt when reached. 
 *	 Otherwise, the counters will be sampled periodically and the PC will 
 *       be recorded for the first sample that exceeds the threshold. 
 *	 If the value of threshold is 0, profiling will be disabled for 
 *       this event.
 * @param flags
 *    -- bit pattern to control profiling behavior. 
 *	 Defined values are shown in the table above.
 *
 * @retval PAPI_OK 
 * @retval PAPI_EINVAL 
 *	   One or more of the arguments is invalid.
 * @retval PAPI_ENOMEM 
 *	   Insufficient memory to complete the operation.
 * @retval PAPI_ENOEVST 
 *	   The EventSet specified does not exist.
 * @retval PAPI_EISRUN 
 *	   The EventSet is currently counting events.
 * @retval PAPI_ECNFLCT 
 *	   The underlying counter hardware can not count this event and other 
 *	   events in the EventSet simultaneously.
 * @retval PAPI_ENOEVNT 
 *	   The PAPI preset is not available on the underlying hardware. 
 *
 *	PAPI_profil() provides hardware event statistics by profiling 
 *      the occurrence of specified hardware counter events.
 *	It is designed to mimic the UNIX SVR4 profil call.
 *	
 *	The statistics are generated by creating a histogram of hardware 
 *      counter event overflows vs. program counter addresses for the current 
 *      process. The histogram is defined for a specific region of program 
 *      code to be profiled, and the identified region is logically broken up 
 *      into a set of equal size subdivisions, each of which corresponds to a 
 *      count in the histogram. 
 *	
 *	With each hardware event overflow, the current subdivision is 
 *      identified and its corresponding histogram count is incremented. 
 *	These counts establish a relative measure of how many hardware counter 
 *	events are occurring in each code subdivision.
 *	
 *	The resulting histogram counts for a profiled region can be used to 
 *	identify those program addresses that generate a disproportionately 
 *	high percentage of the event of interest.
 *
 *	Events to be profiled are specified with the EventSet and 
 *      EventCode parameters.   More than one event can be simultaneously 
 *      profiled by calling PAPI_profil() 
 *	several times with different EventCode values. 
 *	Profiling can be turned off for a given event by calling PAPI_profil() 
 *	with a threshold value of 0. 
 *
 *	@par Representative values for the scale variable
 *  @manonly
 * HEX      DECIMAL  DEFININTION  
 * 0x20000  131072   Maps precisely one instruction address to a unique bucket in buf.  
 * 0x10000   65536   Maps precisely two instruction addresses to a unique bucket in buf.  
 * 0x0FFFF   65535   Maps approximately two instruction addresses to a unique bucket in buf.  
 * 0x08000   32768   Maps every four instruction addresses to a bucket in buf.  
 * 0x04000   16384   Maps every eight instruction addresses to a bucket in buf.  
 * 0x00002       2   Maps all instruction addresses to the same bucket in buf.  
 * 0x00001       1   Undefined.  
 * 0x00000       0   Undefined.  
 * @endmanonly
 * @htmlonly
 * <table class="doxtable">
 * <tr><th>HEX</th>     <th>DECIMAL</th>  <th>DEFINITION</th></tr>
 * <tr><td>0x20000</td>	<td> 131072</td>  <td>Maps precisely one instruction address to a unique bucket in buf.</td></tr>
 * <tr><td>0x10000</td>	<td>  65536</td>  <td>Maps precisely two instruction addresses to a unique bucket in buf.</td></tr>
 * <tr><td>0xFFFF</td>	<td>  65535</td> <td>Maps approximately two instruction addresses to a unique bucket in buf.</td></tr>
 * <tr><td>0x8000</td>	<td>  32768</td> <td>Maps every four instruction addresses to a bucket in buf.</td></tr>
 * <tr><td>0x4000</td>	<td>  16384</td> <td>Maps every eight instruction addresses to a bucket in buf.</td></tr>
 * <tr><td>0x0002</td>	<td>      2</td> <td>Maps all instruction addresses to the same bucket in buf.</td></tr>
 * <tr><td>0x0001</td>	<td>      1</td> <td>Undefined.</td></tr>
 * <tr><td>0x0000</td>	<td>      0</td> <td>Undefined. </td></tr>
 * </table>
 * @endhtmlonly
 *
 *	Historically, the scale factor was introduced to allow the 
 *      allocation of buffers smaller than the code size to be profiled. 
 *	Data and instruction sizes were assumed to be multiples of 16-bits. 
 *	These assumptions are no longer necessarily true. 
 *	PAPI_profil() has preserved the traditional definition of 
 *      scale where appropriate, but deprecated the definitions for 0 and 1 
 *      (disable scaling) and extended the range of scale to include 
 *      65536 and 131072 to allow for exactly two 
 *	addresses and exactly one address per profiling bucket.
 *
 *	The value of bufsiz is computed as follows:
 *	
 *	bufsiz = (end - start)*(bucket_size/2)*(scale/65536) where
 * @arg bufsiz - the size of the buffer in bytes
 * @arg end, start - the ending and starting addresses of the profiled region
 * @arg bucket_size - the size of each bucket in bytes; 2, 4, or 8 as defined in flags 
 *
 *	@par Defined bits for the flags variable:
 * @arg PAPI_PROFIL_POSIX	Default type of profiling, similar to profil (3).@n
 * @arg PAPI_PROFIL_RANDOM	Drop a random 25% of the samples.@n
 * @arg PAPI_PROFIL_WEIGHTED	Weight the samples by their value.@n
 * @arg PAPI_PROFIL_COMPRESS	Ignore samples as values in the hash buckets get big.@n
 * @arg PAPI_PROFIL_BUCKET_16	Use unsigned short (16 bit) buckets, This is the default bucket.@n
 * @arg PAPI_PROFIL_BUCKET_32	Use unsigned int (32 bit) buckets.@n
 * @arg PAPI_PROFIL_BUCKET_64	Use unsigned long long (64 bit) buckets.@n
 * @arg PAPI_PROFIL_FORCE_SW	Force software overflow in profiling. @n
 *
 * @par Example
 * @code
 * int retval;
 * unsigned long length;
 * PAPI_exe_info_t *prginfo;
 * unsigned short *profbuf;
 *
 * if ((prginfo = PAPI_get_executable_info()) == NULL)
 *    handle_error(1);
 *
 * length = (unsigned long)(prginfo->text_end - prginfo->text_start);
 *
 * profbuf = (unsigned short *)malloc(length);
 * if (profbuf == NULL)
 *    handle_error(1);
 * memset(profbuf,0x00,length);
 *
 * if ((retval = PAPI_profil(profbuf, length, start, 65536, EventSet,
 *     PAPI_FP_INS, 1000000, PAPI_PROFIL_POSIX | PAPI_PROFIL_BUCKET_16)) 
 *    != PAPI_OK)
 *    handle_error(retval);
 * @endcode
 *
 * @bug If you call PAPI_profil, PAPI allocates buffer space that will not be 
 *      freed if you call PAPI_shutdown or PAPI_cleanup_eventset. 
 *      To clean all memory, you must call PAPI_profil on the Events with 
 *      a 0 threshold. 
 *
 * @see PAPI_overflow 
 * @see PAPI_sprofil
 *
 */
int
PAPI_profil( void *buf, unsigned bufsiz, vptr_t offset,
			 unsigned scale, int EventSet, int EventCode, int threshold,
			 int flags )
{
	APIDBG( "Entry: buf: %p, bufsiz: %d, offset: %p, scale: %u, EventSet: %d, EventCode: %#x, threshold: %d, flags: %#x\n", buf, bufsiz, offset, scale, EventSet, EventCode, threshold, flags);
	EventSetInfo_t *ESI;
	int i;
	int retval;

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* scale factors are checked for validity in PAPI_sprofil */

	if ( threshold > 0 ) {
		PAPI_sprofil_t *prof;

		for ( i = 0; i < ESI->profile.event_counter; i++ ) {
			if ( ESI->profile.EventCode[i] == EventCode )
				break;
		}

		if ( i == ESI->profile.event_counter ) {
			prof =
				( PAPI_sprofil_t * ) papi_malloc( sizeof ( PAPI_sprofil_t ) );
			memset( prof, 0x0, sizeof ( PAPI_sprofil_t ) );
			prof->pr_base = buf;
			prof->pr_size = bufsiz;
			prof->pr_off = offset;
			prof->pr_scale = scale;

			retval =
				PAPI_sprofil( prof, 1, EventSet, EventCode, threshold, flags );

			if ( retval != PAPI_OK )
				papi_free( prof );
		} else {
			prof = ESI->profile.prof[i];
			prof->pr_base = buf;
			prof->pr_size = bufsiz;
			prof->pr_off = offset;
			prof->pr_scale = scale;
			retval =
				PAPI_sprofil( prof, 1, EventSet, EventCode, threshold, flags );
		}
		papi_return( retval );
	}

	for ( i = 0; i < ESI->profile.event_counter; i++ ) {
		if ( ESI->profile.EventCode[i] == EventCode )
			break;
	}
	/* EventCode not found */
	if ( i == ESI->profile.event_counter )
		papi_return( PAPI_EINVAL );

	papi_free( ESI->profile.prof[i] );
	ESI->profile.prof[i] = NULL;

	papi_return( PAPI_sprofil( NULL, 0, EventSet, EventCode, 0, flags ) );
}

/* This function sets the low level default granularity
   for all newly manufactured eventsets. The first function
   preserves API compatibility and assumes component 0;
   The second function takes a component argument. */

/** @class PAPI_set_granularity
 *	@brief Set the default counting granularity for eventsets bound to the cpu component.
 *
 *	@par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_set_granularity( int granularity );
 *
 *	@param -- granularity one of the following constants as defined in the papi.h header file
 *	@arg PAPI_GRN_THR	-- Count each individual thread
 *	@arg PAPI_GRN_PROC	-- Count each individual process
 *	@arg PAPI_GRN_PROCG	-- Count each individual process group
 *	@arg PAPI_GRN_SYS	-- Count the current CPU
 *	@arg PAPI_GRN_SYS_CPU	-- Count all CPUs individually
 *	@arg PAPI_GRN_MIN	-- The finest available granularity
 *	@arg PAPI_GRN_MAX	-- The coarsest available granularity 
 *  @manonly
 *  @endmanonly
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *  @manonly
 *  @endmanonly
 *
 *	PAPI_set_granularity sets the default counting granularity for all new 
 *	event sets created by PAPI_create_eventset. 
 *	This call implicitly sets the granularity for the cpu component 
 *	(component 0) and is included to preserve backward compatibility. 
 *
 *	@par Example:
 *	@code
int ret;

// Initialize the library
ret = PAPI_library_init(PAPI_VER_CURRENT);
if (ret > 0 && ret != PAPI_VER_CURRENT) {
  fprintf(stderr,"PAPI library version mismatch!\n");
  exit(1); 
}
if (ret < 0) handle_error(ret);

// Set the default granularity for the cpu component
ret = PAPI_set_granularity(PAPI_GRN_PROC);
if (ret != PAPI_OK) handle_error(ret);
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see  PAPI_set_cmp_granularity PAPI_set_domain PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_granularity( int granularity )
{
	return ( PAPI_set_cmp_granularity( granularity, 0 ) );
}

/** @class PAPI_set_cmp_granularity
 *	@brief Set the default counting granularity for eventsets bound to the specified component.
 *
 *	@par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_set_cmp_granularity( int granularity, int cidx );
 *
 *	@param granularity one of the following constants as defined in the papi.h header file
 *	@arg PAPI_GRN_THR	Count each individual thread
 *	@arg PAPI_GRN_PROC	Count each individual process
 *	@arg PAPI_GRN_PROCG	Count each individual process group
 *	@arg PAPI_GRN_SYS	Count the current CPU
 *	@arg PAPI_GRN_SYS_CPU	Count all CPUs individually
 *	@arg PAPI_GRN_MIN	The finest available granularity
 *	@arg PAPI_GRN_MAX	The coarsest available granularity
 *
 *	@param cidx
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *  @manonly
 *  @endmanonly
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *  @manonly
 *  @endmanonly
 *
 *	PAPI_set_cmp_granularity sets the default counting granularity for all new 
 *	event sets, and requires an explicit component argument. 
 *	Event sets that are already in existence are not affected. 
 *
 *	To change the granularity of an existing event set, please see PAPI_set_opt. 
 *	The reader should note that the granularity of an event set affects only 
 *	the mode in which the counter continues to run. 
 *
 *	@par Example:
 *	@code
int ret;

// Initialize the library
ret = PAPI_library_init(PAPI_VER_CURRENT);
if (ret > 0 && ret != PAPI_VER_CURRENT) {
  fprintf(stderr,"PAPI library version mismatch!\n");
  exit(1); 
}
if (ret < 0) handle_error(ret);

// Set the default granularity for the cpu component
ret = PAPI_set_cmp_granularity(PAPI_GRN_PROC, 0);
if (ret != PAPI_OK) handle_error(ret);
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see  PAPI_set_granularity PAPI_set_domain PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_cmp_granularity( int granularity, int cidx )
{
	PAPI_option_t ptr;

	memset( &ptr, 0, sizeof ( ptr ) );
	ptr.defgranularity.def_cidx = cidx;
	ptr.defgranularity.granularity = granularity;
	papi_return( PAPI_set_opt( PAPI_DEFGRN, &ptr ) );
}

/* This function sets the low level default counting domain
   for all newly manufactured eventsets. The first function
   preserves API compatibility and assumes component 0;
   The second function takes a component argument. */

/** @class PAPI_set_domain
 *	@brief Set the default counting domain for new event sets bound to the cpu component.
 *
 *	@par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_set_domain( int domain );
 *
 *	@param domain one of the following constants as defined in the papi.h header file
 *	@arg PAPI_DOM_USER User context counted
 *	@arg PAPI_DOM_KERNEL  Kernel/OS context counted
 *	@arg PAPI_DOM_OTHER Exception/transient mode counted
 *	@arg PAPI_DOM_SUPERVISOR Supervisor/hypervisor context counted
 *	@arg PAPI_DOM_ALL All above contexts counted
 *	@arg PAPI_DOM_MIN The smallest available context
 *	@arg PAPI_DOM_MAX The largest available context 
 *  @manonly
 *  @endmanonly
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *  @manonly
 *  @endmanonly
 * 
 *	PAPI_set_domain sets the default counting domain for all new event sets 
 *	created by PAPI_create_eventset in all threads. 
 *	This call implicitly sets the domain for the cpu component (component 0) 
 *	and is included to preserve backward compatibility. 
 *
 *	@par Example:
 *	@code
int ret;

// Initialize the library
ret = PAPI_library_init(PAPI_VER_CURRENT);
if (ret > 0 && ret != PAPI_VER_CURRENT) {
  fprintf(stderr,"PAPI library version mismatch!\n");
  exit(1); 
}
if (ret < 0) handle_error(ret);

// Set the default domain for the cpu component
ret = PAPI_set_domain(PAPI_DOM_KERNEL);
if (ret != PAPI_OK) handle_error(ret);
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see PAPI_set_cmp_domain PAPI_set_granularity PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_domain( int domain )
{
	return ( PAPI_set_cmp_domain( domain, 0 ) );
}

/** @class PAPI_set_cmp_domain
 *	@brief Set the default counting domain for new event sets bound to the specified component.
 *
 *	@par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_set_cmp_domain( int domain, int  cidx );
 *
 *	@param domain one of the following constants as defined in the papi.h header file
 *	@arg PAPI_DOM_USER User context counted
 *	@arg PAPI_DOM_KERNEL  Kernel/OS context counted
 *	@arg PAPI_DOM_OTHER Exception/transient mode counted
 *	@arg PAPI_DOM_SUPERVISOR Supervisor/hypervisor context counted
 *	@arg PAPI_DOM_ALL All above contexts counted
 *	@arg PAPI_DOM_MIN The smallest available context
 *	@arg PAPI_DOM_MAX The largest available context 
 *	@arg PAPI_DOM_HWSPEC Something other than CPU like stuff. Individual components can decode
 *  low order bits for more meaning
 *
 *	@param cidx
 *		An integer identifier for a component. 
 *		By convention, component 0 is always the cpu component. 
 *  @manonly
 *  @endmanonly
 *
 *	@retval PAPI_OK 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOCMP 
 *		The argument cidx is not a valid component.
 *  @manonly
 *  @endmanonly
 *
 *	PAPI_set_cmp_domain sets the default counting domain for all new event sets 
 *	in all threads, and requires an explicit component argument. 
 *	Event sets that are already in existence are not affected. 
 *	To change the domain of an existing event set, please see PAPI_set_opt.
 *	The reader should note that the domain of an event set affects only the 
 *	mode in which the counter continues to run. 
 *	Counts are still aggregated for the current process, and not for any other 
 *	processes in the system. 
 *	Thus when requesting PAPI_DOM_KERNEL , the user is asking for events that 
 *	occur on behalf of the process, inside the kernel. 
 *
 *	@par Example:
 *	@code
int ret;

// Initialize the library
ret = PAPI_library_init(PAPI_VER_CURRENT);
if (ret > 0 && ret != PAPI_VER_CURRENT) {
  fprintf(stderr,"PAPI library version mismatch!\n");
  exit(1); 
}
if (ret < 0) handle_error(ret);

// Set the default domain for the cpu component
ret = PAPI_set_cmp_domain(PAPI_DOM_KERNEL,0);
if (ret != PAPI_OK) handle_error(ret);
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *	@see PAPI_set_domain PAPI_set_granularity PAPI_set_opt PAPI_get_opt
 */
int
PAPI_set_cmp_domain( int domain, int cidx )
{
	PAPI_option_t ptr;

	memset( &ptr, 0, sizeof ( ptr ) );
	ptr.defdomain.def_cidx = cidx;
	ptr.defdomain.domain = domain;
	papi_return( PAPI_set_opt( PAPI_DEFDOM, &ptr ) );
}

/**	@class PAPI_add_events
 *	@brief add multiple PAPI presets or native hardware events to an event set 
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	int PAPI_add_events( int  EventSet, int * EventCodes, int  number );
 *
 *	PAPI_add_event adds one event to a PAPI Event Set. PAPI_add_events does 
 *	the same, but for an array of events. @n
 *	A hardware event can be either a PAPI preset or a native hardware event code.
 *	For a list of PAPI preset events, see PAPI_presets or run the avail test case
 *	in the PAPI distribution. PAPI presets can be passed to PAPI_query_event to see
 *	if they exist on the underlying architecture.
 *	For a list of native events available on current platform, run native_avail
 *	test case in the PAPI distribution. For the encoding of native events,
 *	see PAPI_event_name_to_code to learn how to generate native code for the
 *	supported native event on the underlying architecture.
 *
 *	@param EventSet
 *		An integer handle for a PAPI Event Set as created by PAPI_create_eventset.
 *	@param *EventCode 
 *		An array of defined events.
 *	@param number 
 *		An integer indicating the number of events in the array *EventCode.
 *		It should be noted that PAPI_add_events can partially succeed, 
 *		exactly like PAPI_remove_events. 
 *
 *	@retval Positive-Integer
 *		The number of consecutive elements that succeeded before the error. 
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOMEM 
 *		Insufficient memory to complete the operation.
 *	@retval PAPI_ENOEVST 
 *		The event set specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The event set is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other events 
 *		in the event set simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware.
 *	@retval PAPI_EBUG 
 *		Internal error, please send mail to the developers. 
 *
 *	@par Examples:
 *	@code
 *	int EventSet = PAPI_NULL;
 *	unsigned int native = 0x0;
 *	if ( PAPI_create_eventset( &EventSet ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add Total Instructions Executed to our EventSet
 *	if ( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK )
 *	handle_error( 1 );
 *	// Add native event PM_CYC to EventSet
 *	if ( PAPI_event_name_to_code( "PM_CYC", &native ) != PAPI_OK )
 *	handle_error( 1 );
 *	if ( PAPI_add_event( EventSet, native ) != PAPI_OK )
 *	handle_error( 1 );
 *	@endcode
 *
 *	@bug
 *	The vector function should take a pointer to a length argument so a proper 
 *	return value can be set upon partial success.
 *
 *	@see PAPI_cleanup_eventset @n
 *	PAPI_destroy_eventset @n
 *	PAPI_event_code_to_name @n
 *	PAPI_remove_events @n
 *	PAPI_query_event @n
 *	PAPI_presets @n
 *	PAPI_native @n
 *	PAPI_remove_event
 */
int
PAPI_add_events( int EventSet, int *Events, int number )
{
	APIDBG( "Entry: EventSet: %d, Events: %p, number: %d\n", EventSet, Events, number);
	int i, retval;

	if ( ( Events == NULL ) || ( number <= 0 ) )
		papi_return( PAPI_EINVAL );

	for ( i = 0; i < number; i++ ) {
		retval = PAPI_add_event( EventSet, Events[i] );
		if ( retval != PAPI_OK ) {
			if ( i == 0 )
				papi_return( retval );
			else
				return ( i );
		}
	}
	return ( PAPI_OK );
}

/** @class PAPI_remove_events
 * @brief Remove an array of hardware event codes from a PAPI event set.
 *
 * A hardware event can be either a PAPI Preset or a native hardware event code. 
 * For a list of PAPI preset events, see PAPI_presets or run the papi_avail utility in the PAPI distribution. 
 * PAPI Presets can be passed to PAPI_query_event to see if they exist on the underlying architecture. 
 * For a list of native events available on current platform, run papi_native_avail in the PAPI distribution. 
 * It should be noted that PAPI_remove_events can partially succeed, exactly like PAPI_add_events. 
 *
 *	@par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_remove_events( int  EventSet, int * EventCode, int  number );
 *
 *	@param EventSet
 *		an integer handle for a PAPI event set as created by PAPI_create_eventset
 *	@param *Events
 *		an array of defined events
 *	@param number
 *		an integer indicating the number of events in the array *EventCode 
 *
 *	@retval Positive integer 
 *		The number of consecutive elements that succeeded before the error.
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *	@retval PAPI_ENOEVST 
 *		The EventSet specified does not exist.
 *	@retval PAPI_EISRUN 
 *		The EventSet is currently counting events.
 *	@retval PAPI_ECNFLCT 
 *		The underlying counter hardware can not count this event and other 
 *		events in the EventSet simultaneously.
 *	@retval PAPI_ENOEVNT 
 *		The PAPI preset is not available on the underlying hardware. 
 *
 *	@par Example:
 *	@code
int EventSet = PAPI_NULL;
int Events[] = {PAPI_TOT_INS, PAPI_FP_OPS};
int ret;
 
 // Create an empty EventSet
ret = PAPI_create_eventset(&EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Add two events to our EventSet
ret = PAPI_add_events(EventSet, Events, 2);
if (ret != PAPI_OK) handle_error(ret);

// Start counting
ret = PAPI_start(EventSet);
if (ret != PAPI_OK) handle_error(ret);

// Stop counting, ignore values
ret = PAPI_stop(EventSet, NULL);
if (ret != PAPI_OK) handle_error(ret);

// Remove event
ret = PAPI_remove_events(EventSet, Events, 2);
if (ret != PAPI_OK) handle_error(ret);
 *	@endcode
 *
 *  @bug The last argument should be a pointer so the count can be returned on partial success in addition
 *  to a real error code.
 *
 *	@see PAPI_cleanup_eventset PAPI_destroy_eventset PAPI_event_name_to_code 
 *		PAPI_presets PAPI_add_event PAPI_add_events
 */
int
PAPI_remove_events( int EventSet, int *Events, int number )
{
	APIDBG( "Entry: EventSet: %d, Events: %p, number: %d\n", EventSet, Events, number);
	int i, retval;

	if ( ( Events == NULL ) || ( number <= 0 ) )
		papi_return( PAPI_EINVAL );

	for ( i = 0; i < number; i++ ) {
		retval = PAPI_remove_event( EventSet, Events[i] );
		if ( retval != PAPI_OK ) {
			if ( i == 0 )
				papi_return( retval );
			else
				return ( i );
		}
	}
	return ( PAPI_OK );
}

/**	@class PAPI_list_events
 *	@brief list the events in an event set
 *
 *	PAPI_list_events() returns an array of events and a count of the
 *  total number of events in an event set.
 *	This call assumes an initialized PAPI library and a successfully created event set.
 *
 * @par C Interface
 * \#include <papi.h> @n
 * int PAPI_list_events(int EventSet, int *Events, int *number);
*
 *	@param[in] EventSet
 *		An integer handle for a PAPI event set as created by PAPI_create_eventset 
 *	@param[in,out] *Events 
 *		A pointer to a preallocated array of codes for events, such as PAPI_INT_INS. 
 *		No more than *number codes will be stored into the array.
 *	@param[in,out] *number 
 *		On input, the size of the Events array, or maximum number of event codes
 *		to be returned. A value of 0 can be used to probe an event set.
 *		On output, the number of events actually in the event set.
 *		This value may be greater than the actually stored number of event codes. 
 *
 *	@retval PAPI_EINVAL
 *	@retval PAPI_ENOEVST
 *	
 *	@par Examples:
 *	@code
 		if (PAPI_event_name_to_code("PAPI_TOT_INS",&EventCode) != PAPI_OK)
 		exit(1);
 		if (PAPI_add_event(EventSet, EventCode) != PAPI_OK)
 		exit(1);
 		Convert a second event name to an event code 
 		if (PAPI_event_name_to_code("PAPI_L1_LDM",&EventCode) != PAPI_OK)
 		exit(1);
 		if (PAPI_add_event(EventSet, EventCode) != PAPI_OK)
 		exit(1);
 		number = 0;
 		if(PAPI_list_events(EventSet, NULL, &number))
 		exit(1);
 		if(number != 2)
 		exit(1);
 		if(PAPI_list_events(EventSet, Events, &number))
 		exit(1);
 *	@endcode
 *	@see PAPI_event_code_to_name 
 *	@see PAPI_event_name_to_code 
 *	@see PAPI_add_event
 *	@see PAPI_create_eventset
 */
int
PAPI_list_events( int EventSet, int *Events, int *number )
{
	APIDBG( "Entry: EventSet: %d, Events: %p, number: %p\n", EventSet, Events, number);
	EventSetInfo_t *ESI;
	int i, j;

	if ( *number < 0 )
		papi_return( PAPI_EINVAL );

	if ( ( Events == NULL ) && ( *number > 0 ) )
		papi_return( PAPI_EINVAL );

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( !ESI )
		papi_return( PAPI_ENOEVST );

	if ( ( Events == NULL ) || ( *number == 0 ) ) {
		*number = ESI->NumberOfEvents;
		papi_return( PAPI_OK );
	}

	for ( i = 0, j = 0; j < ESI->NumberOfEvents; i++ ) {
		if ( ( int ) ESI->EventInfoArray[i].event_code != PAPI_NULL ) {
			Events[j] = ( int ) ESI->EventInfoArray[i].event_code;
			j++;
			if ( j == *number )
				break;
		}
	}

	*number = j;

	return ( PAPI_OK );
}

/* xxx This is OS dependent, not component dependent, right? */
/** @class PAPI_get_dmem_info
 *	@brief Get information about the dynamic memory usage of the current program. 
 *
 *	@par C Prototype:
 *		\#include <papi.h> @n
 *		int PAPI_get_dmem_info( PAPI_dmem_info_t *dest );
 *
 *	@param dest
 *		structure to be filled in @ref PAPI_dmem_info_t
 *	
 *	@retval PAPI_ECMP
 *		The function is not implemented for the current component.
 *	@retval PAPI_EINVAL 
 *		Any value in the structure or array may be undefined as indicated by 
 *		this error value.
 *	@retval PAPI_SYS 
 *		A system error occurred.
 *
 *	@note This function is only implemented for the Linux operating system.
 *	This function takes a pointer to a PAPI_dmem_info_t structure 
 *	and returns with the structure fields filled in. 
 *	A value of PAPI_EINVAL in any field indicates an undefined parameter. 
 *
 *	@see PAPI_get_executable_info PAPI_get_hardware_info PAPI_get_opt PAPI_library_init
 */
int
PAPI_get_dmem_info( PAPI_dmem_info_t * dest )
{
	if ( dest == NULL )
		return PAPI_EINVAL;

	memset( ( void * ) dest, 0x0, sizeof ( PAPI_dmem_info_t ) );
	return ( _papi_os_vector.get_dmem_info( dest ) );
}


/** @class PAPI_get_executable_info
 *	@brief Get the executable's address space info.
 *
 *	@par C Interface:
 *	\#include <papi.h> @n
 *	const PAPI_exe_info_t *PAPI_get_executable_info( void );
 *
 *	This function returns a pointer to a structure containing information 
 *	about the current program.
 *
 *	@param fullname
 *		Fully qualified path + filename of the executable.
 *	@param name
 *		Filename of the executable with no path information.
 *	@param text_start, text_end
 *		Start and End addresses of program text segment.
 *	@param data_start, data_end
 *		Start and End addresses of program data segment.
 *	@param bss_start, bss_end
 *		Start and End addresses of program bss segment.
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. 
 *
 *	@par Examples:
 *	@code
 *	const PAPI_exe_info_t *prginfo = NULL;
 *	if ( ( prginfo = PAPI_get_executable_info( ) ) == NULL )
 *	exit( 1 );
 *	printf( "Path+Program: %s\n", exeinfo->fullname );
 *	printf( "Program: %s\n", exeinfo->address_info.name );
 *	printf( "Text start: %p, Text end: %p\n", exeinfo->address_info.text_start, exeinfo->address_info.text_end) ;
 *	printf( "Data start: %p, Data end: %p\n", exeinfo->address_info.data_start, exeinfo->address_info.data_end );
 *	printf( "Bss start: %p, Bss end: %p\n", exeinfo->address_info.bss_start, exeinfo->address_info.bss_end );
 *	@endcode
 *
 *	@see PAPI_get_opt 
 *	@see PAPI_get_hardware_info 
 *	@see PAPI_exe_info_t
 */
const PAPI_exe_info_t *
PAPI_get_executable_info( void )
{
	PAPI_option_t ptr;
	int retval;

	memset( &ptr, 0, sizeof ( ptr ) );
	retval = PAPI_get_opt( PAPI_EXEINFO, &ptr );
	if ( retval == PAPI_OK )
		return ( ptr.exe_info );
	else
		return ( NULL );
}

/** @class PAPI_get_shared_lib_info
 *	@brief Get address info about the shared libraries used by the process. 
 *
 *	In C, this function returns a pointer to a structure containing information 
 *	about the shared library used by the program. 
 *	There is no Fortran equivalent call. 
 *	@note This data will be incorporated into the PAPI_get_executable_info call in the future. PAPI_get_shared_lib_info will be deprecated and should be used with caution.
 *
 *	@bug If called before initialization the behavior of the routine is undefined.
 *
 *	@see PAPI_shlib_info_t
 *	@see PAPI_get_hardware_info
 *	@see PAPI_get_executable_info 
 *	@see PAPI_get_dmem_info 
 *	@see PAPI_get_opt PAPI_library_init
 */
const PAPI_shlib_info_t *
PAPI_get_shared_lib_info( void )
{
	PAPI_option_t ptr;
	int retval;

	memset( &ptr, 0, sizeof ( ptr ) );
	retval = PAPI_get_opt( PAPI_SHLIBINFO, &ptr );
	if ( retval == PAPI_OK )
		return ( ptr.shlib_info );
	else
		return ( NULL );
}
/**	@class PAPI_get_hardware_info 
 *	@brief get information about the system hardware
 *
 *	In C, this function returns a pointer to a structure containing information about the hardware on which the program runs. 
 *       In Fortran, the values of the structure are returned explicitly.
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid.
 *
 *	@bug
 *		If called before initialization the behavior of the routine is undefined. 
 *	
 *	@note The C structure contains detailed information about cache and TLB sizes. 
 *		This information is not available from Fortran.
 *
 *	@par Examples:
 *	@code
 		const PAPI_hw_info_t *hwinfo = NULL;
 		if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)	
 		exit(1);
 		if ((hwinfo = PAPI_get_hardware_info()) == NULL)
 		exit(1);
 		printf("%d CPUs at %f Mhz.\en",hwinfo->totalcpus,hwinfo->mhz);
 *	@endcode	
 *
 *	@see PAPI_hw_info_t
 *	@see PAPI_get_executable_info, PAPI_get_opt, PAPI_get_dmem_info, PAPI_library_init
 */
const PAPI_hw_info_t *
PAPI_get_hardware_info( void )
{
	PAPI_option_t ptr;
	int retval;

	memset( &ptr, 0, sizeof ( ptr ) );
	retval = PAPI_get_opt( PAPI_HWINFO, &ptr );
	if ( retval == PAPI_OK )
		return ( ptr.hw_info );
	else
		return ( NULL );
}


/* The next 4 timing functions always use component 0 */

/**	@class PAPI_get_real_cyc
 *	@brief get real time counter value in clock cycles 
 * 	Returns the total real time passed since some arbitrary starting point. 
 *	The time is returned in clock cycles. 
 *	This call is equivalent to wall clock time.
 *		  
 *	@par Examples:
 *	@code
 		s = PAPI_get_real_cyc();
 		your_slow_code();
 		e = PAPI_get_real_cyc();
 		printf("Wallclock cycles: %lld\en",e-s);
 *	@endcode	
 *	@see PAPIF  PAPI PAPI_get_virt_usec PAPI_get_virt_cyc PAPI_library_init
 */
long long
PAPI_get_real_cyc( void )
{
	return ( _papi_os_vector.get_real_cycles(  ) );
}

/** @class PAPI_get_real_nsec
 *	@brief Get real time counter value in nanoseconds.
 *
 *	This function returns the total real time passed since some arbitrary 
 *	starting point. 
 *	The time is returned in nanoseconds. 
 *	This call is equivalent to wall clock time.
 *
 *	@see PAPI_get_virt_usec 
 *	@see PAPI_get_virt_cyc 
 *	@see PAPI_library_init
 */

/* FIXME */
long long
PAPI_get_real_nsec( void )
{
  return ( ( _papi_os_vector.get_real_nsec(  )));

}

/**	@class PAPI_get_real_usec
 *	@brief get real time counter value in microseconds 
 *
 *	This function returns the total real time passed since some arbitrary 
 *	starting point. 
 *	The time is returned in microseconds. 
 *	This call is equivalent to wall clock time.
 *	@par Examples:
 *	@code
		s = PAPI_get_real_cyc();
		your_slow_code();
		e = PAPI_get_real_cyc();
		printf("Wallclock cycles: %lld\en",e-s);
 *	@endcode
 *	@see PAPIF
 *	@see PAPI
 *	@see PAPI_get_virt_usec 
 *	@see PAPI_get_virt_cyc 
 *	@see PAPI_library_init
 */
long long
PAPI_get_real_usec( void )
{
	return ( _papi_os_vector.get_real_usec(  ) );
}

/**	@class PAPI_get_virt_cyc 
 *	@brief get virtual time counter value in clock cycles 
 *
 *	@retval PAPI_ECNFLCT 
 *		If there is no master event set. 
 *		This will happen if the library has not been initialized, or 	
 *		for threaded applications, if there has been no thread id 
 *		function defined by the  		PAPI_thread_init function.
 *	@retval PAPI_ENOMEM
 *		For threaded applications, if there has not yet been any thread 
 *		specific master event created for the current thread, and if 
 *		the allocation of such an event set fails, the call will return 
 *		PAPI_ENOMEM or PAPI_ESYS . 
 *
 *	This function returns the total number of virtual units from some 
 *	arbitrary starting point. 
 *	Virtual units accrue every time the process is running in user-mode on 
 *	behalf of the process. 
 *	Like the real time counters, this count is guaranteed to exist on every platform 
 *	PAPI supports. 
 *	However on some platforms, the resolution can be as bad as 1/Hz as defined 
 *	by the operating system.
 *	@par Examples:
 *	@code
 		s = PAPI_get_virt_cyc();
 		your_slow_code();
 		e = PAPI_get_virt_cyc();
 		printf("Process has run for cycles: %lld\en",e-s);
 *	@endcode
 */
long long
PAPI_get_virt_cyc( void )
{

	return ( ( long long ) _papi_os_vector.get_virt_cycles( ) );
}

/** @class PAPI_get_virt_nsec
 *	@brief Get virtual time counter values in nanoseconds.
 *
 *	@retval PAPI_ECNFLCT 
 *		If there is no master event set. 
 *		This will happen if the library has not been initialized, or for threaded 
 *		applications, if there has been no thread id function defined by the 
 *		PAPI_thread_init function.
 *	@retval PAPI_ENOMEM
 *		For threaded applications, if there has not yet been any thread specific
 *		master event created for the current thread, and if the allocation of 
 *		such an event set fails, the call will return PAPI_ENOMEM or PAPI_ESYS . 
 *
 *	This function returns the total number of virtual units from some 
 *	arbitrary starting point. 
 *	Virtual units accrue every time the process is running in user-mode on 
 *	behalf of the process. 
 *	Like the real time counters, this count is guaranteed to exist on every platform 
 *	PAPI supports. 
 *	However on some platforms, the resolution can be as bad as 1/Hz as defined 
 *	by the operating system. 
 *
 */
long long
PAPI_get_virt_nsec( void )
{

  return ( ( _papi_os_vector.get_virt_nsec()));

}

/**	@class PAPI_get_virt_usec
 *	@brief get virtual time counter values in microseconds 
 *
 *	@retval PAPI_ECNFLCT 
 *		If there is no master event set. 
 *		This will happen if the library has not been initialized, or for threaded 
 *		applications, if there has been no thread id function defined by the 
 *		PAPI_thread_init function.
 *	@retval PAPI_ENOMEM
 *		For threaded applications, if there has not yet been any thread 
 *		specific master event created for the current thread, and if the 
 *		allocation of such an event set fails, the call will return PAPI_ENOMEM or PAPI_ESYS . 
 *
 *	This function returns the total number of virtual units from some 
 *	arbitrary starting point. 
 *	Virtual units accrue every time the process is running in user-mode on 
 *	behalf of the process. 
 *	Like the real time counters, this count is guaranteed to exist on every 
 *	platform PAPI supports. However on some platforms, the resolution can be 
 *	as bad as 1/Hz as defined by the operating system.
 *	@par Examples:
 *	@code
 		s = PAPI_get_virt_cyc();
 		your_slow_code();
 		e = PAPI_get_virt_cyc();
 		printf("Process has run for cycles: %lld\en",e-s);
 *	@endcode
 *	@see PAPIF
 *	@see PAPI
 *	@see PAPI
 *	@see PAPI_get_real_cyc
 *	@see PAPI_get_virt_cyc

 */
long long
PAPI_get_virt_usec( void )
{

	return ( ( long long ) _papi_os_vector.get_virt_usec() );
}

/** @class PAPI_lock
 *  @brief Lock one of two mutex variables defined in papi.h.
 *
 *  PAPI_lock() grabs access to one of the two PAPI mutex variables. 
 *  This function is provided to the user to have a platform independent call 
 *  to a (hopefully) efficiently implemented mutex.
 *
 *  @par C Interface:
 *  \#include <papi.h> @n
 *  void PAPI_lock(int lock);
 *
 *  @param[in] lock
 *    -- an integer value specifying one of the two user locks: PAPI_USR1_LOCK or PAPI_USR2_LOCK 
 *
 *  @returns
 *      There is no return value for this call. 
 *      Upon return from  PAPI_lock the current thread has acquired 
 *      exclusive access to the specified PAPI mutex.
 *
 *  @see PAPI_unlock 
 *  @see PAPI_thread_init
 */
int
PAPI_lock( int lck )
{
	if ( ( lck < 0 ) || ( lck >= PAPI_MAX_LOCK ) )
		papi_return( PAPI_EINVAL );

	papi_return( _papi_hwi_lock( lck ) );
}

/** @class PAPI_unlock
 *	@brief Unlock one of the mutex variables defined in papi.h.
 *
 *	@param lck
 *		an integer value specifying one of the two user locks: PAPI_USR1_LOCK 
 *		or PAPI_USR2_LOCK 
 *
 *	PAPI_unlock() unlocks the mutex acquired by a call to PAPI_lock .
 *
 *	@see PAPI_thread_init
 */
int
PAPI_unlock( int lck )
{
	if ( ( lck < 0 ) || ( lck >= PAPI_MAX_LOCK ) )
		papi_return( PAPI_EINVAL );

	papi_return( _papi_hwi_unlock( lck ) );
}

/**	@class PAPI_is_initialized
 *	@brief check for initialization
 *	@retval PAPI_NOT_INITED
 *		Library has not been initialized
 *	@retval PAPI_LOW_LEVEL_INITED
 *		Low level has called library init
 *	@retval PAPI_HIGH_LEVEL_INITED
 *		High level has called library init 
 *	@retval PAPI_THREAD_LEVEL_INITED	
 *		Threads have been inited 
 *	
 *	@param version
		 upon initialization, PAPI checks the argument against the internal value of PAPI_VER_CURRENT when the library was compiled. 
 *	This guards against portability problems when updating the PAPI shared libraries on your system.
 *	@par Examples:
 *	@code
 		int retval;
 		retval = PAPI_library_init(PAPI_VER_CURRENT);
 		if (retval != PAPI_VER_CURRENT && retval > 0) {
 		fprintf(stderr,"PAPI library version mismatch!\en");
 		exit(1); }
 		if (retval < 0)
 		handle_error(retval);
 		retval = PAPI_is_initialized();
 		if (retval != PAPI_LOW_LEVEL_INITED)
 		handle_error(retval);
 *	@endcode
 *	PAPI_is_initialized() returns the status of the PAPI library. 
 *	The PAPI library can be in one of four states, as described under RETURN VALUES. 
 *	@bug	If you don't call this before using any of the low level PAPI calls, your application could core dump.
 *	@see PAPI 
 *	@see PAPI_thread_init
 */
int
PAPI_is_initialized( void )
{
	return ( init_level );
}

/* This function maps the overflow_vector to event indexes in the event
   set, so that user can know which PAPI event overflowed.
   int *array---- an array of event indexes in eventset; the first index
                  maps to the highest set bit in overflow_vector
   int *number--- this is an input/output parameter, user should put the
                  size of the array into this parameter, after the function
                  is executed, the number of indexes in *array is written
                  to this parameter
*/

/**	@class PAPI_get_overflow_event_index
 *	@brief converts an overflow vector into an array of indexes to overflowing events 
 *	@param EventSet
 *		an integer handle to a PAPI event set as created by PAPI_create_eventset
 *	@param overflow_vector
 *		a vector with bits set for each counter that overflowed. 
 *		This vector is passed by the system to the overflow handler routine.
 *	@param *array
 *		an array of indexes for events in EventSet. 
 *		No more than *number indexes will be stored into the array.
 *	@param *number 
 *		On input the variable determines the size of the array. 
 *		On output the variable contains the number of indexes in the array. 
 *
 *	@retval PAPI_EINVAL 
 *		One or more of the arguments is invalid. This could occur if the overflow_vector is empty (zero), if the array or number pointers are NULL, if the value of number is less than one, or if the EventSet is empty.
 *	@retval PAPI_ENOEVST
		The EventSet specified does not exist.
 *	@par Examples
 *	@code
 		void handler(int EventSet, void *address, long_long overflow_vector, void *context){
 		int Events[4], number, i;
 		int total = 0, retval;
 		printf("Overflow #%d\n  Handler(%d) Overflow at %p! vector=%#llx\n",
 		total, EventSet, address, overflow_vector);
 		total++;
 		number = 4;
 		retval = PAPI_get_overflow_event_index(EventSet,
 		overflow_vector, Events, &number);
 		if(retval == PAPI_OK)
 		for(i=0; i<number; i++) printf("Event index[%d] = %d", i, Events[i]);}
 *	@endcode
 *	@bug This function may not return all overflowing events if used with software-driven overflow of multiple derived events.
 *	PAPI_get_overflow_event_index decomposes an overflow_vector into an event 
 *	index array in which the first element corresponds to the least significant set bit in overflow_vector and so on. Based on overflow_vector, the user can only tell which physical counters overflowed. Using this function, the user can map overflowing counters to specific events in the event set. An array is used in this function to support the possibility of multiple simultaneous overflow events.
 *
 *	@see PAPI_overflow
 */
int
PAPI_get_overflow_event_index( int EventSet, long long overflow_vector,
							   int *array, int *number )
{
	APIDBG( "Entry: EventSet: %d, overflow_vector: %lld, array: %p, number: %p\n", EventSet, overflow_vector, array, number);
	EventSetInfo_t *ESI;
	int set_bit, j, pos;
	int count = 0, k;

	if ( overflow_vector == ( long long ) 0 )
		papi_return( PAPI_EINVAL );

	if ( ( array == NULL ) || ( number == NULL ) )
		papi_return( PAPI_EINVAL );

	if ( *number < 1 )
		papi_return( PAPI_EINVAL );

	ESI = _papi_hwi_lookup_EventSet( EventSet );
	if ( ESI == NULL )
		papi_return( PAPI_ENOEVST );

	/* in case the eventset is empty */
	if ( ESI->NumberOfEvents == 0 )
		papi_return( PAPI_EINVAL );

	while ( ( set_bit = ffsll( overflow_vector ) ) ) {
		set_bit -= 1;
		overflow_vector ^= ( long long ) 1 << set_bit;
		for ( j = 0; j < ESI->NumberOfEvents; j++ ) {
			for ( k = 0, pos = 0; k < PAPI_EVENTS_IN_DERIVED_EVENT && pos >= 0; k++ ) {
				pos = ESI->EventInfoArray[j].pos[k];
				if ( ( set_bit == pos ) &&
					 ( ( ESI->EventInfoArray[j].derived == NOT_DERIVED ) ||
					   ( ESI->EventInfoArray[j].derived == DERIVED_CMPD ) ) ) {
					array[count++] = j;
					if ( count == *number )
						return PAPI_OK;

					break;
				}
			}
		}
	}
	*number = count;
	return PAPI_OK;
}


/**	@class PAPI_get_event_component
 *	@brief return component an event belongs to
 *	@retval ENOCMP
 *		component does not exist
 *	
 *	@param EventCode
 *              EventCode for which we want to know the component index
 *	@par Examples:
 *	@code
 		int cidx,eventcode;
 		cidx = PAPI_get_event_component(eventcode);
 *	@endcode
 *	PAPI_get_event_component() returns the component an event
 *      belongs to.
 *	@bug	Doesn't work for preset events
 *	@see  PAPI_get_event_info
 */
int
PAPI_get_event_component( int EventCode)
{
    APIDBG( "Entry: EventCode: %#x\n", EventCode);
    return _papi_hwi_component_index( EventCode);
}

/**	@class PAPI_get_component_index
 *	@brief returns the component index for the named component
 *	@retval ENOCMP
 *		component does not exist
 *
 *	@param name
 *              name of component to find index for
 *	@par Examples:
 *	@code
 		int cidx;
 		cidx = PAPI_get_component_index("cuda");
		if (cidx==PAPI_OK) {
                   printf("The CUDA component is cidx %d\n",cidx);
                }
 *	@endcode
 *	PAPI_get_component_index() returns the component index of
 *      the named component.  This is useful for finding out if
 *      a specified component exists.
 *	@bug	Doesn't work for preset events
 *	@see  PAPI_get_event_component
 */
int  PAPI_get_component_index(const char *name)
{
	APIDBG( "Entry: name: %s\n", name);
  int cidx;

  const PAPI_component_info_t *cinfo;

  for(cidx=0;cidx<papi_num_components;cidx++) {

     cinfo=PAPI_get_component_info(cidx); 
     if (cinfo==NULL) return PAPI_ENOCMP;

     if (!strcmp(name,cinfo->name)) {
        return cidx;
     }
  }

  return PAPI_ENOCMP;
}


/**	@class PAPI_disable_component
 *	@brief disables the specified component
 *	@retval ENOCMP
 *		component does not exist
 *      @retval ENOINIT
 *              cannot disable as PAPI has already been initialized
 *
 *	@param cidx
 *              component index of component to be disabled
 *	@par Examples:
 *	@code
               int cidx, result;

               cidx = PAPI_get_component_index("example");

               if (cidx>=0) {
                  result = PAPI_disable_component(cidx);
                  if (result==PAPI_OK)
                     printf("The example component is disabled\n");
               }
               // ... 
               PAPI_library_init();
 *	@endcode
 *      PAPI_disable_component() allows the user to disable components
 *      before PAPI_library_init() time.  This is useful if the user
 *      knows they do not wish to use events from that component and
 *      want to reduce the PAPI library overhead.
 *    
 *      PAPI_disable_component() must be called before
 *      PAPI_library_init().
 *
 *	@see  PAPI_get_event_component
 *      @see  PAPI_library_init
 */
int
PAPI_disable_component( int cidx )
{
	APIDBG( "Entry: cidx: %d\n", cidx);

   const PAPI_component_info_t *cinfo;

   /* Can only run before PAPI_library_init() is called */
   if (init_level != PAPI_NOT_INITED) {
      return PAPI_ENOINIT;
   }
     
   cinfo=PAPI_get_component_info(cidx); 
   if (cinfo==NULL) return PAPI_ENOCMP;

   ((PAPI_component_info_t *)cinfo)->disabled=1;
   strcpy(((PAPI_component_info_t *)cinfo)->disabled_reason,
	       "Disabled by PAPI_disable_component()");

   return PAPI_OK;
 
}

/** \class PAPI_disable_component_by_name
 *	\brief disables the named component
 *	\retval ENOCMP
 *		component does not exist
 *	\retval ENOINIT
 *		unable to disable the component, the library has already been initialized
 *	\param component_name
 *		name of the component to disable.
 *	\par Example:
 *	\code
	int result;
	result = PAPI_disable_component_by_name("example");
	if (result==PAPI_OK)
		printf("component \"example\" has been disabled\n");
	//...
	PAPI_library_init(PAPI_VER_CURRENT);
 *	\endcode
 *	PAPI_disable_component_by_name() allows the user to disable a component
 *	before PAPI_library_init() time. This is useful if the user knows they do
 *	not with to use events from that component and want to reduce the PAPI
 *	library overhead. 
 *
 *	PAPI_disable_component_by_name() must be called before PAPI_library_init().
 *
 *	\bug none known
 *	\see PAPI_library_init
 *	\see PAPI_disable_component
*/
int
PAPI_disable_component_by_name(const char *name )
{
	APIDBG( "Entry: name: %s\n", name);
	int cidx;

	/* I can only be called before init time */
	if (init_level!=PAPI_NOT_INITED) {
		return PAPI_ENOINIT;
	}

	cidx = PAPI_get_component_index(name);
	if (cidx>=0) {
		return PAPI_disable_component(cidx);
	} 

	return PAPI_ENOCMP;
}

/** \class PAPI_enum_dev_type
 *  \brief returns handle of next device type
 *  \retval ENOCMP
 *      component does not exist
 *  \retval EINVAL
        end of device type list
 *  \param enum_modifier
 *      device type modifier, used to filter out enumerated device types
 *  \par Example:
 *  \code
    enum {
        PAPI_DEV_TYPE_ENUM__FIRST,
        PAPI_DEV_TYPE_ENUM__CPU,
        PAPI_DEV_TYPE_ENUM__CUDA,
        PAPI_DEV_TYPE_ENUM__ROCM,
        PAPI_DEV_TYPE_ENUM__ALL
    };

    void *handle;
    const char *vendor_name;
    int enum_modifier = PAPI_DEV_TYPE_ENUM__CPU | PAPI_DEV_TYPE_ENUM__CUDA;
    while (PAPI_OK == PAPI_enum_dev_type(enum_modifier, &handle)) {
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__CHAR_NAME, &vendor_name);
        ...
    }
 *  \endcode
 *  PAPI_enum_dev_type() allows the user to access all device types in the system.
 *  It takes an enumerator modifier that allows users to enumerate only devices of
 *  a predefined type and it returns an opaque handler that users can pass to other
 *  functions in order to query device type attributes.
 *
 *  \bug none known
 *  \see PAPI_get_dev_type_attr
 *  \see PAPI_get_dev_attr
 */
int
PAPI_enum_dev_type(int enum_modifier, void **handle)
{
    return _papi_hwi_enum_dev_type(enum_modifier, handle);
}

/** \class PAPI_get_dev_type_attr
 *  \brief returns device type attributes
 *  \retval ENOSUPP
 *      invalid attribute
 *  \param handle
 *      opaque handle for device, obtained through PAPI_enum_dev_type
 *  \param attr
 *      device type attribute to query
 *  \param val
 *      value of the requested device type attribute
 *  \par Example:
 *  \code
    typedef enum {
        PAPI_DEV_TYPE_ATTR__INT_PAPI_ID,    // PAPI defined device type id
        PAPI_DEV_TYPE_ATTR__INT_VENDOR_ID,  // Vendor defined id
        PAPI_DEV_TYPE_ATTR__CHAR_NAME,      // Vendor name
        PAPI_DEV_TYPE_ATTR__INT_COUNT,      // Devices of that type and vendor
        PAPI_DEV_TYPE_ATTR__CHAR_STATUS,    // Status string for the device type
    } PAPI_dev_type_attr_e;

    typedef enum {
        PAPI_DEV_TYPE_ID__CPU,   // Device id for CPUs
        PAPI_DEV_TYPE_ID__CUDA,  // Device id for Nvidia GPUs
        PAPI_DEV_TYPE_ID__ROCM,  // Device id for AMD GPUs
    } PAPI_dev_type_id_e;

    void *handle;
    int id;
    int enum_modifier = PAPI_DEV_TYPE_ENUM__ALL;
    while (PAPI_OK == PAPI_enum_dev_type(enum_modifier, &handle)) {
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_PAPI_ID, &id);
        switch (id) {
            case PAPI_DEV_TYPE_ID__CPU:
                // query cpu attributes
                break;
            case PAPI_DEV_TYPE_ID__CUDA:
                // query nvidia gpu attributes
                break;
            case PAPI_DEV_TYPE_ID__ROCM:
                // query amd gpu attributes
                break;
            default: ...
        }
    }
 *  \endcode
 *  PAPI_get_dev_type_attr() allows the user to query all device type attributes.
 *  It takes a device type handle, returned by PAPI_enum_dev_type, and an attribute
 *  to be queried for the device type and returns the attribute value.
 *
 *  \bug none known
 *  \see PAPI_enum_dev_type
 *  \see PAPI_get_dev_attr
 */
int
PAPI_get_dev_type_attr(void *handle, PAPI_dev_type_attr_e attr, void *val)
{
    return _papi_hwi_get_dev_type_attr(handle, attr, val);
}

/** \class PAPI_get_dev_attr
 *  \brief returns device attributes
 *  \retval ENOSUPP
 *      invalid/unsupported attribute
 *  \param handle
 *      opaque handle for device, obtained through PAPI_enum_dev_type
 *  \param id
 *      integer identifier of queried device
 *  \param attr
 *      device attribute to query
 *  \param val
 *      value of the requested device attribute
 *  \par Example:
 *  \code
    typedef enum {
        PAPI_DEV_ATTR__CPU_CHAR_NAME,
        PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_ASSOC,
        PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_ASSOC,
        PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_ASSOC,
        PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_ASSOC,
        PAPI_DEV_ATTR__CPU_UINT_SOCKET_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_NUMA_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_CORE_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_THREAD_COUNT,
        PAPI_DEV_ATTR__CPU_UINT_FAMILY,
        PAPI_DEV_ATTR__CPU_UINT_MODEL,
        PAPI_DEV_ATTR__CPU_UINT_STEPPING,
        PAPI_DEV_ATTR__CPU_UINT_NUMA_MEM_SIZE,
        PAPI_DEV_ATTR__CPU_UINT_THR_NUMA_AFFINITY,
        PAPI_DEV_ATTR__CPU_UINT_THR_PER_NUMA,
        PAPI_DEV_ATTR__CUDA_ULONG_UID,
        PAPI_DEV_ATTR__CUDA_CHAR_DEVICE_NAME,
        PAPI_DEV_ATTR__CUDA_UINT_WARP_SIZE,
        PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_BLK,
        PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_SM,
        PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_X,
        PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Y,
        PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Z,
        PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_X,
        PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Y,
        PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Z,
        PAPI_DEV_ATTR__CUDA_UINT_THR_PER_BLK,
        PAPI_DEV_ATTR__CUDA_UINT_SM_COUNT,
        PAPI_DEV_ATTR__CUDA_UINT_MULTI_KERNEL,
        PAPI_DEV_ATTR__CUDA_UINT_MAP_HOST_MEM,
        PAPI_DEV_ATTR__CUDA_UINT_MEMCPY_OVERLAP,
        PAPI_DEV_ATTR__CUDA_UINT_UNIFIED_ADDR,
        PAPI_DEV_ATTR__CUDA_UINT_MANAGED_MEM,
        PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MAJOR,
        PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MINOR,
        PAPI_DEV_ATTR__CUDA_UINT_BLK_PER_SM,
        PAPI_DEV_ATTR__ROCM_ULONG_UID,
        PAPI_DEV_ATTR__ROCM_CHAR_DEVICE_NAME,
        PAPI_DEV_ATTR__ROCM_UINT_WAVEFRONT_SIZE,
        PAPI_DEV_ATTR__ROCM_UINT_WORKGROUP_SIZE,
        PAPI_DEV_ATTR__ROCM_UINT_WAVE_PER_CU,
        PAPI_DEV_ATTR__ROCM_UINT_SHM_PER_WG,
        PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_X,
        PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Y,
        PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Z,
        PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_X,
        PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Y,
        PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Z,
        PAPI_DEV_ATTR__ROCM_UINT_CU_COUNT,
        PAPI_DEV_ATTR__ROCM_UINT_SIMD_PER_CU,
        PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MAJOR,
        PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MINOR,
    } PAPI_dev_attr_e;

    void *handle;
    int id;
    int count;
    int enum_modifier = PAPI_DEV_TYPE_ENUM__CPU | PAPI_DEV_TYPE_ENUM__CUDA;
    while (PAPI_OK == PAPI_enum_dev_type(enum_modifier, &handle)) {
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_PAPI_ID, &id);
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_COUNT, &count);
        if (PAPI_DEV_TYPE_ID__CUDA == id) {
            for (int i = 0; i < count; ++i) {
                unsigned int warp_size;
                unsigned int cc_major, cc_minor;
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_WARP_SIZE, &warp_size);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MAJOR, &cc_major);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MINOR, &cc_minor);
                ...
            }
        }
    }
 *  \endcode
 *  PAPI_get_dev_type_attr() allows the user to query all device type attributes.
 *  It takes a device type handle, returned by PAPI_enum_dev_type, the device sequential id
 *  and an attribute to be queried for the device and returns the attribute value.
 *
 *  \bug none known
 *  \see PAPI_enum_dev_type
 *  \see PAPI_get_dev_attr
 */
int
PAPI_get_dev_attr(void *handle, int id, PAPI_dev_attr_e attr, void *val)
{
    return _papi_hwi_get_dev_attr(handle, id, attr, val);
}
