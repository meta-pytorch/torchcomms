/** 
 * @file    sw_multiplex.c
 * @author  Philip Mucci
 *          mucci@cs.utk.edu
 * @author  John May
 *          johnmay@llnl.gov
 * @author  Nils Smeds
 *          smeds@pdc.kth.se
 * @author  Haihang You
 *          you@cs.utk.edu 
 * @author  Kevin London
 *	        london@cs.utk.edu
 * @author  Maynard Johnson
 *          maynardj@us.ibm.com
 * @author  Dan Terpstra
 *	        terpstra@cs.utk.edu
 */

/** xxxx Will this stuff run unmodified on multiple components?
    What happens when several components are counting multiplexed?
*/

/* disable this to return to the pre 4.1.1 behavior */
#define MPX_NONDECR_HYBRID

/* Nils Smeds */

/* This MPX update modifies the behaviour of the multiplexing in PAPI.
 * The previous versions of the multiplexing based the value returned
 * from PAPI_reads on the total counts achieved since the PAPI_start
 * of the multiplexed event. This count was used as the basis of the
 * extrapolation using the proportion of time that this particular
 * event was active to the total time the multiplexed event was
 * active. However, a typical usage of PAPI is to measure over
 * sections of code by starting the event once and by comparing
 * the values returned by subsequent calls to PAPI_read. The difference
 * in counts is used as the measure of occurred events in the code
 * section between the calls. 
 *
 * When multiplexing is used in this fashion the time proportion used
 * for extrapolation might appear inconsistent. The time fraction used
 * at each PAPI_read is the total time fraction since PAPI_start. If the
 * counter values achieved in each multiplex of the event varies
 * largely, or if the time slices are varying in length, discrepancies
 * to the behaviour without multiplexing might occur.
 *
 * In this version the extrapolation is made on a local time scale. At
 * each completed time slice the event extrapolates the achieved count
 * to a extrapolated count for the time since this event was last sliced
 * out up to the current point in time. There will still be occasions
 * when two consecutive PAPI_read will yield decreasing results, but all
 * extrapolations are being made on time local data. If time slicing
 * varies or if the count rate varies this implementation is expected to
 * be more "accurate" in a loose and here unspecified meaning.
 *
 * The short description of the changes is that the running events has
 * new fields count_estimate, rate_estimate and prev_total_c. The mpx
 * events have had the meaning of start_values and stop_values modified
 * to mean extrapolated start value and extrapolated stop value.
 */

/* 
Portions of the following code are
Copyright (c) 2009, Lawrence Livermore National Security, LLC.  
Produced at the Lawrence Livermore National Laboratory  
Written by John May, johnmay@llnl.gov
LLNL-CODE-421124
All rights reserved.  
  
Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:  
  
 • Redistributions of source code must retain the above copyright
notice, this list of conditions and the disclaimer below. 

 • Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the disclaimer (as
noted below) in the documentation and/or other materials provided
with the distribution. 

 • Neither the name of the LLNS/LLNL nor the names of its
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.  
  
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;  LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON  ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.  
  
  
Additional BSD Notice  
  
1. This notice is required to be provided under our contract with
the U.S.  Department of Energy (DOE).  This work was produced at
Lawrence Livermore National Laboratory under Contract No.
DE-AC52-07NA27344 with the DOE.  
  
2. Neither the United States Government nor Lawrence Livermore
National Security, LLC nor any of their employees, makes any
warranty, express or implied, or assumes any liability or
responsibility for the accuracy, completeness, or usefulness of
any information, apparatus, product, or process disclosed, or
represents that its use would not infringe privately-owned
rights.  
  
3.  Also, reference herein to any specific commercial products,
process, or services by trade name, trademark, manufacturer or
otherwise does not necessarily constitute or imply its
endorsement, recommendation, or favoring by the United States
Government or Lawrence Livermore National Security, LLC.  The
views and opinions of authors expressed herein do not necessarily
state or reflect those of the United States Government or
Lawrence Livermore National Security, LLC, and shall not be used
for advertising or product endorsement purposes.  
 */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#define MPX_MINCYC 25000

/* Globals for this file. */

/** List of threads that are multiplexing. */

static Threadlist *tlist = NULL;
static unsigned int randomseed;

/* Timer stuff */

#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <unistd.h> 
#include <assert.h>

static sigset_t sigreset;
static struct itimerval itime;
static const struct itimerval itimestop = { {0, 0}, {0, 0} };
static struct sigaction oaction;

/* END Globals */

#ifdef PTHREADS
/** Number of threads that have been signaled */
static int threads_responding = 0;

static pthread_once_t mpx_once_control = PTHREAD_ONCE_INIT;
static pthread_mutex_t tlistlock;
static pthread_key_t master_events_key;
static pthread_key_t thread_record_key;
static MasterEvent *global_master_events;
static void *global_process_record;
#endif

/* Forward prototypes */

static void mpx_remove_unused( MasterEvent ** head );
static void mpx_delete_events( MPX_EventSet * );
static void mpx_delete_one_event( MPX_EventSet * mpx_events, int Event );
static int mpx_insert_events( MPX_EventSet *, int *event_list, int num_events,
							  int domain, int granularity );
static void mpx_handler( int signal );

inline_static void
mpx_hold( void )
{
	sigprocmask( SIG_BLOCK, &sigreset, NULL );
	MPXDBG( "signal held\n" );
}

inline_static void
mpx_release( void )
{
	MPXDBG( "signal released\n" );
	sigprocmask( SIG_UNBLOCK, &sigreset, NULL );
}

static void
mpx_init_timers( int interval )
{
	/* Fill in the interval timer values now to save a
	 * little time later.
	 */
#ifdef OUTSIDE_PAPI
	interval = MPX_DEFAULT_INTERVAL;
#endif

#ifdef REGENERATE
	/* Signal handler restarts the timer every time it runs */
	itime.it_interval.tv_sec = 0;
	itime.it_interval.tv_usec = 0;
	itime.it_value.tv_sec = 0;
	itime.it_value.tv_usec = interval;
#else
	/* Timer resets itself automatically */
	itime.it_interval.tv_sec = 0;
	itime.it_interval.tv_usec = interval;
	itime.it_value.tv_sec = 0;
	itime.it_value.tv_usec = interval;
#endif

	sigemptyset( &sigreset );
	sigaddset( &sigreset, _papi_os_info.itimer_sig );
}

static int
mpx_startup_itimer( void )
{
	struct sigaction sigact;

	/* Set up the signal handler and the timer that triggers it */

	MPXDBG( "PID %d\n", getpid(  ) );
	memset( &sigact, 0, sizeof ( sigact ) );
	sigact.sa_flags = SA_RESTART;
	sigact.sa_handler = mpx_handler;

	if ( sigaction( _papi_os_info.itimer_sig, &sigact, NULL ) == -1 ) {
		PAPIERROR( "sigaction start errno %d", errno );
		return PAPI_ESYS;
	}

	if ( setitimer( _papi_os_info.itimer_num, &itime, NULL ) == -1 ) {
		sigaction( _papi_os_info.itimer_sig, &oaction, NULL );
		PAPIERROR( "setitimer start errno %d", errno );
		return PAPI_ESYS;
	}
	return ( PAPI_OK );
}

static void
mpx_restore_signal( void )
{
	MPXDBG( "restore signal\n" );
	if ( _papi_os_info.itimer_sig != PAPI_NULL ) {
		if ( signal( _papi_os_info.itimer_sig, SIG_IGN ) == SIG_ERR )
			PAPIERROR( "sigaction stop errno %d", errno );
	}
}

static void
mpx_shutdown_itimer( void )
{
	MPXDBG( "setitimer off\n" );
	if ( _papi_os_info.itimer_num != PAPI_NULL ) {
		if ( setitimer( _papi_os_info.itimer_num,
			   ( struct itimerval * ) &itimestop, NULL ) == -1 )
			PAPIERROR( "setitimer stop errno %d", errno );
	}
}

static MasterEvent *
get_my_threads_master_event_list( void )
{
	Threadlist *t = tlist;
	unsigned long tid;

	MPXDBG( "tlist is %p\n", tlist );
	if ( tlist == NULL )
		return NULL;

	if ( _papi_hwi_thread_id_fn == NULL )
		return ( tlist->head );

	tid = _papi_hwi_thread_id_fn(  );
	unsigned long pid = ( unsigned long ) getpid(  );

	while ( t ) {
		if ( t->tid == tid || ( ( tid == 0 ) && ( t->tid == pid ) ) )
			return ( t->head );
		t = t->next;
	}
	return ( NULL );
}

static MPX_EventSet *
mpx_malloc( Threadlist * t )
{
	MPX_EventSet *newset =
		( MPX_EventSet * ) papi_malloc( sizeof ( MPX_EventSet ) );
	if ( newset == NULL )
		return ( NULL );
	memset( newset, 0, sizeof ( MPX_EventSet ) );
	newset->status = MPX_STOPPED;
	newset->mythr = t;
	return ( newset );
}

int
mpx_add_event( MPX_EventSet ** mpx_events, int EventCode, int domain,
			   int granularity )
{
	MPX_EventSet *newset = *mpx_events;
	int retval, alloced_newset = 0;
	Threadlist *t;

	/* Get the global list of threads */

	MPXDBG("Adding %p %#x\n",newset,EventCode);

	_papi_hwi_lock( MULTIPLEX_LOCK );
	t = tlist;

	/* If there are no threads in the list at all, then allocate the new Threadlist */

	if ( t == NULL ) {
	  new_thread:
		t = ( Threadlist * ) papi_malloc( sizeof ( Threadlist ) );
		if ( t == NULL ) {
			_papi_hwi_unlock( MULTIPLEX_LOCK );
			return ( PAPI_ENOMEM );
		}

		/* If we're actually threaded, fill the 
		 * field with the thread_id otherwise
		 * use getpid() as a placeholder. */

		if ( _papi_hwi_thread_id_fn ) {
			MPXDBG( "New thread at %p\n", t );
			t->tid = _papi_hwi_thread_id_fn(  );
		} else {
			MPXDBG( "New process at %p\n", t );
			t->tid = ( unsigned long ) getpid(  );
		}

		/* Fill in the fields */

		t->head = NULL;
		t->cur_event = NULL;
		t->next = tlist;
		tlist = t;
		MPXDBG( "New head is at %p(%lu).\n", tlist,
				( long unsigned ) tlist->tid );
		/* alloced_thread = 1; */
	} else if ( _papi_hwi_thread_id_fn ) {

		/* If we are threaded, AND there exists threads in the list, 
		 *  then try to find our thread in the list. */

		unsigned long tid = _papi_hwi_thread_id_fn(  );

		while ( t ) {
			if ( t->tid == tid ) {
				MPXDBG( "Found thread %#lx\n", t->tid );
				break;
			}
			t = t->next;
		}

		/* Our thread is not in the list, so make a new
		 * thread entry. */

		if ( t == NULL ) {
			MPXDBG( "New thread %lx\n", tid );
			goto new_thread;
		}
	}

	/* Now t & tlist points to our thread, also at the head of the list */

	/* Allocate a the MPX_EventSet if necessary */

	if ( newset == NULL ) {
		newset = mpx_malloc( t );
		if ( newset == NULL ) {
			_papi_hwi_unlock( MULTIPLEX_LOCK );
			return ( PAPI_ENOMEM );
		}
		alloced_newset = 1;
	}

	/* Now we're finished playing with the thread list */

	_papi_hwi_unlock( MULTIPLEX_LOCK );

	/* Removed newset->num_events++, moved to mpx_insert_events() */

	mpx_hold(  );

	/* Create PAPI events (if they don't already exist) and link
	 * the new event set to them, add them to the master list for
	 the thread, reset master event list for this thread */

	retval = mpx_insert_events( newset, &EventCode, 1, 
				    domain, granularity );
	if ( retval != PAPI_OK ) {
		if ( alloced_newset ) {
			papi_free( newset );
			newset = NULL;
		}
	}

	mpx_release(  );

	/* Output the new or existing EventSet */

	*mpx_events = newset;

	return retval;
}

int
mpx_remove_event( MPX_EventSet ** mpx_events, int EventCode )
{
	mpx_hold(  );
	if ( *mpx_events )
		mpx_delete_one_event( *mpx_events, EventCode );
	mpx_release(  );
	return ( PAPI_OK );
}

#ifdef MPX_DEBUG_TIMER
static long long lastcall;
#endif


#ifdef _POWER6
/* POWER6 can always count PM_RUN_CYC on counter 6 in domain
   PAPI_DOM_ALL, and can count it on other domains on counters
   1 and 2 along with a very limited number of other native
   events */
int _PNE_PM_RUN_CYC;
#define SCALE_EVENT _PNE_PM_RUN_CYC
#else
#define SCALE_EVENT PAPI_TOT_CYC
#endif


static void
mpx_handler( int signal )
{
	int retval;
	MasterEvent *mev, *head;
	Threadlist *me = NULL;
#ifdef REGENERATE
	int lastthread;
#endif
#ifdef MPX_DEBUG_OVERHEAD
	long long usec;
	int didwork = 0;
	usec = PAPI_get_real_usec(  );
#endif
#ifdef MPX_DEBUG_TIMER
	long long thiscall;
#endif

	signal = signal;		 /* unused */

	MPXDBG( "Handler in thread\n" );

	/* This handler can be invoked either when a timer expires
	 * or when another thread in this handler responding to the
	 * timer signals other threads.  We have to distinguish
	 * these two cases so that we don't get infinite loop of 
	 * handler calls.  To do that, we look at the value of
	 * threads_responding.  We assume that only one thread can
	 * be active in this signal handler at a time, since the
	 * invoking signal is blocked while the handler is active.
	 * If threads_responding == 0, the current thread caught
	 * the original timer signal.  (This thread may not have
	 * any active event lists itself, though.)  This first
	 * thread sends a signal to each of the other threads in
	 * our list of threads that have master events lists.  If
	 * threads_responding != 0, then this thread was signaled
	 * by another thread.  We decrement that value and look
	 * for an active events.  threads_responding should
	 * reach zero when all active threads have handled their
	 * signal.  It's probably possible for a thread to die
	 * before it responds to a signal; if that happens,
	 * threads_responding won't reach zero until the next
	 * timer signal happens.  Then the signalled thread won't
	 * signal any other threads.  If that happens only
	 * occasionally, there should be no harm.  Likewise if
	 * a new thread is added that fails to get signalled.
	 * As for locking, we have to lock this list to prevent
	 * another thread from modifying it, but if *this* thread
	 * is trying to update the list (from another function) and
	 * is signaled while it holds the lock, we will have deadlock.
	 * Therefore, noninterrupt functions that update *this* list
	 * must disable the signal that invokes this handler.
	 */

#ifdef PTHREADS
	_papi_hwi_lock( MULTIPLEX_LOCK );

	if ( threads_responding == 0 ) {	/* this thread caught the timer sig */
		/* Signal the other threads with event lists */
#ifdef MPX_DEBUG_TIMER
		thiscall = _papi_hwd_get_real_usec(  );
		MPXDBG( "last signal was %lld usec ago\n", thiscall - lastcall );
		lastcall = thiscall;
#endif
		MPXDBG( "%#x caught it, tlist is %p\n", self, tlist );
		for ( t = tlist; t != NULL; t = t->next ) {
			if ( pthread_equal( t->thr, self ) == 0 ) {
				++threads_responding;
				retval = pthread_kill( t->thr, _papi_os_info.itimer_sig );
				assert( retval == 0 );
#ifdef MPX_DEBUG_SIGNALS
				MPXDBG( "%#x signaling %#x\n", self, t->thr );
#endif
			}
		}
	} else {
#ifdef MPX_DEBUG_SIGNALS
		MPXDBG( "%#x was tapped, tr = %d\n", self, threads_responding );
#endif
		--threads_responding;
	}
#ifdef REGENERATE
	lastthread = ( threads_responding == 0 );
#endif
	_papi_hwi_unlock( MULTIPLEX_LOCK );
#endif

	/* See if this thread has an active event list */
	head = get_my_threads_master_event_list(  );
	if ( head != NULL ) {

		/* Get the thread header for this master event set.  It's
		 * always in the first record of the set (and maybe in others)
		 * if any record in the set is active.
		 */
		me = head->mythr;

		/* Find the event that's currently active, stop and read
		 * it, then start the next event in the list.
		 * No need to lock the list because other functions
		 * disable the timer interrupt before they update the list.
		 */
		if ( me != NULL && me->cur_event != NULL ) {
			long long counts[2];
			MasterEvent *cur_event = me->cur_event;
			long long cycles = 0, total_cycles = 0;

			retval = PAPI_stop( cur_event->papi_event, counts );
			MPXDBG( "retval=%d, cur_event=%p, I'm tid=%lx\n",
					retval, cur_event, me->tid );

			if ( retval == PAPI_OK ) {
				MPXDBG( "counts[0] = %lld counts[1] = %lld\n", counts[0],
						counts[1] );

				cur_event->count += counts[0];
				cycles = ( cur_event->pi.event_type == SCALE_EVENT )
					? counts[0] : counts[1];

				me->total_c += cycles;
				total_cycles = me->total_c - cur_event->prev_total_c;
				cur_event->prev_total_c = me->total_c;

				/* If it's a rate, count occurrences & average later */
				if ( !cur_event->is_a_rate ) {
					cur_event->cycles += cycles;
					if ( cycles >= MPX_MINCYC ) {	/* Only update current rate on a decent slice */
						cur_event->rate_estimate =
							( double ) counts[0] / ( double ) cycles;
					}
					cur_event->count_estimate +=
						( long long ) ( ( double ) total_cycles *
										cur_event->rate_estimate );
                                        MPXDBG("New estimate = %lld (%lld cycles * %lf rate)\n",
                                               cur_event->count_estimate,total_cycles,
                                               cur_event->rate_estimate);
				} else {
					/* Make sure we ran long enough to get a useful measurement (otherwise
					 * potentially inaccurate rate measurements get averaged in with
					 * the same weight as longer, more accurate ones.)
					 */
					if ( cycles >= MPX_MINCYC ) {
						cur_event->cycles += 1;
					} else {
						cur_event->count -= counts[0];
					}
				}
			} else {
				MPXDBG( "%lx retval = %d, skipping\n", me->tid, retval );
				MPXDBG( "%lx value = %lld cycles = %lld\n\n",
						me->tid, cur_event->count, cur_event->cycles );
			}

			MPXDBG
				( "tid(%lx): value = %lld (%lld) cycles = %lld (%lld) rate = %lf\n\n",
				  me->tid, cur_event->count, cur_event->count_estimate,
				  cur_event->cycles, total_cycles, cur_event->rate_estimate );
			/* Start running the next event; look for the
			 * next one in the list that's marked active.
			 * It's possible that this event is the only
			 * one active; if so, we should restart it,
			 * but only after considerating all the other
			 * possible events.
			 */
			if ( ( retval != PAPI_OK ) ||
				 ( ( retval == PAPI_OK ) && ( cycles >= MPX_MINCYC ) ) ) {
				for ( mev =
					  ( cur_event->next == NULL ) ? head : cur_event->next;
					  mev != cur_event;
					  mev = ( mev->next == NULL ) ? head : mev->next ) {
					/* Found the next one to start */
					if ( mev->active ) {
						me->cur_event = mev;
						break;
					}
				}
			}

			if ( me->cur_event->active ) {
				retval = PAPI_start( me->cur_event->papi_event );
			}
#ifdef MPX_DEBUG_OVERHEAD
			didwork = 1;
#endif
		}
	}
#ifdef ANY_THREAD_GETS_SIGNAL
	else {
		Threadlist *t;
		for ( t = tlist; t != NULL; t = t->next ) {
			if ( ( t->tid == _papi_hwi_thread_id_fn(  ) ) ||
				 ( t->head == NULL ) )
				continue;
			MPXDBG( "forwarding signal to thread %lx\n", t->tid );
			retval = ( *_papi_hwi_thread_kill_fn ) ( t->tid, _papi_os_info.itimer_sig );
			if ( retval != 0 ) {
				MPXDBG( "forwarding signal to thread %lx returned %d\n",
						t->tid, retval );
			}
		}
	}
#endif

#ifdef REGENERATE
	/* Regenerating the signal each time through has the
	 * disadvantage that if any thread ever drops a signal,
	 * the whole time slicing system will stop.  Using
	 * an automatically regenerated signal may have the
	 * disadvantage that a new signal can arrive very
	 * soon after all the threads have finished handling
	 * the last one, so the interval may be too small for
	 * accurate data collection.  However, using the
	 * MIN_CYCLES check above should alleviate this.
	 */
	/* Reset the timer once all threads have responded */
	if ( lastthread ) {
		retval = setitimer( _papi_os_info.itimer_num, &itime, NULL );
		assert( retval == 0 );
#ifdef MPX_DEBUG_TIMER
		MPXDBG( "timer restarted by %lx\n", me->tid );
#endif
	}
#endif

#ifdef MPX_DEBUG_OVERHEAD
	usec = _papi_hwd_get_real_usec(  ) - usec;
	MPXDBG( "handler %#x did %swork in %lld usec\n",
			self, ( didwork ? "" : "no " ), usec );
#endif
}

int
MPX_add_events( MPX_EventSet ** mpx_events, int *event_list, int num_events,
				int domain, int granularity )
{
	int i, retval = PAPI_OK;

	for ( i = 0; i < num_events; i++ ) {
		retval =
			mpx_add_event( mpx_events, event_list[i], domain, granularity );

		if ( retval != PAPI_OK )
			return ( retval );
	}
	return ( retval );
}

int
MPX_start( MPX_EventSet * mpx_events )
{
	int retval = PAPI_OK;
	int i;
	long long values[2];
	long long cycles_this_slice, current_thread_mpx_c = 0;
	Threadlist *t;

	t = mpx_events->mythr;

	mpx_hold(  );

	if ( t->cur_event && t->cur_event->active ) {
		current_thread_mpx_c += t->total_c;
		retval = PAPI_read( t->cur_event->papi_event, values );
		assert( retval == PAPI_OK );
		if ( retval == PAPI_OK ) {
			cycles_this_slice = ( t->cur_event->pi.event_type == SCALE_EVENT )
				? values[0] : values[1];
		} else {
			values[0] = values[1] = 0;
			cycles_this_slice = 0;
		}

	} else {
		values[0] = values[1] = 0;
		cycles_this_slice = 0;
	}

	/* Make all events in this set active, and for those
	 * already active, get the current count and cycles.
	 */
	for ( i = 0; i < mpx_events->num_events; i++ ) {
		MasterEvent *mev = mpx_events->mev[i];

		if ( mev->active++ ) {
			mpx_events->start_values[i] = mev->count_estimate;
			mpx_events->start_hc[i] = mev->cycles;

			/* If this happens to be the currently-running
			 * event, add in the current amounts from this
			 * time slice.  If it's a rate, though, don't
			 * bother since the event might not have been
			 * running long enough to get an accurate count.
			 */
			if ( t->cur_event && !( t->cur_event->is_a_rate ) ) {
#ifdef MPX_NONDECR_HYBRID
				if ( mev != t->cur_event ) {	/* This event is not running this slice */
					mpx_events->start_values[i] +=
						( long long ) ( mev->rate_estimate *
										( cycles_this_slice + t->total_c -
										  mev->prev_total_c ) );
				} else {	 /* The event is running, use current value + estimate */
					if ( cycles_this_slice >= MPX_MINCYC )
						mpx_events->start_values[i] += values[0] + ( long long )
							( ( values[0] / ( double ) cycles_this_slice ) *
							  ( t->total_c - mev->prev_total_c ) );
					else	 /* Use previous rate if the event has run too short time */
						mpx_events->start_values[i] += values[0] + ( long long )
							( mev->rate_estimate *
							  ( t->total_c - mev->prev_total_c ) );
				}
#endif
			} else {
				mpx_events->start_values[i] = mev->count;
			}
		} else {
			/* The = 0 isn't actually necessary; we only need
			 * to sync up the mpx event to the master event,
			 * but it seems safe to set the mev to 0 here, and
			 * that gives us a change to avoid (very unlikely)
			 * rollover problems for events used repeatedly over
			 * a long time.
			 */
			mpx_events->start_values[i] = 0;
			mpx_events->stop_values[i] = 0;
			mpx_events->start_hc[i] = mev->cycles = 0;
			mev->count_estimate = 0;
			mev->rate_estimate = 0.0;
			mev->prev_total_c = current_thread_mpx_c;
			mev->count = 0;
		}
		/* Adjust start value to include events and cycles
		 * counted previously for this event set.
		 */
	}

	mpx_events->status = MPX_RUNNING;

	/* Start first counter if one isn't already running */
	if ( t->cur_event == NULL ) {
		/* Pick an events at random to start. */
		int index = ( rand_r( &randomseed ) % mpx_events->num_events );
		t->cur_event = mpx_events->mev[index];
		t->total_c = 0;
		t->cur_event->prev_total_c = 0;
		mpx_events->start_c = 0;
		retval = PAPI_start( mpx_events->mev[index]->papi_event );
		assert( retval == PAPI_OK );
	} else {
		/* If an event is already running, record the starting cycle
		 * count for mpx_events, which is the accumlated cycle count
		 * for the master event set plus the cycles for this time
		 * slice.
		 */
		mpx_events->start_c = t->total_c + cycles_this_slice;
	}

#if defined(DEBUG)
	if ( ISLEVEL( DEBUG_MULTIPLEX ) ) {
		MPXDBG( "%s:%d:: start_c=%lld  thread->total_c=%lld\n", __FILE__,
				__LINE__, mpx_events->start_c, t->total_c );
		for ( i = 0; i < mpx_events->num_events; i++ ) {
			MPXDBG
				( "%s:%d:: start_values[%d]=%lld  estimate=%lld rate=%g last active=%lld\n",
				  __FILE__, __LINE__, i, mpx_events->start_values[i],
				  mpx_events->mev[i]->count_estimate,
				  mpx_events->mev[i]->rate_estimate,
				  mpx_events->mev[i]->prev_total_c );
		}
	}
#endif

	mpx_release(  );

	retval = mpx_startup_itimer(  );

	return retval;
}

int
MPX_read( MPX_EventSet * mpx_events, long long *values, int called_by_stop )
{
	int i;
	int retval;
	long long last_value[2];
	long long cycles_this_slice = 0;
	MasterEvent *cur_event;
	Threadlist *thread_data;

	if ( mpx_events->status == MPX_RUNNING ) {

		/* Hold timer interrupts while we read values */
		mpx_hold(  );

		thread_data = mpx_events->mythr;
		cur_event = thread_data->cur_event;

		retval = PAPI_read( cur_event->papi_event, last_value );
		if ( retval != PAPI_OK )
			return retval;

		cycles_this_slice = ( cur_event->pi.event_type == SCALE_EVENT )
			? last_value[0] : last_value[1];

		/* Save the current counter values and get
		 * the lastest data for the current event
		 */
		for ( i = 0; i < mpx_events->num_events; i++ ) {
			MasterEvent *mev = mpx_events->mev[i];

			if ( !( mev->is_a_rate ) ) {
				mpx_events->stop_values[i] = mev->count_estimate;
			}
			else {
				mpx_events->stop_values[i] = mev->count;
			}
#ifdef MPX_NONDECR_HYBRID
			/* If we are called from MPX_stop() then      */
                        /* adjust the final values based on the       */
                        /* cycles elapsed since the last read         */
                        /* otherwise, don't do this as it can cause   */
                        /* decreasing values if read is called again  */
                        /* before another sample happens.             */

                        if (called_by_stop) {

			   /* Extrapolate data up to the current time 
			    * only if it's not a rate measurement 
			    */
			   if ( !( mev->is_a_rate ) ) {
			      if ( mev != thread_data->cur_event ) {
				 mpx_events->stop_values[i] +=
						( long long ) ( mev->rate_estimate *
										( cycles_this_slice +
										  thread_data->total_c -
										  mev->prev_total_c ) );
				 MPXDBG
						( "%s:%d:: Inactive %d, stop values=%lld (est. %lld, rate %g, cycles %lld)\n",
						  __FILE__, __LINE__, i, mpx_events->stop_values[i],
						  mev->count_estimate, mev->rate_estimate,
						  cycles_this_slice + thread_data->total_c -
						  mev->prev_total_c );
			      } else {
				 mpx_events->stop_values[i] += last_value[0] +
						( long long ) ( mev->rate_estimate *
										( thread_data->total_c -
										  mev->prev_total_c ) );
				 MPXDBG
						( "%s:%d:: -Active- %d, stop values=%lld (est. %lld, rate %g, cycles %lld)\n",
						  __FILE__, __LINE__, i, mpx_events->stop_values[i],
						  mev->count_estimate, mev->rate_estimate,
						  thread_data->total_c - mev->prev_total_c );
			      }
			   }
			}
#endif
		}

		mpx_events->stop_c = thread_data->total_c + cycles_this_slice;

		/* Restore the interrupt */
		mpx_release(  );
	}

	/* Store the values in user array. */
	for ( i = 0; i < mpx_events->num_events; i++ ) {
		MasterEvent *mev = mpx_events->mev[i];
		long long elapsed_slices = 0;
		long long elapsed_values = mpx_events->stop_values[i]
			- mpx_events->start_values[i];

		/* For rates, cycles contains the number of measurements,
		 * not the number of cycles, so just divide to compute
		 * an average value.  This assumes that the rate was
		 * constant over the whole measurement period.
		 */
		values[i] = elapsed_values;
		if ( mev->is_a_rate ) {
			/* Handler counts */
			elapsed_slices = mev->cycles - mpx_events->start_hc[i];
			values[i] =
				elapsed_slices ? ( elapsed_values / elapsed_slices ) : 0;
		}
		MPXDBG( "%s:%d:: event %d, values=%lld ( %lld - %lld), cycles %lld\n",
				__FILE__, __LINE__, i,
				elapsed_values,
				mpx_events->stop_values[i], mpx_events->start_values[i],
				mev->is_a_rate ? elapsed_slices : 0 );
	}

	return PAPI_OK;
}

int
MPX_reset( MPX_EventSet * mpx_events )
{
	int i, retval;
	long long values[PAPI_MAX_SW_MPX_EVENTS];

	/* Get the current values from MPX_read */
	retval = MPX_read( mpx_events, values, 0 );
	if ( retval != PAPI_OK )
		return retval;

	/* Disable timer interrupt */
	mpx_hold(  );

	/* Make counters read zero by setting the start values
	 * to the current counter values.
	 */
	for ( i = 0; i < mpx_events->num_events; i++ ) {
		MasterEvent *mev = mpx_events->mev[i];

		if ( mev->is_a_rate ) {
			mpx_events->start_values[i] = mev->count;
		} else {
			mpx_events->start_values[i] += values[i];
		}
		mpx_events->start_hc[i] = mev->cycles;
	}

	/* Set the start time for this set to the current cycle count */
	mpx_events->start_c = mpx_events->stop_c;

	/* Restart the interrupt */
	mpx_release(  );

	return PAPI_OK;
}

int
MPX_stop( MPX_EventSet * mpx_events, long long *values )
{
	int i, cur_mpx_event;
	int retval = PAPI_OK;
	long long dummy_value[2];
	long long dummy_mpx_values[PAPI_MAX_SW_MPX_EVENTS];
	/* long long cycles_this_slice, total_cycles; */
	MasterEvent *cur_event = NULL, *head;
	Threadlist *thr = NULL;

	if ( mpx_events == NULL )
		return PAPI_EINVAL;
	if ( mpx_events->status != MPX_RUNNING )
		return PAPI_ENOTRUN;

	/* Read the counter values, this updates mpx_events->stop_values[] */
	MPXDBG( "Start\n" );
	if ( values == NULL )
	  retval = MPX_read( mpx_events, dummy_mpx_values, 1 );
	else
	  retval = MPX_read( mpx_events, values, 1 );

	/* Block timer interrupts while modifying active events */
	mpx_hold(  );

	/* Get the master event list for this thread. */
	head = get_my_threads_master_event_list(  );
	if (!head) {
	  retval=PAPI_EBUG;
	  goto exit_mpx_stop;
	}

	/* Get this threads data structure */
	thr = head->mythr;
	cur_event = thr->cur_event;

	/* This would be a good spot to "hold" the counter and then restart
	 * it at the end, but PAPI_start resets counters so it is not possible
	 */

	/* Run through all the events decrement their activity counters. */
	cur_mpx_event = -1;
	for ( i = 0; i < mpx_events->num_events; i++ ) {
		--mpx_events->mev[i]->active;
		if ( mpx_events->mev[i] == cur_event )
			cur_mpx_event = i;
	}

	/* One event in this set is currently running, if this was the
	 * last active event set using this event, we need to start the next 
	 * event if there still is one left in the queue
	 */
	if ( cur_mpx_event > -1 ) {
		MasterEvent *tmp, *mev = mpx_events->mev[cur_mpx_event];

		if ( mev->active == 0 ) {
			/* Event is now inactive; stop it 
			 * There is no need to update master event set 
			 * counters as this is the last active user
			 */
			retval = PAPI_stop( mev->papi_event, dummy_value );
			mev->rate_estimate = 0.0;

			/* Fall-back value if none is found */
			thr->cur_event = NULL;
			/* Now find a new cur_event */
			for ( tmp = ( cur_event->next == NULL ) ? head : cur_event->next;
				  tmp != cur_event;
				  tmp = ( tmp->next == NULL ) ? head : tmp->next ) {
				if ( tmp->active ) {	/* Found the next one to start */
					thr->cur_event = tmp;
					break;
				}
			}

			if ( thr->cur_event != NULL ) {
				retval = PAPI_start( thr->cur_event->papi_event );
				assert( retval == PAPI_OK );
			} else {
				mpx_shutdown_itimer(  );
			}
		}
	}
	mpx_events->status = MPX_STOPPED;

exit_mpx_stop:
	MPXDBG( "End\n" );

	/* Restore the timer (for other event sets that may be running) */
	mpx_release(  );

	return retval;
}

int
MPX_cleanup( MPX_EventSet ** mpx_events )
{
#ifdef PTHREADS
	int retval;
#endif

	if ( mpx_events == NULL )
	   return PAPI_EINVAL;

	if ( *mpx_events == NULL )
	   return PAPI_OK;

	if (( *mpx_events )->status == MPX_RUNNING )
	   return PAPI_EINVAL;

	mpx_hold(  );

	/* Remove master events from this event set and from
	 * the master list, if necessary.
	 */
	mpx_delete_events( *mpx_events );

	mpx_release(  );

	/* Free all the memory */

	papi_free( *mpx_events );

	*mpx_events = NULL;
	return PAPI_OK;
}

void
MPX_shutdown( void )
{
	MPXDBG( "%d\n", getpid(  ) );
	mpx_shutdown_itimer(  );
	mpx_restore_signal(  );

	if ( tlist ) {
	       Threadlist *next,*t=tlist;

		while(t!=NULL) {
		   next=t->next;
		   papi_free( t );
		   t = next;			
		}
		tlist = NULL;
	}
}

int
mpx_check( int EventSet )
{
	/* Currently, there is only the need for one mpx check: if
	 * running on POWER6/perfctr platform, the domain must
	 * include user, kernel, and supervisor, since the scale
	 * event uses the dedicated counter #6, PM_RUN_CYC, which
	 * cannot be controlled on a domain level.
	 */
	EventSetInfo_t *ESI = _papi_hwi_lookup_EventSet( EventSet );

	if (ESI==NULL) return PAPI_EBUG;

	if ( strstr( _papi_hwd[ESI->CmpIdx]->cmp_info.name, "perfctr.c" ) == NULL )
		return PAPI_OK;

	if ( strcmp( _papi_hwi_system_info.hw_info.model_string, "POWER6" ) == 0 ) {
		unsigned int chk_domain =
			PAPI_DOM_USER + PAPI_DOM_KERNEL + PAPI_DOM_SUPERVISOR;

		if ( ( ESI->domain.domain & chk_domain ) != chk_domain ) {
			PAPIERROR
				( "This platform requires PAPI_DOM_USER+PAPI_DOM_KERNEL+PAPI_DOM_SUPERVISOR\n"
				  "to be set in the domain when using multiplexing.  Instead, found %#x\n",
				  ESI->domain.domain );
			return ( PAPI_EINVAL_DOM );
		}
	}
	return PAPI_OK;
}

int
mpx_init( int interval_ns )
{
#if defined(PTHREADS) || defined(_POWER6)
	int retval;
#endif

#ifdef _POWER6
	retval = PAPI_event_name_to_code( "PM_RUN_CYC", &_PNE_PM_RUN_CYC );
	if ( retval != PAPI_OK )
		return ( retval );
#endif
	tlist = NULL;
	mpx_hold(  );
	mpx_shutdown_itimer(  );
	mpx_init_timers( interval_ns / 1000 );

	return ( PAPI_OK );
}

/** Inserts a list of events into the master event list, 
   and adds new mev pointers to the MPX_EventSet. 
   MUST BE CALLED WITH THE TIMER INTERRUPT DISABLED */

static int
mpx_insert_events( MPX_EventSet *mpx_events, int *event_list,
		   int num_events, int domain, int granularity )
{
	int i, retval = 0, num_events_success = 0;
	MasterEvent *mev;
	PAPI_option_t options;
	MasterEvent **head = &mpx_events->mythr->head;

	MPXDBG("Inserting %p %d\n",mpx_events,mpx_events->num_events );

	/* Make sure we don't overrun our buffers */
	if (mpx_events->num_events + num_events > PAPI_MAX_SW_MPX_EVENTS) {
	   return PAPI_ECOUNT;
	}

	/* For each event, see if there is already a corresponding
	 * event in the master set for this thread.  If not, add it.
	 */
	for ( i = 0; i < num_events; i++ ) {

		/* Look for a matching event in the master list */
		for( mev = *head; mev != NULL; mev = mev->next ) {
		   if ( (mev->pi.event_type == event_list[i]) && 
			(mev->pi.domain == domain) &&
			(mev->pi.granularity == granularity ))
				break;
		}

		/* No matching event in the list; add a new one */
		if ( mev == NULL ) {
		   mev = (MasterEvent *) papi_malloc( sizeof ( MasterEvent ) );
		   if ( mev == NULL ) {
		      return PAPI_ENOMEM;
		   }

		   mev->pi.event_type = event_list[i];
		   mev->pi.domain = domain;
		   mev->pi.granularity = granularity;
		   mev->uses = mev->active = 0;
		   mev->prev_total_c = mev->count = mev->cycles = 0;
		   mev->rate_estimate = 0.0;
		   mev->count_estimate = 0;
		   mev->is_a_rate = 0;
		   mev->papi_event = PAPI_NULL;
			
		   retval = PAPI_create_eventset( &( mev->papi_event ) );
		   if ( retval != PAPI_OK ) {
		      MPXDBG( "Event %d could not be counted.\n", 
			      event_list[i] );
		      goto bail;
		   }

		   retval = PAPI_add_event( mev->papi_event, event_list[i] );
		   if ( retval != PAPI_OK ) {
		      MPXDBG( "Event %d could not be counted.\n", 
			      event_list[i] );
		      goto bail;
		   }

		   /* Always count total cycles so we can scale results.
		    * If user just requested cycles, 
		    * don't add that event again. */

		   if ( event_list[i] != SCALE_EVENT ) {
		      retval = PAPI_add_event( mev->papi_event, SCALE_EVENT );
		      if ( retval != PAPI_OK ) {
			 MPXDBG( "Scale event could not be counted "
				 "at the same time.\n" );
			 goto bail;
		      }
		   }
			
		   /* Set the options for the event set */
		   memset( &options, 0x0, sizeof ( options ) );
		   options.domain.eventset = mev->papi_event;
		   options.domain.domain = domain;
		   retval = PAPI_set_opt( PAPI_DOMAIN, &options );
		   if ( retval != PAPI_OK ) {
		      MPXDBG( "PAPI_set_opt(PAPI_DOMAIN, ...) = %d\n", 
			      retval );
		      goto bail;
		   }

		   memset( &options, 0x0, sizeof ( options ) );
		   options.granularity.eventset = mev->papi_event;
		   options.granularity.granularity = granularity;
		   retval = PAPI_set_opt( PAPI_GRANUL, &options );
		   if ( retval != PAPI_OK ) {
		      if ( retval != PAPI_ECMP ) {
			 /* ignore component errors because they typically mean
			    "not supported by the component" */
			 MPXDBG( "PAPI_set_opt(PAPI_GRANUL, ...) = %d\n", 
				 retval );
			 goto bail;
		      }
		   }


		   /* Chain the event set into the 
		    * master list of event sets used in
		    * multiplexing. */

		    mev->next = *head;
		    *head = mev;

		}

		/* If we created a new event set, or we found a matching
		 * eventset already in the list, then add the pointer in
		 * the master list to this threads list. Then we bump the
		 * number of successfully added events. */
	MPXDBG("Inserting now %p %d\n",mpx_events,mpx_events->num_events );

		mpx_events->mev[mpx_events->num_events + num_events_success] = mev;
		mpx_events->mev[mpx_events->num_events + num_events_success]->uses++;
		num_events_success++;

	}

	/* Always be sure the head master event points to the thread */
	if ( *head != NULL ) {
		( *head )->mythr = mpx_events->mythr;
	}
	MPXDBG( "%d of %d events were added.\n", num_events_success, num_events );
	mpx_events->num_events += num_events_success;
	return ( PAPI_OK );

  bail:
	/* If there is a current mev, it is currently not linked into the list
	 * of multiplexing events, so we can just delete that
	 */
	if ( mev && mev->papi_event ) {
	   if (PAPI_cleanup_eventset( mev->papi_event )!=PAPI_OK) {
	     PAPIERROR("Cleanup eventset\n");
	   }
	   if (PAPI_destroy_eventset( &( mev->papi_event )) !=PAPI_OK) {
	     PAPIERROR("Destroy eventset\n");
	   }
	}
	if ( mev )
		papi_free( mev );
	mev = NULL;

	/* Decrease the usage count of events */
	for ( i = 0; i < num_events_success; i++ ) {
		mpx_events->mev[mpx_events->num_events + i]->uses--;
	}

	/* Run the garbage collector to remove unused events */
	if ( num_events_success )
		mpx_remove_unused( head );

	return ( retval );
}

/** Remove events from an mpx event set (and from the
 * master event set for this thread, if the events are unused).
 * MUST BE CALLED WITH THE SIGNAL HANDLER DISABLED
 */
static void
mpx_delete_events( MPX_EventSet * mpx_events )
{
	int i;
	MasterEvent *mev;

	/* First decrement the reference counter for each master
	 * event in this event set, then see if the master events
	 * can be deleted.
	 */
	for ( i = 0; i < mpx_events->num_events; i++ ) {
		mev = mpx_events->mev[i];
		--mev->uses;
		mpx_events->mev[i] = NULL;
		/* If it's no longer used, it should not be active! */
		assert( mev->uses || !( mev->active ) );
	}
	mpx_events->num_events = 0;
	mpx_remove_unused( &mpx_events->mythr->head );
}

/** Remove one event from an mpx event set (and from the
 * master event set for this thread, if the events are unused).
 * MUST BE CALLED WITH THE SIGNAL HANDLER DISABLED
 */
static void
mpx_delete_one_event( MPX_EventSet * mpx_events, int Event )
{
	int i;
	MasterEvent *mev;

	/* First decrement the reference counter for each master
	 * event in this event set, then see if the master events
	 * can be deleted.
	 */
	for ( i = 0; i < mpx_events->num_events; i++ ) {
		mev = mpx_events->mev[i];
		if ( mev->pi.event_type == Event ) {
			--mev->uses;
			mpx_events->num_events--;
			mpx_events->mev[i] = NULL;
			/* If it's no longer used, it should not be active! */
			assert( mev->uses || !( mev->active ) );
			break;
		}
	}

	/* If we removed an event that is not last in the list we
	 * need to compact the event list
	 */

	for ( ; i < mpx_events->num_events; i++ ) {
		mpx_events->mev[i] = mpx_events->mev[i + 1];
		mpx_events->start_values[i] = mpx_events->start_values[i + 1];
		mpx_events->stop_values[i] = mpx_events->stop_values[i + 1];
		mpx_events->start_hc[i] = mpx_events->start_hc[i + 1];
	}
	mpx_events->mev[i] = NULL;

	mpx_remove_unused( &mpx_events->mythr->head );

}

/** Remove events that are not used any longer from the run 
 * list of events to multiplex by the handler
 * MUST BE CALLED WITH THE SIGNAL HANDLER DISABLED
 */
static void
mpx_remove_unused( MasterEvent ** head )
{
	MasterEvent *mev, *lastmev = NULL, *nextmev;
	Threadlist *thr = ( *head == NULL ) ? NULL : ( *head )->mythr;
	int retval;

	/* Clean up and remove unused master events. */
	for ( mev = *head; mev != NULL; mev = nextmev ) {
		nextmev = mev->next; /* get link before mev is freed */
		if ( !mev->uses ) {
			if ( lastmev == NULL ) {	/* this was the head event */
				*head = nextmev;
			} else {
				lastmev->next = nextmev;
			}
			retval=PAPI_cleanup_eventset( mev->papi_event );
			retval=PAPI_destroy_eventset( &( mev->papi_event ) );
			if (retval!=PAPI_OK) PAPIERROR("Error destroying event\n");
			papi_free( mev );
		} else {
			lastmev = mev;
		}
	}

	/* Always be sure the head master event points to the thread */
	if ( *head != NULL ) {
		( *head )->mythr = thr;
	}
}
