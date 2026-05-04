/** @file threads.h
 *  CVS: $Id$
 *  @author ??
 */

#ifndef PAPI_THREADS_H
#define PAPI_THREADS_H

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#ifdef HAVE_THREAD_LOCAL_STORAGE
#define THREAD_LOCAL_STORAGE_KEYWORD HAVE_THREAD_LOCAL_STORAGE
#else
#define THREAD_LOCAL_STORAGE_KEYWORD
#endif

#if defined(ANY_THREAD_GETS_SIGNAL) && !defined(_AIX)
#error "lookup_and_set_thread_symbols and _papi_hwi_broadcast_signal have only been tested on AIX"
#endif

typedef struct _ThreadInfo
{
	unsigned long int tid;
	unsigned long int allocator_tid;
	struct _ThreadInfo *next;
	hwd_context_t **context;
	void *thread_storage[PAPI_MAX_TLS];
	EventSetInfo_t **running_eventset;
	EventSetInfo_t *from_esi;          /* ESI used for last update this control state */
	int wants_signal;

   // The current event code can be stored here prior to 
   // component calls and cleared after the component returns.
   unsigned int tls_papi_event_code;
   int tls_papi_event_code_changed;
} ThreadInfo_t;

/** The list of threads, gets initialized to master process with TID of getpid() 
 *	@internal */

extern volatile ThreadInfo_t *_papi_hwi_thread_head;

/* If we have TLS, this variable ALWAYS points to our thread descriptor. It's like magic! */

#if defined(HAVE_THREAD_LOCAL_STORAGE)
extern THREAD_LOCAL_STORAGE_KEYWORD ThreadInfo_t *_papi_hwi_my_thread;
#endif

/** Function that returns an unsigned long int thread identifier 
 *	@internal */

extern unsigned long int ( *_papi_hwi_thread_id_fn ) ( void );

/** Function that sends a signal to other threads 
 *	@internal */

extern int ( *_papi_hwi_thread_kill_fn ) ( int, int );

extern int _papi_hwi_initialize_thread( ThreadInfo_t ** dest, int tid );
extern int _papi_hwi_init_global_threads( void );
extern int _papi_hwi_shutdown_thread( ThreadInfo_t * thread, int force );
extern int _papi_hwi_shutdown_global_threads( void );
extern int _papi_hwi_broadcast_signal( unsigned int mytid );
extern int _papi_hwi_set_thread_id_fn( unsigned long int ( *id_fn ) ( void ) );

inline_static int
_papi_hwi_lock( int lck )
{
	if ( _papi_hwi_thread_id_fn ) {
		_papi_hwd_lock( lck );
		THRDBG( "Lock %d\n", lck );
	} else {
		( void ) lck;		 /* unused if !defined(DEBUG) */
		THRDBG( "Skipped lock %d\n", lck );
	}

	return ( PAPI_OK );
}

inline_static int
_papi_hwi_unlock( int lck )
{
	if ( _papi_hwi_thread_id_fn ) {
		_papi_hwd_unlock( lck );
		THRDBG( "Unlock %d\n", lck );
	} else {
		( void ) lck;		 /* unused if !defined(DEBUG) */
		THRDBG( "Skipped unlock %d\n", lck );
	}

	return ( PAPI_OK );
}

inline_static ThreadInfo_t *
_papi_hwi_lookup_thread( int custom_tid )
{

	unsigned long int tid;
	ThreadInfo_t *tmp;


	if (custom_tid==0) {
#ifdef HAVE_THREAD_LOCAL_STORAGE
	   THRDBG( "TLS returning %p\n", _papi_hwi_my_thread );
	   return ( _papi_hwi_my_thread );
#else
	   if ( _papi_hwi_thread_id_fn == NULL ) {
	      THRDBG( "Threads not initialized, returning master thread at %p\n",
				_papi_hwi_thread_head );
	      return ( ( ThreadInfo_t * ) _papi_hwi_thread_head );
	   }

	   tid = ( *_papi_hwi_thread_id_fn ) (  );
#endif
	}
	else {
	  tid=custom_tid;
	}
	THRDBG( "Threads initialized, looking for thread %#lx\n", tid );

	_papi_hwi_lock( THREADS_LOCK );

	tmp = ( ThreadInfo_t * ) _papi_hwi_thread_head;
	while ( tmp != NULL ) {
		THRDBG( "Examining thread tid %#lx at %p\n", tmp->tid, tmp );
		if ( tmp->tid == tid )
			break;
		tmp = tmp->next;
		if ( tmp == _papi_hwi_thread_head ) {
			tmp = NULL;
			break;
		}
	}

	if ( tmp ) {
		_papi_hwi_thread_head = tmp;
		THRDBG( "Found thread %ld at %p\n", tid, tmp );
	} else {
		THRDBG( "Did not find tid %ld\n", tid );
	}

	_papi_hwi_unlock( THREADS_LOCK );
	return ( tmp );

}

inline_static int
_papi_hwi_lookup_or_create_thread( ThreadInfo_t ** here, int tid )
{
	ThreadInfo_t *tmp = _papi_hwi_lookup_thread( tid );
	int retval = PAPI_OK;

	if ( tmp == NULL )
	  retval = _papi_hwi_initialize_thread( &tmp, tid );

	if ( retval == PAPI_OK )
		*here = tmp;

	return ( retval );
}

/* Prototypes */
void _papi_hwi_shutdown_the_thread_list( void );
void _papi_hwi_cleanup_thread_list( void );
int _papi_hwi_insert_in_thread_list( ThreadInfo_t * ptr );
ThreadInfo_t *_papi_hwi_lookup_in_thread_list(  );
void _papi_hwi_shutdown_the_thread_list( void );
int _papi_hwi_get_thr_context( void ** );
int _papi_hwi_gather_all_thrspec_data( int tag, PAPI_all_thr_spec_t * where );
unsigned long _papi_gettid(void);
unsigned long _papi_getpid(void);

#endif
