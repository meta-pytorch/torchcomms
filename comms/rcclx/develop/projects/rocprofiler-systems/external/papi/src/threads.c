/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    threads.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
*/

/* This file contains thread allocation and bookkeeping functions */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include <string.h>
#include <unistd.h>

/*****************/
/* BEGIN GLOBALS */
/*****************/

/* The following globals get initialized and cleared by:
   extern int _papi_hwi_init_global_threads(void);
   extern int _papi_hwi_shutdown_thread(ThreadInfo_t *thread); */

/* list of threads, gets initialized to master process with TID of getpid() */

volatile ThreadInfo_t *_papi_hwi_thread_head;

/* If we have TLS, this variable ALWAYS points to our thread descriptor. It's like magic! */

#if defined(HAVE_THREAD_LOCAL_STORAGE)
THREAD_LOCAL_STORAGE_KEYWORD ThreadInfo_t *_papi_hwi_my_thread;
#endif

/* Function that returns and unsigned long thread identifier */

unsigned long ( *_papi_hwi_thread_id_fn ) ( void );

/* Function that sends a signal to other threads */

#ifdef ANY_THREAD_GETS_SIGNAL
int ( *_papi_hwi_thread_kill_fn ) ( int, int );
#endif

/*****************/
/*  END  GLOBALS */
/*****************/

static int
lookup_and_set_thread_symbols( void )
{
#if defined(ANY_THREAD_GETS_SIGNAL)
	int retval;
	char *error_ptc = NULL, *error_ptk = NULL;
	void *symbol_ptc = NULL, *symbol_ptk = NULL, *handle = NULL;

	handle = dlopen( NULL, RTLD_LAZY );
	if ( handle == NULL ) {
		PAPIERROR( "Error from dlopen(NULL, RTLD_LAZY): %d %s", errno,
				   dlerror(  ) );
		return ( PAPI_ESYS );
	}

	symbol_ptc = dlsym( handle, "pthread_self" );
	if ( symbol_ptc == NULL ) {
		error_ptc = dlerror(  );
		THRDBG( "dlsym(%p,pthread_self) returned NULL: %s\n",
				( error_ptc ? error_ptc : "No error, NULL symbol!" ) );
	}

	symbol_ptk = dlsym( handle, "pthread_kill" );
	if ( symbol_ptk == NULL ) {
		error_ptk = dlerror(  );
		THRDBG( "dlsym(%p,pthread_kill) returned NULL: %s\n",
				( error_ptk ? error_ptk : "No error, NULL symbol!" ) );
	}

	dlclose( handle );

	if ( !( ( _papi_hwi_thread_kill_fn && _papi_hwi_thread_id_fn ) ||
			( !_papi_hwi_thread_kill_fn && !_papi_hwi_thread_id_fn ) ) )
		return ( PAPI_EMISC );

	_papi_hwi_thread_kill_fn = ( int ( * )( int, int ) ) symbol_ptk;
	_papi_hwi_thread_id_fn = ( unsigned long ( * )( void ) ) symbol_ptc;
#endif
	return ( PAPI_OK );
}

static ThreadInfo_t *
allocate_thread( int tid )
{
	ThreadInfo_t *thread;
	int i;

	/* The Thread EventSet is special. It is not in the EventSet list, but is pointed
	   to by each EventSet of that particular thread. */

	thread = ( ThreadInfo_t * ) papi_malloc( sizeof ( ThreadInfo_t ) );
	if ( thread == NULL )
		return ( NULL );
	memset( thread, 0x00, sizeof ( ThreadInfo_t ) );

	thread->context =
		( hwd_context_t ** ) papi_malloc( sizeof ( hwd_context_t * ) *
										  ( size_t ) papi_num_components );
	if ( !thread->context ) {
		papi_free( thread );
		return ( NULL );
	}

	thread->running_eventset =
		( EventSetInfo_t ** ) papi_malloc( sizeof ( EventSetInfo_t * ) *
										   ( size_t ) papi_num_components );
	if ( !thread->running_eventset ) {
		papi_free( thread->context );
		papi_free( thread );
		return ( NULL );
	}

	for ( i = 0; i < papi_num_components; i++ ) {
		thread->context[i] =
			( void * ) papi_malloc( ( size_t ) _papi_hwd[i]->size.context );
		thread->running_eventset[i] = NULL;
		if ( thread->context[i] == NULL ) {
			for ( i--; i >= 0; i-- )
				papi_free( thread->context[i] );
			papi_free( thread->running_eventset );
			papi_free( thread->context );
			papi_free( thread );
			return ( NULL );
		}
		memset( thread->context[i], 0x00,
				( size_t ) _papi_hwd[i]->size.context );
	}

	if ( _papi_hwi_thread_id_fn ) {
           thread->tid = ( *_papi_hwi_thread_id_fn ) (  );
	}
	else {
	   thread->tid = ( unsigned long ) getpid(  );
	}

	thread->allocator_tid=thread->tid;

	if (tid == 0 ) {
	}
	else {
	  thread->tid=tid;
	}

	THRDBG( "Allocated thread %ld at %p, allocator: %ld\n", thread->tid, 
		thread,
		thread->allocator_tid );

	return thread;
}

static void
free_thread( ThreadInfo_t ** thread )
{
	int i;
	THRDBG( "Freeing thread %ld at %p\n", ( *thread )->tid, *thread );

	for ( i = 0; i < papi_num_components; i++ ) {
		if ( ( *thread )->context[i] )
			papi_free( ( *thread )->context[i] );
	}

	if ( ( *thread )->context )
		papi_free( ( *thread )->context );

	if ( ( *thread )->running_eventset )
		papi_free( ( *thread )->running_eventset );

	memset( *thread, 0x00, sizeof ( ThreadInfo_t ) );
	papi_free( *thread );
	*thread = NULL;
}

static void
insert_thread( ThreadInfo_t * entry, int tid )
{
	_papi_hwi_lock( THREADS_LOCK );

	if ( _papi_hwi_thread_head == NULL ) {	/* 0 elements */
		THRDBG( "_papi_hwi_thread_head is NULL\n" );
		entry->next = entry;
	} else if ( _papi_hwi_thread_head->next == _papi_hwi_thread_head ) {	/* 1 elements */
		THRDBG( "_papi_hwi_thread_head was thread %ld at %p\n",
				_papi_hwi_thread_head->tid, _papi_hwi_thread_head );
		_papi_hwi_thread_head->next = entry;
		entry->next = ( ThreadInfo_t * ) _papi_hwi_thread_head;
	} else {				 /* 2+ elements */

		THRDBG( "_papi_hwi_thread_head was thread %ld at %p\n",
				_papi_hwi_thread_head->tid, _papi_hwi_thread_head );
		entry->next = _papi_hwi_thread_head->next;
		_papi_hwi_thread_head->next = entry;
	}

	_papi_hwi_thread_head = entry;

	THRDBG( "_papi_hwi_thread_head now thread %ld at %p\n",
			_papi_hwi_thread_head->tid, _papi_hwi_thread_head );

	_papi_hwi_unlock( THREADS_LOCK );

#if defined(HAVE_THREAD_LOCAL_STORAGE)
	/* Don't set the current local thread if we are a fake attach thread */
        if (tid==0) {
	   _papi_hwi_my_thread = entry;
	   THRDBG( "TLS for thread %ld is now %p\n", entry->tid,
			_papi_hwi_my_thread );
	}
#else
	( void ) tid;
#endif
}

static int
remove_thread( ThreadInfo_t * entry )
{
	ThreadInfo_t *tmp = NULL, *prev = NULL;

	_papi_hwi_lock( THREADS_LOCK );

	THRDBG( "_papi_hwi_thread_head was thread %ld at %p\n",
			_papi_hwi_thread_head->tid, _papi_hwi_thread_head );

	/* Find the preceding element and the matched element,
	   short circuit if we've seen the head twice */

	for ( tmp = ( ThreadInfo_t * ) _papi_hwi_thread_head;
		  ( entry != tmp ) || ( prev == NULL ); tmp = tmp->next ) {
		prev = tmp;
	}

	if ( tmp != entry ) {
		THRDBG( "Thread %ld at %p was not found in the thread list!\n",
				entry->tid, entry );
		return ( PAPI_EBUG );
	}

	/* Only 1 element in list */

	if ( prev == tmp ) {
		_papi_hwi_thread_head = NULL;
		tmp->next = NULL;
		THRDBG( "_papi_hwi_thread_head now NULL\n" );
	} else {
		prev->next = tmp->next;
		/* If we're removing the head, better advance it! */
		if ( _papi_hwi_thread_head == tmp ) {
			_papi_hwi_thread_head = tmp->next;
			THRDBG( "_papi_hwi_thread_head now thread %ld at %p\n",
					_papi_hwi_thread_head->tid, _papi_hwi_thread_head );
		}
		THRDBG( "Removed thread %p from list\n", tmp );
	}

	_papi_hwi_unlock( THREADS_LOCK );

#if defined(HAVE_THREAD_LOCAL_STORAGE)
	_papi_hwi_my_thread = NULL;
	THRDBG( "TLS for thread %ld is now %p\n", entry->tid,
			_papi_hwi_my_thread );
#endif

	return PAPI_OK;
}

int
_papi_hwi_initialize_thread( ThreadInfo_t ** dest, int tid )
{
	int retval;
	ThreadInfo_t *thread;
	int i;

	if ( ( thread = allocate_thread( tid  ) ) == NULL ) {
		*dest = NULL;
		return PAPI_ENOMEM;
	}

   /* init event memory variables, used by papi_internal.c  */
   thread->tls_papi_event_code = -1;
   thread->tls_papi_event_code_changed = -1;

	/* Call the component to fill in anything special. */

	for ( i = 0; i < papi_num_components; i++ ) {
	    if (_papi_hwd[i]->cmp_info.disabled &&
            _papi_hwd[i]->cmp_info.disabled != PAPI_EDELAY_INIT)
            continue;
	    retval = _papi_hwd[i]->init_thread( thread->context[i] );
	    if ( retval ) {
	       free_thread( &thread );
	       *dest = NULL;
	       return retval;
	    }
	}

	insert_thread( thread, tid );

	*dest = thread;
	return PAPI_OK;
}

#if defined(ANY_THREAD_GETS_SIGNAL)

/* This is ONLY defined for systems that enable ANY_THREAD_GETS_SIGNAL
   since we must forward signals sent to non-PAPI threads.

   This is NOT compatible with thread local storage, since to broadcast
   the signal, we need a list of threads. */

int
_papi_hwi_broadcast_signal( unsigned int mytid )
{
	int i, retval, didsomething = 0;
	volatile ThreadInfo_t *foo = NULL;

	_papi_hwi_lock( THREADS_LOCK );

	for ( foo = _papi_hwi_thread_head; foo != NULL; foo = foo->next ) {
		/* xxxx Should this be hardcoded to index 0 or walk the list or what? */
		for ( i = 0; i < papi_num_components; i++ ) {
			if ( ( foo->tid != mytid ) && ( foo->running_eventset[i] ) &&
				 ( foo->running_eventset[i]->
				   state & ( PAPI_OVERFLOWING | PAPI_MULTIPLEXING ) ) ) {
				/* xxxx mpx_info inside _papi_mdi_t _papi_hwi_system_info is commented out.
				   See papi_internal.h for details. The multiplex_timer_sig value is now part of that structure */
			  THRDBG("Thread %ld sending signal %d to thread %ld\n",mytid,foo->tid,
				  (foo->running_eventset[i]->state & PAPI_OVERFLOWING ? _papi_hwd[i]->cmp_info.hardware_intr_sig : _papi_os_info.itimer_sig));
			  retval = (*_papi_hwi_thread_kill_fn)(foo->tid, 
				  (foo->running_eventset[i]->state & PAPI_OVERFLOWING ? _papi_hwd[i]->cmp_info.hardware_intr_sig : _papi_os_info.itimer_sig));
			  if (retval != 0)
				return(PAPI_EMISC);
			}
		}
		if ( foo->next == _papi_hwi_thread_head )
		  break;
	}
	_papi_hwi_unlock( THREADS_LOCK );

	return ( PAPI_OK );
}
#endif

/* This is undefined for systems that enable ANY_THREAD_GETS_SIGNAL
   since we always must enable threads for safety. */

int
_papi_hwi_set_thread_id_fn( unsigned long ( *id_fn ) ( void ) )
{
#if !defined(ANY_THREAD_GETS_SIGNAL)
	/* Check for multiple threads still in the list, if so, we can't change it */

	if ( _papi_hwi_thread_head->next != _papi_hwi_thread_head )
		return ( PAPI_EINVAL );

	/* We can't change the thread id function from one to another, 
	   only NULL to non-NULL and vice versa. */

	if ( ( id_fn != NULL ) && ( _papi_hwi_thread_id_fn != NULL ) )
		return ( PAPI_EINVAL );

	_papi_hwi_thread_id_fn = id_fn;

	THRDBG( "Set new thread id function to %p\n", id_fn );

	if ( id_fn )
		_papi_hwi_thread_head->tid = ( *_papi_hwi_thread_id_fn ) (  );
	else
		_papi_hwi_thread_head->tid = ( unsigned long ) getpid(  );

	THRDBG( "New master tid is %ld\n", _papi_hwi_thread_head->tid );
#else
	THRDBG( "Skipping set of thread id function\n" );
#endif

	return PAPI_OK;
}


static int _papi_hwi_thread_free_eventsets(long tid) {

   EventSetInfo_t *ESI;
   ThreadInfo_t *master;
   DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
   int i;

   master = _papi_hwi_lookup_thread( tid );

   _papi_hwi_lock( INTERNAL_LOCK );

   for( i = 0; i < map->totalSlots; i++ ) {
      ESI = map->dataSlotArray[i];
      if ( ( ESI ) && (ESI->master!=NULL) ) {

	 if ( ESI->master == master ) {
	    THRDBG("Attempting to remove %d from tid %ld\n",ESI->EventSetIndex,tid);

	    /* Code copied from _papi_hwi_remove_EventSet(ESI);      */
	    _papi_hwi_free_EventSet( ESI );
	    map->dataSlotArray[i] = NULL;
	    map->availSlots++;
	    map->fullSlots--;
	 }
      }
   }

   _papi_hwi_unlock( INTERNAL_LOCK );

   return PAPI_OK;
}


int
_papi_hwi_shutdown_thread( ThreadInfo_t * thread, int force_shutdown )
{
	int retval = PAPI_OK;
	unsigned long tid;
	int i, failure = 0;

   /* Clear event memory variables */
   thread->tls_papi_event_code = -1;
   thread->tls_papi_event_code_changed = -1;

   /* Get thread id */
	if ( _papi_hwi_thread_id_fn )
		tid = ( *_papi_hwi_thread_id_fn ) (  );
	else
		tid = ( unsigned long ) getpid(  );

        THRDBG("Want to shutdown thread %ld, alloc %ld, our_tid: %ld\n",
	       thread->tid,
	       thread->allocator_tid,
	       tid);

	if ((thread->tid==tid) || ( thread->allocator_tid == tid ) || force_shutdown) {

                _papi_hwi_thread_free_eventsets(tid);

		remove_thread( thread );
		THRDBG( "Shutting down thread %ld at %p\n", thread->tid, thread );
		for( i = 0; i < papi_num_components; i++ ) {
		   if (_papi_hwd[i]->cmp_info.disabled &&
               _papi_hwd[i]->cmp_info.disabled != PAPI_EDELAY_INIT)
               continue;
		   retval = _papi_hwd[i]->shutdown_thread( thread->context[i]);
		   if ( retval != PAPI_OK ) failure = retval;
		}
		free_thread( &thread );
		return ( failure );
	}

	THRDBG( "Skipping shutdown thread %ld at %p, thread %ld not allocator!\n",
			thread->tid, thread, tid );
	return PAPI_EBUG;
}

/* THESE MUST BE CALLED WITH A GLOBAL LOCK */

int
_papi_hwi_shutdown_global_threads( void )
{
        int err,num_threads,i;
	ThreadInfo_t *tmp,*next;
	unsigned long our_tid;

	tmp = _papi_hwi_lookup_thread( 0 );

	if ( tmp == NULL ) {
	   THRDBG( "Did not find my thread for shutdown!\n" );
	   err = PAPI_EBUG;
	}
	else {
	   our_tid=tmp->tid;
	   (void)our_tid;

	   THRDBG("Shutting down %ld\n",our_tid);

	   err = _papi_hwi_shutdown_thread( tmp, 1 );

	   /* count threads */
	   tmp = ( ThreadInfo_t * ) _papi_hwi_thread_head;
	   num_threads=0;
	   while(tmp!=NULL) {
	      num_threads++;
	      if (tmp->next==_papi_hwi_thread_head) break;
	      tmp=tmp->next;
	   }

	   /* Shut down all threads allocated by this thread */
	   /* Urgh it's a circular list where we removed in the loop  */
	   /* so the only sane way to do it is get a count in advance */
	   tmp = ( ThreadInfo_t * ) _papi_hwi_thread_head;

	   for(i=0;i<num_threads;i++) {

	      next=tmp->next;

	      THRDBG("looking at #%d %ld our_tid: %ld alloc_tid: %ld\n",
		     i,tmp->tid,our_tid,tmp->allocator_tid);

		 THRDBG("Also removing thread %ld\n",tmp->tid);
	         err = _papi_hwi_shutdown_thread( tmp, 1 );

	      tmp=next;

	   }
	}


#ifdef DEBUG
	if ( ISLEVEL( DEBUG_THREADS ) ) {
		if ( _papi_hwi_thread_head ) {
			THRDBG( "Thread head %p still exists!\n", _papi_hwi_thread_head );
		}
	}
#endif

#if defined(HAVE_THREAD_LOCAL_STORAGE)
	_papi_hwi_my_thread = NULL;
#endif
	_papi_hwi_thread_head = NULL;
	_papi_hwi_thread_id_fn = NULL;
#if defined(ANY_THREAD_GETS_SIGNAL)
	_papi_hwi_thread_kill_fn = NULL;
#endif

	return err;
}

int
_papi_hwi_init_global_threads( void )
{
	int retval;
	ThreadInfo_t *tmp;

	_papi_hwi_lock( GLOBAL_LOCK );

#if defined(HAVE_THREAD_LOCAL_STORAGE)
	_papi_hwi_my_thread = NULL;
#endif
	_papi_hwi_thread_head = NULL;
	_papi_hwi_thread_id_fn = NULL;
#if defined(ANY_THREAD_GETS_SIGNAL)
	_papi_hwi_thread_kill_fn = NULL;
#endif

	retval = _papi_hwi_initialize_thread( &tmp , 0);
	if ( retval == PAPI_OK ) {
	   retval = lookup_and_set_thread_symbols(  );
	}

	_papi_hwi_unlock( GLOBAL_LOCK );

	return ( retval );
}

int
_papi_hwi_gather_all_thrspec_data( int tag, PAPI_all_thr_spec_t * where )
{
	int didsomething = 0;
	ThreadInfo_t *foo = NULL;

	_papi_hwi_lock( THREADS_LOCK );

	for ( foo = ( ThreadInfo_t * ) _papi_hwi_thread_head; foo != NULL;
		  foo = foo->next ) {
		/* If we want thread ID's */
		if ( where->id )
			memcpy( &where->id[didsomething], &foo->tid,
					sizeof ( where->id[didsomething] ) );

		/* If we want data pointers */
		if ( where->data )
			where->data[didsomething] = foo->thread_storage[tag];

		didsomething++;

		if ( ( where->id ) || ( where->data ) ) {
			if ( didsomething >= where->num )
				break;
		}

		if ( foo->next == _papi_hwi_thread_head )
			break;
	}

	where->num = didsomething;
	_papi_hwi_unlock( THREADS_LOCK );

	return ( PAPI_OK );

}

#if defined(__APPLE__)
  #include <pthread.h>
  unsigned long _papi_gettid(void)
  {
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);  // macOS-specific thread ID function
    return (unsigned long)tid;
  }
#elif defined(__NR_gettid) && !defined(HAVE_GETTID)
  #include <syscall.h>
  #include <unistd.h>
  unsigned long _papi_gettid(void)
  {
    return (unsigned long)(syscall(__NR_gettid));
  } 
#elif defined(HAVE_GETTID)
  #include <sys/types.h>
  unsigned long _papi_gettid(void)
  {
    return (unsigned long)(gettid());
  }
#elif defined(HAVE_SYSCALL_GETTID)
  #include <syscall.h>
  #include <sys/types.h>
  unsigned long _papi_gettid(void)
  {
    return (unsigned long)(syscall(SYS_gettid));
  }
#else
  #include <sys/types.h>
  #include <unistd.h>
  /* Fall-back on getpid for tid if not available. */
  unsigned long _papi_gettid(void)
  {
    return (unsigned long)(getpid());
  }
#endif

unsigned long _papi_getpid(void)
{
    return (unsigned long) getpid();
}
