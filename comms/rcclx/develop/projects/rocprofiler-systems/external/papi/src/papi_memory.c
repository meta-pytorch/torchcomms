/**
 * @file    papi_memory.c
 * @author  Kevin London
 *          london@cs.utk.edu
 * PAPI memory allocation provides for checking and maintenance of all memory
 * allocated through this interface. Implemented as a series of wrappers around
 * standard C memory allocation routines, _papi_malloc and associated functions
 * add a prolog and optional epilog to each malloc'd pointer.
 * The prolog, sized to preserve memory alignment, contains a pointer to a 
 * linked list of pmem_t structures that describe every block of memory 
 * allocated through these calls.
 * The optional epilog is enabled if DEBUG is defined, and contains 
 * a distinctive pattern that allows checking for pointer overflow.
 */

#define IN_MEM_FILE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "papi_lock.h"
#include "papi_memory.h"
#include "papi_internal.h"


/** Define the amount of extra memory at the beginning of the alloc'd pointer.
 * This is usually the size of a pointer, but in some cases needs to be bigger
 * to preserve data alignment.
 */
#define MEM_PROLOG (2*sizeof(void *))

/* If you are tracing memory, then DEBUG must be set also. */
#ifdef DEBUG
/** Define the amount of extra memory at the end of the alloc'd pointer.
 * Also define the contents: 0xCACA
 */
#define MEM_EPILOG 4
#define MEM_EPILOG_1 0xC
#define MEM_EPILOG_2 0xA
#define MEM_EPILOG_3 0xC
#define MEM_EPILOG_4 0xA
#endif

/* Local global variables */
static pmem_t *mem_head = NULL;

/* Local Prototypes */
static pmem_t *get_mem_ptr( void *ptr );
static pmem_t *init_mem_ptr( void *, int, char *, int );
static void insert_mem_ptr( pmem_t * );
static void remove_mem_ptr( pmem_t * );
static int set_epilog( pmem_t * mem_ptr );

/**********************************************************************
 * Exposed papi versions of std memory management routines:           *
 *  _papi_realloc                                                     *
 *  _papi_calloc                                                      *
 *  _papi_malloc                                                      *
 *  _papi_strdup                                                      *
 *  _papi_free                                                        *
 *  _papi_valid_free                                                  *
 * Exposed useful papi memory maintenance routines:                   *
 *  _papi_mem_print_info                                              *
 *  _papi_mem_print_stats                                             *
 *  _papi_mem_overhead                                                *
 *  _papi_mem_cleanup_all                                             *
 *  _papi_mem_check_buf_overflow                                      *
 *  _papi_mem_check_all_overflow                                      *
 **********************************************************************/

/** _papi_realloc -- given a pointer returned by _papi_malloc, returns a pointer
 * to the related pmem_t structure describing this pointer.
 * Checks for NULL pointers and returns NULL if error.
 */
void *
_papi_realloc( char *file, int line, void *ptr, size_t size )
{
	size_t nsize = size + MEM_PROLOG;
	pmem_t *mem_ptr;
	void *nptr;

#ifdef DEBUG
	nsize += MEM_EPILOG;
	_papi_hwi_lock( MEMORY_LOCK );
	_papi_mem_check_all_overflow(  );
#endif

	if ( !ptr )
		return ( _papi_malloc( file, line, size ) );

	mem_ptr = get_mem_ptr( ptr );
	nptr = ( pmem_t * ) realloc( ( ( char * ) ptr - MEM_PROLOG ), nsize );

	if ( !nptr )
		return ( NULL );

	mem_ptr->size = ( int ) size;
	mem_ptr->ptr = ( char * ) nptr + MEM_PROLOG;
#ifdef DEBUG
	strncpy( mem_ptr->file, file, DEBUG_FILE_LEN );
	mem_ptr->file[DEBUG_FILE_LEN - 1] = '\0';
	mem_ptr->line = line;
	set_epilog( mem_ptr );
	_papi_hwi_unlock( MEMORY_LOCK );
#endif
	MEMDBG( "%p: Re-allocated: %lu bytes from File: %s  Line: %d\n",
			mem_ptr->ptr, ( unsigned long ) size, file, line );
	return ( mem_ptr->ptr );
}

void *
_papi_calloc( char *file, int line, size_t nmemb, size_t size )
{
	void *ptr = _papi_malloc( file, line, size * nmemb );

	if ( !ptr )
		return ( NULL );
	memset( ptr, 0, size * nmemb );
	return ( ptr );
}

void *
_papi_malloc( char *file, int line, size_t size )
{
	void *ptr;
	void **tmp;
	pmem_t *mem_ptr;
	size_t nsize = size + MEM_PROLOG;

#ifdef DEBUG
	nsize += MEM_EPILOG;
#endif

	if ( size == 0 ) {
		MEMDBG( "Attempting to allocate %lu bytes from File: %s  Line: %d\n",
				( unsigned long ) size, file, line );
		return ( NULL );
	}

	ptr = ( void * ) malloc( nsize );

	if ( !ptr )
		return ( NULL );
	else {
		if ( ( mem_ptr =
			   init_mem_ptr( ( char * ) ptr + MEM_PROLOG, ( int ) size, file,
							 line ) ) == NULL ) {
			free( ptr );
			return ( NULL );
		}
		tmp = ptr;
		*tmp = mem_ptr;
		ptr = mem_ptr->ptr;
		mem_ptr->ptr = ptr;
		_papi_hwi_lock( MEMORY_LOCK );
		insert_mem_ptr( mem_ptr );
		set_epilog( mem_ptr );
		_papi_hwi_unlock( MEMORY_LOCK );

		MEMDBG( "%p: Allocated %lu bytes from File: %s  Line: %d\n",
				mem_ptr->ptr, ( unsigned long ) size, file, line );
		return ( ptr );
	}
	return ( NULL );
}

char *
_papi_strdup( char *file, int line, const char *s )
{
	size_t size;
	char *ptr;

	if ( !s )
		return ( NULL );

	/* String Length +1 for \0 */
	size = strlen( s ) + 1;
	ptr = ( char * ) _papi_malloc( file, line, size );

	if ( !ptr )
		return ( NULL );

	memcpy( ptr, s, size );
	return ( ptr );
}

/** Only frees the memory if PAPI malloced it 
  * returns 1 if pointer was valid; 0 if not */
int
_papi_valid_free( char *file, int line, void *ptr )
{
	pmem_t *tmp;
	int valid = 0;

	if ( !ptr ) {
		( void ) file;
		( void ) line;
		return ( 0 );
	}

	_papi_hwi_lock( MEMORY_LOCK );

	for ( tmp = mem_head; tmp; tmp = tmp->next ) {
		if ( ptr == tmp->ptr ) {
			pmem_t *mem_ptr = get_mem_ptr( ptr );

			if ( mem_ptr ) {
				MEMDBG( "%p: Freeing %d bytes from File: %s  Line: %d\n",
						mem_ptr->ptr, mem_ptr->size, file, line );
				remove_mem_ptr( mem_ptr );
				_papi_mem_check_all_overflow(  );
			}

			valid = 1;
			break;
		}
	}

	_papi_hwi_unlock( MEMORY_LOCK );
	return ( valid );
}

/** Frees up the ptr */
void
_papi_free( char *file, int line, void *ptr )
{
	pmem_t *mem_ptr = get_mem_ptr( ptr );

	if ( !mem_ptr ) {
		( void ) file;
		( void ) line;
		return;
	}

	MEMDBG( "%p: Freeing %d bytes from File: %s  Line: %d\n", mem_ptr->ptr,
			mem_ptr->size, file, line );

	_papi_hwi_lock( MEMORY_LOCK );
	remove_mem_ptr( mem_ptr );
	_papi_mem_check_all_overflow(  );
	_papi_hwi_unlock( MEMORY_LOCK );
}

/** Print information about the memory including file and location it came from */
void
_papi_mem_print_info( void *ptr )
{
	pmem_t *mem_ptr = get_mem_ptr( ptr );

#ifdef DEBUG
	fprintf( stderr, "%p: Allocated %d bytes from File: %s  Line: %d\n", ptr,
			 mem_ptr->size, mem_ptr->file, mem_ptr->line );
#else
	fprintf( stderr, "%p: Allocated %d bytes\n", ptr, mem_ptr->size );
#endif
	return;
}

/** Print out all memory information */
void
_papi_mem_print_stats(  )
{
	pmem_t *tmp = NULL;

	_papi_hwi_lock( MEMORY_LOCK );
	for ( tmp = mem_head; tmp; tmp = tmp->next ) {
		_papi_mem_print_info( tmp->ptr );
	}
	_papi_hwi_unlock( MEMORY_LOCK );
}

/** Return the amount of memory overhead of the PAPI library and the memory system
 * PAPI_MEM_LIB_OVERHEAD is the library overhead
 * PAPI_MEM_OVERHEAD is the memory overhead
 * They both can be | together
 * This only includes "malloc'd memory"
 */
int
_papi_mem_overhead( int type )
{
	pmem_t *ptr = NULL;
	int size = 0;

	_papi_hwi_lock( MEMORY_LOCK );
	for ( ptr = mem_head; ptr; ptr = ptr->next ) {
		if ( type & PAPI_MEM_LIB_OVERHEAD )
			size += ptr->size;
		if ( type & PAPI_MEM_OVERHEAD ) {
			size += ( int ) sizeof ( pmem_t );
			size += ( int ) MEM_PROLOG;
#ifdef DEBUG
			size += ( int ) MEM_EPILOG;
#endif
		}
	}
	_papi_hwi_unlock( MEMORY_LOCK );
	return size;
}

/** Clean all memory up and print out memory leak information to stderr */
void
_papi_mem_cleanup_all(  )
{
	pmem_t *ptr = NULL, *tmp = NULL;
#ifdef DEBUG
	int cnt = 0;
#endif

	_papi_hwi_lock( MEMORY_LOCK );
	_papi_mem_check_all_overflow(  );

	for ( ptr = mem_head; ptr; ptr = tmp ) {
		tmp = ptr->next;
#ifdef DEBUG
		LEAKDBG( "MEMORY LEAK: %p of %d bytes, from File: %s Line: %d\n",
				 ptr->ptr, ptr->size, ptr->file, ptr->line );
		cnt += ptr->size;
#endif

		remove_mem_ptr( ptr );
	}
	_papi_hwi_unlock( MEMORY_LOCK );
#ifdef DEBUG
	if ( 0 != cnt ) { 
		LEAKDBG( "TOTAL MEMORY LEAK: %d bytes.\n", cnt );
	}
#endif
}

/* Loop through memory structures and look for buffer overflows 
 * returns the number of overflows detected
 */

/**********************************************************************
 * Private helper routines for papi memory management                 *
 **********************************************************************/

/* Given a pointer returned by _papi_malloc, returns a pointer
 * to the related pmem_t structure describing this pointer.
 * Checks for NULL pointers and returns NULL if error.
 */
static pmem_t *
get_mem_ptr( void *ptr )
{
	pmem_t **tmp_ptr = ( pmem_t ** ) ( ( char * ) ptr - MEM_PROLOG );
	pmem_t *mem_ptr;

	if ( !tmp_ptr || !ptr )
		return ( NULL );

	mem_ptr = *tmp_ptr;
	return ( mem_ptr );
}

/* Allocate and initialize a memory pointer */
pmem_t *
init_mem_ptr( void *ptr, int size, char *file, int line )
{
	pmem_t *mem_ptr = NULL;
	if ( ( mem_ptr = ( pmem_t * ) malloc( sizeof ( pmem_t ) ) ) == NULL )
		return ( NULL );

	mem_ptr->ptr = ptr;
	mem_ptr->size = size;
	mem_ptr->next = NULL;
	mem_ptr->prev = NULL;
#ifdef DEBUG
	strncpy( mem_ptr->file, file, DEBUG_FILE_LEN );
	mem_ptr->file[DEBUG_FILE_LEN - 1] = '\0';
	mem_ptr->line = line;
#else
	( void ) file;			 /*unused */
	( void ) line;			 /*unused */
#endif
	return ( mem_ptr );
}

/* Insert the memory information 
 * Do not lock these routines, but lock in routines using these
 */
static void
insert_mem_ptr( pmem_t * ptr )
{
	if ( !ptr )
		return;

	if ( !mem_head ) {
		mem_head = ptr;
		ptr->next = NULL;
		ptr->prev = NULL;
	} else {
		mem_head->prev = ptr;
		ptr->next = mem_head;
		mem_head = ptr;
	}
	return;
}

/* Remove the memory information pointer and free the memory 
 * Do not using locking in this routine, instead lock around 
 * the sections of code that use this call.
 */
static void
remove_mem_ptr( pmem_t * ptr )
{
	if ( !ptr )
		return;

	if ( ptr->prev )
		ptr->prev->next = ptr->next;
	if ( ptr->next )
		ptr->next->prev = ptr->prev;
	if ( ptr == mem_head )
		mem_head = ptr->next;
	free( ptr );
}

static int
set_epilog( pmem_t * mem_ptr )
{
#ifdef DEBUG
	char *chptr = ( char * ) mem_ptr->ptr + mem_ptr->size;
	*chptr++ = MEM_EPILOG_1;
	*chptr++ = MEM_EPILOG_2;
	*chptr++ = MEM_EPILOG_3;
	*chptr++ = MEM_EPILOG_4;
	return ( _papi_mem_check_all_overflow(  ) );
#else
	( void ) mem_ptr;		 /*unused */
#endif
	return ( 0 );
}

/* Check for memory buffer overflows */
#ifdef DEBUG
static int
_papi_mem_check_buf_overflow( pmem_t * tmp )
{
	int fnd = 0;
	char *ptr;
	char *tptr;

	if ( !tmp )
		return ( 0 );

	tptr = tmp->ptr;
	tptr += tmp->size;

	/* Move to the buffer overflow padding */
	ptr = ( ( char * ) tmp->ptr ) + tmp->size;
	if ( *ptr++ != MEM_EPILOG_1 )
		fnd = 1;
	else if ( *ptr++ != MEM_EPILOG_2 )
		fnd = 2;
	else if ( *ptr++ != MEM_EPILOG_3 )
		fnd = 3;
	else if ( *ptr++ != MEM_EPILOG_4 )
		fnd = 4;

	if ( fnd ) {
		LEAKDBG( "Buffer Overflow[%d] for %p allocated from %s at line %d\n",
				 fnd, tmp->ptr, tmp->file, tmp->line );
	}
	return ( fnd );
}
#endif

int
_papi_mem_check_all_overflow(  )
{
	int fnd = 0;
#ifdef DEBUG
	pmem_t *tmp;

	for ( tmp = mem_head; tmp; tmp = tmp->next ) {
		if ( _papi_mem_check_buf_overflow( tmp ) )
			fnd++;
	}

	if ( fnd ) {
		LEAKDBG( "%d Total Buffer overflows detected!\n", fnd );
	}
#endif
	return ( fnd );
}
