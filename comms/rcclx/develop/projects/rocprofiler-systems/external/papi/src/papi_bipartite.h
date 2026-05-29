/*
* File:    papi_bipartite.h
* Author:  Dan Terpstra
*          terpstra@eecs.utk.edu
* Mods:    
*          
*/
/* This file contains one function: _papi_bipartite_alloc()
   Its role is to act as an execution harness for implementing a recursive
   Modified Bipartite Graph allocation of counter resources for those
   platforms that don't have built-in smart counter allocation.
   It is intended to be #included in the cpu component source to minimize
   other disruption to the build process.
   
   This routine presumes the existence of a half dozen "bpt_" helper routines. 
   Prototypes for these routines are given below.
 
    success  return 1
    fail     return 0
*/

/* This function examines the event to determine
    if it can be mapped to counter ctr.
    Returns true if it can, false if it can't. */
static int
_bpt_map_avail( hwd_reg_alloc_t * dst, int ctr );

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.  */
static void
_bpt_map_set( hwd_reg_alloc_t * dst, int ctr );

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.  */
static int
_bpt_map_exclusive( hwd_reg_alloc_t * dst );

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.  */
static int
_bpt_map_shared( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src );

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
static void
_bpt_map_preempt( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src );

static void
_bpt_map_update( hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src );


static int
_papi_bipartite_alloc( hwd_reg_alloc_t * event_list, int count, int cidx )
{
	int i, j;
	char *ptr = ( char * ) event_list;
	int idx_q[count];				   /* queue of indexes of lowest rank events */
	int map_q[count];				   /* queue of mapped events (TRUE if mapped) */
	int head, tail;
	int size = _papi_hwd[cidx]->size.reg_alloc;

	/* build a queue of indexes to all events 
	   that live on one counter only (rank == 1) */
	head = 0;				 /* points to top of queue */
	tail = 0;				 /* points to bottom of queue */
	for ( i = 0; i < count; i++ ) {
		map_q[i] = 0;
		if ( _bpt_map_exclusive( ( hwd_reg_alloc_t * ) & ptr[size * i] ) )
			idx_q[tail++] = i;
	}
	/* scan the single counter queue looking for events that share counters.
	   If two events can live only on one counter, return failure.
	   If the second event lives on more than one counter, remove shared counter
	   from its selector and reduce its rank. 
	   Mark first event as mapped to its counter. */
	while ( head < tail ) {
		for ( i = 0; i < count; i++ ) {
			if ( i != idx_q[head] ) {
				if ( _bpt_map_shared( ( hwd_reg_alloc_t * ) & ptr[size * i],
									 ( hwd_reg_alloc_t * ) & ptr[size *
																 idx_q
																 [head]] ) ) {
					/* both share a counter; if second is exclusive, mapping fails */
					if ( _bpt_map_exclusive( ( hwd_reg_alloc_t * ) &
											ptr[size * i] ) )
						return 0;
					else {
						_bpt_map_preempt( ( hwd_reg_alloc_t * ) &
											 ptr[size * i],
											 ( hwd_reg_alloc_t * ) & ptr[size *
																		 idx_q
																		 [head]] );
						if ( _bpt_map_exclusive( ( hwd_reg_alloc_t * ) &
												ptr[size * i] ) )
							idx_q[tail++] = i;
					}
				}
			}
		}
		map_q[idx_q[head]] = 1;	/* mark this event as mapped */
		head++;
	}
	if ( tail == count ) {
		return 1;			 /* idx_q includes all events; everything is successfully mapped */
	} else {
		char *rest_event_list;
		char *copy_rest_event_list;
		int remainder;

		rest_event_list =
			papi_calloc(  _papi_hwd[cidx]->cmp_info.num_cntrs, 
				      size );

		copy_rest_event_list =
		        papi_calloc( _papi_hwd[cidx]->cmp_info.num_cntrs,
				     size );

		if ( !rest_event_list || !copy_rest_event_list ) {
			if ( rest_event_list )
				papi_free( rest_event_list );
			if ( copy_rest_event_list )
				papi_free( copy_rest_event_list );
			return ( 0 );
		}

		/* copy all unmapped events to a second list and make a backup */
		for ( i = 0, j = 0; i < count; i++ ) {
			if ( map_q[i] == 0 ) {
				memcpy( &copy_rest_event_list[size * j++], &ptr[size * i],
						( size_t ) size );
			}
		}
		remainder = j;

		memcpy( rest_event_list, copy_rest_event_list,
				( size_t ) size * ( size_t ) remainder );

		/* try each possible mapping until you fail or find one that works */
		for ( i = 0; i < _papi_hwd[cidx]->cmp_info.num_cntrs; i++ ) {
			/* for the first unmapped event, try every possible counter */
			if ( _bpt_map_avail( ( hwd_reg_alloc_t * ) rest_event_list, i ) ) {
				 _bpt_map_set( ( hwd_reg_alloc_t * ) rest_event_list, i );
				/* remove selected counter from all other unmapped events */
				for ( j = 1; j < remainder; j++ ) {
					if ( _bpt_map_shared( ( hwd_reg_alloc_t * ) &
										 rest_event_list[size * j],
										 ( hwd_reg_alloc_t * )
										 rest_event_list ) )
						_bpt_map_preempt( ( hwd_reg_alloc_t * ) &
											 rest_event_list[size * j],
											 ( hwd_reg_alloc_t * )
											 rest_event_list );
				}
				/* if recursive call to allocation works, break out of the loop */
				if ( _papi_bipartite_alloc
					 ( ( hwd_reg_alloc_t * ) rest_event_list, remainder,
					   cidx ) )
					break;

				/* recursive mapping failed; copy the backup list and try the next combination */
				memcpy( rest_event_list, copy_rest_event_list,
						( size_t ) size * ( size_t ) remainder );
			}
		}
		if ( i == _papi_hwd[cidx]->cmp_info.num_cntrs ) {
			papi_free( rest_event_list );
			papi_free( copy_rest_event_list );
			return 0;		 /* fail to find mapping */
		}
		for ( i = 0, j = 0; i < count; i++ ) {
			if ( map_q[i] == 0 )
				_bpt_map_update( ( hwd_reg_alloc_t * ) & ptr[size * i],
									( hwd_reg_alloc_t * ) & rest_event_list[size
																			*
																			j++] );
		}
		papi_free( rest_event_list );
		papi_free( copy_rest_event_list );
		return 1;
	}
}

