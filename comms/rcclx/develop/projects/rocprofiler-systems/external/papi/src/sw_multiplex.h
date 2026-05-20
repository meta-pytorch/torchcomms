#ifndef MULTIPLEX_H
#define MULTIPLEX_H

#define PAPI_MAX_SW_MPX_EVENTS 32

/* Structure contained in the EventSet structure that 
   holds information about multiplexing. */

typedef enum
  { MPX_STOPPED, MPX_RUNNING } MPX_status;

/** Structure contained in the EventSet structure that 
    holds information about multiplexing.
    @internal */

typedef struct _MPX_EventSet {
  MPX_status status;
  /** Pointer to this thread's structure */
  struct _threadlist *mythr;
  /** Pointers to this EventSet's MPX entries in the master list for this thread */
  struct _masterevent *(mev[PAPI_MAX_SW_MPX_EVENTS]);
  /** Number of entries in above list */
  int num_events;
  /** Not sure... */
  long long start_c, stop_c;
  long long start_values[PAPI_MAX_SW_MPX_EVENTS];
  long long stop_values[PAPI_MAX_SW_MPX_EVENTS];
  long long start_hc[PAPI_MAX_SW_MPX_EVENTS];
} MPX_EventSet;

typedef struct EventSetMultiplexInfo {
  MPX_EventSet *mpx_evset;
  int ns;
  int flags;
} EventSetMultiplexInfo_t;

int mpx_check( int EventSet );
int mpx_init( int );
int mpx_add_event( MPX_EventSet **, int EventCode, int domain,
		   int granularity );
int mpx_remove_event( MPX_EventSet **, int EventCode );
int MPX_add_events( MPX_EventSet ** mpx_events, int *event_list, int num_events,
		    int domain, int granularity );
int MPX_stop( MPX_EventSet * mpx_events, long long *values );
int MPX_cleanup( MPX_EventSet ** mpx_events );
void MPX_shutdown( void );
int MPX_reset( MPX_EventSet * mpx_events );
int MPX_read( MPX_EventSet * mpx_events, long long *values, int called_by_stop );
int MPX_start( MPX_EventSet * mpx_events );

#endif /* MULTIPLEX_H */
