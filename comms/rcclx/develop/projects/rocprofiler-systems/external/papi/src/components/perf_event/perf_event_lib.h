/* Various definitions */

/* This is arbitrary.  Typically you can add up to ~1000 before */
/* you run out of fds                                           */
#define PERF_EVENT_MAX_MPX_COUNTERS 384

/* We really don't need fancy definitions for these */

typedef struct
{
  int group_leader_fd;            /* fd of group leader                   */
  int event_fd;                   /* fd of event                          */
  int event_opened;               /* event successfully opened            */
  int profiling;                  /* event is profiling                   */
  int sampling;			  /* event is a sampling event            */
  uint32_t nr_mmap_pages;         /* number pages in the mmap buffer      */
  void *mmap_buf;                 /* used for control/profiling           */
  uint64_t tail;                  /* current read location in mmap buffer */
  uint64_t mask;                  /* mask used for wrapping the pages     */
  int cpu;                        /* cpu associated with this event       */
  struct perf_event_attr attr;    /* perf_event config structure          */
} pe_event_info_t;


typedef struct {
  int num_events;                 /* number of events in control state */
  unsigned int domain;            /* control-state wide domain         */
  unsigned int granularity;       /* granularity                       */
  unsigned int multiplexed;       /* multiplexing enable               */
  unsigned int overflow;          /* overflow enable                   */
  unsigned int inherit;           /* inherit enable                    */
  unsigned int overflow_signal;   /* overflow signal                   */
  unsigned int attached;          /* attached to a process             */
  int cidx;                       /* current component                 */
  int cpu;                        /* which cpu to measure              */
  pid_t tid;                      /* thread we are monitoring          */
  pe_event_info_t events[PERF_EVENT_MAX_MPX_COUNTERS];
  long long counts[PERF_EVENT_MAX_MPX_COUNTERS];
  unsigned int reset_flag;
  long long reset_counts[PERF_EVENT_MAX_MPX_COUNTERS];
} pe_control_t;


typedef struct {
  int initialized;                /* are we initialized?           */
  int state;                      /* are we opened and/or running? */
  int cidx;                       /* our component id              */
  struct native_event_table_t *event_table; /* our event table     */
} pe_context_t;


