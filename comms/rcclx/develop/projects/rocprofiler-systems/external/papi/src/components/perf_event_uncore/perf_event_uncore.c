/*
* File:    perf_event_uncore.c
*
* Author:  Vince Weaver
*          vincent.weaver@maine.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
*          Modified the perf_event_uncore component to use PFM_OS_PERF_EVENT_EXT mode in libpfm4.
*          This adds several new event masks, including cpu=, u=, and k= which give the user
*          the ability to set cpu number to use or control the domain (user, kernel, or both)
*          in which the counter should be incremented.  These are event masks so it is now
*          possible to have multiple events in the same event set that count activity from
*          differennt cpu's or count activity in different domains.
*/

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syscall.h>
#include <sys/utsname.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

/* PAPI-specific includes */
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "extras.h"

/* libpfm4 includes */
#include "papi_libpfm4_events.h"
#include "components/perf_event/pe_libpfm4_events.h"
#include "perfmon/pfmlib.h"
#include PEINCLUDE

/* Linux-specific includes */
#include "mb.h"
#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"
#include "linux-context.h"

#include "components/perf_event/perf_event_lib.h"

/* Forward declaration */
papi_vector_t _perf_event_uncore_vector;

/* Globals */
struct native_event_table_t uncore_native_event_table;
static int our_cidx;
//int
//_peu_libpfm4_get_cidx() {
//	return our_cidx;
//}

/* Defines for ctx->state */
#define PERF_EVENTS_OPENED  0x01
#define PERF_EVENTS_RUNNING 0x02

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

static int _peu_set_domain( hwd_control_state_t *ctl, int domain);
static int _peu_shutdown_component( void );



/******************************************************************/
/******** Kernel Version Dependent Routines  **********************/
/******************************************************************/

/* The read format on perf_event varies based on various flags that */
/* are passed into it.  This helper avoids copying this logic       */
/* multiple places.                                                 */
static unsigned int
get_read_format( unsigned int multiplex,
		 unsigned int inherit,
		 int format_group )
{
   unsigned int format = 0;

   /* if we need read format options for multiplexing, add them now */
   if (multiplex) {
      format |= PERF_FORMAT_TOTAL_TIME_ENABLED;
      format |= PERF_FORMAT_TOTAL_TIME_RUNNING;
   }

   /* If we are not using inherit, add the group read options     */
   if (!inherit) {
      if (format_group) {
	 format |= PERF_FORMAT_GROUP;
      }
   }

   SUBDBG("multiplex: %d, inherit: %d, group_leader: %d, format: %#x\n",
	  multiplex, inherit, format_group, format);

   return format;
}

/*****************************************************************/
/********* End Kernel-version Dependent Routines  ****************/
/*****************************************************************/

/********************************************************************/
/* Low-level perf_event calls                                       */
/********************************************************************/

/* In case headers aren't new enough to have __NR_perf_event_open */
#ifndef __NR_perf_event_open

#ifdef __powerpc__
#define __NR_perf_event_open	319
#elif defined(__x86_64__)
#define __NR_perf_event_open	298
#elif defined(__i386__)
#define __NR_perf_event_open	336
#elif defined(__arm__)          366+0x900000
#define __NR_perf_event_open
#endif

#endif

static long
sys_perf_event_open( struct perf_event_attr *hw_event, pid_t pid, int cpu,
					   int group_fd, unsigned long flags )
{
   int ret;

   SUBDBG("sys_perf_event_open(hw_event: %p, pid: %d, cpu: %d, group_fd: %d, flags: %lx\n",hw_event,pid,cpu,group_fd,flags);
   SUBDBG("   type: %d\n",hw_event->type);
   SUBDBG("   size: %d\n",hw_event->size);
   SUBDBG("   config: %#"PRIx64" (%"PRIu64")\n",hw_event->config,
	  hw_event->config);
   SUBDBG("   sample_period: %"PRIu64"\n",hw_event->sample_period);
   SUBDBG("   sample_type: %"PRIu64"\n",hw_event->sample_type);
   SUBDBG("   read_format: %"PRIu64"\n",hw_event->read_format);
   SUBDBG("   disabled: %d\n",hw_event->disabled);
   SUBDBG("   inherit: %d\n",hw_event->inherit);
   SUBDBG("   pinned: %d\n",hw_event->pinned);
   SUBDBG("   exclusive: %d\n",hw_event->exclusive);
   SUBDBG("   exclude_user: %d\n",hw_event->exclude_user);
   SUBDBG("   exclude_kernel: %d\n",hw_event->exclude_kernel);
   SUBDBG("   exclude_hv: %d\n",hw_event->exclude_hv);
   SUBDBG("   exclude_idle: %d\n",hw_event->exclude_idle);
   SUBDBG("   mmap: %d\n",hw_event->mmap);
   SUBDBG("   comm: %d\n",hw_event->comm);
   SUBDBG("   freq: %d\n",hw_event->freq);
   SUBDBG("   inherit_stat: %d\n",hw_event->inherit_stat);
   SUBDBG("   enable_on_exec: %d\n",hw_event->enable_on_exec);
   SUBDBG("   task: %d\n",hw_event->task);
   SUBDBG("   watermark: %d\n",hw_event->watermark);
   SUBDBG("   precise_ip: %d\n",hw_event->precise_ip);
   SUBDBG("   mmap_data: %d\n",hw_event->mmap_data);
   SUBDBG("   sample_id_all: %d\n",hw_event->sample_id_all);
   SUBDBG("   exclude_host: %d\n",hw_event->exclude_host);
   SUBDBG("   exclude_guest: %d\n",hw_event->exclude_guest);
   SUBDBG("   exclude_callchain_kernel: %d\n",hw_event->exclude_callchain_kernel);
   SUBDBG("   exclude_callchain_user: %d\n",hw_event->exclude_callchain_user);
   SUBDBG("   wakeup_watermark: %d\n",hw_event->wakeup_watermark);
   SUBDBG("   bp_type: %d\n",hw_event->bp_type);
   SUBDBG("   config1: %#lx (%lu)\n",hw_event->config1,hw_event->config1);
   SUBDBG("   config2: %#lx (%lu)\n",hw_event->config2,hw_event->config2);
   SUBDBG("   branch_sample_type: %lu\n",hw_event->branch_sample_type);
   SUBDBG("   sample_regs_user: %lu\n",hw_event->sample_regs_user);
   SUBDBG("   sample_stack_user: %d\n",hw_event->sample_stack_user);

	ret = syscall( __NR_perf_event_open, hw_event, pid, cpu, group_fd, flags );
	SUBDBG("Returned %d %d %s\n",ret,
	       ret<0?errno:0,
	       ret<0?strerror(errno):" ");
	return ret;
}


static int map_perf_event_errors_to_papi(int perf_event_error) {

   int ret;

   /* These mappings are approximate.
      EINVAL in particular can mean lots of different things */
   switch(perf_event_error) {
      case EPERM:
      case EACCES:
           ret = PAPI_EPERM;
	   break;
      case ENODEV:
      case EOPNOTSUPP:
	   ret = PAPI_ENOSUPP;
           break;
      case ENOENT:
	   ret = PAPI_ENOEVNT;
           break;
      case ENOSYS:
      case EAGAIN:
      case EBUSY:
      case E2BIG:
	   ret = PAPI_ESYS;
	   break;
      case ENOMEM:
	   ret = PAPI_ENOMEM;
	   break;
      case EINVAL:
      default:
	   ret = PAPI_EINVAL;
           break;
   }
   return ret;
}

/* Maximum size we ever expect to read from a perf_event fd   */
/*  (this is the number of 64-bit values)                     */
/* We use this to size the read buffers                       */
/* The three is for event count, time_enabled, time_running   */
/*  and the counter term is count value and count id for each */
/*  possible counter value.                                   */
#define READ_BUFFER_SIZE (3 + (2 * PERF_EVENT_MAX_MPX_COUNTERS))



/* KERNEL_CHECKS_SCHEDUABILITY_UPON_OPEN is a work-around for kernel arch */
/* implementations (e.g. x86 before 2.6.33) which don't do a static event */
/* scheduability check in sys_perf_event_open.  It is also needed if the  */
/* kernel is stealing an event, such as when NMI watchdog is enabled.     */

static int
check_scheduability( pe_context_t *ctx, pe_control_t *ctl)
{
   SUBDBG("ENTER: ctx: %p, ctl: %p\n", ctx, ctl);
   int retval = 0, cnt = -1;
   ( void ) ctx;	     /*unused */
   long long papi_pe_buffer[READ_BUFFER_SIZE];
   int i;

   /* If the kernel isn't tracking scheduability right       */
   /* Then we need to start/stop/read to force the event     */
   /* to be scheduled and see if an error condition happens. */

   /* start all events */
   for( i = 0; i < ctl->num_events; i++) {
      retval = ioctl( ctl->events[i].event_fd, PERF_EVENT_IOC_ENABLE, NULL );
      if (retval == -1) {
	 SUBDBG("EXIT: Enable failed event index: %d, num_events: %d, return PAPI_ESYS\n", i, ctl->num_events);
	 return PAPI_ESYS;
      }
   }

   /* stop all events */
   for( i = 0; i < ctl->num_events; i++) {
      retval = ioctl(ctl->events[i].event_fd, PERF_EVENT_IOC_DISABLE, NULL );
      if (retval == -1) {
	 SUBDBG("EXIT: Disable failed: event index: %d, num_events: %d, return PAPI_ESYS\n", i, ctl->num_events);
	 return PAPI_ESYS;
      }
   }

   /* See if a read of each event returns results */
   for( i = 0; i < ctl->num_events; i++) {
      cnt = read( ctl->events[i].event_fd, papi_pe_buffer, sizeof(papi_pe_buffer));
      if ( cnt == -1 ) {
	 SUBDBG( "EXIT: read failed: event index: %d, num_events: %d, return PAPI_ESYS.  Should never happen.\n", i, ctl->num_events);
	 return PAPI_ESYS;
      }

      if ( cnt == 0 ) {
	 /* We read 0 bytes if we could not schedule the event */
	 /* The kernel should have detected this at open       */
	 /* but various bugs (including NMI watchdog)          */
	 /* result in this behavior                            */

	 SUBDBG( "EXIT: read returned 0: event index: %d, num_events: %d, return PAPI_ECNFLCT.\n", i, ctl->num_events);
	 return PAPI_ECNFLCT;
      }
   }

   /* Reset all of the counters (opened so far) back to zero      */
   /* from the above brief enable/disable call pair.              */

   /* We have to reset all events because reset of group leader      */
   /* does not reset all.                                            */
   /* we assume that the events are being added one by one and that  */
   /* we do not need to reset higher events (doing so may reset ones */
   /* that have not been initialized yet.                            */

   /* Note... PERF_EVENT_IOC_RESET does not reset time running       */
   /* info if multiplexing, so we should avoid coming here if        */
   /* we are multiplexing the event.                                 */
   for( i = 0; i < ctl->num_events; i++) {
      retval=ioctl( ctl->events[i].event_fd, PERF_EVENT_IOC_RESET, NULL );
      if (retval == -1) {
	 SUBDBG("EXIT: Reset failed: event index: %d, num_events: %d, return PAPI_ESYS\n", i, ctl->num_events);
	 return PAPI_ESYS;
      }
   }
   SUBDBG("EXIT: return PAPI_OK\n");
   return PAPI_OK;
}


/* Open all events in the control state */
static int
open_pe_events( pe_context_t *ctx, pe_control_t *ctl )
{

   int i, ret = PAPI_OK;
   long pid;

   if (ctl->granularity==PAPI_GRN_SYS) {
      pid = -1;
   }
   else {
      pid = ctl->tid;
   }

   for( i = 0; i < ctl->num_events; i++ ) {

      ctl->events[i].event_opened=0;

      /* set up the attr structure.  We don't set up all fields here */
      /* as some have already been set up previously.                */

/*
 * The following code controls how the uncore component interfaces with the 
 * kernel for uncore events.  The code inside the ifdef will use grouping of 
 * uncore events which can make the cost of reading the results more efficient.
 * The problem with it is that the uncore component supports 20 different uncore 
 * PMU's.  The kernel requires that all events in a group must be for the same PMU.
 * This means that with grouping enabled papi applications can count events on only
 * one of the 20 PMU's during a run.
 * 
 * The code inside the else clause treats each event in the event set as 
 * independent.  When running in this mode the kernel allows the papi multiple 
 * uncore PMU's at the same time.
 * 
 * Example:
 *  An application wants to measure all the L3 cache write requests.
 *  The event to do this is part of a cbox pmu (there are 8 cbox pmu's).
 *  When built with the code in the ifdef, the application would have to be 
 *    run 8 times and count write requests from one pmu at a time.
 *  When built with the code in the else, the write requests in all 8 cbox 
 *    pmu's could be counted in the same run.
 * 
 */
// #define GROUPIT 1       // remove the comment on this line to force event grouping
#ifdef GROUPIT
      /* group leader (event 0) is special                */
      /* If we're multiplexed, everyone is a group leader */
      if (( i == 0 ) || (ctl->multiplexed)) {
         ctl->events[i].attr.pinned = !ctl->multiplexed;
	 ctl->events[i].attr.disabled = 1;
	 ctl->events[i].group_leader_fd=-1;
         ctl->events[i].attr.read_format = get_read_format(ctl->multiplexed,
							   ctl->inherit,
							   !ctl->multiplexed );
      } else {
	 ctl->events[i].attr.pinned=0;
	 ctl->events[i].attr.disabled = 0;
	 ctl->events[i].group_leader_fd=ctl->events[0].event_fd,
         ctl->events[i].attr.read_format = get_read_format(ctl->multiplexed,
							   ctl->inherit,
							   0 );
      }
#else
             ctl->events[i].attr.pinned = !ctl->multiplexed;
         	 ctl->events[i].attr.disabled = 1;
         	 ctl->inherit = 1;
         	 ctl->events[i].group_leader_fd=-1;
             ctl->events[i].attr.read_format = get_read_format(ctl->multiplexed, ctl->inherit, 0 );
#endif


      /* try to open */
      ctl->events[i].event_fd = sys_perf_event_open( &ctl->events[i].attr,
						     pid,
						     ctl->events[i].cpu,
			       ctl->events[i].group_leader_fd,
						     0 /* flags */
						     );

      /* Try to match Linux errors to PAPI errors */
      if ( ctl->events[i].event_fd == -1 ) {
	 SUBDBG("sys_perf_event_open returned error on event #%d."
		"  Error: %s\n",
		i, strerror( errno ) );
         ret=map_perf_event_errors_to_papi(errno);

    goto open_peu_cleanup;
      }

      SUBDBG ("sys_perf_event_open: tid: %ld, cpu_num: %d,"
              " group_leader/fd: %d, event_fd: %d,"
              " read_format: %"PRIu64"\n",
	      pid, ctl->events[i].cpu, ctl->events[i].group_leader_fd,
	      ctl->events[i].event_fd, ctl->events[i].attr.read_format);

      ctl->events[i].event_opened=1;
   }


   /* in many situations the kernel will indicate we opened fine */
   /* yet things will fail later.  So we need to double check    */
   /* we actually can use the events we've set up.               */

   /* This is not necessary if we are multiplexing, and in fact */
   /* we cannot do this properly if multiplexed because         */
   /* PERF_EVENT_IOC_RESET does not reset the time running info */
   if (!ctl->multiplexed) {
	ret = check_scheduability( ctx, ctl);

	if ( ret != PAPI_OK ) {
	    /* the last event did open, so we need to bump the counter */
	    /* before doing the cleanup                                */
	    i++;
	    goto open_peu_cleanup;
	}
   }

   /* Now that we've successfully opened all of the events, do whatever  */
   /* "tune-up" is needed to attach the mmap'd buffers, signal handlers, */
   /* and so on.                                                         */
   for ( i = 0; i < ctl->num_events; i++ ) {

      /* No sampling if uncore */
      ctl->events[i].mmap_buf = NULL;
   }

   /* Set num_evts only if completely successful */
   ctx->state |= PERF_EVENTS_OPENED;

   return PAPI_OK;

open_peu_cleanup:
   /* We encountered an error, close up the fds we successfully opened.  */
   /* We go backward in an attempt to close group leaders last, although */
   /* That's probably not strictly necessary.                            */
   while ( i > 0 ) {
      i--;
      if (ctl->events[i].event_fd>=0) {
	 close( ctl->events[i].event_fd );
	 ctl->events[i].event_opened=0;
      }
   }

   return ret;
}

/* Close all of the opened events */
static int
close_pe_events( pe_context_t *ctx, pe_control_t *ctl )
{
   int i;
   int num_closed=0;
   int events_not_opened=0;

   /* should this be a more serious error? */
   if ( ctx->state & PERF_EVENTS_RUNNING ) {
      SUBDBG("Closing without stopping first\n");
   }

   /* Close child events first */
   for( i=0; i<ctl->num_events; i++ ) {

      if (ctl->events[i].event_opened) {

         if (ctl->events[i].group_leader_fd!=-1) {
            if ( ctl->events[i].mmap_buf ) {
	       if ( munmap ( ctl->events[i].mmap_buf,
		             ctl->events[i].nr_mmap_pages * getpagesize() ) ) {
	          PAPIERROR( "munmap of fd = %d returned error: %s",
			     ctl->events[i].event_fd, strerror( errno ) );
	          return PAPI_ESYS;
	       }
	    }

            if ( close( ctl->events[i].event_fd ) ) {
	       PAPIERROR( "close of fd = %d returned error: %s",
		       ctl->events[i].event_fd, strerror( errno ) );
	       return PAPI_ESYS;
	    } else {
	       num_closed++;
	    }
	    ctl->events[i].event_opened=0;
	 }
      }
      else {
	events_not_opened++;
      }
   }

   /* Close the group leaders last */
   for( i=0; i<ctl->num_events; i++ ) {

      if (ctl->events[i].event_opened) {

         if (ctl->events[i].group_leader_fd==-1) {
            if ( ctl->events[i].mmap_buf ) {
	       if ( munmap ( ctl->events[i].mmap_buf,
		             ctl->events[i].nr_mmap_pages * getpagesize() ) ) {
	          PAPIERROR( "munmap of fd = %d returned error: %s",
			     ctl->events[i].event_fd, strerror( errno ) );
	          return PAPI_ESYS;
	       }
	    }


            if ( close( ctl->events[i].event_fd ) ) {
	       PAPIERROR( "close of fd = %d returned error: %s",
		       ctl->events[i].event_fd, strerror( errno ) );
	       return PAPI_ESYS;
	    } else {
	       num_closed++;
	    }
	    ctl->events[i].event_opened=0;
	 }
      }
   }


   if (ctl->num_events!=num_closed) {
      if (ctl->num_events!=(num_closed+events_not_opened)) {
         PAPIERROR("Didn't close all events: "
		   "Closed %d Not Opened: %d Expected %d\n",
		   num_closed,events_not_opened,ctl->num_events);
         return PAPI_EBUG;
      }
   }

   ctl->num_events=0;

   ctx->state &= ~PERF_EVENTS_OPENED;

   return PAPI_OK;
}




/********************************************************************/
/* Component Interface                                              */
/********************************************************************/



/* Initialize a thread */
static int
_peu_init_thread( hwd_context_t *hwd_ctx )
{

  pe_context_t *pe_ctx = ( pe_context_t *) hwd_ctx;

  /* clear the context structure and mark as initialized */
  memset( pe_ctx, 0, sizeof ( pe_context_t ) );
  pe_ctx->initialized=1;

  pe_ctx->event_table=&uncore_native_event_table;
  pe_ctx->cidx=our_cidx;

  return PAPI_OK;
}

/* Initialize a new control state */
static int
_peu_init_control_state( hwd_control_state_t *ctl )
{
  pe_control_t *pe_ctl = ( pe_control_t *) ctl;

  /* clear the contents */
  memset( pe_ctl, 0, sizeof ( pe_control_t ) );

  /* Set the default domain */
  _peu_set_domain( ctl, _perf_event_uncore_vector.cmp_info.default_domain );

  /* Set the default granularity */
  pe_ctl->granularity=_perf_event_uncore_vector.cmp_info.default_granularity;

  pe_ctl->cidx=our_cidx;

  /* Set cpu number in the control block to show events */
  /* are not tied to specific cpu                       */
  pe_ctl->cpu = -1;
  return PAPI_OK;
}



/* Initialize the perf_event uncore component */
static int
_peu_init_component( int cidx )
{

   int retval;
   int paranoid_level;

   FILE *fff;
   char *strCpy;

   our_cidx=cidx;

   /* The is the official way to detect if perf_event support exists */
   /* The file is called perf_counter_paranoid on 2.6.31             */
   /* currently we are lazy and do not support 2.6.31 kernels        */

   fff=fopen("/proc/sys/kernel/perf_event_paranoid","r");
   if (fff==NULL) {
     strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "perf_event support not detected",PAPI_MAX_STR_LEN);
     if (strCpy == NULL) HANDLE_STRING_ERROR;
     retval = PAPI_ECMP;
     goto fn_fail;
   }
   retval=fscanf(fff,"%d",&paranoid_level);
   if (retval!=1) fprintf(stderr,"Error reading paranoid level\n");
   fclose(fff);


   /* Run the libpfm4-specific setup */

   retval = _papi_libpfm4_init(_papi_hwd[cidx]);
   if (retval) {
     strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	     "Error initializing libpfm4",PAPI_MAX_STR_LEN);
     _peu_shutdown_component( );
     if (strCpy == NULL) HANDLE_STRING_ERROR;
     retval = PAPI_ECMP;
     goto fn_fail;
   }


   /* Run the uncore specific libpfm4 setup */

   retval = _peu_libpfm4_init(_papi_hwd[cidx], our_cidx,
			       &uncore_native_event_table,
                               PMU_TYPE_UNCORE);
   if (retval) {
     strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	     "Error setting up libpfm4",PAPI_MAX_STR_LEN);
     _peu_shutdown_component( );
     if (strCpy == NULL) HANDLE_STRING_ERROR;
     retval = PAPI_ECMP;
     goto fn_fail;
   }

   /* Check if no uncore events found */

   if (_papi_hwd[cidx]->cmp_info.num_native_events==0) {
     strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	     "No uncore PMUs or events found",PAPI_MAX_STR_LEN);
     _peu_shutdown_component( );
     if (strCpy == NULL) HANDLE_STRING_ERROR;
     retval = PAPI_ECMP;
     goto fn_fail;
   }

   /* Check if we have enough permissions for uncore */

   /* 2 means no kernel measurements allowed   */
   /* 1 means normal counter access            */
   /* 0 means you can access CPU-specific data */
   /* -1 means no restrictions                 */

   if ((paranoid_level>0) && (getuid()!=0)) {
      strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
	    "Insufficient permissions for uncore access.  Set /proc/sys/kernel/perf_event_paranoid to 0 or run as root.",
	    PAPI_MAX_STR_LEN);
      _peu_shutdown_component( );
     if (strCpy == NULL) HANDLE_STRING_ERROR;
     retval = PAPI_ECMP;
     goto fn_fail;
   }

  fn_exit:
   _papi_hwd[cidx]->cmp_info.disabled = retval;
   return retval;
  fn_fail:
   goto fn_exit;

}

/* Shutdown the perf_event component */
static int
_peu_shutdown_component( void ) {

	/* deallocate our event table */
	_pe_libpfm4_shutdown(&_perf_event_uncore_vector,
				&uncore_native_event_table);

	/* Shutdown libpfm4 */
	_papi_libpfm4_shutdown(&_perf_event_uncore_vector);

	return PAPI_OK;
}

/* This function clears the current contents of the control structure and
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

int
_peu_update_control_state( hwd_control_state_t *ctl,
			       NativeInfo_t *native,
			       int count, hwd_context_t *ctx )
{
	int i;
	int j;
	int ret;
	int skipped_events=0;
	struct native_event_t *ntv_evt;
   pe_context_t *pe_ctx = ( pe_context_t *) ctx;
   pe_control_t *pe_ctl = ( pe_control_t *) ctl;

   /* close all of the existing fds and start over again */
   /* In theory we could have finer-grained control and know if             */
   /* things were changed, but it's easier to tear things down and rebuild. */
   close_pe_events( pe_ctx, pe_ctl );

   /* Calling with count==0 should be OK, it's how things are deallocated */
   /* when an eventset is destroyed.                                      */
   if ( count == 0 ) {
      SUBDBG( "Called with count == 0\n" );
      return PAPI_OK;
   }

   /* set up all the events */
   for( i = 0; i < count; i++ ) {
      if ( native ) {
			// get the native event pointer used for this papi event
			int ntv_idx = _papi_hwi_get_ntv_idx((unsigned)(native[i].ni_papi_code));
			if (ntv_idx < -1) {
				SUBDBG("papi_event_code: %#x known by papi but not by the component\n", native[i].ni_papi_code);
				continue;
			}
			// if native index is -1, then we have an event without a mask and need to find the right native index to use
			if (ntv_idx == -1) {
				// find the native event index we want by matching for the right papi event code
				for (j=0 ; j<pe_ctx->event_table->num_native_events ; j++) {
					if (pe_ctx->event_table->native_events[j].papi_event_code == native[i].ni_papi_code) {
						ntv_idx = j;
					}
				}
			}

			// if native index is still negative, we did not find event we wanted so just return error
			if (ntv_idx < 0) {
				SUBDBG("papi_event_code: %#x not found in native event tables\n", native[i].ni_papi_code);
				continue;
			}

			// this native index is positive so there was a mask with the event, the ntv_idx identifies which native event to use
			ntv_evt = (struct native_event_t *)(&(pe_ctx->event_table->native_events[ntv_idx]));

			SUBDBG("ntv_evt: %p\n", ntv_evt);

			SUBDBG("i: %d, pe_ctx->event_table->num_native_events: %d\n", i, pe_ctx->event_table->num_native_events);

	    	// Move this events hardware config values and other attributes to the perf_events attribute structure
			memcpy (&pe_ctl->events[i].attr, &ntv_evt->attr, sizeof(perf_event_attr_t));

			// may need to update the attribute structure with information from event set level domain settings (values set by PAPI_set_domain)
			// only done if the event mask which controls each counting domain was not provided

			// get pointer to allocated name, will be NULL when adding preset events to event set
			char *aName = ntv_evt->allocated_name;
			if ((aName == NULL)  ||  (strstr(aName, ":u=") == NULL)) {
				SUBDBG("set exclude_user attribute from eventset level domain flags, encode: %d, eventset: %d\n", pe_ctl->events[i].attr.exclude_user, !(pe_ctl->domain & PAPI_DOM_USER));
				pe_ctl->events[i].attr.exclude_user = !(pe_ctl->domain & PAPI_DOM_USER);
			}
			if ((aName == NULL)  ||  (strstr(aName, ":k=") == NULL)) {
				SUBDBG("set exclude_kernel attribute from eventset level domain flags, encode: %d, eventset: %d\n", pe_ctl->events[i].attr.exclude_kernel, !(pe_ctl->domain & PAPI_DOM_KERNEL));
				pe_ctl->events[i].attr.exclude_kernel = !(pe_ctl->domain & PAPI_DOM_KERNEL);
			}

			// set the cpu number provided with an event mask if there was one (will be -1 if mask not provided)
			pe_ctl->events[i].cpu = ntv_evt->cpu;
			// if cpu event mask not provided, then set the cpu to use to what may have been set on call to PAPI_set_opt (will still be -1 if not called)
			if (pe_ctl->events[i].cpu == -1) {
				pe_ctl->events[i].cpu = pe_ctl->cpu;
			}
      } else {
    	  // This case happens when called from _pe_set_overflow and _pe_ctl
          // Those callers put things directly into the pe_ctl structure so it is already set for the open call
      }

      // Copy the inherit flag into the attribute block that will be passed to the kernel
      pe_ctl->events[i].attr.inherit = pe_ctl->inherit;

      /* Set the position in the native structure */
      /* We just set up events linearly           */
      if ( native ) {
    	  native[i].ni_position = i;
    	  SUBDBG( "&native[%d]: %p, ni_papi_code: %#x, ni_event: %#x, ni_position: %d, ni_owners: %d\n",
			i, &(native[i]), native[i].ni_papi_code, native[i].ni_event, native[i].ni_position, native[i].ni_owners);
      }
   }

	if (count <= skipped_events) {
		SUBDBG("EXIT: No events to count, they all contained invalid umasks\n");
		return PAPI_ENOEVNT;
	}

  pe_ctl->num_events = count - skipped_events;

   /* actuall open the events */
   /* (why is this a separate function?) */
   ret = open_pe_events( pe_ctx, pe_ctl );
   if ( ret != PAPI_OK ) {
      SUBDBG("open_pe_events failed\n");
      /* Restore values ? */
      return ret;
   }

   SUBDBG( "EXIT: PAPI_OK\n" );
   return PAPI_OK;
}

/********************************************************************/
/********************************************************************/
/* Start with functions that are exported via the module interface  */
/********************************************************************/
/********************************************************************/


/* set the domain. perf_events allows per-event control of this, papi allows it to be set at the event level or at the event set level. */
/* this will set the event set level domain values but they only get used if no event level domain mask (u= or k=) was specified. */
static int
_peu_set_domain( hwd_control_state_t *ctl, int domain)
{
   pe_control_t *pe_ctl = ( pe_control_t *) ctl;

   SUBDBG("old control domain %d, new domain %d\n",
	  pe_ctl->domain,domain);

   pe_ctl->domain = domain;
   return PAPI_OK;
}

/* Shutdown a thread */
static int
_peu_shutdown_thread( hwd_context_t *ctx )
{
    pe_context_t *pe_ctx = ( pe_context_t *) ctx;

    pe_ctx->initialized=0;

    return PAPI_OK;
}


/* reset the hardware counters */
/* Note: PAPI_reset() does not necessarily call this */
/* unless the events are actually running.           */
static int
_peu_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
   int i, ret;
   pe_control_t *pe_ctl = ( pe_control_t *) ctl;

   ( void ) ctx;			 /*unused */

   /* We need to reset all of the events, not just the group leaders */
   for( i = 0; i < pe_ctl->num_events; i++ ) {
      ret = ioctl( pe_ctl->events[i].event_fd, PERF_EVENT_IOC_RESET, NULL );
      if ( ret == -1 ) {
	 PAPIERROR("ioctl(%d, PERF_EVENT_IOC_RESET, NULL) "
		   "returned error, Linux says: %s",
		   pe_ctl->events[i].event_fd, strerror( errno ) );
	 return PAPI_ESYS;
      }
   }

   return PAPI_OK;
}


/* write (set) the hardware counters */
/* Current we do not support this.   */
static int
_peu_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
		long long *from )
{
   ( void ) ctx;			 /*unused */
   ( void ) ctl;			 /*unused */
   ( void ) from;			 /*unused */
   /*
    * Counters cannot be written.  Do we need to virtualize the
    * counters so that they can be written, or perhaps modify code so that
    * they can be written? FIXME ?
    */

    return PAPI_ENOSUPP;
}

/*
 * perf_event provides a complicated read interface.
 *  the info returned by read() varies depending on whether
 *  you have PERF_FORMAT_GROUP, PERF_FORMAT_TOTAL_TIME_ENABLED,
 *  PERF_FORMAT_TOTAL_TIME_RUNNING, or PERF_FORMAT_ID set
 *
 * To simplify things we just always ask for everything.  This might
 * lead to overhead when reading more than we need, but it makes the
 * read code a lot simpler than the original implementation we had here.
 *
 * For more info on the layout see include/linux/perf_event.h
 *
 */

static int
_peu_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
	       long long **events, int flags )
{
    SUBDBG("ENTER: ctx: %p, ctl: %p, events: %p, flags: %#x\n", ctx, ctl, events, flags);

   ( void ) flags;			 /*unused */
   int i, ret = -1;
   /* pe_context_t *pe_ctx = ( pe_context_t *) ctx; */ 
   (void) ctx; /*unused*/
   pe_control_t *pe_ctl = ( pe_control_t *) ctl;
   long long papi_pe_buffer[READ_BUFFER_SIZE];
   long long tot_time_running, tot_time_enabled, scale;

   /* Handle case where we are multiplexing */
   if (pe_ctl->multiplexed) {

      /* currently we handle multiplexing by having individual events */
      /* so we read from each in turn.                                */

      for ( i = 0; i < pe_ctl->num_events; i++ ) {

         ret = read( pe_ctl->events[i].event_fd, papi_pe_buffer,
		    sizeof ( papi_pe_buffer ) );
         if ( ret == -1 ) {
	    PAPIERROR("read returned an error: %s", strerror( errno ));
       SUBDBG("EXIT: PAPI_ESYS\n");
	    return PAPI_ESYS;
	 }

	 /* We should read 3 64-bit values from the counter */
	 if (ret<(signed)(3*sizeof(long long))) {
	    PAPIERROR("Error!  short read!\n");
       SUBDBG("EXIT: PAPI_ESYS\n");
	    return PAPI_ESYS;
	 }

         SUBDBG("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n",
	        pe_ctl->events[i].event_fd,
		(long)pe_ctl->tid, pe_ctl->events[i].cpu, ret);
         SUBDBG("read: %lld %lld %lld\n",papi_pe_buffer[0],
	        papi_pe_buffer[1],papi_pe_buffer[2]);

         tot_time_enabled = papi_pe_buffer[1];
         tot_time_running = papi_pe_buffer[2];

         SUBDBG("count[%d] = (papi_pe_buffer[%d] %lld * "
		"tot_time_enabled %lld) / tot_time_running %lld\n",
		i, 0,papi_pe_buffer[0],
		tot_time_enabled,tot_time_running);

         if (tot_time_running == tot_time_enabled) {
	    /* No scaling needed */
	    pe_ctl->counts[i] = papi_pe_buffer[0];
         } else if (tot_time_running && tot_time_enabled) {
	    /* Scale factor of 100 to avoid overflows when computing */
	    /*enabled/running */

	    scale = (tot_time_enabled * 100LL) / tot_time_running;
	    scale = scale * papi_pe_buffer[0];
	    scale = scale / 100LL;
	    pe_ctl->counts[i] = scale;
	 } else {
	   /* This should not happen, but Phil reports it sometime does. */
	    SUBDBG("perf_event kernel bug(?) count, enabled, "
		   "running: %lld, %lld, %lld\n",
		   papi_pe_buffer[0],tot_time_enabled,
		   tot_time_running);

	    pe_ctl->counts[i] = papi_pe_buffer[0];
	 }
      }
   }

   /* Handle cases where we cannot use FORMAT GROUP */
   else if (pe_ctl->inherit) {

      /* we must read each counter individually */
      for ( i = 0; i < pe_ctl->num_events; i++ ) {

         ret = read( pe_ctl->events[i].event_fd, papi_pe_buffer, 
		    sizeof ( papi_pe_buffer ) );
         if ( ret == -1 ) {
	    PAPIERROR("read returned an error: %s", strerror( errno ));
       SUBDBG("EXIT: PAPI_ESYS\n");
	    return PAPI_ESYS;
	 }

	 /* we should read one 64-bit value from each counter */
	 if (ret!=sizeof(long long)) {
	    PAPIERROR("Error!  short read!\n");
	    PAPIERROR("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n",
		   pe_ctl->events[i].event_fd,
		   (long)pe_ctl->tid, pe_ctl->events[i].cpu, ret);
       SUBDBG("EXIT: PAPI_ESYS\n");
	    return PAPI_ESYS;
	 }

         SUBDBG("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n",
	        pe_ctl->events[i].event_fd, (long)pe_ctl->tid,
		pe_ctl->events[i].cpu, ret);
         SUBDBG("read: %lld\n",papi_pe_buffer[0]);

	 pe_ctl->counts[i] = papi_pe_buffer[0];
      }
   }


   /* Handle cases where we are using FORMAT_GROUP   */
   /* We assume only one group leader, in position 0 */

   else {
      if (pe_ctl->events[0].group_leader_fd!=-1) {
	 PAPIERROR("Was expecting group leader!\n");
      }

      ret = read( pe_ctl->events[0].event_fd, papi_pe_buffer,
		  sizeof ( papi_pe_buffer ) );

      if ( ret == -1 ) {
	 PAPIERROR("read returned an error: %s", strerror( errno ));
       SUBDBG("EXIT: PAPI_ESYS\n");
	 return PAPI_ESYS;
      }

      /* we read 1 64-bit value (number of events) then     */
      /* num_events more 64-bit values that hold the counts */
      if (ret<(signed)((1+pe_ctl->num_events)*sizeof(long long))) {
	 PAPIERROR("Error! short read!\n");
       SUBDBG("EXIT: PAPI_ESYS\n");
	 return PAPI_ESYS;
      }

      SUBDBG("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n",
	     pe_ctl->events[0].event_fd,
	     (long)pe_ctl->tid, pe_ctl->events[0].cpu, ret);
      {
	 int j;
	 for(j=0;j<ret/8;j++) {
            SUBDBG("read %d: %lld\n",j,papi_pe_buffer[j]);
	 }
      }

      /* Make sure the kernel agrees with how many events we have */
      if (papi_pe_buffer[0]!=pe_ctl->num_events) {
	 PAPIERROR("Error!  Wrong number of events!\n");
       SUBDBG("EXIT: PAPI_ESYS\n");
	 return PAPI_ESYS;
      }

      /* put the count values in their proper location */
      for(i=0;i<pe_ctl->num_events;i++) {
         pe_ctl->counts[i] = papi_pe_buffer[1+i];
      }
   }

   /* point PAPI to the values we read */
   *events = pe_ctl->counts;

   SUBDBG("EXIT: PAPI_OK\n");
   return PAPI_OK;
}

/* Start counting events */
static int
_peu_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
   int ret;
   int i;
   int did_something = 0;
   pe_context_t *pe_ctx = ( pe_context_t *) ctx;
   pe_control_t *pe_ctl = ( pe_control_t *) ctl;

   /* Reset the counters first.  Is this necessary? */
   ret = _peu_reset( pe_ctx, pe_ctl );
   if ( ret ) {
      return ret;
   }

   /* Enable all of the group leaders                */
   /* All group leaders have a group_leader_fd of -1 */
   for( i = 0; i < pe_ctl->num_events; i++ ) {
      if (pe_ctl->events[i].group_leader_fd == -1) {
	 SUBDBG("ioctl(enable): fd: %d\n", pe_ctl->events[i].event_fd);
	 ret=ioctl( pe_ctl->events[i].event_fd, PERF_EVENT_IOC_ENABLE, NULL) ; 

	 /* ioctls always return -1 on failure */
         if (ret == -1) {
            PAPIERROR("ioctl(PERF_EVENT_IOC_ENABLE) failed.\n");
            return PAPI_ESYS;
	 }

	 did_something++;
      } 
   }

   if (!did_something) {
      PAPIERROR("Did not enable any counters.\n");
      return PAPI_EBUG;
   }

   pe_ctx->state |= PERF_EVENTS_RUNNING;

   return PAPI_OK;

}

/* Stop all of the counters */
static int
_peu_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

   int ret;
   int i;
   pe_context_t *pe_ctx = ( pe_context_t *) ctx;
   pe_control_t *pe_ctl = ( pe_control_t *) ctl;

   /* Just disable the group leaders */
   for ( i = 0; i < pe_ctl->num_events; i++ ) {
      if ( pe_ctl->events[i].group_leader_fd == -1 ) {
	 ret=ioctl( pe_ctl->events[i].event_fd, PERF_EVENT_IOC_DISABLE, NULL);
	 if ( ret == -1 ) {
	    PAPIERROR( "ioctl(%d, PERF_EVENT_IOC_DISABLE, NULL) "
		       "returned error, Linux says: %s",
		       pe_ctl->events[i].event_fd, strerror( errno ) );
	    return PAPI_EBUG;
	 }
      }
   }

   pe_ctx->state &= ~PERF_EVENTS_RUNNING;

   return PAPI_OK;
}

/* Set various options on a control state */
static int
_peu_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
   int ret;
   pe_context_t *pe_ctx = ( pe_context_t *) ctx;
   pe_control_t *pe_ctl = NULL;

   switch ( code ) {
      case PAPI_MULTIPLEX:
	   pe_ctl = ( pe_control_t * ) ( option->multiplex.ESI->ctl_state );

	   pe_ctl->multiplexed = 1;
	   ret = _peu_update_control_state( pe_ctl, NULL,
						pe_ctl->num_events, pe_ctx );
	   if (ret != PAPI_OK) {
	      pe_ctl->multiplexed = 0;
	   }
	   return ret;

      case PAPI_ATTACH:
	   pe_ctl = ( pe_control_t * ) ( option->attach.ESI->ctl_state );

	   pe_ctl->tid = option->attach.tid;

	   /* If events have been already been added, something may */
	   /* have been done to the kernel, so update */
	   ret =_peu_update_control_state( pe_ctl, NULL,
						pe_ctl->num_events, pe_ctx);

	   return ret;

      case PAPI_DETACH:
	   pe_ctl = ( pe_control_t *) ( option->attach.ESI->ctl_state );

	   pe_ctl->tid = 0;
	   return PAPI_OK;

      case PAPI_CPU_ATTACH:
	   pe_ctl = ( pe_control_t *) ( option->cpu.ESI->ctl_state );

	   /* this tells the kernel not to count for a thread   */
	   /* should we warn if we try to set both?  perf_event */
	   /* will reject it.                                   */
	   pe_ctl->tid = -1;

	   pe_ctl->cpu = option->cpu.cpu_num;

	   return PAPI_OK;

      case PAPI_DOMAIN:
	   pe_ctl = ( pe_control_t *) ( option->domain.ESI->ctl_state );

	   /* looks like we are allowed, so set event set level counting domains */
       pe_ctl->domain = option->domain.domain;
	   return PAPI_OK;

      case PAPI_GRANUL:
	   pe_ctl = (pe_control_t *) ( option->granularity.ESI->ctl_state );

	   /* FIXME: we really don't support this yet */

           switch ( option->granularity.granularity  ) {
              case PAPI_GRN_PROCG:
              case PAPI_GRN_SYS_CPU:
              case PAPI_GRN_PROC:
		   return PAPI_ECMP;

	      /* Currently we only support thread and CPU granularity */
              case PAPI_GRN_SYS:
	 	   pe_ctl->granularity=PAPI_GRN_SYS;
		   break;

              case PAPI_GRN_THR:
	 	   pe_ctl->granularity=PAPI_GRN_THR;
		   break;


              default:
		   return PAPI_EINVAL;
	   }
           return PAPI_OK;

      case PAPI_INHERIT:
	   pe_ctl = (pe_control_t *) ( option->inherit.ESI->ctl_state );

	   if (option->inherit.inherit) {
	      /* children will inherit counters */
	      pe_ctl->inherit = 1;
	   } else {
	      /* children won't inherit counters */
	      pe_ctl->inherit = 0;
	   }
	   return PAPI_OK;

      case PAPI_DATA_ADDRESS:
	   return PAPI_ENOSUPP;

      case PAPI_INSTR_ADDRESS:
	   return PAPI_ENOSUPP;

      case PAPI_DEF_ITIMER:
	   return PAPI_ENOSUPP;

      case PAPI_DEF_MPX_NS:
	   return PAPI_ENOSUPP;

      case PAPI_DEF_ITIMER_NS:
	   return PAPI_ENOSUPP;

      default:
	   return PAPI_ENOSUPP;
   }
}


static int
_peu_ntv_enum_events( unsigned int *PapiEventCode, int modifier )
{

  if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;


  return _pe_libpfm4_ntv_enum_events(PapiEventCode, modifier, our_cidx,
                                       &uncore_native_event_table);
}

static int
_peu_ntv_name_to_code( const char *name, unsigned int *event_code) {

  if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

  return _pe_libpfm4_ntv_name_to_code(name,event_code, our_cidx,
                                        &uncore_native_event_table);
}

static int
_peu_ntv_code_to_name(unsigned int EventCode,
                          char *ntv_name, int len) {

   if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

   return _pe_libpfm4_ntv_code_to_name(EventCode,
                                         ntv_name, len, 
					 &uncore_native_event_table);
}

static int
_peu_ntv_code_to_descr( unsigned int EventCode,
                            char *ntv_descr, int len) {

   if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

   return _pe_libpfm4_ntv_code_to_descr(EventCode,ntv_descr,len,
                                          &uncore_native_event_table);
}

static int
_peu_ntv_code_to_info(unsigned int EventCode,
                          PAPI_event_info_t *info) {

  if (_perf_event_uncore_vector.cmp_info.disabled) return PAPI_ENOEVNT;

  return _pe_libpfm4_ntv_code_to_info(EventCode, info,
                                        &uncore_native_event_table);
}

/* Our component vector */

papi_vector_t _perf_event_uncore_vector = {
   .cmp_info = {
       /* component information (unspecified values initialized to 0) */
      .name = "perf_event_uncore",
      .short_name = "peu",
      .version = "5.0",
      .description = "Linux perf_event CPU uncore and northbridge",

      .default_domain = PAPI_DOM_ALL,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR,
      .default_granularity = PAPI_GRN_SYS,
      .available_granularities = PAPI_GRN_SYS,

      .num_mpx_cntrs = PERF_EVENT_MAX_MPX_COUNTERS,

      /* component specific cmp_info initializations */
      .fast_virtual_timer = 0,
      .attach = 1,
      .attach_must_ptrace = 1,
      .cpu = 1,
      .inherit = 1,
      .cntr_umasks = 1,

  },

  /* sizes of framework-opaque component-private structures */
  .size = {
      .context = sizeof ( pe_context_t ),
      .control_state = sizeof ( pe_control_t ),
      .reg_value = sizeof ( int ),
      .reg_alloc = sizeof ( int ),
  },

  /* function pointers in this component */
  .init_component =        _peu_init_component,
  .shutdown_component =    _peu_shutdown_component,
  .init_thread =           _peu_init_thread,
  .init_control_state =    _peu_init_control_state,
  .start =                 _peu_start,
  .stop =                  _peu_stop,
  .read =                  _peu_read,
  .shutdown_thread =       _peu_shutdown_thread,
  .ctl =                   _peu_ctl,
  .update_control_state =  _peu_update_control_state,
  .set_domain =            _peu_set_domain,
  .reset =                 _peu_reset,
  .write =                 _peu_write,

  /* from counter name mapper */
  .ntv_enum_events =   _peu_ntv_enum_events,
  .ntv_name_to_code =  _peu_ntv_name_to_code,
  .ntv_code_to_name =  _peu_ntv_code_to_name,
  .ntv_code_to_descr = _peu_ntv_code_to_descr,
  .ntv_code_to_info =  _peu_ntv_code_to_info,
};


