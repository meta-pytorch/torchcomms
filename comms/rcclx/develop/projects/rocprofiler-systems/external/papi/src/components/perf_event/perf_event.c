/*
* File:    perf_event.c
*
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
*          - based upon perfmon.c written by -
*          Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
* Mods:    Vince Weaver
*          vweaver1@eecs.utk.edu
* Mods:	   Philip Mucci
*	   mucci@eecs.utk.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
*          Modified the perf_event component to use PFM_OS_PERF_EVENT_EXT mode in libpfm4.
*          This adds several new event masks, including cpu=, u=, and k= which give the user
*          the ability to set cpu number to use or control the domain (user, kernel, or both)
*          in which the counter should be incremented.  These are event masks so it is now 
*          possible to have multiple events in the same event set that count activity from 
*          differennt cpu's or count activity in different domains.
*/


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
#include "pe_libpfm4_events.h"
#include "perfmon/pfmlib.h"
#include PEINCLUDE

/* Linux-specific includes */
#include "mb.h"
#include "linux-memory.h"
#include "linux-timer.h"
#include "linux-common.h"
#include "linux-context.h"

#include "perf_event_lib.h"
#include "perf_helpers.h"

/* Set to enable pre-Linux 2.6.34 perf_event workarounds   */
/* If disabling them gets no complaints then we can remove */
/* These in a future version of PAPI.                      */
#define OBSOLETE_WORKAROUNDS 0

/* Defines for ctx->state */
#define PERF_EVENTS_OPENED  0x01
#define PERF_EVENTS_RUNNING 0x02

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

/* Forward declaration */
papi_vector_t _perf_event_vector;

/* Globals */
struct native_event_table_t perf_native_event_table;
static int our_cidx;
static int exclude_guest_unsupported;

/* The kernel developers say to never use a refresh value of 0        */
/* See https://lkml.org/lkml/2011/5/24/172                            */
/* However, on some platforms (like Power) a value of 1 does not work */
/* We're still tracking down why this happens.                        */

#if defined(__powerpc__)
#define PAPI_REFRESH_VALUE 0
#else
#define PAPI_REFRESH_VALUE 1
#endif

static int _pe_set_domain( hwd_control_state_t *ctl, int domain);

#if (OBSOLETE_WORKAROUNDS==1)

/* Check for processor support */
/* Can be used for generic checking, though in general we only     */
/* check for pentium4 here because support was broken for multiple */
/* kernel releases and the usual standard detections did not       */
/* handle this.  So we check for pentium 4 explicitly.             */
static int
processor_supported(int vendor, int family) {

   /* Error out if kernel too early to support p4 */
   if (( vendor == PAPI_VENDOR_INTEL ) && (family == 15)) {
      if (_papi_os_info.os_version < LINUX_VERSION(2,6,35)) {
         PAPIERROR("Pentium 4 not supported on kernels before 2.6.35");
         return PAPI_ENOSUPP;
      }
   }
   return PAPI_OK;
}

#endif

/* Fix up the config based on what CPU/Vendor we are running on */
static int
pe_vendor_fixups(papi_vector_t *vector)
{
     /* powerpc */
     /* On IBM and Power6 Machines default domain should include supervisor */
  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_IBM ) {
     vector->cmp_info.available_domains |=
                  PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     if (strcmp(_papi_hwi_system_info.hw_info.model_string, "POWER6" ) == 0 ) {
        vector->cmp_info.default_domain =
                  PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
     }
  }

  if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_MIPS ) {
     vector->cmp_info.available_domains |= PAPI_DOM_KERNEL;
  }

  if ((_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL) ||
      (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD)) {
     vector->cmp_info.fast_real_timer = 1;
  }

	/* ARM */
	/* If implementer is ARM Limited. */
	if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_ARM_ARM) {

		/* Some ARMv7 and earlier could not measure	*/
		/* KERNEL and USER separately.			*/

		/* Whitelist CortexA7 and CortexA15		*/
		/* There might be more				*/

		if ((_papi_hwi_system_info.hw_info.cpuid_family < 8) &&
			(_papi_hwi_system_info.hw_info.cpuid_model!=0xc07) &&
			(_papi_hwi_system_info.hw_info.cpuid_model!=0xc0f)) {

			vector->cmp_info.available_domains |=
				PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
			vector->cmp_info.default_domain =
				PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR;
		}
	}

	/* CRAY */
	if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_CRAY ) {
		vector->cmp_info.available_domains |= PAPI_DOM_OTHER;
	}

	return PAPI_OK;
}



/******************************************************************/
/******** Kernel Version Dependent Routines  **********************/
/******************************************************************/


/* PERF_FORMAT_GROUP allows reading an entire group's counts at once   */
/* before 2.6.34 PERF_FORMAT_GROUP did not work when reading results   */
/*  from attached processes.  We are lazy and disable it for all cases */
/*  commit was:  050735b08ca8a016bbace4445fa025b88fee770b              */

static int
bug_format_group(void) {


#if (OBSOLETE_WORKAROUNDS==1)
	if (_papi_os_info.os_version < LINUX_VERSION(2,6,34)) return 1;
#endif

	/* MIPS, as of version 3.1, does not support this properly */
	/* FIXME: is this still true? */

#if defined(__mips__)
  return 1;
#endif

  return 0;

}

#if (OBSOLETE_WORKAROUNDS==1)


/* There's a bug prior to Linux 2.6.33 where if you are using */
/* PERF_FORMAT_GROUP, the TOTAL_TIME_ENABLED and              */
/* TOTAL_TIME_RUNNING fields will be zero unless you disable  */
/* the counters first                                         */
static int
bug_sync_read(void) {

  if (_papi_os_info.os_version < LINUX_VERSION(2,6,33)) return 1;

  return 0;

}

#endif

/* Set the F_SETOWN_EX flag on the fd.                          */
/* This affects which thread an overflow signal gets sent to    */
/* Handled in a subroutine to handle the fact that the behavior */
/* is dependent on kernel version.                              */
static int
fcntl_setown_fd(int fd) {

	int ret;
	struct f_owner_ex fown_ex;

	/* F_SETOWN_EX is not available until 2.6.32 */
	/* but PAPI perf_event support didn't work on 2.6.31 anyay */

	/* set ownership of the descriptor */
	fown_ex.type = F_OWNER_TID;
	fown_ex.pid  = mygettid();
	ret = fcntl(fd, F_SETOWN_EX, (unsigned long)&fown_ex );

	if ( ret == -1 ) {
		PAPIERROR( "cannot fcntl(F_SETOWN_EX) on %d: %s",
			fd, strerror( errno ) );
		return PAPI_ESYS;
	}
	return PAPI_OK;
}

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

   /* if our kernel supports it and we are not using inherit, */
   /* add the group read options                              */
   if ( (!bug_format_group()) && !inherit) {
      if (format_group) {
	 format |= PERF_FORMAT_GROUP;
      }
   }

   SUBDBG("multiplex: %d, inherit: %d, group_leader: %d, format: %#x\n",
	  multiplex, inherit, format_group, format);

   return format;
}


/* attr.exclude_guest is enabled by default in recent libpfm4 */
/* however older kernels will reject events with it set */
/* because the reserved field is not all zeros */
void
check_exclude_guest( void )
{
	int ev_fd;
	struct perf_event_attr attr;

	exclude_guest_unsupported=0;

	/* First check that we can open a plain instructions event */
	memset(&attr, 0 , sizeof(attr));
	attr.config = PERF_COUNT_HW_INSTRUCTIONS;
	attr.exclude_kernel=1;

	ev_fd = sys_perf_event_open( &attr, 0, -1, -1, 0 );
	if ( ev_fd == -1 ) {
		PAPIERROR("Couldn't open hw_instructions in exclude_guest=0 test. " 
			      "Set /proc/sys/kernel/perf_event_paranoid to 2 (or less) or run as root.");
		return;
	}
	close(ev_fd);

	/* Now try again with excude_guest */
	memset(&attr, 0 , sizeof(attr));
	attr.config = PERF_COUNT_HW_INSTRUCTIONS;
	attr.exclude_kernel=1;
	attr.exclude_guest=1;

	ev_fd = sys_perf_event_open( &attr, 0, -1, -1, 0 );
	if ( ev_fd == -1 ) {
		if (errno==EINVAL) {
			exclude_guest_unsupported=1;
		}
		else {
		  PAPIERROR("Couldn't open hw_instructions in exclude_guest=1 test. "
			        "Set /proc/sys/kernel/perf_event_paranoid to 2 (or less) or run as root.");
		}
	} else {
		exclude_guest_unsupported=0;
		close(ev_fd);
	}

	return;
}

/*****************************************************************/
/********* End Kernel-version Dependent Routines  ****************/
/*****************************************************************/

/*****************************************************************/
/********* Begin perf_event low-level code ***********************/
/*****************************************************************/

static void perf_event_dump_attr( struct perf_event_attr *hw_event,
	pid_t pid, int cpu, int group_fd, unsigned long int flags) {

	/* Mark parameters as not used                   */
	/* In the common case (no SUBDBG) the function   */
	/* compiles into an empty function and complains */
	/* about unused variables.                       */
	(void)hw_event;
	(void)pid;
	(void)cpu;
	(void)group_fd;
	(void)flags;

	SUBDBG("sys_perf_event_open(hw_event: %p, pid: %d, cpu: %d, "
		"group_fd: %d, flags: %lx\n",
		hw_event, pid, cpu, group_fd, flags);
	SUBDBG("   type: %d\n",hw_event->type);
	SUBDBG("   size: %d\n",hw_event->size);
	SUBDBG("   config: %"PRIx64" (%"PRIu64")\n",
		hw_event->config, hw_event->config);
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
	SUBDBG("   exclude_callchain_kernel: %d\n",
		hw_event->exclude_callchain_kernel);
	SUBDBG("   exclude_callchain_user: %d\n",
		hw_event->exclude_callchain_user);
	SUBDBG("   wakeup_events: %"PRIx32" (%"PRIu32")\n",
		hw_event->wakeup_events, hw_event->wakeup_events);
	SUBDBG("   bp_type: %"PRIx32" (%"PRIu32")\n",
		hw_event->bp_type, hw_event->bp_type);
	SUBDBG("   config1: %"PRIx64" (%"PRIu64")\n",
		hw_event->config1, hw_event->config1);
	SUBDBG("   config2: %"PRIx64" (%"PRIu64")\n",
		hw_event->config2, hw_event->config2);
	SUBDBG("   branch_sample_type: %"PRIx64" (%"PRIu64")\n",
		hw_event->branch_sample_type, hw_event->branch_sample_type);
	SUBDBG("   sample_regs_user: %"PRIx64" (%"PRIu64")\n",
		hw_event->sample_regs_user, hw_event->sample_regs_user);
	SUBDBG("   sample_stack_user: %"PRIx32" (%"PRIu32")\n",
		hw_event->sample_stack_user, hw_event->sample_stack_user);
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
	case ESRCH:	/* If cannnot find process to attach to */
      case ENOSYS:
      case EAGAIN:
      case EBUSY:
      case E2BIG:	/* Only happens if attr is the wrong size somehow */
      case EBADF:	/* We are attempting to group with an invalid file descriptor */
	   ret = PAPI_ESYS;
	   break;
      case ENOMEM:
	   ret = PAPI_ENOMEM;
	   break;
      case EMFILE:	/* Out of file descriptors.  Typically max out at 1024 */
           ret = PAPI_ECOUNT;
           break;
      case EINVAL:
      default:
	   ret = PAPI_EINVAL;
           break;
   }
   return ret;
}


/** Check if the current set of options is supported by  */
/*  perf_events.                                         */
/*  We do this by temporarily opening an event with the  */
/*  desired options then closing it again.  We use the   */
/*  PERF_COUNT_HW_INSTRUCTION event as a dummy event     */
/*  on the assumption it is available on all             */
/*  platforms.                                           */

static int
check_permissions( unsigned long tid,
		   unsigned int cpu_num,
		   unsigned int domain,
		   unsigned int granularity,
		   unsigned int multiplex,
		   unsigned int inherit )
{
   int ev_fd;
   struct perf_event_attr attr;

   long pid;

   /* clearing this will set a type of hardware and to count all domains */
   memset(&attr, '\0', sizeof(attr));
   attr.read_format = get_read_format(multiplex, inherit, 1);

   /* set the event id (config field) to instructios */
   /* (an event that should always exist)            */
   /* This was cycles but that is missing on Niagara */
   attr.config = PERF_COUNT_HW_INSTRUCTIONS;

   /* now set up domains this event set will be counting */
   if (!(domain & PAPI_DOM_SUPERVISOR)) {
      attr.exclude_hv = 1;
   }
   if (!(domain & PAPI_DOM_USER)) {
      attr.exclude_user = 1;
   }
   if (!(domain & PAPI_DOM_KERNEL)) {
      attr.exclude_kernel = 1;
   }

   if (granularity==PAPI_GRN_SYS) {
      pid = -1;
   } else {
      pid = tid;
   }

   SUBDBG("Calling sys_perf_event_open() from check_permissions\n");

	perf_event_dump_attr( &attr, pid, cpu_num, -1, 0 );

   ev_fd = sys_perf_event_open( &attr, pid, cpu_num, -1, 0 );
   if ( ev_fd == -1 ) {
      SUBDBG("sys_perf_event_open returned error.  Linux says, %s", 
	     strerror( errno ) );
      return map_perf_event_errors_to_papi(errno);
   }

   /* now close it, this was just to make sure we have permissions */
   /* to set these options                                         */
   close(ev_fd);
   return PAPI_OK;
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
check_scheduability( pe_context_t *ctx, pe_control_t *ctl, int idx )
{
   int retval = 0, cnt = -1;
   ( void ) ctx;			 /*unused */
   long long papi_pe_buffer[READ_BUFFER_SIZE];
   int i,group_leader_fd;

   /* If the kernel isn't tracking scheduability right       */
   /* Then we need to start/stop/read to force the event     */
   /* to be scheduled and see if an error condition happens. */

   /* get the proper fd to start */
   group_leader_fd=ctl->events[idx].group_leader_fd;
   if (group_leader_fd==-1) group_leader_fd=ctl->events[idx].event_fd;

   /* start the event */
   retval = ioctl( group_leader_fd, PERF_EVENT_IOC_ENABLE, NULL );
   if (retval == -1) {
      PAPIERROR("ioctl(PERF_EVENT_IOC_ENABLE) failed");
      return PAPI_ESYS;
   }

   /* stop the event */
   retval = ioctl(group_leader_fd, PERF_EVENT_IOC_DISABLE, NULL );
   if (retval == -1) {
      PAPIERROR( "ioctl(PERF_EVENT_IOC_DISABLE) failed" );
      return PAPI_ESYS;
   }

   /* See if a read returns any results */
   cnt = read( group_leader_fd, papi_pe_buffer, sizeof(papi_pe_buffer));
   if ( cnt == -1 ) {
      SUBDBG( "read returned an error!  Should never happen.\n" );
      return PAPI_ESYS;
   }

   if ( cnt == 0 ) {
      /* We read 0 bytes if we could not schedule the event */
      /* The kernel should have detected this at open       */
      /* but various bugs (including NMI watchdog)          */
      /* result in this behavior                            */

      return PAPI_ECNFLCT;

   } else {

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
      for( i = 0; i < idx; i++) {
	 retval=ioctl( ctl->events[i].event_fd, PERF_EVENT_IOC_RESET, NULL );
	 if (retval == -1) {
	    PAPIERROR( "ioctl(PERF_EVENT_IOC_RESET) #%d/%d %d "
		       "(fd %d)failed",
		       i,ctl->num_events,idx,ctl->events[i].event_fd);
	    return PAPI_ESYS;
	 }
      }
   }
   return PAPI_OK;
}


/* Do some extra work on a perf_event fd if we're doing sampling  */
/* This mostly means setting up the mmap buffer.                  */
static int
configure_fd_for_sampling( pe_control_t *ctl, int evt_idx )
{
   int ret;
   int fd = ctl->events[evt_idx].event_fd;

   /* Register that we would like a SIGIO notification when a mmap'd page */
   /* becomes full.                                                       */
   ret = fcntl( fd, F_SETFL, O_ASYNC | O_NONBLOCK );
   if ( ret ) {
      PAPIERROR ( "fcntl(%d, F_SETFL, O_ASYNC | O_NONBLOCK) "
		  "returned error: %s", fd, strerror( errno ) );
      return PAPI_ESYS;
   }

   /* Set the F_SETOWN_EX flag on the fd.                          */
   /* This affects which thread an overflow signal gets sent to.   */
   ret=fcntl_setown_fd(fd);
   if (ret!=PAPI_OK) return ret;

   /* Set FD_CLOEXEC.  Otherwise if we do an exec with an overflow */
   /* running, the overflow handler will continue into the exec()'d*/
   /* process and kill it because no signal handler is set up.     */
   ret=fcntl(fd, F_SETFD, FD_CLOEXEC);
   if (ret) {
      return PAPI_ESYS;
   }

   /* when you explicitely declare that you want a particular signal,  */
   /* even with you use the default signal, the kernel will send more  */
   /* information concerning the event to the signal handler.          */
   /*                                                                  */
   /* In particular, it will send the file descriptor from which the   */
   /* event is originating which can be quite useful when monitoring   */
   /* multiple tasks from a single thread.                             */
   ret = fcntl( fd, F_SETSIG, ctl->overflow_signal );
   if ( ret == -1 ) {
      PAPIERROR( "cannot fcntl(F_SETSIG,%d) on %d: %s",
		 ctl->overflow_signal, fd,
		 strerror( errno ) );
      return PAPI_ESYS;
   }

	return PAPI_OK;
}

static int
set_up_mmap( pe_control_t *ctl, int evt_idx)
{

	void *buf_addr;
	int fd = ctl->events[evt_idx].event_fd;

	/* mmap() the sample buffer */
	buf_addr = mmap( NULL,
			ctl->events[evt_idx].nr_mmap_pages * getpagesize(),
			PROT_READ | PROT_WRITE,
			MAP_SHARED,
			fd, 0 );

	/* This may happen if we go over the limit in	*/
	/* /proc/sys/kernel/perf_event_mlock_kb		*/
	/* which defaults to 516k			*/
	/* with regular rdpmc events on 4k page archs	*/
	/* this is roughly 128 events			*/

	/* We sholdn't fail, just fall back to non-rdpmc	*/
	/* Although not sure what happens if it's a sample	*/
	/* event that fails to mmap.				*/

	if ( buf_addr == MAP_FAILED ) {
		SUBDBG( "mmap(NULL,%d,%d,%d,%d,0): %s",
			ctl->events[evt_idx].nr_mmap_pages * getpagesize(),
			PROT_READ | PROT_WRITE,
			MAP_SHARED,
			fd, strerror( errno ) );

		ctl->events[evt_idx].mmap_buf = NULL;

		/* Easier to just globally disable this, as it should	*/
		/* be a fairly uncommon case hopefully.			*/
		if (_perf_event_vector.cmp_info.fast_counter_read) {
			PAPIERROR("Can't mmap, disabling fast_counter_read\n");
			_perf_event_vector.cmp_info.fast_counter_read=0;
		}
		return PAPI_ESYS;
	}

	SUBDBG( "Sample buffer for fd %d is located at %p\n", fd, buf_addr );

	/* Set up the mmap buffer and its associated helpers */
	ctl->events[evt_idx].mmap_buf = (struct perf_counter_mmap_page *) buf_addr;
	ctl->events[evt_idx].tail = 0;
	ctl->events[evt_idx].mask =
		( ctl->events[evt_idx].nr_mmap_pages - 1 ) * getpagesize() - 1;

	return PAPI_OK;
}



/* Request user access for arm64 */
static inline void arm64_request_user_access(struct perf_event_attr *hw_event)
{
	hw_event->config1=0x2;        /* Request user access */
}

/* Open all events in the control state */
static int
open_pe_events( pe_context_t *ctx, pe_control_t *ctl )
{

	int i, ret = PAPI_OK;
	long pid;


	/* Set the pid setting */
	/* If attached, this is the pid of process we are attached to. */
	/* If GRN_THRD then it is 0 meaning current process only */
	/* If GRN_SYS then it is -1 meaning all procs on this CPU */
	/* Note if GRN_SYS then CPU must be specified, not -1 */

	if (ctl->attached) {
		pid = ctl->tid;
	}
	else {
		if (ctl->granularity==PAPI_GRN_SYS) {
			pid = -1;
		}
		else {
			pid = 0;
		}
	}

	for( i = 0; i < ctl->num_events; i++ ) {

		ctl->events[i].event_opened=0;

		/* set up the attr structure.			*/
		/* We don't set up all fields here		*/
		/* as some have already been set up previously.	*/

		/* Handle the broken exclude_guest problem */
		/* libpfm4 sets this by default (PEBS events depend on it) */
		/* but on older kernels that dont know about exclude_guest */
		/* perf_event_open() will error out as a "reserved"        */
		/* unknown bit is set to 1.                                */
		/* Do we need to also watch for exclude_host, exclude_idle */
		/* exclude_callchain*?					   */
		if ((ctl->events[i].attr.exclude_guest) &&
			(exclude_guest_unsupported)) {
			SUBDBG("Disabling exclude_guest in event %d\n",i);
			ctl->events[i].attr.exclude_guest=0;
		}

		/* group leader (event 0) is special                */
		/* If we're multiplexed, everyone is a group leader */
		if (( i == 0 ) || (ctl->multiplexed)) {
			ctl->events[i].attr.pinned = !ctl->multiplexed;
			ctl->events[i].attr.disabled = 1;
#if defined(__aarch64__)
			if (_perf_event_vector.cmp_info.fast_counter_read) {
				arm64_request_user_access(&ctl->events[i].attr);
			}
#endif
			ctl->events[i].group_leader_fd=-1;
			ctl->events[i].attr.read_format = get_read_format(
							ctl->multiplexed,
							ctl->inherit,
							!ctl->multiplexed );
		} else {
			ctl->events[i].attr.pinned=0;
			ctl->events[i].attr.disabled = 0;
#if defined(__aarch64__)
			if (_perf_event_vector.cmp_info.fast_counter_read) {
				arm64_request_user_access(&ctl->events[i].attr);
			}
#endif
			ctl->events[i].group_leader_fd=ctl->events[0].event_fd;
			ctl->events[i].attr.read_format = get_read_format(
							ctl->multiplexed,
							ctl->inherit,
							0 );
		}

		/* try to open */
		perf_event_dump_attr(
				&ctl->events[i].attr,
				pid,
				ctl->events[i].cpu,
				ctl->events[i].group_leader_fd,
				0 /* flags */ );

		ctl->events[i].event_fd = sys_perf_event_open(
				&ctl->events[i].attr,
				pid,
				ctl->events[i].cpu,
				ctl->events[i].group_leader_fd,
				0 /* flags */ );

		/* Try to match Linux errors to PAPI errors */
		if ( ctl->events[i].event_fd == -1 ) {
			SUBDBG("sys_perf_event_open returned error "
				"on event #%d.  Error: %s\n",
				i, strerror( errno ) );
			ret=map_perf_event_errors_to_papi(errno);

			goto open_pe_cleanup;
		}

		SUBDBG ("sys_perf_event_open: tid: %ld, cpu_num: %d,"
			" group_leader/fd: %d, event_fd: %d,"
			" read_format: %"PRIu64"\n",
			pid, ctl->events[i].cpu,
			ctl->events[i].group_leader_fd,
			ctl->events[i].event_fd,
			ctl->events[i].attr.read_format);


		/* in many situations the kernel will indicate we opened fine */
		/* yet things will fail later.  So we need to double check    */
		/* we actually can use the events we've set up.               */

		/* This is not necessary if we are multiplexing, and in fact */
		/* we cannot do this properly if multiplexed because         */
		/* PERF_EVENT_IOC_RESET does not reset the time running info */
		if (!ctl->multiplexed) {
			ret = check_scheduability( ctx, ctl, i );

			if ( ret != PAPI_OK ) {
				/* the last event did open, so we need to    */
				/* bump the counter before doing the cleanup */
				i++;
				goto open_pe_cleanup;
			}
		}
		ctl->events[i].event_opened=1;
	}

	/* Now that we've successfully opened all of the events, do whatever  */
	/* "tune-up" is needed to attach the mmap'd buffers, signal handlers, */
	/* and so on.                                                         */


	/* Make things easier and give each event a mmap() buffer */
	/* Keeping separate tracking for rdpmc vs regular events  */
	/* Would be a pain.  Also perf always gives every event a */
	/* mmap buffer.						  */

	for ( i = 0; i < ctl->num_events; i++ ) {

		/* Can't mmap() inherited events :( */
		if (ctl->inherit) {
			ctl->events[i].nr_mmap_pages = 0;
			ctl->events[i].mmap_buf = NULL;
		}
		else {
			/* Just a guess at how many pages would make this   */
			/* relatively efficient.                            */
			/* Note that it's "1 +" because of the need for a   */
			/* control page, and the number following the "+"   */
			/* must be a power of 2 (1, 4, 8, 16, etc) or zero. */
			/* This is required to optimize dealing with        */
			/* circular buffer wrapping of the mapped pages.    */
			if (ctl->events[i].sampling) {
				ctl->events[i].nr_mmap_pages = 1 + 2;
			}
			else if (_perf_event_vector.cmp_info.fast_counter_read) {
				ctl->events[i].nr_mmap_pages = 1;
			}
			else {
				ctl->events[i].nr_mmap_pages = 0;
			}

			/* Set up the MMAP sample pages */
			if (ctl->events[i].nr_mmap_pages) {
				set_up_mmap(ctl,i);
			} else {
				ctl->events[i].mmap_buf = NULL;
			}
		}
	}

	for ( i = 0; i < ctl->num_events; i++ ) {

		/* If sampling is enabled, hook up signal handler */
		if (ctl->events[i].attr.sample_period) {

			ret = configure_fd_for_sampling( ctl, i );
			if ( ret != PAPI_OK ) {
				/* We failed, and all of the fds are open */
				/* so we need to clean up all of them */
				i = ctl->num_events;
				goto open_pe_cleanup;
			}
		}
	}

	/* Set num_evts only if completely successful */
	ctx->state |= PERF_EVENTS_OPENED;

	return PAPI_OK;

open_pe_cleanup:
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

/* TODO: make code clearer -- vmw */
static int
close_event( pe_event_info_t *event )
{
	int munmap_error=0,close_error=0;

	if ( event->mmap_buf ) {
		if (event->nr_mmap_pages==0) {
			PAPIERROR("munmap and num pages is zero");
		}
		if ( munmap ( event->mmap_buf,
				event->nr_mmap_pages * getpagesize() ) ) {
			PAPIERROR( "munmap of fd = %d returned error: %s",
							event->event_fd,
							strerror( errno ) );
			event->mmap_buf=NULL;
			munmap_error=1;
		}
	}
	if ( close( event->event_fd ) ) {
		PAPIERROR( "close of fd = %d returned error: %s",
			event->event_fd, strerror( errno ) );
		close_error=1;
	}

	event->event_opened=0;

	if ((close_error || munmap_error)) {
		return PAPI_ESYS;
	}

	return 0;
}

/* Close all of the opened events */
static int
close_pe_events( pe_context_t *ctx, pe_control_t *ctl )
{
	int i,result;
	int num_closed=0;
	int events_not_opened=0;

	/* should this be a more serious error? */
	if ( ctx->state & PERF_EVENTS_RUNNING ) {
		SUBDBG("Closing without stopping first\n");
	}

	/* Close child events first */
	/* Is that necessary? -- vmw */
	for( i=0; i<ctl->num_events; i++ ) {
		if (ctl->events[i].event_opened) {
			if (ctl->events[i].group_leader_fd!=-1) {
				result=close_event(&ctl->events[i]);
				if (result!=0) return result;
				else num_closed++;
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
				result=close_event(&ctl->events[i]);
				if (result!=0) return result;
				else num_closed++;
			}
		}
	}

	if (ctl->num_events!=num_closed) {
		if (ctl->num_events!=(num_closed+events_not_opened)) {
			PAPIERROR("Didn't close all events: "
				"Closed %d Not Opened: %d Expected %d",
				num_closed,events_not_opened,ctl->num_events);
			return PAPI_EBUG;
		}
	}

	ctl->num_events=0;

	ctx->state &= ~PERF_EVENTS_OPENED;

	return PAPI_OK;
}


/********************************************************************/
/********************************************************************/
/*     Functions that are exported via the component interface      */
/********************************************************************/
/********************************************************************/

/********************* DOMAIN RELATED *******************************/


/* set the domain. */
/* perf_events allows per-event control of this, */
/* papi allows it to be set at the event level or at the event set level. */
/* this will set the event set level domain values */
/* but they only get used if no event level domain mask (u= or k=) */
/* was specified. */
static int
_pe_set_domain( hwd_control_state_t *ctl, int domain)
{
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;

	SUBDBG("old control domain %d, new domain %d\n", pe_ctl->domain,domain);
	pe_ctl->domain = domain;
	return PAPI_OK;
}


/********************* THREAD RELATED *******************************/


/* Shutdown a thread */
static int
_pe_shutdown_thread( hwd_context_t *ctx )
{
	pe_context_t *pe_ctx = ( pe_context_t *) ctx;

	pe_ctx->initialized=0;

	return PAPI_OK;
}

/* Initialize a thread */
static int
_pe_init_thread( hwd_context_t *hwd_ctx )
{

	pe_context_t *pe_ctx = ( pe_context_t *) hwd_ctx;

	/* clear the context structure and mark as initialized */
	memset( pe_ctx, 0, sizeof ( pe_context_t ) );
	pe_ctx->initialized=1;
	pe_ctx->event_table=&perf_native_event_table;
	pe_ctx->cidx=our_cidx;

	return PAPI_OK;
}



/**************************** COUNTER RELATED *******************/


/* reset the hardware counters */
/* Note: PAPI_reset() does not necessarily call this */
/* unless the events are actually running.           */
static int
_pe_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	int i, ret;
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;

	( void ) ctx;			 /*unused */

	/* We need to reset all of the events, not just the group leaders */
	for( i = 0; i < pe_ctl->num_events; i++ ) {
		if (_perf_event_vector.cmp_info.fast_counter_read) {
			ret = ioctl( pe_ctl->events[i].event_fd, 
					PERF_EVENT_IOC_RESET, NULL );
			pe_ctl->reset_counts[i] = mmap_read_reset_count(
					pe_ctl->events[i].mmap_buf);
			pe_ctl->reset_flag = 1;
		} else {
			ret = ioctl( pe_ctl->events[i].event_fd, 
					PERF_EVENT_IOC_RESET, NULL );
		}
		if ( ret == -1 ) {
			PAPIERROR("ioctl(%d, PERF_EVENT_IOC_RESET, NULL) "
					"returned error, Linux says: %s",
					pe_ctl->events[i].event_fd,
					strerror( errno ) );
			return PAPI_ESYS;
		}
	}

	return PAPI_OK;
}


/* write (set) the hardware counters */
/* Currently we do not support this.   */
static int
_pe_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
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
 * For more info on the layout see include/uapi/linux/perf_event.h
 *
 */


/* When we read with rdpmc, we must read each counter individually */
/* Because of this we don't need separate multiplexing support */
/* This is all handled by mmap_read_self() */
static int
_pe_rdpmc_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
		long long **events, int flags )
{
	long long papi_pe_buffer[READ_BUFFER_SIZE];

	SUBDBG("ENTER: ctx: %p, ctl: %p, events: %p, flags: %#x\n",
		ctx, ctl, events, flags);

	( void ) flags;			/*unused */
	( void ) ctx;			/*unused */
	( void ) papi_pe_buffer;	/*unused */
	int i;
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;
	unsigned long long count, enabled = 0, running = 0, adjusted;
	int errors=0;

	/* we must read each counter individually */
	for ( i = 0; i < pe_ctl->num_events; i++ ) {

		count = mmap_read_self(pe_ctl->events[i].mmap_buf,
						pe_ctl->reset_flag,
						pe_ctl->reset_counts[i],
						&enabled,&running);

		if (count==0xffffffffffffffffULL) {
			errors++;
		}

		/* Handle multiplexing case */
		if (enabled == running) {
			/* no adjustment needed */
		}
		else if (enabled && running) {
			adjusted = (enabled * 128LL) / running;
			adjusted = adjusted * count;
			adjusted = adjusted / 128LL;
			count = adjusted;
		} else {
			/* This should not happen, but we have had it reported */
			SUBDBG("perf_event kernel bug(?) count, enabled, "
				"running: %lld, %lld, %lld\n",
				papi_pe_buffer[0],enabled,running);

		}

		pe_ctl->counts[i] = count;
	}
	/* point PAPI to the values we read */
	*events = pe_ctl->counts;

	SUBDBG("EXIT: *events: %p\n", *events);

	if (errors) return PAPI_ESYS;

	return PAPI_OK;
}


static int
_pe_read_multiplexed( pe_control_t *pe_ctl )
{
	int i,ret=-1;
	long long papi_pe_buffer[READ_BUFFER_SIZE];
	long long tot_time_running, tot_time_enabled, scale;

	/* perf_event does not support FORMAT_GROUP on multiplex */
	/* so we have to handle separate events when multiplexing */

	for ( i = 0; i < pe_ctl->num_events; i++ ) {

		ret = read( pe_ctl->events[i].event_fd,
				papi_pe_buffer,
				sizeof ( papi_pe_buffer ) );
		if ( ret == -1 ) {
			PAPIERROR("read returned an error: %s",
					strerror( errno ));
			return PAPI_ESYS;
		}

		/* We should read 3 64-bit values from the counter */
		if (ret<(signed)(3*sizeof(long long))) {
			PAPIERROR("Error!  short read");
			return PAPI_ESYS;
		}

		SUBDBG("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n",
				pe_ctl->events[i].event_fd,
				(long)pe_ctl->tid, pe_ctl->events[i].cpu, ret);
		SUBDBG("read: %lld %lld %lld\n",
				papi_pe_buffer[0],
				papi_pe_buffer[1],
				papi_pe_buffer[2]);

		tot_time_enabled = papi_pe_buffer[1];
		tot_time_running = papi_pe_buffer[2];

		SUBDBG("count[%d] = (papi_pe_buffer[%d] %lld * "
				"tot_time_enabled %lld) / "
				"tot_time_running %lld\n",
				i, 0,papi_pe_buffer[0],
				tot_time_enabled,tot_time_running);

		if (tot_time_running == tot_time_enabled) {
			/* No scaling needed */
			pe_ctl->counts[i] = papi_pe_buffer[0];
		} else if (tot_time_running && tot_time_enabled) {
			/* Scale to give better results */
			/* avoid truncation.            */
			/* Why use 100?  Would 128 be faster? */
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
	return PAPI_OK;
}

/* For cases where we can't group counters together */
/* But must read them out individually */
/* This includes when INHERIT is set, as well as various bugs */

static int
_pe_read_nogroup( pe_control_t *pe_ctl ) {

	int i,ret=-1;
	long long papi_pe_buffer[READ_BUFFER_SIZE];

	/* we must read each counter individually */
	for ( i = 0; i < pe_ctl->num_events; i++ ) {
		ret = read( pe_ctl->events[i].event_fd,
				papi_pe_buffer,
				sizeof ( papi_pe_buffer ) );
		if ( ret == -1 ) {
			PAPIERROR("read returned an error: %s",
				strerror( errno ));
			return PAPI_ESYS;
		}

		/* we should read one 64-bit value from each counter */
		if (ret!=sizeof(long long)) {
			PAPIERROR("Error!  short read");
			PAPIERROR("read: fd: %2d, tid: %ld, cpu: %d, ret: %d",
				pe_ctl->events[i].event_fd,
				(long)pe_ctl->tid, pe_ctl->events[i].cpu, ret);
			return PAPI_ESYS;
		}

		SUBDBG("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n",
			pe_ctl->events[i].event_fd, (long)pe_ctl->tid,
			pe_ctl->events[i].cpu, ret);
		SUBDBG("read: %lld\n",papi_pe_buffer[0]);

		pe_ctl->counts[i] = papi_pe_buffer[0];
	}

	return PAPI_OK;

}

static int
_pe_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
	       long long **events, int flags )
{
	SUBDBG("ENTER: ctx: %p, ctl: %p, events: %p, flags: %#x\n",
		ctx, ctl, events, flags);

	( void ) flags;			 /*unused */
	( void ) ctx;			 /*unused */
	int i, j, ret = -1;
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;
	long long papi_pe_buffer[READ_BUFFER_SIZE];
	int result;

	/* Handle fast case */
	/* FIXME: we fallback to slow reads if *any* event in eventset fails */
	/*        in theory we could only fall back for the one event        */
	/*        but that makes the code more complicated.                  */
	if ((_perf_event_vector.cmp_info.fast_counter_read) &&
		(!pe_ctl->inherit) &&
		(!pe_ctl->attached) &&
		(pe_ctl->granularity==PAPI_GRN_THR)) {
		result=_pe_rdpmc_read( ctx, ctl, events, flags);
		/* if successful we are done, otherwise fall back to read */
		if (result==PAPI_OK) return PAPI_OK;
	}

	/* Handle case where we are multiplexing */
	if (pe_ctl->multiplexed) {
		_pe_read_multiplexed(pe_ctl);
	}

	/* Handle cases where we cannot use FORMAT GROUP */
	else if (bug_format_group() || pe_ctl->inherit) {
		_pe_read_nogroup(pe_ctl);
	}

	/* Handle common case where we are using FORMAT_GROUP	*/
	/* We assume only one group leader, in position 0	*/

	/* By reading the leader file descriptor, we get a series */
	/* of 64-bit values.  The first is the total number of    */
	/* events, followed by the counts for them.               */

	else {
		if (pe_ctl->events[0].group_leader_fd!=-1) {
			PAPIERROR("Was expecting group leader");
		}

		ret = read( pe_ctl->events[0].event_fd,
			papi_pe_buffer,
			sizeof ( papi_pe_buffer ) );

		if ( ret == -1 ) {
			PAPIERROR("read returned an error: %s",
				strerror( errno ));
			return PAPI_ESYS;
		}

		/* we read 1 64-bit value (number of events) then     */
		/* num_events more 64-bit values that hold the counts */
		if (ret<(signed)((1+pe_ctl->num_events)*sizeof(long long))) {
			PAPIERROR("Error! short read");
			return PAPI_ESYS;
		}

		SUBDBG("read: fd: %2d, tid: %ld, cpu: %d, ret: %d\n",
			pe_ctl->events[0].event_fd,
			(long)pe_ctl->tid, pe_ctl->events[0].cpu, ret);

		for(j=0;j<ret/8;j++) {
			SUBDBG("read %d: %lld\n",j,papi_pe_buffer[j]);
		}

		/* Make sure the kernel agrees with how many events we have */
		if (papi_pe_buffer[0]!=pe_ctl->num_events) {
			PAPIERROR("Error!  Wrong number of events");
			return PAPI_ESYS;
		}

		/* put the count values in their proper location */
		for(i=0;i<pe_ctl->num_events;i++) {
			pe_ctl->counts[i] = papi_pe_buffer[1+i];
		}
	}

	/* point PAPI to the values we read */
	*events = pe_ctl->counts;

	SUBDBG("EXIT: *events: %p\n", *events);

	return PAPI_OK;
}

#if (OBSOLETE_WORKAROUNDS==1)
/* On kernels before 2.6.33 the TOTAL_TIME_ENABLED and TOTAL_TIME_RUNNING */
/* fields are always 0 unless the counter is disabled.  So if we are on   */
/* one of these kernels, then we must disable events before reading.      */
/* Elsewhere though we disable multiplexing on kernels before 2.6.34 */
/* so maybe this isn't even necessary.                               */
static int
_pe_read_bug_sync( hwd_context_t *ctx, hwd_control_state_t *ctl,
	       long long **events, int flags )
{

	( void ) flags;			 /*unused */
	int i, ret = -1;
	pe_context_t *pe_ctx = ( pe_context_t *) ctx;
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;
	int result;

	if ( pe_ctx->state & PERF_EVENTS_RUNNING ) {
		 for ( i = 0; i < pe_ctl->num_events; i++ ) {
			/* disable only the group leaders */
			if ( pe_ctl->events[i].group_leader_fd == -1 ) {
				ret = ioctl( pe_ctl->events[i].event_fd,
					PERF_EVENT_IOC_DISABLE, NULL );
				if ( ret == -1 ) {
					PAPIERROR("ioctl(PERF_EVENT_IOC_DISABLE) "
						"returned an error: ", strerror( errno ));
					return PAPI_ESYS;
				}
			}
		}
	}

	result=_pe_read( ctx, ctl, events, flags );

	/* If we disabled the counters due to the sync_read_bug(), */
	/* then we need to re-enable them now.                     */

	if ( pe_ctx->state & PERF_EVENTS_RUNNING ) {
		for ( i = 0; i < pe_ctl->num_events; i++ ) {
			if ( pe_ctl->events[i].group_leader_fd == -1 ) {
				/* this should refresh any overflow counters too */
				ret = ioctl( pe_ctl->events[i].event_fd,
					PERF_EVENT_IOC_ENABLE, NULL );
				if ( ret == -1 ) {
					/* Should never happen */
					PAPIERROR("ioctl(PERF_EVENT_IOC_ENABLE) returned an error: ",
						strerror( errno ));
					return PAPI_ESYS;
				}
			}
		}
	}

	return result;
}

#endif

/* Start counting events */
static int
_pe_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	int ret;
	int i;
	int did_something = 0;
	pe_context_t *pe_ctx = ( pe_context_t *) ctx;
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;

	/* Reset the counters first.  Is this necessary? */
	ret = _pe_reset( pe_ctx, pe_ctl );
	if ( ret ) {
		return ret;
	}

	/* Enable all of the group leaders                */
	/* All group leaders have a group_leader_fd of -1 */
	for( i = 0; i < pe_ctl->num_events; i++ ) {
		if (pe_ctl->events[i].group_leader_fd == -1) {
			SUBDBG("ioctl(enable): fd: %d\n",
				pe_ctl->events[i].event_fd);
			ret=ioctl( pe_ctl->events[i].event_fd,
				PERF_EVENT_IOC_ENABLE, NULL) ;
			if (_perf_event_vector.cmp_info.fast_counter_read) {
				pe_ctl->reset_counts[i] = 0LL;
				pe_ctl->reset_flag = 0;
			}

			/* ioctls always return -1 on failure */
			if (ret == -1) {
				PAPIERROR("ioctl(PERF_EVENT_IOC_ENABLE) failed");
				return PAPI_ESYS;
			}

			did_something++;
		}
	}

	if (!did_something) {
		PAPIERROR("Did not enable any counters");
		return PAPI_EBUG;
	}

	pe_ctx->state |= PERF_EVENTS_RUNNING;

	return PAPI_OK;

}

/* Stop all of the counters */
static int
_pe_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	SUBDBG( "ENTER: ctx: %p, ctl: %p\n", ctx, ctl);

	int ret;
	int i;
	pe_context_t *pe_ctx = ( pe_context_t *) ctx;
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;

	/* Just disable the group leaders */
	for ( i = 0; i < pe_ctl->num_events; i++ ) {
		if ( pe_ctl->events[i].group_leader_fd == -1 ) {
			ret=ioctl( pe_ctl->events[i].event_fd,
				PERF_EVENT_IOC_DISABLE, NULL);
			if ( ret == -1 ) {
				PAPIERROR( "ioctl(%d, PERF_EVENT_IOC_DISABLE, NULL) "
					"returned error, Linux says: %s",
					pe_ctl->events[i].event_fd, strerror( errno ) );
				return PAPI_EBUG;
			}
		}
	}

	pe_ctx->state &= ~PERF_EVENTS_RUNNING;

	SUBDBG( "EXIT:\n");

	return PAPI_OK;
}





/*********************** CONTROL STATE RELATED *******************/


/* This function clears the current contents of the control structure and
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

static int
_pe_update_control_state( hwd_control_state_t *ctl,
			       NativeInfo_t *native,
			       int count, hwd_context_t *ctx )
{
	SUBDBG( "ENTER: ctl: %p, native: %p, count: %d, ctx: %p\n",
		ctl, native, count, ctx);
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
		SUBDBG( "EXIT: Called with count == 0\n" );
		return PAPI_OK;
	}

	/* set up all the events */
	for( i = 0; i < count; i++ ) {
		if ( native ) {
			/* get the native event pointer used for this papi event */
			int ntv_idx = _papi_hwi_get_ntv_idx((unsigned)(native[i].ni_papi_code));
			if (ntv_idx < -1) {
				SUBDBG("papi_event_code: %#x known by papi but not by the component\n", native[i].ni_papi_code);
				continue;
			}
			/* if native index is -1, then we have an event without a mask and need to find the right native index to use */
			if (ntv_idx == -1) {
				/* find the native event index we want by matching for the right papi event code */
				for (j=0 ; j<pe_ctx->event_table->num_native_events ; j++) {
					if (pe_ctx->event_table->native_events[j].papi_event_code == native[i].ni_papi_code) {
						ntv_idx = j;
					}
				}
			}

			/* if native index is still negative, we did not find event we wanted so just return error */
			if (ntv_idx < 0) {
				SUBDBG("papi_event_code: %#x not found in native event tables\n", native[i].ni_papi_code);
				continue;
			}

			/* this native index is positive so there was a mask with the event, the ntv_idx identifies which native event to use */
			ntv_evt = (struct native_event_t *)(&(pe_ctx->event_table->native_events[ntv_idx]));
			SUBDBG("ntv_evt: %p\n", ntv_evt);

			SUBDBG("i: %d, pe_ctx->event_table->num_native_events: %d\n", i, pe_ctx->event_table->num_native_events);

			/* Move this events hardware config values and other attributes to the perf_events attribute structure */
			memcpy (&pe_ctl->events[i].attr, &ntv_evt->attr, sizeof(perf_event_attr_t));

			/* may need to update the attribute structure with information from event set level domain settings (values set by PAPI_set_domain) */
			/* only done if the event mask which controls each counting domain was not provided */

			/* get pointer to allocated name, will be NULL when adding preset events to event set */
			char *aName = ntv_evt->allocated_name;
			if ((aName == NULL)  ||  (strstr(aName, ":u=") == NULL)) {
				SUBDBG("set exclude_user attribute from eventset level domain flags, encode: %d, eventset: %d\n", pe_ctl->events[i].attr.exclude_user, !(pe_ctl->domain & PAPI_DOM_USER));
				pe_ctl->events[i].attr.exclude_user = !(pe_ctl->domain & PAPI_DOM_USER);
			}
			if ((aName == NULL)  ||  (strstr(aName, ":k=") == NULL)) {
				SUBDBG("set exclude_kernel attribute from eventset level domain flags, encode: %d, eventset: %d\n", pe_ctl->events[i].attr.exclude_kernel, !(pe_ctl->domain & PAPI_DOM_KERNEL));
				pe_ctl->events[i].attr.exclude_kernel = !(pe_ctl->domain & PAPI_DOM_KERNEL);
			}

			// libpfm4 supports mh (monitor host) and mg (monitor guest) event masks
			// perf_events supports exclude_hv and exclude_idle attributes
			// PAPI_set_domain supports PAPI_DOM_SUPERVISOR and PAPI_DOM_OTHER domain attributes
			// not sure how these perf_event attributes, and PAPI domain attributes relate to each other
			// if that can be figured out then there should probably be code here to set some perf_events attributes based on what was set in a PAPI_set_domain call
			// the code sample below is one possibility
//			if (strstr(ntv_evt->allocated_name, ":mg=") == NULL) {
//				SUBDBG("set exclude_hv attribute from eventset level domain flags, encode: %d, eventset: %d\n", pe_ctl->events[i].attr.exclude_hv, !(pe_ctl->domain & PAPI_DOM_SUPERVISOR));
//				pe_ctl->events[i].attr.exclude_hv = !(pe_ctl->domain & PAPI_DOM_SUPERVISOR);
//			}


			// set the cpu number provided with an event mask if there was one (will be -1 if mask not provided)
			pe_ctl->events[i].cpu = ntv_evt->cpu;
			// if cpu event mask not provided, then set the cpu to use to what may have been set on call to PAPI_set_opt (will still be -1 if not called)
			if (pe_ctl->events[i].cpu == -1) {
				pe_ctl->events[i].cpu = pe_ctl->cpu;
			}
      } else {
    	  /* This case happens when called from _pe_set_overflow and _pe_ctl */
          /* Those callers put things directly into the pe_ctl structure so it is already set for the open call */
      }

      /* Copy the inherit flag into the attribute block that will be passed to the kernel */
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

	/* actually open the events */
	ret = open_pe_events( pe_ctx, pe_ctl );
	if ( ret != PAPI_OK ) {
		SUBDBG("EXIT: open_pe_events returned: %d\n", ret);
      		/* Restore values ? */
		return ret;
	}

	SUBDBG( "EXIT: PAPI_OK\n" );
	return PAPI_OK;
}

/* Set various options on a control state */
static int
_pe_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
   int ret;
   pe_context_t *pe_ctx = ( pe_context_t *) ctx;
   pe_control_t *pe_ctl = NULL;

   switch ( code ) {
      case PAPI_MULTIPLEX:
	   pe_ctl = ( pe_control_t * ) ( option->multiplex.ESI->ctl_state );
	   ret = check_permissions( pe_ctl->tid, pe_ctl->cpu, pe_ctl->domain,
				    pe_ctl->granularity,
				    1, pe_ctl->inherit );
           if (ret != PAPI_OK) {
	      return ret;
	   }

	   /* looks like we are allowed, so set multiplexed attribute */
	   pe_ctl->multiplexed = 1;
	   ret = _pe_update_control_state( pe_ctl, NULL,
						pe_ctl->num_events, pe_ctx );
	   if (ret != PAPI_OK) {
	      pe_ctl->multiplexed = 0;
	   }
	   return ret;

      case PAPI_ATTACH:
	   pe_ctl = ( pe_control_t * ) ( option->attach.ESI->ctl_state );
	   ret = check_permissions( option->attach.tid, pe_ctl->cpu,
				  pe_ctl->domain, pe_ctl->granularity,
				  pe_ctl->multiplexed,
				    pe_ctl->inherit );
	   if (ret != PAPI_OK) {
	      return ret;
	   }

	   pe_ctl->attached = 1;
	   pe_ctl->tid = option->attach.tid;

	   /* If events have been already been added, something may */
	   /* have been done to the kernel, so update */
	   ret =_pe_update_control_state( pe_ctl, NULL,
						pe_ctl->num_events, pe_ctx);

	   return ret;

      case PAPI_DETACH:
	   pe_ctl = ( pe_control_t *) ( option->attach.ESI->ctl_state );

	   pe_ctl->attached = 0;
	   pe_ctl->tid = 0;

	   return PAPI_OK;

      case PAPI_CPU_ATTACH:
	   pe_ctl = ( pe_control_t *) ( option->cpu.ESI->ctl_state );
	   ret = check_permissions( pe_ctl->tid, option->cpu.cpu_num,
				    pe_ctl->domain, pe_ctl->granularity,
				    pe_ctl->multiplexed,
				    pe_ctl->inherit );
           if (ret != PAPI_OK) {
	       return ret;
	   }
	   /* looks like we are allowed so set cpu number */

	   pe_ctl->cpu = option->cpu.cpu_num;

	   return PAPI_OK;

      case PAPI_DOMAIN:
	   pe_ctl = ( pe_control_t *) ( option->domain.ESI->ctl_state );
	   ret = check_permissions( pe_ctl->tid, pe_ctl->cpu,
				    option->domain.domain,
				    pe_ctl->granularity,
				    pe_ctl->multiplexed,
				    pe_ctl->inherit );
           if (ret != PAPI_OK) {
	      return ret;
	   }
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
		   pe_ctl->cpu=_papi_getcpu();
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
	   ret = check_permissions( pe_ctl->tid, pe_ctl->cpu, pe_ctl->domain,
				  pe_ctl->granularity, pe_ctl->multiplexed,
				    option->inherit.inherit );
           if (ret != PAPI_OK) {
	      return ret;
	   }
	   /* looks like we are allowed, so set the requested inheritance */
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
#if 0
	   pe_ctl = (pe_control_t *) (option->address_range.ESI->ctl_state);
	   ret = set_default_domain( pe_ctl, option->address_range.domain );
	   if ( ret != PAPI_OK ) {
	      return ret;
	   }
	   set_drange( pe_ctx, pe_ctl, option );
	   return PAPI_OK;
#endif
      case PAPI_INSTR_ADDRESS:
	   return PAPI_ENOSUPP;
#if 0
	   pe_ctl = (pe_control_t *) (option->address_range.ESI->ctl_state);
	   ret = set_default_domain( pe_ctl, option->address_range.domain );
	   if ( ret != PAPI_OK ) {
	      return ret;
	   }
	   set_irange( pe_ctx, pe_ctl, option );
	   return PAPI_OK;
#endif

      case PAPI_DEF_ITIMER:
	   /* What should we be checking for here?                   */
	   /* This seems like it should be OS-specific not component */
	   /* specific.                                              */

	   return PAPI_OK;

      case PAPI_DEF_MPX_NS:
	   /* Defining a given ns per set is not current supported */
	   return PAPI_ENOSUPP;

      case PAPI_DEF_ITIMER_NS:
	   /* We don't support this... */
	   return PAPI_OK;

      default:
	   return PAPI_ENOSUPP;
   }
}


/* Initialize a new control state */
static int
_pe_init_control_state( hwd_control_state_t *ctl )
{
	pe_control_t *pe_ctl = ( pe_control_t *) ctl;

	/* clear the contents */
	memset( pe_ctl, 0, sizeof ( pe_control_t ) );

	/* Set the domain */
	_pe_set_domain( ctl, _perf_event_vector.cmp_info.default_domain );

	/* default granularity */
	pe_ctl->granularity= _perf_event_vector.cmp_info.default_granularity;

	/* overflow signal */
	pe_ctl->overflow_signal=_perf_event_vector.cmp_info.hardware_intr_sig;

	pe_ctl->cidx=our_cidx;

	/* Set cpu number in the control block to show events */
	/* are not tied to specific cpu                       */
	pe_ctl->cpu = -1;

	return PAPI_OK;
}


/****************** EVENT NAME HANDLING CODE *****************/

static int
_pe_ntv_enum_events( unsigned int *PapiEventCode, int modifier )
{
	return _pe_libpfm4_ntv_enum_events(PapiEventCode, modifier, our_cidx,
			&perf_native_event_table);
}

static int
_pe_ntv_name_to_code( const char *name, unsigned int *event_code)
{
	return _pe_libpfm4_ntv_name_to_code(name,event_code, our_cidx,
			&perf_native_event_table);
}

static int
_pe_ntv_code_to_name(unsigned int EventCode,
			char *ntv_name, int len)
{
	return _pe_libpfm4_ntv_code_to_name(EventCode,
					ntv_name, len,
					&perf_native_event_table);
}

static int
_pe_ntv_code_to_descr( unsigned int EventCode,
			char *ntv_descr, int len)
{

	return _pe_libpfm4_ntv_code_to_descr(EventCode,ntv_descr,len,
					&perf_native_event_table);
}

static int
_pe_ntv_code_to_info(unsigned int EventCode,
			PAPI_event_info_t *info) {

	return _pe_libpfm4_ntv_code_to_info(EventCode, info,
					&perf_native_event_table);
}


/*********************** SAMPLING / PROFILING *******************/


/* Find a native event specified by a profile index */
static int
find_profile_index( EventSetInfo_t *ESI, int evt_idx, int *flags,
		unsigned int *native_index, int *profile_index )
{
	int pos, esi_index, count;

	for ( count = 0; count < ESI->profile.event_counter; count++ ) {
		esi_index = ESI->profile.EventIndex[count];
		pos = ESI->EventInfoArray[esi_index].pos[0];

		if ( pos == evt_idx ) {
			*profile_index = count;
			*native_index = ESI->NativeInfoArray[pos].ni_event &
					PAPI_NATIVE_AND_MASK;
			*flags = ESI->profile.flags;
			SUBDBG( "Native event %d is at profile index %d, flags %d\n",
				*native_index, *profile_index, *flags );
			return PAPI_OK;
		}
	}
	PAPIERROR( "wrong count: %d vs. ESI->profile.event_counter %d",
			count, ESI->profile.event_counter );
	return PAPI_EBUG;
}


/* What exactly does this do? */
static int
process_smpl_buf( int evt_idx, ThreadInfo_t **thr, int cidx )
{
	int ret, flags, profile_index;
	unsigned native_index;
	pe_control_t *ctl;

	ret = find_profile_index( ( *thr )->running_eventset[cidx], evt_idx,
			&flags, &native_index, &profile_index );
	if ( ret != PAPI_OK ) {
		return ret;
	}

	ctl= (*thr)->running_eventset[cidx]->ctl_state;

	mmap_read( cidx, thr, &(ctl->events[evt_idx]), profile_index );

	return PAPI_OK;
}

/*
 * This function is used when hardware overflows are working or when
 * software overflows are forced
 */

static void
_pe_dispatch_timer( int n, hwd_siginfo_t *info, void *uc)
{
	( void ) n;                           /*unused */
	_papi_hwi_context_t hw_context;
	int found_evt_idx = -1, fd = info->si_fd;
	vptr_t address;
	ThreadInfo_t *thread = _papi_hwi_lookup_thread( 0 );
	int i;
	pe_control_t *ctl;
	int cidx = _perf_event_vector.cmp_info.CmpIdx;

	if ( thread == NULL ) {
		PAPIERROR( "thread == NULL in _papi_pe_dispatch_timer for fd %d!", fd );
		return;
	}

	if ( thread->running_eventset[cidx] == NULL ) {
		PAPIERROR( "thread->running_eventset == NULL in "
				"_papi_pe_dispatch_timer for fd %d!",fd );
		return;
	}

	if ( thread->running_eventset[cidx]->overflow.flags == 0 ) {
		PAPIERROR( "thread->running_eventset->overflow.flags == 0 in "
			"_papi_pe_dispatch_timer for fd %d!", fd );
		return;
	}

	hw_context.si = info;
	hw_context.ucontext = ( hwd_ucontext_t * ) uc;

	if ( thread->running_eventset[cidx]->overflow.flags &
			PAPI_OVERFLOW_FORCE_SW ) {
		address = GET_OVERFLOW_ADDRESS( hw_context );
		_papi_hwi_dispatch_overflow_signal( ( void * ) &hw_context,
					address, NULL, 0,
					0, &thread, cidx );
		return;
	}

	if ( thread->running_eventset[cidx]->overflow.flags !=
		PAPI_OVERFLOW_HARDWARE ) {
			PAPIERROR( "thread->running_eventset->overflow.flags "
				"is set to something other than "
				"PAPI_OVERFLOW_HARDWARE or "
				"PAPI_OVERFLOW_FORCE_SW for fd %d (%#x)",
				fd,
				thread->running_eventset[cidx]->overflow.flags);
	}

	/* convoluted way to get ctl */
	ctl= thread->running_eventset[cidx]->ctl_state;

	/* See if the fd is one that's part of the this thread's context */
	for( i=0; i < ctl->num_events; i++ ) {
		if ( fd == ctl->events[i].event_fd ) {
			found_evt_idx = i;
			break;
		}
	}

	if ( found_evt_idx == -1 ) {
		PAPIERROR( "Unable to find fd %d among the open event fds "
				"_papi_hwi_dispatch_timer!", fd );
		return;
	}

	if (ioctl( fd, PERF_EVENT_IOC_DISABLE, NULL ) == -1 ) {
		PAPIERROR("ioctl(PERF_EVENT_IOC_DISABLE) failed");
	}

	if ( ( thread->running_eventset[cidx]->state & PAPI_PROFILING ) &&
		!( thread->running_eventset[cidx]->profile.flags &
		PAPI_PROFIL_FORCE_SW ) ) {
		process_smpl_buf( found_evt_idx, &thread, cidx );
	}
	else {
		uint64_t ip;
		unsigned int head;
		pe_event_info_t *pe = &(ctl->events[found_evt_idx]);
		unsigned char *data = ((unsigned char*)pe->mmap_buf) + getpagesize(  );

	/*
	* Read up the most recent IP from the sample in the mmap buffer.  To
	* do this, we make the assumption that all of the records in the
	* mmap buffer are the same size, and that they all contain the IP as
	* their only record element.  This means that we can use the
	* data_head element from the user page and move backward one record
	* from that point and read the data.  Since we don't actually need
	* to access the header of the record, we can just subtract 8 (size
	* of the IP) from data_head and read up that word from the mmap
	* buffer.  After we subtract 8, we account for mmap buffer wrapping
	* by AND'ing this offset with the buffer mask.
	*/
		head = mmap_read_head( pe );

		if ( head == 0 ) {
			PAPIERROR( "Attempting to access memory "
				"which may be inaccessable" );
			return;
		}
		ip = *( uint64_t * ) ( data + ( ( head - 8 ) & pe->mask ) );
	/*
	* Update the tail to the current head pointer.
	*
	* Note: that if we were to read the record at the tail pointer,
	* rather than the one at the head (as you might otherwise think
	* would be natural), we could run into problems.  Signals don't
	* stack well on Linux, particularly if not using RT signals, and if
	* they come in rapidly enough, we can lose some.  Overtime, the head
	* could catch up to the tail and monitoring would be stopped, and
	* since no more signals are coming in, this problem will never be
	* resolved, resulting in a complete loss of overflow notification
	* from that point on.  So the solution we use here will result in
	* only the most recent IP value being read every time there are two
	* or more samples in the buffer (for that one overflow signal).  But
	* the handler will always bring up the tail, so the head should
	* never run into the tail.
	*/
		mmap_write_tail( pe, head );

	/*
	* The fourth parameter is supposed to be a vector of bits indicating
	* the overflowed hardware counters, but it's not really clear that
	* it's useful, because the actual hardware counters used are not
	* exposed to the PAPI user.  For now, I'm just going to set the bit
	* that indicates which event register in the array overflowed.  The
	* result is that the overflow vector will not be identical to the
	* perfmon implementation, and part of that is due to the fact that
	* which hardware register is actually being used is opaque at the
	* user level (the kernel event dispatcher hides that info).
	*/

		_papi_hwi_dispatch_overflow_signal( ( void * ) &hw_context,
					( vptr_t ) ( unsigned long ) ip,
					NULL, ( 1 << found_evt_idx ), 0,
					&thread, cidx );

	}

	/* Restart the counters */
	if (ioctl( fd, PERF_EVENT_IOC_REFRESH, PAPI_REFRESH_VALUE ) == -1) {
		PAPIERROR("overflow refresh failed");
	}
}

/* Stop profiling */
/* FIXME: does this actually stop anything? */
/* It looks like it is only actually called from PAPI_stop() */
/* So the event will be destroyed soon after anyway. */
static int
_pe_stop_profiling( ThreadInfo_t *thread, EventSetInfo_t *ESI )
{
	int i, ret = PAPI_OK;
	pe_control_t *ctl;
	int cidx;

	ctl=ESI->ctl_state;

	cidx=ctl->cidx;

	/* Loop through all of the events and process those which have mmap */
	/* buffers attached.                                                */
	for ( i = 0; i < ctl->num_events; i++ ) {
		/* Use the mmap_buf field as an indicator */
		/* of this fd being used for profiling.   */
		if ( ctl->events[i].profiling ) {
			/* Process any remaining samples in the sample buffer */
			ret = process_smpl_buf( i, &thread, cidx );
			if ( ret ) {
				PAPIERROR( "process_smpl_buf returned error %d", ret );
				return ret;
			}
			ctl->events[i].profiling=0;
		}
	}

	return ret;
}

/* Set up an event to cause overflow */
/* If threshold==0 then disable overflow for that event */
static int
_pe_set_overflow( EventSetInfo_t *ESI, int EventIndex, int threshold )
{
	SUBDBG("ENTER: ESI: %p, EventIndex: %d, threshold: %d\n",
		ESI, EventIndex, threshold);

	pe_context_t *ctx;
	pe_control_t *ctl = (pe_control_t *) ( ESI->ctl_state );
	int i, evt_idx, found_non_zero_sample_period = 0, retval = PAPI_OK;
	int cidx;

	cidx = ctl->cidx;
	ctx = ( pe_context_t *) ( ESI->master->context[cidx] );

	/* pos[0] is the first native event */
	/* derived events might be made up of multiple native events */
	evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

	SUBDBG("Attempting to set overflow for index %d (%d) of EventSet %d\n",
		evt_idx,EventIndex,ESI->EventSetIndex);

	if (evt_idx<0) {
		SUBDBG("EXIT: evt_idx: %d\n", evt_idx);
		return PAPI_EINVAL;
	}

	/* It's an error to disable overflow if it wasn't set in the	*/
	/* first place.							*/
	if (( threshold == 0 ) &&
		( ctl->events[evt_idx].attr.sample_period == 0 ) ) {
			SUBDBG("EXIT: PAPI_EINVAL, Tried to clear "
				"sample threshold when it was not set\n");
			return PAPI_EINVAL;
	}

	/* Set the sample period to threshold */
	ctl->events[evt_idx].attr.sample_period = threshold;

	if (threshold == 0) {
		ctl->events[evt_idx].sampling = 0;
	}
	else {
		ctl->events[evt_idx].sampling = 1;

		/* Setting wakeup_events to one means issue a wakeup on every */
		/* counter overflow (not mmap page overflow).                 */
		ctl->events[evt_idx].attr.wakeup_events = 1;
		/* We need the IP to pass to the overflow handler */
		ctl->events[evt_idx].attr.sample_type = PERF_SAMPLE_IP;
	}


	/* Check to see if any events in the EventSet are setup to sample */
	/* Do we actually handle multiple overflow events at once? --vmw  */
	for ( i = 0; i < ctl->num_events; i++ ) {
		if ( ctl->events[i].attr.sample_period ) {
			found_non_zero_sample_period = 1;
			break;
		}
	}

	if ( found_non_zero_sample_period ) {
		/* turn on internal overflow flag for this event set */
		ctl->overflow = 1;

		/* Enable the signal handler */
		retval = _papi_hwi_start_signal(
				    ctl->overflow_signal,
				    1, ctl->cidx );
		if (retval != PAPI_OK) {
			SUBDBG("Call to _papi_hwi_start_signal "
				"returned: %d\n", retval);
		}
	} else {

		/* turn off internal overflow flag for this event set */
		ctl->overflow = 0;

		/* Remove the signal handler, if there are no remaining */
		/* non-zero sample_periods set                          */
		retval = _papi_hwi_stop_signal(ctl->overflow_signal);
		if ( retval != PAPI_OK ) {
			SUBDBG("Call to _papi_hwi_stop_signal "
				"returned: %d\n", retval);
			return retval;
		}
	}

	retval = _pe_update_control_state( ctl, NULL,
				((pe_control_t *)(ESI->ctl_state) )->num_events,
				ctx );

	SUBDBG("EXIT: return: %d\n", retval);

	return retval;
}

/* Enable/disable profiling */
/* If threshold is zero, we disable */
static int
_pe_set_profile( EventSetInfo_t *ESI, int EventIndex, int threshold )
{
	int ret;
	int evt_idx;
	pe_control_t *ctl = ( pe_control_t *) ( ESI->ctl_state );

	/* Since you can't profile on a derived event,	*/
	/* the event is always the first and only event	*/
	/* in the native event list.			*/
	evt_idx = ESI->EventInfoArray[EventIndex].pos[0];

	/* If threshold is zero we want to *disable*    */
	/* profiling on the event                       */
	if ( threshold == 0 ) {
//		SUBDBG( "MUNMAP(%p,%"PRIu64")\n",
//			ctl->events[evt_idx].mmap_buf,
//			( uint64_t ) ctl->events[evt_idx].nr_mmap_pages *
//			getpagesize() );

//		if ( ctl->events[evt_idx].mmap_buf ) {
//			munmap( ctl->events[evt_idx].mmap_buf,
//				ctl->events[evt_idx].nr_mmap_pages *
//				getpagesize() );
//		}
//		ctl->events[evt_idx].mmap_buf = NULL;
//		ctl->events[evt_idx].nr_mmap_pages = 0;

		/* no longer sample on IP */
		ctl->events[evt_idx].attr.sample_type &= ~PERF_SAMPLE_IP;

		/* Clear any residual overflow flags */
		/* ??? old warning says "This should be handled somewhere else" */
		ESI->state &= ~( PAPI_OVERFLOWING );
		ESI->overflow.flags &= ~( PAPI_OVERFLOW_HARDWARE );

		ctl->events[evt_idx].profiling=0;

	} else {

		/* Otherwise, we are *enabling* profiling */

		/* Look up the native event code */

		if ( ESI->profile.flags & (PAPI_PROFIL_DATA_EAR |
						PAPI_PROFIL_INST_EAR)) {
			/* Not supported yet... */
			return PAPI_ENOSUPP;
		}

		if ( ESI->profile.flags & PAPI_PROFIL_RANDOM ) {
			/* This requires an ability to randomly alter the    */
			/* sample_period within a given range.		     */
			/* Linux currently does not have this ability. FIXME */
			return PAPI_ENOSUPP;
		}
		ctl->events[evt_idx].profiling=1;
	}

	ret = _pe_set_overflow( ESI, EventIndex, threshold );
	if ( ret != PAPI_OK ) return ret;

	return PAPI_OK;
}


/************ INITIALIZATION / SHUTDOWN CODE *********************/


/* Shutdown the perf_event component */
static int
_pe_shutdown_component( void ) {

	/* deallocate our event table */
	_pe_libpfm4_shutdown(&_perf_event_vector, &perf_native_event_table);

	/* Shutdown libpfm4 */
	_papi_libpfm4_shutdown(&_perf_event_vector);

	return PAPI_OK;
}


#if defined(__aarch64__)
/* Check access PMU counter from User space for arm64 support */
static int _pe_detect_arm64_access(void) {

	FILE *fff;
	int perf_user_access;
	int retval;

	fff=fopen("/proc/sys/kernel/perf_user_access","r");
	if (fff==NULL) {
		return 0;
	}

	/* 1 means you can access PMU counter from User space */
	/* 0 means you can not access PMU counter from User space */
	retval=fscanf(fff,"%d",&perf_user_access);
	if (retval!=1) fprintf(stderr,"Error reading /proc/sys/kernel/perf_user_access\n");
	fclose(fff);

	return perf_user_access;
}
#endif

/* Check the mmap page for rdpmc support */
static int _pe_detect_rdpmc(void) {

	struct perf_event_attr pe;
	int fd,rdpmc_exists=1;
	void *addr;
	struct perf_event_mmap_page *our_mmap;
	int page_size=getpagesize();
#if defined(__aarch64__)
	int retval;
#endif

#if defined(__i386__) || defined (__x86_64__) || defined(__aarch64__)
#else
	/* We support rdpmc on x86 and arm64 for now */
        return 0;
#endif

	/* There were various subtle bugs in rdpmc support before	*/
	/* the Linux 4.13 release.					*/
	if (_papi_os_info.os_version < LINUX_VERSION(4,13,0)) {
		return 0;
	}

#if defined(__aarch64__)
	/* Detect if we can use PMU counter from User space for arm64 */
	retval = _pe_detect_arm64_access();
	if (retval == 0) {
		return 0;
	}
#endif

	/* Create a fake instructions event so we can read a mmap page */
	memset(&pe,0,sizeof(struct perf_event_attr));

	pe.type=PERF_TYPE_HARDWARE;
	pe.size=sizeof(struct perf_event_attr);
	pe.config=PERF_COUNT_HW_INSTRUCTIONS;
#if defined(__aarch64__)
	arm64_request_user_access(&pe);
#endif
	pe.exclude_kernel=1;
	pe.disabled=1;

	perf_event_dump_attr(&pe,0,-1,-1,0);
	fd=sys_perf_event_open(&pe,0,-1,-1,0);

	/* This hopefully won't happen? */
	/* Though there is a chance this is the first */
	/* attempt to open a perf_event */
	if (fd<0) {
		SUBDBG("FAILED perf_event_open trying to detect rdpmc support");
		return PAPI_ESYS;
	}

	/* create the mmap page */
	addr=mmap(NULL, page_size, PROT_READ, MAP_SHARED,fd,0);
	if (addr == MAP_FAILED) {
		SUBDBG("FAILED mmap trying to detect rdpmc support");
		close(fd);
		return PAPI_ESYS;
	}

	/* get the rdpmc info from the mmap page */
	our_mmap=(struct perf_event_mmap_page *)addr;

	/* If cap_usr_rdpmc bit is set to 1, we have support! */
	if (our_mmap->cap_usr_rdpmc!=0) {
		rdpmc_exists=1;
	}
	else if ((!our_mmap->cap_bit0_is_deprecated) && (our_mmap->cap_bit0)) {
		/* 3.4 to 3.11 had somewhat broken rdpmc support */
		/* This convoluted test is the "official" way to detect this */
		/* To make things easier we don't support these kernels */
		rdpmc_exists=0;
	}
	else {
		rdpmc_exists=0;
	}

	/* close the fake event */
	munmap(addr,page_size);
	close(fd);

	return rdpmc_exists;

}


static int
_pe_handle_paranoid(papi_vector_t *component) {

	FILE *fff;
	int paranoid_level;
	int retval;
   char *strCpy;

	/* The is the official way to detect if perf_event support exists */
	/* The file is called perf_counter_paranoid on 2.6.31             */
	/* currently we are lazy and do not support 2.6.31 kernels        */

	fff=fopen("/proc/sys/kernel/perf_event_paranoid","r");
	if (fff==NULL) {
		strCpy=strncpy(component->cmp_info.disabled_reason,
			"perf_event support not detected",PAPI_MAX_STR_LEN);
      if (strCpy == NULL) HANDLE_STRING_ERROR;
		return PAPI_ECMP;
	}

	/* 3 (vendor patch) means completely disabled */
	/* 2 means no kernel measurements allowed   */
	/* 1 means normal counter access            */
	/* 0 means you can access CPU-specific data */
	/* -1 means no restrictions                 */
	retval=fscanf(fff,"%d",&paranoid_level);
	if (retval!=1) fprintf(stderr,"Error reading paranoid level\n");
	fclose(fff);

	if (paranoid_level==3) {
		strCpy=strncpy(component->cmp_info.disabled_reason,
			"perf_event support disabled by Linux with paranoid=3",PAPI_MAX_STR_LEN);
      if (strCpy == NULL) HANDLE_STRING_ERROR;
		return PAPI_ECMP;
	}

	if ((paranoid_level==2) && (getuid()!=0)) {
		SUBDBG("/proc/sys/kernel/perf_event_paranoid prohibits kernel counts");
		component->cmp_info.available_domains &=~PAPI_DOM_KERNEL;
	}

	return PAPI_OK;

}

#if (OBSOLETE_WORKAROUNDS==1)
/* Version based workarounds */
/* perf_event has many bugs */
/* PAPI has to work around a number of them, but for the most part */
/* all of those were fixed by Linux 2.6.34 (May 2010) */
/* Unfortunately it's not easy to auto-detect for these so we were */
/* going by uname() version number */
/* To complicate things, some vendors like Redhat backport fixes */
/* So even though their kernel reports as 2.6.32 it has the fixes */
/* As of PAPI 5.6 we're going to default to disabling the workarounds */
/* I'm going to leave them here, ifdefed out, for the time being */
static int
_pe_version_workarounds(papi_vector_t *component) {

	/* Kernel multiplexing is broken prior to kernel 2.6.34 */
	/* The fix was probably git commit:                     */
	/*     45e16a6834b6af098702e5ea6c9a40de42ff77d8         */
	if (_papi_os_info.os_version < LINUX_VERSION(2,6,34)) {
		component->cmp_info.kernel_multiplex = 0;
		component->cmp_info.num_mpx_cntrs = PAPI_MAX_SW_MPX_EVENTS;
	}

	/* Check that processor is supported */
	if (processor_supported(_papi_hwi_system_info.hw_info.vendor,
			_papi_hwi_system_info.hw_info.cpuid_family)!=PAPI_OK) {
		fprintf(stderr,"warning, your processor is unsupported\n");
		/* should not return error, as software events should still work */
	}

	/* Update the default function pointers */
	/* Based on features/bugs               */
	if (bug_sync_read()) {
		component->read = _pe_read_bug_sync;
	}

	return PAPI_OK;

}

#endif




/* Initialize the perf_event component */
static int
_pe_init_component( int cidx )
{

	int retval;
   char *strCpy;

	our_cidx=cidx;

	/* Update component behavior based on paranoid setting */
	retval=_pe_handle_paranoid(_papi_hwd[cidx]);
   
	if (retval!=PAPI_OK) goto fn_fail; // disabled_reason handled by _pe_handle_paranoid.

#if (OBSOLETE_WORKAROUNDS==1)
	/* Handle any kernel version related workarounds */
	_pe_version_workarounds(_papi_hwd[cidx]);
#endif

	/* Setup mmtimers, if appropriate */
	retval=mmtimer_setup();
	if (retval) {
		strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
			"Error initializing mmtimer",PAPI_MAX_STR_LEN);
      if (strCpy == NULL) HANDLE_STRING_ERROR;
		goto fn_fail;
	}

	/* Set the overflow signal */
	_papi_hwd[cidx]->cmp_info.hardware_intr_sig = SIGRTMIN + 2;

	/* Run Vendor-specific fixups */
	pe_vendor_fixups(_papi_hwd[cidx]);

	/* Detect if we can use rdpmc (or equivalent) */
	retval=_pe_detect_rdpmc();
	_papi_hwd[cidx]->cmp_info.fast_counter_read = retval;
	if (retval < 0 ) {
		/* Don't actually fail here, as could be a surivable bug? */
		/* If perf_event_open/mmap truly are failing we will      */
		/* likely catch it pretty quickly elsewhere.              */
		_papi_hwd[cidx]->cmp_info.fast_counter_read = 0;
	}

#if (USE_PERFEVENT_RDPMC==1)

#else
	/* Force fast_counter_read off if --enable-perfevent-rdpmc=no */
	_papi_hwd[cidx]->cmp_info.fast_counter_read = 0;
#endif

	/* Run the libpfm4-specific setup */
	retval = _papi_libpfm4_init(_papi_hwd[cidx]);
	if (retval) {
		strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
			"Error initializing libpfm4",PAPI_MAX_STR_LEN);
      if (strCpy == NULL) HANDLE_STRING_ERROR;
		goto fn_fail;

	}

	/* Now that libpfm4 is initialized */
	/* Try to setup the perf_event component events */

	retval = _pe_libpfm4_init(_papi_hwd[cidx], cidx,
				&perf_native_event_table,
				PMU_TYPE_CORE | PMU_TYPE_OS);
	if (retval) {
		switch(retval) {
			case PAPI_ENOMEM:
				strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
					"Error libpfm4 memory allocation",
					PAPI_MAX_STR_LEN);
            if (strCpy == NULL) HANDLE_STRING_ERROR;
				break;
			case PAPI_ENOSUPP:
				strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
					"Error libpfm4 no PMUs found",
					PAPI_MAX_STR_LEN);
            if (strCpy == NULL) HANDLE_STRING_ERROR;
				break;
			case PAPI_ECMP:
				strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
					"Error libpfm4 no default PMU found",
					PAPI_MAX_STR_LEN);
            if (strCpy == NULL) HANDLE_STRING_ERROR;
				break;
			case PAPI_ECOUNT:
				strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
					"Error libpfm4 too many default PMUs found",
					PAPI_MAX_STR_LEN);
            if (strCpy == NULL) HANDLE_STRING_ERROR;
				break;
			case PAPI_ENOEVNT:
				strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
					"Error loading preset events",
					PAPI_MAX_STR_LEN);
            if (strCpy == NULL) HANDLE_STRING_ERROR;
				break;
			default:
				strCpy=strncpy(_papi_hwd[cidx]->cmp_info.disabled_reason,
					"Unknown libpfm4 related error",
					PAPI_MAX_STR_LEN);
            if (strCpy == NULL) HANDLE_STRING_ERROR;

		}
        goto fn_fail;
	}

	/* Detect NMI watchdog which can steal counters */
	/* FIXME: on Intel we should also halve the count if SMT enabled */
	if (_linux_detect_nmi_watchdog()) {
		if (_papi_hwd[cidx]->cmp_info.num_cntrs>0) {
			_papi_hwd[cidx]->cmp_info.num_cntrs--;
		}
		SUBDBG("The Linux nmi_watchdog is using one of the performance "
			"counters, reducing the total number available.\n");
	}

	/* check for exclude_guest issue */
	check_exclude_guest();

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;

}



/* Our component vector */

papi_vector_t _perf_event_vector = {
   .cmp_info = {
       /* component information (unspecified values initialized to 0) */
      .name = "perf_event",
      .short_name = "perf",
      .version = "5.0",
      .description = "Linux perf_event CPU counters",

      .default_domain = PAPI_DOM_USER,
      .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL | PAPI_DOM_SUPERVISOR,
      .default_granularity = PAPI_GRN_THR,
      .available_granularities = PAPI_GRN_THR | PAPI_GRN_SYS,

      .hardware_intr = 1,
      .kernel_profile = 1,

      /* component specific cmp_info initializations */
      .fast_virtual_timer = 0,
      .attach = 1,
      .attach_must_ptrace = 1,
      .cpu = 1,
      .inherit = 1,
      .cntr_umasks = 1,

	.kernel_multiplex = 1,
	.num_mpx_cntrs = PERF_EVENT_MAX_MPX_COUNTERS,


  },

  /* sizes of framework-opaque component-private structures */
  .size = {
      .context = sizeof ( pe_context_t ),
      .control_state = sizeof ( pe_control_t ),
      .reg_value = sizeof ( int ),
      .reg_alloc = sizeof ( int ),
  },

  /* function pointers in this component */
  .init_component =        _pe_init_component,
  .shutdown_component =    _pe_shutdown_component,
  .init_thread =           _pe_init_thread,
  .init_control_state =    _pe_init_control_state,
  .dispatch_timer =        _pe_dispatch_timer,

  /* function pointers from the shared perf_event lib */
  .start =                 _pe_start,
  .stop =                  _pe_stop,
  .read =                  _pe_read,
  .shutdown_thread =       _pe_shutdown_thread,
  .ctl =                   _pe_ctl,
  .update_control_state =  _pe_update_control_state,
  .set_domain =            _pe_set_domain,
  .reset =                 _pe_reset,
  .set_overflow =          _pe_set_overflow,
  .set_profile =           _pe_set_profile,
  .stop_profiling =        _pe_stop_profiling,
  .write =                 _pe_write,


  /* from counter name mapper */
  .ntv_enum_events =   _pe_ntv_enum_events,
  .ntv_name_to_code =  _pe_ntv_name_to_code,
  .ntv_code_to_name =  _pe_ntv_code_to_name,
  .ntv_code_to_descr = _pe_ntv_code_to_descr,
  .ntv_code_to_info =  _pe_ntv_code_to_info,
};
