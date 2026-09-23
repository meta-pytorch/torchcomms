/**
 * @file    linux-mx.c
 * @brief A component for Myricom MX (Myrinet Express)
 */


#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>

#define MX_MAX_COUNTERS 100
#define MX_MAX_COUNTER_TERMS  MX_MAX_COUNTERS

#define LINELEN 128

typedef struct MX_register
{
	/* indicate which counters this event can live on */
	unsigned int selector;
} MX_register_t;

typedef struct MX_native_event_entry
{
	/* description of the resources required by this native event */
	MX_register_t resources;
	/* If it exists, then this is the name of this event */
	char *name;
	/* If it exists, then this is the description of this event */
	char *description;
} MX_native_event_entry_t;

typedef struct MX_reg_alloc
{
	MX_register_t ra_bits;
} MX_reg_alloc_t;

typedef struct MX_control_state
{
  long long start_count[MX_MAX_COUNTERS];
  long long current_count[MX_MAX_COUNTERS];
  long long difference[MX_MAX_COUNTERS];
  int which_counter[MX_MAX_COUNTERS];
  int num_events;
} MX_control_state_t;

typedef struct MX_context
{
	MX_control_state_t state;
} MX_context_t;


static const MX_native_event_entry_t mx_native_table[] = {
	{{1,  }, "LANAI_UPTIME", "Lanai uptime (seconds)"},
	{{2,  }, "COUNTERS_UPTIME", "Counters uptime (seconds)"},
	{{3,  }, "BAD_CRC8", "Bad CRC8 (Port 0)"},
	{{4,  }, "BAD_CRC32", "Bad CRC32 (Port 0)"},
	{{5,  }, "UNSTRIPPED_ROUTE", "Unstripped route (Port 0)"},
	{{6,  }, "PKT_DESC_INVALID", "pkt_desc_invalid (Port 0)"},
	{{7,  }, "RECV_PKT_ERRORS", "recv_pkt_errors (Port 0)"},
	{{8,  }, "PKT_MISROUTED", "pkt_misrouted (Port 0)"},
	{{9,  }, "DATA_SRC_UNKNOWN", "data_src_unknown"},
	{{10,  }, "DATA_BAD_ENDPT", "data_bad_endpt"},
	{{11,  }, "DATA_ENDPT_CLOSED", "data_endpt_closed"},
	{{12,  }, "DATA_BAD_SESSION", "data_bad_session"},
	{{13,  }, "PUSH_BAD_WINDOW", "push_bad_window"},
	{{14,  }, "PUSH_DUPLICATE", "push_duplicate"},
	{{15,  }, "PUSH_OBSOLETE", "push_obsolete"},
	{{16,  }, "PUSH_RACE_DRIVER", "push_race_driver"},
	{{17,  }, "PUSH_BAD_SEND_HANDLE_MAGIC", "push_bad_send_handle_magic"},
	{{18,  }, "PUSH_BAD_SRC_MAGIC", "push_bad_src_magic"},
	{{19,  }, "PULL_OBSOLETE", "pull_obsolete"},
	{{20,  }, "PULL_NOTIFY_OBSOLETE", "pull_notify_obsolete"},
	{{21,  }, "PULL_RACE_DRIVER", "pull_race_driver"},
	{{22,  }, "ACK_BAD_TYPE", "ack_bad_type"},
	{{23,  }, "ACK_BAD_MAGIC", "ack_bad_magic"},
	{{24,  }, "ACK_RESEND_RACE", "ack_resend_race"},
	{{25,  }, "LATE_ACK", "Late ack"},
	{{26,  }, "ACK_NACK_FRAMES_IN_PIPE", "ack_nack_frames_in_pipe"},
	{{27,  }, "NACK_BAD_ENDPT", "nack_bad_endpt"},
	{{28,  }, "NACK_ENDPT_CLOSED", "nack_endpt_closed"},
	{{29,  }, "NACK_BAD_SESSION", "nack_bad_session"},
	{{30,  }, "NACK_BAD_RDMAWIN", "nack_bad_rdmawin"},
	{{31,  }, "NACK_EVENTQ_FULL", "nack_eventq_full"},
	{{32,  }, "SEND_BAD_RDMAWIN", "send_bad_rdmawin"},
	{{33,  }, "CONNECT_TIMEOUT", "connect_timeout"},
	{{34,  }, "CONNECT_SRC_UNKNOWN", "connect_src_unknown"},
	{{35,  }, "QUERY_BAD_MAGIC", "query_bad_magic"},
	{{36,  }, "QUERY_TIMED_OUT", "query_timed_out"},
	{{37,  }, "QUERY_SRC_UNKNOWN", "query_src_unknown"},
	{{38,  }, "RAW_SENDS", "Raw sends (Port 0)"},
	{{39,  }, "RAW_RECEIVES", "Raw receives (Port 0)"},
	{{40,  }, "RAW_OVERSIZED_PACKETS", "Raw oversized packets (Port 0)"},
	{{41,  }, "RAW_RECV_OVERRUN", "raw_recv_overrun"},
	{{42,  }, "RAW_DISABLED", "raw_disabled"},
	{{43,  }, "CONNECT_SEND", "connect_send"},
	{{44,  }, "CONNECT_RECV", "connect_recv"},
	{{45,  }, "ACK_SEND", "ack_send (Port 0)"},
	{{46,  }, "ACK_RECV", "ack_recv (Port 0)"},
	{{47,  }, "PUSH_SEND", "push_send (Port 0)"},
	{{48,  }, "PUSH_RECV", "push_recv (Port 0)"},
	{{49,  }, "QUERY_SEND", "query_send (Port 0)"},
	{{50,  }, "QUERY_RECV", "query_recv (Port 0)"},
	{{51,  }, "REPLY_SEND", "reply_send (Port 0)"},
	{{52,  }, "REPLY_RECV", "reply_recv (Port 0)"},
	{{53,  }, "QUERY_UNKNOWN", "query_unknown (Port 0)"},
/*   {{ 54,  }, "QUERY_UNKNOWN", "query_unknown (Port 0)"},*/
	{{55,  }, "DATA_SEND_NULL", "data_send_null (Port 0)"},
	{{56,  }, "DATA_SEND_SMALL", "data_send_small (Port 0)"},
	{{57,  }, "DATA_SEND_MEDIUM", "data_send_medium (Port 0)"},
	{{58,  }, "DATA_SEND_RNDV", "data_send_rndv (Port 0)"},
	{{59,  }, "DATA_SEND_PULL", "data_send_pull (Port 0)"},
	{{60,  }, "DATA_RECV_NULL", "data_recv_null (Port 0)"},
	{{61,  }, "DATA_RECV_SMALL_INLINE", "data_recv_small_inline (Port 0)"},
	{{62,  }, "DATA_RECV_SMALL_COPY", "data_recv_small_copy (Port 0)"},
	{{63,  }, "DATA_RECV_MEDIUM", "data_recv_medium (Port 0)"},
	{{64,  }, "DATA_RECV_RNDV", "data_recv_rndv (Port 0)"},
	{{65,  }, "DATA_RECV_PULL", "data_recv_pull (Port 0)"},
	{{66,  }, "ETHER_SEND_UNICAST_CNT", "ether_send_unicast_cnt (Port 0)"},
	{{67,  }, "ETHER_SEND_MULTICAST_CNT", "ether_send_multicast_cnt (Port 0)"},
	{{68,  }, "ETHER_RECV_SMALL_CNT", "ether_recv_small_cnt (Port 0)"},
	{{69,  }, "ETHER_RECV_BIG_CNT", "ether_recv_big_cnt (Port 0)"},
	{{70,  }, "ETHER_OVERRUN", "ether_overrun"},
	{{71,  }, "ETHER_OVERSIZED", "ether_oversized"},
	{{72,  }, "DATA_RECV_NO_CREDITS", "data_recv_no_credits"},
	{{73,  }, "PACKETS_RECENT", "Packets resent"},
	{{74,  }, "PACKETS_DROPPED", "Packets dropped (data send side)"},
	{{75,  }, "MAPPER_ROUTES_UPDATE", "Mapper routes update"},
	{{76,  }, "ROUTE_DISPERSION", "Route dispersion (Port 0)"},
	{{77,  }, "OUT_OF_SEND_HANDLES", "out_of_send_handles"},
	{{78,  }, "OUT_OF_PULL_HANDLES", "out_of_pull_handles"},
	{{79,  }, "OUT_OF_PUSH_HANDLES", "out_of_push_handles"},
	{{80,  }, "MEDIUM_CONT_RACE", "medium_cont_race"},
	{{81,  }, "CMD_TYPE_UNKNOWN", "cmd_type_unknown"},
	{{82,  }, "UREQ_TYPE_UNKNOWN", "ureq_type_unknown"},
	{{83,  }, "INTERRUPTS_OVERRUN", "Interrupts overrun"},
	{{84,  }, "WAITING_FOR_INTERRUPT_DMA", "Waiting for interrupt DMA"},
	{{85,  }, "WAITING_FOR_INTERRUPT_ACK", "Waiting for interrupt Ack"},
	{{86,  }, "WAITING_FOR_INTERRUPT_TIMER", "Waiting for interrupt Timer"},
	{{87,  }, "SLABS_RECYCLING", "Slabs recycling"},
	{{88,  }, "SLABS_PRESSURE", "Slabs pressure"},
	{{89,  }, "SLABS_STARVATION", "Slabs starvation"},
	{{90,  }, "OUT_OF_RDMA_HANDLES", "out_of_rdma handles"},
	{{91,  }, "EVENTQ_FULL", "eventq_full"},
	{{92,  }, "BUFFER_DROP", "buffer_drop (Port 0)"},
	{{93,  }, "MEMORY_DROP", "memory_drop (Port 0)"},
	{{94,  }, "HARDWARE_FLOW_CONTROL", "Hardware flow control (Port 0)"},
	{{95,  }, "SIMULATED_PACKETS_LOST", "(Devel) Simulated packets lost (Port 0)"},
	{{96,  }, "LOGGING_FRAMES_DUMPED", "(Logging) Logging frames dumped"},
	{{97,  }, "WAKE_INTERRUPTS", "Wake interrupts"},
	{{98,  }, "AVERTED_WAKEUP_RACE", "Averted wakeup race"},
	{{99,  }, "DMA_METADATA_RACE", "Dma metadata race"},
	{{0, }, "", ""}
};

static int num_events=0;
papi_vector_t _mx_vector;

static char mx_counters_exe[BUFSIZ];

static int
read_mx_counters( long long *counters )
{
	FILE *fp;
	char line[LINELEN];
	int i, linenum;

	/* Open a pipe to the mx_counters executable */

	fp = popen( mx_counters_exe, "r" );
	if ( !fp ) {
	   perror( "popen" );
	   return PAPI_ECMP;
	}


	/* A line of output looks something similar to:       */
        /* "    Lanai uptime (seconds):     766268 (0xbb13c)" */

	/* This code may fail if number of ports on card > 1 */

	linenum = 0;
	while ( fgets( line, LINELEN, fp ) ) {
	  //	   printf("%s",line);
	   for(i=0; line[i]!= '\0' && i<LINELEN-1;i++) {

	      /* skip to colon */
	      if (line[i]==':') {

	         /* read in value */
	         if (line[i+1]!='\0') {
		   //	    printf("Line %d trying %s",linenum,&line[i+1]);
	            sscanf(&line[i+1],"%lld",&counters[linenum]);
		    linenum++;
		    break;
		 }
	      }
	   }
	   if (linenum>=MX_MAX_COUNTERS) break;
	}

	pclose( fp );

	return PAPI_OK;
}



/*
 * Component setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int
_mx_init_component( int cidx )
{
    int retval = PAPI_OK;
	FILE *fff;
	char *path;
	int checklen, len, pathlen;

	/* detect if MX available */

	path = getenv("PATH");
	pathlen = strlen(path);
	while(pathlen > 0) {
	   len = strcspn(path, ":");
	   if (len < BUFSIZ) {
	      strncpy(mx_counters_exe, path, len);
	   } else {
	      fff = NULL;
	      break;
	   }
	   mx_counters_exe[len] = '\0';
	   checklen = len + strlen("/mx_counters");
	   if (checklen < BUFSIZ) {
	      strcat(mx_counters_exe, "/mx_counters");
	   } else {
	      fff = NULL;
	      break;
	   }
	   fff = fopen(mx_counters_exe, "r");
	   if (fff != NULL) {
	      checklen = checklen + strlen(" 2> /dev/null");
	      if (checklen < BUFSIZ) {
	         strcat(mx_counters_exe, " 2> /dev/null");
	         break;
	      } else {
	         fclose(fff);
	         fff = NULL;
	         break;
	      }
	   }
	   pathlen = pathlen - len - 1;
	   if (pathlen > 0) {
	      path = path + len + 1;
	   }
	}
	if (fff == NULL) {
	   /* mx_counters not found */
	   strncpy(_mx_vector.cmp_info.disabled_reason,
		   "No MX utilities found",PAPI_MAX_STR_LEN);
       retval = PAPI_ECMP;
       goto fn_fail;
	}
	fclose(fff);

	num_events=MX_MAX_COUNTERS;
	_mx_vector.cmp_info.num_native_events=num_events;

	/* Export the component id */
	_mx_vector.cmp_info.CmpIdx = cidx;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}


/*
 * This is called whenever a thread is initialized
 */
static int
_mx_init_thread( hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
	return PAPI_OK;
}


static int
_mx_shutdown_component(void) 
{
  return PAPI_OK;
}

static int
_mx_shutdown_thread( hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
	return PAPI_OK;
}



/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
static int
_mx_init_control_state( hwd_control_state_t *ctl )
{
	( void ) ctl;			 /*unused */

	return PAPI_OK;
}

static int
_mx_update_control_state( hwd_control_state_t *ctl, NativeInfo_t *native,
			  int count, hwd_context_t *ctx )
{
	( void ) ctx;			 /*unused */
	int i, index;
	
	MX_control_state_t *mx_ctl = (MX_control_state_t *)ctl;

	for(i=0; i<count; i++ ) {
	    index = native[i].ni_event;
	    mx_ctl->which_counter[i]=index;
	    //	    printf("Mapping event# %d to HW counter %d (count=%d)\n",
	    //	   i,index,count); 
	    native[i].ni_position = i;
	}

	mx_ctl->num_events=count;

	return PAPI_OK;
}


static int
_mx_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

        long long mx_counters[MX_MAX_COUNTERS];

	( void ) ctx;			 /*unused */

	MX_control_state_t *mx_ctl = (MX_control_state_t *)ctl;
	int i;

	read_mx_counters( mx_counters );

	//	for(i=0;i<MX_MAX_COUNTERS;i++) printf("%d %lld\n",i,mx_counters[i]);

	for(i=0;i<mx_ctl->num_events;i++) {
           mx_ctl->current_count[i]=
	         mx_counters[mx_ctl->which_counter[i]];
	   mx_ctl->start_count[i]=mx_ctl->current_count[i];
	}

	return PAPI_OK;
}


static int
_mx_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	( void ) ctx;			 /*unused */

        long long mx_counters[MX_MAX_COUNTERS];
	MX_control_state_t *mx_ctl = (MX_control_state_t *)ctl;
	int i;

	read_mx_counters( mx_counters );

	for(i=0;i<mx_ctl->num_events;i++) {
           mx_ctl->current_count[i]=
	         mx_counters[mx_ctl->which_counter[i]];
	}

	return PAPI_OK;
}

static int
_mx_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events,
		 int flags )
{
	( void ) ctx;			 /*unused */
	( void ) flags;			 /*unused */
	int i;
        long long mx_counters[MX_MAX_COUNTERS];

	MX_control_state_t *mx_ctl = (MX_control_state_t *)ctl;

	read_mx_counters( mx_counters );

	for ( i = 0; i < mx_ctl->num_events; i++ ) {
            mx_ctl->current_count[i]=
	         mx_counters[mx_ctl->which_counter[i]];
	    mx_ctl->difference[i] = mx_ctl->current_count[i]-
	                               mx_ctl->start_count[i];
	}
	*events = mx_ctl->difference;

	return PAPI_OK;
}


static int
_mx_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	_mx_start( ctx, ctrl );
	return PAPI_OK;
}

/* Unused write function */
/* static int */
/* _mx_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long *from ) */
/* { */
/* 	( void ) ctx;			 /\*unused *\/ */
/* 	( void ) ctrl;			 /\*unused *\/ */
/* 	( void ) from;			 /\*unused *\/ */

/* 	return PAPI_OK; */
/* } */

/*
 * Functions for setting up various options
 */

/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
static int
_mx_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;			 /*unused */
	( void ) code;			 /*unused */
	( void ) option;		 /*unused */

	return PAPI_OK;
}

/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
static int
_mx_set_domain( hwd_control_state_t * cntrl, int domain )
{
	( void ) cntrl;			 /*unused */
	if ( PAPI_DOM_ALL != domain ) {
		return PAPI_EINVAL;
	}

	return PAPI_OK;
}



static int
_mx_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{

	int event=EventCode;

	if (event >=0 && event < num_events) {
	  strncpy( name, mx_native_table[event].name, len );
	  return PAPI_OK;
	}
	return PAPI_ENOEVNT;


}

static int
_mx_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
    int event=EventCode;

    if (event >=0 && event < num_events) {
       strncpy( name, mx_native_table[event].description, len );
       return PAPI_OK;
    }
    return PAPI_ENOEVNT;
}



static int
_mx_ntv_enum_events( unsigned int *EventCode, int modifier )
{

	if ( modifier == PAPI_ENUM_FIRST ) {
	   if (num_events==0) return PAPI_ENOEVNT;
	   *EventCode = 0;
	   return PAPI_OK;
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode;

		if ( mx_native_table[index + 1].resources.selector ) {
			*EventCode = *EventCode + 1;
			return PAPI_OK;
		} else {
			return PAPI_ENOEVNT;
		}
	}
		
        return PAPI_EINVAL;
}


papi_vector_t _mx_vector = {
	.cmp_info = {
	    .name = "mx",
		.short_name = "mx",
	    .version = "1.4",
	    .description = "Myricom MX (Myrinet Express) statistics",
	    .num_mpx_cntrs = MX_MAX_COUNTERS,
	    .num_cntrs = MX_MAX_COUNTERS,
	    .default_domain = PAPI_DOM_ALL,
	    .default_granularity = PAPI_GRN_SYS,
	    .available_granularities = PAPI_GRN_SYS,
	    .hardware_intr_sig = PAPI_INT_SIGNAL,

	    /* component specific cmp_info initializations */
	    .fast_real_timer = 0,
	    .fast_virtual_timer = 0,
	    .attach = 0,
	    .attach_must_ptrace = 0,
	    .available_domains = PAPI_DOM_ALL,
  },

	/* sizes of framework-opaque component-private structures */
	.size = {
	    .context = sizeof ( MX_context_t ),
	    .control_state = sizeof ( MX_control_state_t ),
	    .reg_value = sizeof ( MX_register_t ),
	    .reg_alloc = sizeof ( MX_reg_alloc_t ),
  },
        /* function pointers in this component */
	.init_thread =          _mx_init_thread,
	.init_component =       _mx_init_component,
	.init_control_state =   _mx_init_control_state,
	.start =                _mx_start,
	.stop =                 _mx_stop,
	.read =                 _mx_read,
	.shutdown_thread =      _mx_shutdown_thread,
	.shutdown_component =   _mx_shutdown_component,
	.ctl =                  _mx_ctl,
	.update_control_state = _mx_update_control_state,
	.set_domain =           _mx_set_domain,
	.reset =                _mx_reset,

	.ntv_enum_events =      _mx_ntv_enum_events,
	.ntv_code_to_name =     _mx_ntv_code_to_name,
	.ntv_code_to_descr =    _mx_ntv_code_to_descr,
};
