/**
 * @file    linux-powercap.c
 * @author  Philip Vaccaro
 * @ingroup papi_components
 * @brief powercap component
 *
 * To work, the powercap kernel module must be loaded.
 */

#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}


typedef struct _powercap_register {
    unsigned int selector;
} _powercap_register_t;

typedef struct _powercap_native_event_entry {
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  int socket_id;
  int component_id;
  int event_id;
  int type;
  int return_type;
  _powercap_register_t resources;
} _powercap_native_event_entry_t;

typedef struct _powercap_reg_alloc {
    _powercap_register_t ra_bits;
} _powercap_reg_alloc_t;

static char read_buff[PAPI_MAX_STR_LEN];
static char write_buff[PAPI_MAX_STR_LEN];

static long long max_pkg_energy_count;
static long long max_component_energy_count;

static int num_events=0;

// package events
#define PKG_ENERGY                  0
#define PKG_MAX_ENERGY_RANGE        1
#define PKG_MAX_POWER_A             2
#define PKG_POWER_LIMIT_A           3
#define PKG_TIME_WINDOW_A           4
#define PKG_MAX_POWER_B             5
#define PKG_POWER_LIMIT_B           6
#define PKG_TIME_WINDOW_B           7
#define PKG_ENABLED                 8
#define PKG_NAME                    9

#define PKG_NUM_EVENTS              10
static int   pkg_events[PKG_NUM_EVENTS]        = {PKG_ENERGY, PKG_MAX_ENERGY_RANGE, PKG_MAX_POWER_A, PKG_POWER_LIMIT_A, PKG_TIME_WINDOW_A, PKG_MAX_POWER_B, PKG_POWER_LIMIT_B, PKG_TIME_WINDOW_B, PKG_ENABLED, PKG_NAME};
static char *pkg_event_names[PKG_NUM_EVENTS]   = {"ENERGY_UJ", "MAX_ENERGY_RANGE_UJ", "MAX_POWER_A_UW", "POWER_LIMIT_A_UW", "TIME_WINDOW_A_US", "MAX_POWER_B_UW", "POWER_LIMIT_B_UW", "TIME_WINDOW_B", "ENABLED", "NAME"};
static char *pkg_sys_names[PKG_NUM_EVENTS]     = {"energy_uj", "max_energy_range_uj", "constraint_0_max_power_uw", "constraint_0_power_limit_uw", "constraint_0_time_window_us", "constraint_1_max_power_uw", "constraint_1_power_limit_uw", "constraint_1_time_window_us", "enabled", "name"};
static mode_t   pkg_sys_flags[PKG_NUM_EVENTS]  = {O_RDONLY, O_RDONLY, O_RDONLY, O_RDWR, O_RDONLY, O_RDONLY, O_RDWR, O_RDONLY, O_RDONLY, O_RDONLY};


// non-package events
#define COMPONENT_ENERGY            10
#define COMPONENT_MAX_ENERGY_RANGE  11
#define COMPONENT_MAX_POWER_A       12
#define COMPONENT_POWER_LIMIT_A     13
#define COMPONENT_TIME_WINDOW_A     14
#define COMPONENT_ENABLED           15
#define COMPONENT_NAME              16

#define COMPONENT_NUM_EVENTS        7
static int   component_events[COMPONENT_NUM_EVENTS]      = {COMPONENT_ENERGY, COMPONENT_MAX_ENERGY_RANGE, COMPONENT_MAX_POWER_A, COMPONENT_POWER_LIMIT_A, COMPONENT_TIME_WINDOW_A, COMPONENT_ENABLED, COMPONENT_NAME};
static char *component_event_names[COMPONENT_NUM_EVENTS] = {"ENERGY_UJ", "MAX_ENERGY_RANGE_UJ", "MAX_POWER_A_UW", "POWER_LIMIT_A_UW", "TIME_WINDOW_A_US", "ENABLED", "NAME"};
static char *component_sys_names[COMPONENT_NUM_EVENTS]         = {"energy_uj", "max_energy_range_uj", "constraint_0_max_power_uw", "constraint_0_power_limit_uw", "constraint_0_time_window_us", "enabled", "name"};
static mode_t   component_sys_flags[COMPONENT_NUM_EVENTS]      = {O_RDONLY, O_RDONLY, O_RDONLY, O_RDWR, O_RDONLY, O_RDONLY, O_RDONLY};

#define POWERCAP_MAX_COUNTERS (2 * (PKG_NUM_EVENTS + (3 * COMPONENT_NUM_EVENTS)))

static _powercap_native_event_entry_t powercap_ntv_events[(2 * (PKG_NUM_EVENTS + (3 * COMPONENT_NUM_EVENTS)))];

static int event_fds[POWERCAP_MAX_COUNTERS];

typedef struct _powercap_control_state {
  long long count[POWERCAP_MAX_COUNTERS];
  long long which_counter[POWERCAP_MAX_COUNTERS];
  long long need_difference[POWERCAP_MAX_COUNTERS];
  long long lastupdate;
  int active_counters;
} _powercap_control_state_t;

typedef struct _powercap_context {
  long long start_value[POWERCAP_MAX_COUNTERS];
  _powercap_control_state_t state;
} _powercap_context_t;

papi_vector_t _powercap_vector;

/***************************************************************************/
/******  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *******/
/***************************************************************************/

static long long map_index_to_counter( hwd_control_state_t *ctl,  int index )
{
  _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;

  return control->which_counter[index];
}

/* Null terminated version of strncpy */
static char * _local_strlcpy( char *dst, const char *src, size_t size )
{
  char *retval = strncpy( dst, src, size );
  if ( size>0 ) dst[size-1] = '\0';
  return( retval );
}

static long long read_powercap_value( int index )
{
  int sz = pread(event_fds[index], read_buff, PAPI_MAX_STR_LEN, 0);
  read_buff[sz] = '\0';

  return atoll(read_buff);
}

static int write_powercap_value( int index, long long value )
{
  snprintf(write_buff, sizeof(write_buff), "%lld", value);
  int sz = pwrite(event_fds[index], write_buff, PAPI_MAX_STR_LEN, 0);
  if(sz == -1) {
     perror("Error in pwrite(): ");
  }
  return 1;
}

/************************* PAPI Functions **********************************/

/*
 * This is called whenever a thread is initialized
 */
static int _powercap_init_thread( hwd_context_t *ctx )
{
    ( void ) ctx;
    return PAPI_OK;
}

/*
 * Called when PAPI process is initialized (i.e. PAPI_library_init)
 */
static int _powercap_init_component( int cidx )
{
    int retval = PAPI_OK;
  int num_sockets = -1;
  int s = -1, e = -1, c = -1;
  long unsigned int strErr;

  char events_dir[128];
  char event_path[128];
  char *strCpy;

  DIR *events;

  // get hw info
  const PAPI_hw_info_t *hw_info;
  hw_info=&( _papi_hwi_system_info.hw_info );

  // check if intel processor
  if ( hw_info->vendor!=PAPI_VENDOR_INTEL ) {
    strCpy=strncpy(_powercap_vector.cmp_info.disabled_reason, "Not an Intel processor", PAPI_MAX_STR_LEN);
    if (strCpy == NULL) HANDLE_STRING_ERROR;
    retval = PAPI_ENOSUPP;
    goto fn_fail;
  }

  // store number of sockets for adding events
  num_sockets = hw_info->sockets;

  num_events = 0;
  for(s = 0; s < num_sockets; s++) {

    // Find the directory corresponding to the socket number.  There may be other top-level entries in there
    // besides packages, such as "psys", that mess up the numbering, so we conduct an exhaustive search.
    int s_dir, found = 0;
    for(s_dir = 0; ; s_dir++) {
      strErr=snprintf(event_path, sizeof(event_path), "/sys/class/powercap/intel-rapl:%d/%s", s_dir, pkg_sys_names[PKG_NAME]);
      event_path[sizeof(event_path)-1]=0;
      if (strErr > sizeof(event_path)) HANDLE_STRING_ERROR;

      int event_fd;
      event_fd = open(event_path, pkg_sys_flags[PKG_NAME]);
      if (event_fd == -1) { break; }

      int sz = pread(event_fd, read_buff, PAPI_MAX_STR_LEN, 0);
      read_buff[sz] = '\0';
      close(event_fd);

      if (strncmp(read_buff, "package-", strlen("package-")) == 0 && strtol(read_buff + strlen("package-"), NULL, 10) == s) {
        found = 1;
	break;
      }
    }
    if (!found) { continue; }

    // compose string of a pkg directory path
    strErr=snprintf(events_dir, sizeof(events_dir), "/sys/class/powercap/intel-rapl:%d/", s_dir);
    events_dir[sizeof(events_dir)-1]=0;
    if (strErr > sizeof(events_dir)) HANDLE_STRING_ERROR;

    // open directory to make sure it exists
    events = opendir(events_dir);

    // not a valid pkg/component directory so continue
    if (events == NULL) { continue; }
    closedir(events);                                                // opendir has mallocs; so clean up.

    // loop through pkg events and create powercap event entries
    for (e = 0; e < PKG_NUM_EVENTS; e++) {

      // compose string to individual event
      strErr=snprintf(event_path, sizeof(event_path), "%s%s", events_dir, pkg_sys_names[e]);
      event_path[sizeof(event_path)-1]=0;
      if (strErr > sizeof(event_path)) HANDLE_STRING_ERROR;
      // not a valid pkg event path so continue

      if (access(event_path, R_OK) == -1) { continue; }

      strErr=snprintf(powercap_ntv_events[num_events].name, sizeof(powercap_ntv_events[num_events].name), "%s:ZONE%d", pkg_event_names[e], s);
      powercap_ntv_events[num_events].name[sizeof(powercap_ntv_events[num_events].name)-1]=0;
      if (strErr > sizeof(powercap_ntv_events[num_events].name)) HANDLE_STRING_ERROR;
      //snprintf(powercap_ntv_events[num_events].description, sizeof(powercap_ntv_events[num_events].name), "%s:ZONE%d", pkg_event_names[e], s);
      //snprintf(powercap_ntv_events[num_events].units, sizeof(powercap_ntv_events[num_events].name), "%s:ZONE%d", pkg_event_names[e], s);
      powercap_ntv_events[num_events].return_type = PAPI_DATATYPE_UINT64;
      powercap_ntv_events[num_events].type = pkg_events[e];

      powercap_ntv_events[num_events].resources.selector = num_events + 1;

      event_fds[num_events] = open(event_path, O_SYNC|pkg_sys_flags[e]);

      if(powercap_ntv_events[num_events].type == PKG_NAME) {
        int sz = pread(event_fds[num_events], read_buff, PAPI_MAX_STR_LEN, 0);
        read_buff[sz] = '\0';
        strErr=snprintf(powercap_ntv_events[num_events].description, sizeof(powercap_ntv_events[num_events].description), "%s", read_buff);
        powercap_ntv_events[num_events].description[sizeof(powercap_ntv_events[num_events].description)-1]=0;
        if (strErr > sizeof(powercap_ntv_events[num_events].description)) HANDLE_STRING_ERROR;
      }

      if(powercap_ntv_events[num_events].type == PKG_MAX_ENERGY_RANGE) {
        int sz = pread(event_fds[num_events], read_buff, PAPI_MAX_STR_LEN, 0);
        read_buff[sz] = '\0';
        max_pkg_energy_count = atoll(read_buff);
      }

      num_events++;
    }

    // reset component count for each socket
    c = 0;
    strErr=snprintf(events_dir, sizeof(events_dir), "/sys/class/powercap/intel-rapl:%d:%d/", s_dir, c);
    events_dir[sizeof(events_dir)-1]=0;
    if (strErr > sizeof(events_dir)) HANDLE_STRING_ERROR;
    while((events = opendir(events_dir)) != NULL) {
      closedir(events);                                                // opendir has mallocs; so clean up.

      // loop through pkg events and create powercap event entries
      for (e = 0; e < COMPONENT_NUM_EVENTS; e++) {

        // compose string to individual event
        strErr=snprintf(event_path, sizeof(event_path), "%s%s", events_dir, component_sys_names[e]);
        event_path[sizeof(event_path)-1]=0;
        if (strErr > sizeof(event_path)) HANDLE_STRING_ERROR;

        // not a valid pkg event path so continue
        if (access(event_path, R_OK) == -1) { continue; }

        strErr=snprintf(powercap_ntv_events[num_events].name, sizeof(powercap_ntv_events[num_events].name), "%s:ZONE%d_SUBZONE%d", component_event_names[e], s, c);
        powercap_ntv_events[num_events].name[sizeof(powercap_ntv_events[num_events].name)-1]=0;
        if (strErr > sizeof(powercap_ntv_events[num_events].name)) HANDLE_STRING_ERROR;
        //snprintf(powercap_ntv_events[num_events].description, sizeof(powercap_ntv_events[num_events].name), "%s:ZONE%d_SUBZONE%d", component_event_names[e], s, c);
        //snprintf(powercap_ntv_events[num_events].units, sizeof(powercap_ntv_events[num_events].name), "%s:ZONE%d_SUBZONE%d", component_event_names[e], s, c);
        powercap_ntv_events[num_events].return_type = PAPI_DATATYPE_UINT64;
        powercap_ntv_events[num_events].type = component_events[e];

        powercap_ntv_events[num_events].resources.selector = num_events + 1;

        event_fds[num_events] = open(event_path, O_SYNC|component_sys_flags[e]);

        if(powercap_ntv_events[num_events].type == COMPONENT_NAME) {
          int sz = pread(event_fds[num_events], read_buff, PAPI_MAX_STR_LEN, 0);
          read_buff[sz] = '\0';
          strErr=snprintf(powercap_ntv_events[num_events].description, sizeof(powercap_ntv_events[num_events].description), "%s", read_buff);
          powercap_ntv_events[num_events].description[sizeof(powercap_ntv_events[num_events].description)-1]=0;
          if (strErr > sizeof(powercap_ntv_events[num_events].description)) HANDLE_STRING_ERROR;
        }

        if(powercap_ntv_events[num_events].type == COMPONENT_MAX_ENERGY_RANGE) {
          int sz = pread(event_fds[num_events], read_buff, PAPI_MAX_STR_LEN, 0);
          read_buff[sz] = '\0';
          max_component_energy_count = atoll(read_buff);
        }

        num_events++;
      }

      // test for next component
      c++;

      // compose string of an pkg directory path
      strErr=snprintf(events_dir, sizeof(events_dir), "/sys/class/powercap/intel-rapl:%d:%d/", s_dir, c);
      events_dir[sizeof(events_dir)-1]=0;
      if (strErr > sizeof(events_dir)) HANDLE_STRING_ERROR;
    }
  }

  /* Export the total number of events available */
  _powercap_vector.cmp_info.num_native_events = num_events;
  _powercap_vector.cmp_info.num_cntrs = num_events;
  _powercap_vector.cmp_info.num_mpx_cntrs = num_events;

  /* Export the component id */
  _powercap_vector.cmp_info.CmpIdx = cidx;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
static int _powercap_init_control_state( hwd_control_state_t *ctl )
{
    _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;
    memset( control, 0, sizeof ( _powercap_control_state_t ) );


    /* if an event is a counter, set its corresponding flag to 1  */
    int i;
    for (i = 0; i < num_events; i++) {
        if ((powercap_ntv_events[i].type == PKG_ENERGY) || (powercap_ntv_events[i].type == COMPONENT_ENERGY)) {
            control->need_difference[i] = 1;
        }
    }

    return PAPI_OK;
}

static int _powercap_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    _powercap_context_t* context = ( _powercap_context_t* ) ctx;
    (void) ctl;

    int b;
    for( b = 0; b < num_events; b++ ) {
      context->start_value[b]=read_powercap_value(b);
    }

    return PAPI_OK;
}

static int _powercap_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
  (void) ctx;
  (void) ctl;
  return PAPI_OK;
}

/* Shutdown a thread */
static int
_powercap_shutdown_thread( hwd_context_t *ctx )
{
    ( void ) ctx;
    SUBDBG( "Enter\n" );
    return PAPI_OK;
}


static int
_powercap_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
                long long **events, int flags )
{
  SUBDBG("Enter _powercap_read\n");

  (void) flags;
  _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;
  _powercap_context_t* context = ( _powercap_context_t* ) ctx;

  long long start_val = 0;
  long long curr_val = 0;
  int c, i;

  for( c = 0; c < control->active_counters; c++ ) {
    i = map_index_to_counter(ctl, c);
    start_val = context->start_value[i];
    curr_val = read_powercap_value(i);

    SUBDBG("%d, start value: %lld, current value %lld\n", i, start_val, curr_val);

    if(start_val) {

      /* Make sure an event is a counter. */
      if (control->need_difference[i] == 1) {

        /* Wraparound. */
        if(start_val > curr_val) {
          SUBDBG("Wraparound!\nstart value:\t%lld,\tcurrent value:%lld\n", start_val, curr_val);
          if( powercap_ntv_events[i].type == PKG_ENERGY ) {
            curr_val += (max_pkg_energy_count - start_val);
          } else if( powercap_ntv_events[i].type == COMPONENT_ENERGY ) {
            curr_val += (max_component_energy_count - start_val);
          } else {
            curr_val += (0x100000000 - start_val);
          }
        }

        /* Normal subtraction. */
        else if (start_val < curr_val) {
          SUBDBG("Normal subtraction!\nstart value:\t%lld,\tcurrent value:%lld\n", start_val, curr_val);
          curr_val -= start_val;
        }
        SUBDBG("Final value: %lld\n", curr_val);

      }
    }
    control->count[c]=curr_val;
  }

  *events = ( ( _powercap_control_state_t* ) ctl )->count;

  return PAPI_OK;
}

static int _powercap_write( hwd_context_t * ctx, hwd_control_state_t * ctl, long long *values )
{
    /* write values */
    ( void ) ctx;
    _powercap_control_state_t *control = ( _powercap_control_state_t * ) ctl;

    int c, i;

    for(c=0;c<control->active_counters;c++) {
      i = map_index_to_counter(ctl, c);
      if( (powercap_ntv_events[i].type == PKG_POWER_LIMIT_A) || (powercap_ntv_events[i].type == PKG_POWER_LIMIT_B) ) {
        write_powercap_value(i, values[c]);
      }
    }

    return PAPI_OK;
}
/*
 * Clean up what was setup in powercap_init_component().
 */
static int _powercap_shutdown_component( void )
{
  int i;

  /* Read counters into expected slot */
  for(i=0;i<num_events;i++) {
    close(event_fds[i]);
  }
    return PAPI_OK;
}

/* This function sets various options in the component. The valid
 * codes being passed in are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN,
 * PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
static int
_powercap_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    SUBDBG( "Enter: ctx: %p\n", ctx );
    ( void ) ctx;
    ( void ) code;
    ( void ) option;

    return PAPI_OK;
}


static int _powercap_update_control_state( hwd_control_state_t *ctl,
                                NativeInfo_t *native, int count,
                                hwd_context_t *ctx )
{
  (void) ctx;
  int i, index;

  _powercap_control_state_t* control = ( _powercap_control_state_t* ) ctl;
  control->active_counters = count;

  if (count==0) return PAPI_OK;

  for( i = 0; i < count; i++ ) {
    index = native[i].ni_event;
    control->which_counter[i]=index;
    native[i].ni_position = i;
  }

  return PAPI_OK;

}

static int _powercap_set_domain( hwd_control_state_t *ctl, int domain )
{
    ( void ) ctl;
    if ( PAPI_DOM_ALL != domain )
        return PAPI_EINVAL;

    return PAPI_OK;
}


static int _powercap_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;
    return PAPI_OK;
}

/*
 * Native Event functions
 */
static int _powercap_ntv_enum_events( unsigned int *EventCode, int modifier )
{
  int index;
  switch ( modifier ) {

    case PAPI_ENUM_FIRST:
      *EventCode = 0;
      return PAPI_OK;
    case PAPI_ENUM_EVENTS:index = *EventCode;
      if (index < num_events - 1) {
        *EventCode = *EventCode + 1;
        return PAPI_OK;
      } else {
        return PAPI_ENOEVNT;
      }

    default:return PAPI_EINVAL;
  }
}

/*
 *
 */
static int _powercap_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK;

    if ( index >= 0 && index < num_events ) {
        _local_strlcpy( name, powercap_ntv_events[index].name, len );
        return PAPI_OK;
    } 
    return PAPI_ENOEVNT;
}

/* 
 *
 */
static int _powercap_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
    int index = EventCode;

    if ( index < 0 && index >= num_events ) 
        return PAPI_ENOEVNT;
    _local_strlcpy( name, powercap_ntv_events[index].description, len );
    return PAPI_OK;
}

static int _powercap_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t *info )
{
    int index = EventCode;

    if ( index < 0 || index >= num_events ) 
        return PAPI_ENOEVNT;

    _local_strlcpy( info->symbol, powercap_ntv_events[index].name, PAPI_MAX_STR_LEN );
    _local_strlcpy( info->long_descr, powercap_ntv_events[index].description, PAPI_MAX_STR_LEN );
    _local_strlcpy( info->units, powercap_ntv_events[index].units, PAPI_MIN_STR_LEN-1 );

    info->data_type = powercap_ntv_events[index].return_type;
    return PAPI_OK;
}


papi_vector_t _powercap_vector = {
    .cmp_info = { /* (unspecified values are initialized to 0) */
        .name = "powercap",
        .short_name = "powercap",
        .description = "Linux powercap energy measurements",
        .version = "5.3.0",
        .default_domain = PAPI_DOM_ALL,
        .default_granularity = PAPI_GRN_SYS,
        .available_granularities = PAPI_GRN_SYS,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .available_domains = PAPI_DOM_ALL,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context = sizeof ( _powercap_context_t ),
        .control_state = sizeof ( _powercap_control_state_t ),
        .reg_value = sizeof ( _powercap_register_t ),
        .reg_alloc = sizeof ( _powercap_reg_alloc_t ),
    },
    /* function pointers in this component */
    .init_thread =          _powercap_init_thread,
    .init_component =       _powercap_init_component,
    .init_control_state =   _powercap_init_control_state,
    .update_control_state = _powercap_update_control_state,
    .start =                _powercap_start,
    .stop =                 _powercap_stop,
    .read =                 _powercap_read,
    .write =                _powercap_write,
    .shutdown_thread =      _powercap_shutdown_thread,
    .shutdown_component =   _powercap_shutdown_component,
    .ctl =                  _powercap_ctl,

    .set_domain =           _powercap_set_domain,
    .reset =                _powercap_reset,

    .ntv_enum_events =      _powercap_ntv_enum_events,
    .ntv_code_to_name =     _powercap_ntv_code_to_name,
    .ntv_code_to_descr =    _powercap_ntv_code_to_descr,
    .ntv_code_to_info =     _powercap_ntv_code_to_info,
};
