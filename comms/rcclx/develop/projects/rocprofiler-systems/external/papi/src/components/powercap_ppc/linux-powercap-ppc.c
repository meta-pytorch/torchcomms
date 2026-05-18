/**
 * @file    linux-powercap.c
 * @author  PAPI team UTK/ICL 
 * @ingroup papi_components
 * @brief powercap component
 *
 * To work, the powercap kernel module must be loaded.
 */

#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "linux-powercap-ppc.h"

// The following macro exit if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}


static char read_buff[PAPI_MAX_STR_LEN];
static char write_buff[PAPI_MAX_STR_LEN];

static int num_events=0;

static int   pkg_events[PKG_NUM_EVENTS]
  = {PKG_MIN_POWER, PKG_MAX_POWER, PKG_CUR_POWER};
static const char *pkg_event_names[PKG_NUM_EVENTS]
  = {"MIN_POWER",  "MAX_POWER",  "CURRENT_POWER"};
static const char *pkg_sys_names[PKG_NUM_EVENTS]
  = {"powercap-min", "powercap-max", "powercap-current"};
static const char *pkg_event_descs[PKG_NUM_EVENTS]
  = {"Minimum value allowed for power capping.",
     "Maximum value allowed for power capping.",
     "Current power drawned by package."};
static mode_t      pkg_sys_flags[PKG_NUM_EVENTS]
  = {O_RDONLY,       O_RDONLY,       O_RDWR};

static _powercap_ppc_native_event_entry_t powercap_ppc_ntv_events[(PKG_NUM_EVENTS)];

static int event_fds[POWERCAP_MAX_COUNTERS];

papi_vector_t _powercap_ppc_vector;

/***************************************************************************/
/******  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *******/
/***************************************************************************/

/* Null terminated version of strncpy */
static char *
_local_strlcpy( char *dst, const char *src, size_t size )
{
    char *retval = strncpy( dst, src, size );
    if ( size > 0 ) dst[size-1] = '\0';

    return( retval );
}

static long long
read_powercap_value( int index )
{
    int sz = pread(event_fds[index], read_buff, PAPI_MAX_STR_LEN, 0);
    read_buff[sz] = '\0';

    return atoll(read_buff);
}

static int
write_powercap_value( int index, long long value )
{
    size_t ret = snprintf(write_buff, sizeof(write_buff), "%lld", value);
    if (ret <= 0 || sizeof(write_buff) <= ret)
        return PAPI_ENOSUPP;

    papi_powercap_ppc_lock();
    int sz = pwrite(event_fds[index], write_buff, PAPI_MAX_STR_LEN, 0);
    if ( sz == -1 ) {
        perror("Error in pwrite(): ");
    }
    papi_powercap_ppc_unlock();

    return 1;
}

/************************* PAPI Functions **********************************/

/*
 * This is called whenever a thread is initialized
 */
static int
_powercap_ppc_init_thread( hwd_context_t *ctx )
{
    (void) ctx;

    return PAPI_OK;
}

/*
 * Called when PAPI process is initialized (i.e. PAPI_library_init)
 */
static int
_powercap_ppc_init_component( int cidx )
{
    int retval = PAPI_OK;
    int e = -1;
    char events_dir[128];
    char event_path[128];
    char *strCpy;

    DIR *events;

    const PAPI_hw_info_t *hw_info;
    hw_info=&( _papi_hwi_system_info.hw_info );

    /* check if IBM processor */
    if ( hw_info->vendor!=PAPI_VENDOR_IBM ) {
        strCpy=strncpy(_powercap_ppc_vector.cmp_info.disabled_reason, "Not an IBM Power9 processor", PAPI_MAX_STR_LEN);
        _powercap_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        retval = PAPI_ENOSUPP;
        goto fn_fail;
    }

    num_events = 0;

    /* Check the existence, and correct access modes to pkg directory path */
    size_t ret = snprintf(events_dir, sizeof(events_dir), "/sys/firmware/opal/powercap/system-powercap/");
    if (ret <= 0 || sizeof(events_dir) <= ret) HANDLE_STRING_ERROR;

    if ( NULL == (events = opendir(events_dir)) ) {
        strCpy=strncpy(_powercap_ppc_vector.cmp_info.disabled_reason,
            "Directory /sys/firmware/opal/powercap/system-powercap missing.",
            PAPI_MAX_STR_LEN);
        _powercap_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        retval = PAPI_ENOSUPP;
        goto fn_fail;
    }

    /* opendir needs clean up. */
    closedir(events);

    /* loop through events and create powercap event entries */
    for ( e = 0; e < PKG_NUM_EVENTS; ++e ) {
        /* compose string to individual event */
        size_t ret = snprintf(event_path, sizeof(event_path), "%s%s", events_dir, pkg_sys_names[e]);
        if (ret <= 0 || sizeof(event_path) <= ret) HANDLE_STRING_ERROR;

        /* if it's not a valid pkg event path we skip it */
        if (access(event_path, F_OK) == -1) continue;

        ret = snprintf(powercap_ppc_ntv_events[num_events].name,
                       sizeof(powercap_ppc_ntv_events[num_events].name),
                       "%s", pkg_event_names[e]);
        powercap_ppc_ntv_events[num_events].name[sizeof(powercap_ppc_ntv_events[num_events].name)-1]=0;
        if (ret <= 0 || sizeof(powercap_ppc_ntv_events[num_events].name) <= ret) HANDLE_STRING_ERROR;

        ret = snprintf(powercap_ppc_ntv_events[num_events].description,
                       sizeof(powercap_ppc_ntv_events[num_events].description),
                       "%s", pkg_event_descs[e]);
        powercap_ppc_ntv_events[num_events].description[sizeof(powercap_ppc_ntv_events[num_events].description)-1]=0;
        if (ret <= 0 || sizeof(powercap_ppc_ntv_events[num_events].description) <= ret) HANDLE_STRING_ERROR;
        ret = snprintf(powercap_ppc_ntv_events[num_events].units,
                       sizeof(powercap_ppc_ntv_events[num_events].units), "W");
        powercap_ppc_ntv_events[num_events].units[sizeof(powercap_ppc_ntv_events[num_events].units)-1]=0;
        if (ret <= 0 || sizeof(powercap_ppc_ntv_events[num_events].units) <= ret) HANDLE_STRING_ERROR;

        powercap_ppc_ntv_events[num_events].return_type = PAPI_DATATYPE_INT64;
        powercap_ppc_ntv_events[num_events].type = pkg_events[e];

        powercap_ppc_ntv_events[num_events].resources.selector = num_events + 1;

        event_fds[num_events] = open(event_path, O_SYNC|pkg_sys_flags[e]);

        num_events++;
    }

    /* Export the total number of events available */
    _powercap_ppc_vector.cmp_info.num_native_events = num_events;
    _powercap_ppc_vector.cmp_info.num_cntrs = num_events;
    _powercap_ppc_vector.cmp_info.num_mpx_cntrs = num_events;

    /* Export the component id */
    _powercap_ppc_vector.cmp_info.CmpIdx = cidx;

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
static int
_powercap_ppc_init_control_state( hwd_control_state_t *ctl )
{
    _powercap_ppc_control_state_t* control = ( _powercap_ppc_control_state_t* ) ctl;
    memset( control, 0, sizeof ( _powercap_ppc_control_state_t ) );

    return PAPI_OK;
}

static int
_powercap_ppc_update_control_state( hwd_control_state_t *ctl,
                                    NativeInfo_t *native,
                                    int count,
                                    hwd_context_t *ctx )
{
    (void) ctx;
    int i, index;

    _powercap_ppc_control_state_t* control = ( _powercap_ppc_control_state_t* ) ctl;
    control->active_counters = count;

    for ( i = 0; i < count; ++i ) {
        index = native[i].ni_event;
        control->which_counter[i]=index;
        native[i].ni_position = i;
    }

    return PAPI_OK;
}

/*
 *  There are no counters to start, all three values are instantaneous
 * */
static int
_powercap_ppc_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    (void) ctl;

    return PAPI_OK;
}

static int
_powercap_ppc_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    (void) ctl;

    return PAPI_OK;
}

/* 
 * Shutdown a thread
 * */
static int
_powercap_ppc_shutdown_thread( hwd_context_t *ctx )
{
    (void) ctx;
    SUBDBG( "Enter _powercap_ppc_shutdown_thread\n" );
    return PAPI_OK;
}


static int
_powercap_ppc_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
                    long long **events, int flags )
{
    SUBDBG("Enter _powercap_ppc_read\n");

    (void) flags;
    (void) ctx;
    _powercap_ppc_control_state_t* control = ( _powercap_ppc_control_state_t* ) ctl;

    long long curr_val = 0;

    int c, i;
    for( c = 0; c < control->active_counters; c++ ) {
        i = control->which_counter[c];
        curr_val = read_powercap_value(i);
        SUBDBG("%d, current value %lld\n", i, curr_val);
        control->count[c]=curr_val;
    }

    *events = ( ( _powercap_ppc_control_state_t* ) ctl )->count;

    return PAPI_OK;
}

/*
 * One counter only is writable, the current power one
 * */
static int
_powercap_ppc_write( hwd_context_t * ctx, hwd_control_state_t * ctl, long long *values )
{
    (void) ctx;
    _powercap_ppc_control_state_t *control = ( _powercap_ppc_control_state_t * ) ctl;

    int i;
    for (i = 0; i < control->active_counters; i++) {
      if (PKG_CUR_POWER == powercap_ppc_ntv_events[control->which_counter[i]].type)
        write_powercap_value(control->which_counter[i], values[i]);
    }

    return PAPI_OK;
}

/*
 * Close opened file descriptors.
 */
static int
_powercap_ppc_shutdown_component( void )
{
    int i;
    for( i = 0; i < num_events; i++ ) {
        close(event_fds[i]);
    }

    return PAPI_OK;
}

static int
_powercap_ppc_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    SUBDBG( "Enter: ctx: %p\n", ctx );
    (void) ctx;
    (void) code;
    (void) option;

    return PAPI_OK;
}


static int
_powercap_ppc_set_domain( hwd_control_state_t *ctl, int domain )
{
    (void) ctl;
    if ( PAPI_DOM_ALL != domain )
        return PAPI_EINVAL;

    return PAPI_OK;
}


static int
_powercap_ppc_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    (void) ctl;

    return PAPI_OK;
}

/*
 * Native Event functions
 */
static int
_powercap_ppc_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    int index;
    switch ( modifier ) {
        case PAPI_ENUM_FIRST:
            *EventCode = 0;
            return PAPI_OK;
        case PAPI_ENUM_EVENTS:
            index = *EventCode;
            if ( index < num_events - 1 ) {
                *EventCode = *EventCode + 1;
                return PAPI_OK;
            } else {
                return PAPI_ENOEVNT;
            }

        default:
            return PAPI_EINVAL;
    }
}

/*
 *
 */
static int
_powercap_ppc_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK;

    if ( index >= 0 && index < num_events ) {
        _local_strlcpy( name, powercap_ppc_ntv_events[index].name, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}

static int
_powercap_ppc_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t *info )
{
    int index = EventCode;

    if ( index < 0 || index >= num_events )
        return PAPI_ENOEVNT;

    _local_strlcpy( info->symbol, powercap_ppc_ntv_events[index].name, sizeof( info->symbol ) - 1 );
    _local_strlcpy( info->units, powercap_ppc_ntv_events[index].units, sizeof( info->units ) - 1 );
    _local_strlcpy( info->long_descr, powercap_ppc_ntv_events[index].description, sizeof( info->long_descr ) - 1 );

    info->data_type = powercap_ppc_ntv_events[index].return_type;
    return PAPI_OK;
}

static int
_powercap_ppc_ntv_name_to_code( const char *name, unsigned int *EventCode)
{
    if (!strcmp(name, "MIN_POWER")) *EventCode = 0;
    else if (!strcmp(name, "MAX_POWER")) *EventCode = 1;
    else if (!strcmp(name, "CURRENT_POWER")) *EventCode = 2;
    return PAPI_OK;
}

papi_vector_t _powercap_ppc_vector = {
    .cmp_info = {
        .name = "powercap_ppc",
        .short_name = "powercap_ppc",
        .description = "Linux powercap energy measurements for IBM PowerPC (9) architectures",
        .version = "5.7.0",
        .default_domain = PAPI_DOM_ALL,
        .default_granularity = PAPI_GRN_SYS,
        .available_granularities = PAPI_GRN_SYS,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .available_domains = PAPI_DOM_ALL,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context = sizeof ( _powercap_ppc_context_t ),
        .control_state = sizeof ( _powercap_ppc_control_state_t ),
        .reg_value = sizeof ( _powercap_ppc_register_t ),
        .reg_alloc = sizeof ( _powercap_ppc_reg_alloc_t ),
    },
    /* function pointers in this component */
    .init_thread =          _powercap_ppc_init_thread,
    .init_component =       _powercap_ppc_init_component,
    .init_control_state =   _powercap_ppc_init_control_state,
    .update_control_state = _powercap_ppc_update_control_state,
    .start =                _powercap_ppc_start,
    .stop =                 _powercap_ppc_stop,
    .read =                 _powercap_ppc_read,
    .write =                _powercap_ppc_write,
    .shutdown_thread =      _powercap_ppc_shutdown_thread,
    .shutdown_component =   _powercap_ppc_shutdown_component,
    .ctl =                  _powercap_ppc_ctl,

    .set_domain =           _powercap_ppc_set_domain,
    .reset =                _powercap_ppc_reset,

    .ntv_enum_events =      _powercap_ppc_ntv_enum_events,
    .ntv_name_to_code =     _powercap_ppc_ntv_name_to_code,
    .ntv_code_to_name =     _powercap_ppc_ntv_code_to_name,
    .ntv_code_to_info =     _powercap_ppc_ntv_code_to_info,
};
