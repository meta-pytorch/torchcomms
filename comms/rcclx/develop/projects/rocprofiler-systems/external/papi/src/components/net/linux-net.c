/**
 * @file    linux-net.c
 *
 * @author  Haihang You
 *          you@cs.utk.edu
 *
 * @author  Jose Pedro Oliveira
 *          jpo@di.uminho.pt
 *
 * @ingroup papi_components
 *
 * @brief net component
 *  This file contains the source code for a component that enables
 *  PAPI-C to access network statistics through the /proc file system.
 *  This component will dynamically create a native events table for
 *  all the interfaces listed in /proc/net/dev (16 entries for each
 *  interface).
 */


#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <net/if.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "linux-net.h"


papi_vector_t _net_vector;

/*********************************************************************
 * Private
 ********************************************************************/

/* Network stats refresh latency in usec (default: 1 sec) */
#define NET_REFRESH_LATENCY   1000000

#define NET_PROC_FILE          "/proc/net/dev"

/* /proc/net/dev line size
 * interface name + 8 RX counters + 8 TX counters + separators
 */
#define NET_PROC_MAX_LINE      (IFNAMSIZ + 16 * (20 + 1) + 16)

#define NET_INVALID_RESULT     -1


// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

static NET_native_event_entry_t * _net_native_events=NULL;

static int num_events       = 0;
static int is_initialized   = 0;

static long long _net_register_start[NET_MAX_COUNTERS];
static long long _net_register_current[NET_MAX_COUNTERS];

/* temporary event */
struct temp_event {
    char name[PAPI_MAX_STR_LEN];
    char description[PAPI_MAX_STR_LEN];
    struct temp_event *next;
};
static struct temp_event* root = NULL;

/* /proc/net/dev: network counters by interface */
#define NET_INTERFACE_COUNTERS 16

static const struct net_counters {
    char *name;
    char *description;
} _net_counter_info[NET_INTERFACE_COUNTERS] = {
    /* Receive */
    { "rx:bytes",      "receive bytes"},
    { "rx:packets",    "receive packets"},
    { "rx:errors",     "receive errors"},
    { "rx:dropped",    "receive dropped"},
    { "rx:fifo",       "receive fifo"},
    { "rx:frame",      "receive frame"},
    { "rx:compressed", "receive compressed"},
    { "rx:multicast",  "receive multicast"},
    /* Transmit */
    { "tx:bytes",      "transmit bytes"},
    { "tx:packets",    "transmit packets"},
    { "tx:errors",     "transmit errors"},
    { "tx:dropped",    "transmit dropped"},
    { "tx:fifo",       "transmit fifo"},
    { "tx:colls",      "transmit colls"},
    { "tx:carrier",    "transmit carrier"},
    { "tx:compressed", "transmit compressed"},
};


/*********************************************************************
 ***  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT ****
 ********************************************************************/

/*
 * find all network interfaces listed in /proc/net/dev
 */
static int
generateNetEventList( void )
{
    FILE *fin;
    char line[NET_PROC_MAX_LINE];
    char *retval, *ifname;
    int count = 0;
    struct temp_event *temp;
    struct temp_event *last = NULL;
    int i, j;

    fin = fopen(NET_PROC_FILE, "r");
    if (fin == NULL) {
        SUBDBG("Can't find %s, are you sure the /proc file-system is mounted?\n",
           NET_PROC_FILE);
        snprintf(_net_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
        "Failed to find /proc file-system.");
        _net_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
        return 0;
    }

    /* skip the 2 header lines */
    for (i=0; i<2; i++) {
        retval = fgets (line, NET_PROC_MAX_LINE, fin);
        if (retval == NULL) {
        	fclose(fin);
            SUBDBG("Not enough lines in %s\n", NET_PROC_FILE);
            snprintf(_net_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
            "Not enough lines in '%s'.", NET_PROC_FILE);
            _net_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
            return 0;
        }
    }

    while ((fgets (line, NET_PROC_MAX_LINE, fin)) == line) {

        /* split the interface name from the 16 counters */
        retval = strstr(line, ":");
        if (retval == NULL) {
            SUBDBG("Wrong line format <%s>\n", line);
            continue;
        }

        *retval = '\0';
        ifname = line;
        while (isspace(*ifname)) { ifname++; }

        for (j=0; j<NET_INTERFACE_COUNTERS; j++) {

            /* keep the interface name around */
            temp = (struct temp_event *)papi_malloc(sizeof(struct temp_event));
            if (!temp) {
                PAPIERROR("out of memory!");
                fclose(fin);
                snprintf(_net_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
                "%s failed to allocate %lu bytes for temp_event.", __func__, sizeof(struct temp_event));
                _net_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                return PAPI_ENOMEM;
            }
            temp->next = NULL;

            if (root == NULL) {
                root = temp;
            } else if (last) {
                last->next = temp;
            } else {
                free(temp);
                fclose(fin);
                PAPIERROR("This shouldn't be possible\n");
                snprintf(_net_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN-2,
                "%s Error navigating linked list.", __func__);
                _net_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
                return PAPI_ECMP;
            }
            last = temp;

            size_t str_len = strlen(ifname) + strlen(_net_counter_info[j].name) + 1;
            str_len = (str_len > PAPI_MAX_STR_LEN - 1) ? PAPI_MAX_STR_LEN - 1 : str_len;
            snprintf(temp->name, str_len, "%s:%s",
                    ifname, _net_counter_info[j].name);
            str_len = strlen(ifname) + strlen(_net_counter_info[j].description) + 1;
            str_len = (str_len > PAPI_MAX_STR_LEN - 1) ? PAPI_MAX_STR_LEN - 1 : str_len;
            snprintf(temp->description, str_len, "%s %s",
                    ifname, _net_counter_info[j].description);

            count++;
        }
    }

    fclose(fin);

    return count;
}


static int
getInterfaceBaseIndex(const char *ifname)
{
    int i;

    for ( i=0; i<num_events; i+=NET_INTERFACE_COUNTERS ) {
        if (strncmp(_net_native_events[i].name, ifname, strlen(ifname)) == 0) {
            return i;
        }
    }

    return -1;  /* Not found */
}


static int
read_net_counters( long long *values )
{
    FILE *fin;
    char line[NET_PROC_MAX_LINE];
    char *retval, *ifname, *data;
    int i, nf, if_bidx;

    fin = fopen(NET_PROC_FILE, "r");
    if (fin == NULL) {
        SUBDBG("Can't find %s, are you sure the /proc file-system is mounted?\n",
           NET_PROC_FILE);
        return NET_INVALID_RESULT;
    }

    /* skip the 2 header lines */
    for (i=0; i<2; i++) {
        retval = fgets (line, NET_PROC_MAX_LINE, fin);
        if (retval == NULL) {
            SUBDBG("Not enough lines in %s\n", NET_PROC_FILE);
			fclose(fin);
            return 0;
        }
    }

    while ((fgets (line, NET_PROC_MAX_LINE, fin)) == line) {

        /* split the interface name from its 16 counters */
        retval = strstr(line, ":");
        if (retval == NULL) {
            SUBDBG("Wrong line format <%s>\n", line);
            continue;
        }

        *retval = '\0';
        data = retval + 1;
        ifname = line;
        while (isspace(*ifname)) { ifname++; }

        if_bidx = getInterfaceBaseIndex(ifname);
        if (if_bidx < 0) {
            SUBDBG("Interface <%s> not found\n", ifname);
        } else {
            nf = sscanf( data,
                "%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
                &values[if_bidx + 0],  &values[if_bidx + 1],
                &values[if_bidx + 2],  &values[if_bidx + 3],
                &values[if_bidx + 4],  &values[if_bidx + 5],
                &values[if_bidx + 6],  &values[if_bidx + 7],
                &values[if_bidx + 8],  &values[if_bidx + 9],
                &values[if_bidx + 10], &values[if_bidx + 11],
                &values[if_bidx + 12], &values[if_bidx + 13],
                &values[if_bidx + 14], &values[if_bidx + 15]);

            SUBDBG("\nRead "
                "%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
                values[if_bidx + 0],  values[if_bidx + 1],
                values[if_bidx + 2],  values[if_bidx + 3],
                values[if_bidx + 4],  values[if_bidx + 5],
                values[if_bidx + 6],  values[if_bidx + 7],
                values[if_bidx + 8],  values[if_bidx + 9],
                values[if_bidx + 10], values[if_bidx + 11],
                values[if_bidx + 12], values[if_bidx + 13],
                values[if_bidx + 14], values[if_bidx + 15]);

            if ( nf != NET_INTERFACE_COUNTERS ) {
                /* This shouldn't happen */
                SUBDBG("/proc line with wrong number of fields\n");
            }
        }

    }

    fclose(fin);

    return 0;
}


/*********************************************************************
 ***************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *********
 *********************************************************************/

/*
 * This is called whenever a thread is initialized
 */
static int
_net_init_thread( hwd_context_t *ctx )
{
    ( void ) ctx;

    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int
_net_init_component( int cidx  )
{
    int retval = PAPI_OK;
    int i = 0;
    struct temp_event *t, *last;

    if ( is_initialized )
        goto fn_exit;

    memset(_net_register_start, 0,
    		NET_MAX_COUNTERS*sizeof(_net_register_start[0]));
    memset(_net_register_current, 0, 
    		NET_MAX_COUNTERS*sizeof(_net_register_current[0]));

    is_initialized = 1;

    /* The network interfaces are listed in /proc/net/dev */
    num_events = generateNetEventList();

    if ( num_events < 0 ) {  /* disabled_reason already set */
        retval = PAPI_ECMP;
        goto fn_fail;
    }

    if ( num_events == 0 )  /* No network interfaces found */
        goto fn_exit;

    t = root;
    _net_native_events = (NET_native_event_entry_t*)
        papi_malloc(sizeof(NET_native_event_entry_t) * num_events);
    do {
        int retlen;
        retlen = snprintf(_net_native_events[i].name, PAPI_MAX_STR_LEN, "%s", t->name);
        if (retlen <= 0 || retlen >= PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        retlen = snprintf(_net_native_events[i].description, PAPI_MAX_STR_LEN, "%s", t->description);
        if (retlen <= 0 || retlen >= PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        _net_native_events[i].resources.selector = i + 1;
        last    = t;
        t       = t->next;
        papi_free(last);
        i++;
    } while (t != NULL);
    root = NULL;

    /* Export the total number of events available */
    _net_vector.cmp_info.num_native_events = num_events;

    /* Export the component id */
    _net_vector.cmp_info.CmpIdx = cidx;

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
_net_init_control_state( hwd_control_state_t *ctl )
{
    ( void ) ctl;

    return PAPI_OK;
}


static int
_net_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;

    NET_control_state_t *net_ctl = (NET_control_state_t *) ctl;
    long long now = PAPI_get_real_usec();

    read_net_counters(_net_register_start);
    memcpy(_net_register_current, _net_register_start,
            NET_MAX_COUNTERS * sizeof(_net_register_start[0]));

    /* set initial values to 0 */
    memset(net_ctl->values, 0, NET_MAX_COUNTERS*sizeof(net_ctl->values[0]));
    
    /* Set last access time for caching purposes */
    net_ctl->lastupdate = now;

    return PAPI_OK;
}


static int
_net_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
    long long ** events, int flags )
{
    (void) flags;
    (void) ctx;

    NET_control_state_t *net_ctl = (NET_control_state_t *) ctl;
    long long now = PAPI_get_real_usec();
    int i;

    /* Caching
     * Only read new values from /proc if enough time has passed
     * since the last read.
     */
    if ( now - net_ctl->lastupdate > NET_REFRESH_LATENCY ) {
        read_net_counters(_net_register_current);
        for ( i=0; i<NET_MAX_COUNTERS; i++ ) {
            net_ctl->values[i] = _net_register_current[i] - _net_register_start[i];
        }
        net_ctl->lastupdate = now;
    }
    *events = net_ctl->values;

    return PAPI_OK;
}


static int
_net_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;

    NET_control_state_t *net_ctl = (NET_control_state_t *) ctl;
    long long now = PAPI_get_real_usec();
    int i;

    read_net_counters(_net_register_current);
    for ( i=0; i<NET_MAX_COUNTERS; i++ ) {
        net_ctl->values[i] = _net_register_current[i] - _net_register_start[i];
    }
    net_ctl->lastupdate = now;

    return PAPI_OK;
}


/*
 * Thread shutdown
 */
static int
_net_shutdown_thread( hwd_context_t *ctx )
{
    ( void ) ctx;

    return PAPI_OK;
}


/*
 * Clean up what was setup in net_init_component().
 */
static int
_net_shutdown_component( void )
{
    if ( is_initialized )
    {
      is_initialized = 0;
      if (_net_native_events != NULL)
      {
         papi_free(_net_native_events);
         _net_native_events = NULL;
      }
    }

    return PAPI_OK;
}


/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and
 * PAPI_SET_INHERIT
 */
static int
_net_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    ( void ) ctx;
    ( void ) code;
    ( void ) option;

    return PAPI_OK;
}


static int
_net_update_control_state( hwd_control_state_t *ctl,
        NativeInfo_t *native, int count, hwd_context_t *ctx )
{
    ( void ) ctx;
    ( void ) ctl;

    int i, index;

    for ( i = 0; i < count; i++ ) {
        index = native[i].ni_event;
        native[i].ni_position = _net_native_events[index].resources.selector - 1;
    }

    return PAPI_OK;
}


/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER   is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL    is all of the domains
 */
static int
_net_set_domain( hwd_control_state_t *ctl, int domain )
{
    ( void ) ctl;
    if ( PAPI_DOM_ALL != domain )
        return PAPI_EINVAL;

    return PAPI_OK;
}


int
_net_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;

    return PAPI_OK;
}


/*
 * Native Event functions
 */
static int
_net_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    int index;

    switch ( modifier ) {
        case PAPI_ENUM_FIRST:
            if (num_events==0) {
                return PAPI_ENOEVNT;
            }
            *EventCode = 0;
            return PAPI_OK;
            break;

        case PAPI_ENUM_EVENTS:
            index = *EventCode;
            if ( index < num_events - 1 ) {
                *EventCode = *EventCode + 1;
                return PAPI_OK;
            } else {
                return PAPI_ENOEVNT;
            }
            break;

        default:
            return PAPI_EINVAL;
            break;
    }
    return PAPI_EINVAL;
}


/*
 *
 */
static int
_net_ntv_name_to_code( const char *name, unsigned int *EventCode )
{
    int i;

    for ( i=0; i<num_events; i++) {
        if (strcmp(name, _net_native_events[i].name) == 0) {
	   *EventCode = i;

            return PAPI_OK;
        }
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
static int
_net_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode;

    if ( index >= 0 && index < num_events ) {
        strncpy( name, _net_native_events[index].name, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
static int
_net_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
    int index = EventCode;

    if ( index >= 0 && index < num_events ) {
        strncpy( name, _net_native_events[index].description, len );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
static int
_net_ntv_code_to_bits( unsigned int EventCode, hwd_register_t *bits )
{
    int index = EventCode;

    if ( index >= 0 && index < num_events ) {
        memcpy( ( NET_register_t * ) bits,
                &( _net_native_events[index].resources ),
                sizeof ( NET_register_t ) );
        return PAPI_OK;
    }

    return PAPI_ENOEVNT;
}


/*
 *
 */
papi_vector_t _net_vector = {
    .cmp_info = {
        /* default component information (unspecified values are initialized to 0) */
        .name = "net",
        .short_name = "net",
        .version               = "4.2.1",
	.description = "Linux network driver statistics",
        .num_mpx_cntrs         = NET_MAX_COUNTERS,
        .num_cntrs             = NET_MAX_COUNTERS,
        .default_domain        = PAPI_DOM_ALL,
        .available_domains   = PAPI_DOM_ALL,
        .default_granularity   = PAPI_GRN_SYS,
        .available_granularities = PAPI_GRN_SYS,
        .hardware_intr_sig     = PAPI_INT_SIGNAL,

        /* component specific cmp_info initializations */
        .fast_real_timer       = 0,
        .fast_virtual_timer    = 0,
        .attach                = 0,
        .attach_must_ptrace    = 0,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context               = sizeof ( NET_context_t ),
        .control_state         = sizeof ( NET_control_state_t ),
        .reg_value             = sizeof ( NET_register_t ),
        .reg_alloc             = sizeof ( NET_reg_alloc_t ),
    },

    /* function pointers in this component */
    .init_thread               = _net_init_thread,
    .init_component            = _net_init_component,
    .init_control_state        = _net_init_control_state,
    .start                     = _net_start,
    .stop                      = _net_stop,
    .read                      = _net_read,
    .shutdown_thread           = _net_shutdown_thread,
    .shutdown_component        = _net_shutdown_component,
    .ctl                       = _net_ctl,

    .update_control_state      = _net_update_control_state,
    .set_domain                = _net_set_domain,
    .reset                     = _net_reset,

    .ntv_enum_events           = _net_ntv_enum_events,
    .ntv_name_to_code          = _net_ntv_name_to_code,
    .ntv_code_to_name          = _net_ntv_code_to_name,
    .ntv_code_to_descr         = _net_ntv_code_to_descr,
    .ntv_code_to_bits          = _net_ntv_code_to_bits,
};

/* vim:set ts=4 sw=4 sts=4 et: */
