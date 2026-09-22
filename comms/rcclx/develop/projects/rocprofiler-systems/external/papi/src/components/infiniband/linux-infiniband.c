/** 
 * @file    linux-infiniband.c
 * @author  Heike Jagode 
 *          jagode@icl.utk.edu
 * @author  Gabriel Marin
 *          gmarin@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 *
 * Infiniband component 
 * 
 *
 * @brief 
 *  This file has the source code for a component that enables PAPI-C to access
 *  the infiniband performance monitor through the Linux sysfs interface.
 *  This code will dynamically create a native events table for all the events 
 *  that can be accesed through the sysfs interface. The counters exported by 
 *  this component cannot be reset programatically.
 */


/* Headers required by infiniband */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>
#include <string.h>
#include <dirent.h>
#include <error.h>
#include <time.h>
#include "pscanf.h"

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/
/* this number assumes that there will never be more events than indicated */
#define INFINIBAND_MAX_COUNTERS 256

/** Structure that stores private information of each event */
typedef struct infiniband_register
{
    /* This is used by the framework.It likes it to be !=0 to do somehting */
    unsigned int selector;
} infiniband_register_t;

/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */

typedef struct _ib_device_type
{
    char* dev_name;
    int   dev_port;
    struct _ib_device_type *next;
} ib_device_t;

typedef struct _ib_counter_type
{
    char* ev_name;
    char* ev_file_name;
    ib_device_t* ev_device;
    int extended;   // if this is an extended (64-bit) counter
    struct _ib_counter_type *next;
} ib_counter_t;

static const char *ib_dir_path = "/sys/class/infiniband";

/** This structure is used to build the table of events */
typedef struct _infiniband_native_event_entry
{
    infiniband_register_t resources;
    char *name;
    char *description;
    char* file_name;
    ib_device_t* device;
    int extended;   /* if this is an extended (64-bit) counter */
} infiniband_native_event_entry_t;


typedef struct _infiniband_control_state
{
    long long counts[INFINIBAND_MAX_COUNTERS];
    int being_measured[INFINIBAND_MAX_COUNTERS];
    /* all IB counters need difference, but use a flag for generality */
    int need_difference[INFINIBAND_MAX_COUNTERS];
    long long lastupdate;
} infiniband_control_state_t;


typedef struct _infiniband_context
{
    infiniband_control_state_t state;
    long long start_value[INFINIBAND_MAX_COUNTERS];
} infiniband_context_t;



/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/
/* This table contains the component native events */
static infiniband_native_event_entry_t *infiniband_native_events = 0;
/* number of events in the table*/
static int num_events = 0;


papi_vector_t _infiniband_vector;

/******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
 *****************************************************************************/

static ib_device_t *root_device = 0;
static ib_counter_t *root_counter = 0;

    static char*
make_ib_event_description(const char* input_str, int extended)
{
    int i, len;
    char *desc = 0;
    if (! input_str)
        return (0);

    desc = (char*) papi_calloc(PAPI_MAX_STR_LEN, 1);
    if (desc == 0) {
        PAPIERROR("cannot allocate memory for event description");
        return (0);
    }
    len = strlen(input_str);

    // append additional info to counter description
    int bits = 32;
    if (extended)
        bits = 64;

    int ret;
    // list of event descriptions
    if (strstr(input_str, "rx_atomic_requests")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of received ATOMIC requests for the associated Queue Pairs",
                bits);
    }
    else if (strstr(input_str, "out_of_buffer")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of drops which occurred due to lack of Work Queue Entry for the associated Queue Pairs",
                bits);
    }
    else if (strstr(input_str, "out_of_sequence")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of out of sequence packets received",
                bits);
    }
    else if (strstr(input_str, "lifespan")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Maximum sampling period of the counters in milliseconds",
                bits);
    }
    else if (strstr(input_str, "rx_read_requests")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of received READ requests for the associated Queue Pairs",
                bits);
    }
    else if (strstr(input_str, "rx_write_requests")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of received WRITE requests for the associated Queue Pairs",
                bits);
    }
    else if (strstr(input_str, "port_rcv_data")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of data octets, divided by 4 (lanes), received on all Virtual Lanes. "
                "Multiply this by 4 to get bytes",
                bits);
    }
    else if (strstr(input_str, "port_rcv_packets")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of packets received on all Virtual Lanes from this port, including packets containing errors",
                bits);
    }
    else if (strstr(input_str, "port_multicast_rcv_packets") || strstr(input_str, "multicast_rcv_packets")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of multicast packets, including multicast packets containing errors",
                bits);
    }
    else if (strstr(input_str, "port_unicast_rcv_packets") || strstr(input_str, "unicast_rcv_packets")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of unicast packets, including unicast packets containing errors",
                bits);
    }
    else if (strstr(input_str, "port_xmit_data")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of data octets, divided by 4 (lanes), transmitted on all Virtual Lanes. "
                "Multiply this by 4 to get bytes",
                bits);
    }
    else if (strstr(input_str, "port_xmit_packets")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of packets transmitted on all Virtual Lanes from this port, including packets containing errors",
                bits);
    }
    else if (strstr(input_str, "port_rcv_switch_relay_errors")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of packets received on port that were discarded"
                " because they could not be forwarded by switch relay",
                bits);
    }
    else if (strstr(input_str, "port_rcv_errors")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of packets containing an error that were received on the port",
                bits);
    }
    else if (strstr(input_str, "port_rcv_constraint_errors")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of packets received on the switch physical port that are discarded",
                bits);
    }
    else if (strstr(input_str, "local_link_integrity_errors")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of times the count of local physical errors exceeded threshold",
                bits);
    }
    else if (strstr(input_str, "port_xmit_wait")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of ticks during which port had data to transmit but no data was sent during the entire tick",
                bits);
    }
    else if (strstr(input_str, "port_multicast_xmit_packets") || strstr(input_str, "multicast_xmit_packets")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of multicast packets transmitted on all VLs from port,"
                " including multicast packets containing errors",
                bits);
    }
    else if (strstr(input_str, "port_unicast_xmit_packets") || strstr(input_str, "unicast_xmit_packets")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of unicast packets transmitted on all VLs from port,"
                " including unicast packets containing errors",
                bits);
    }
    else if (strstr(input_str, "port_xmit_discards")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of outbound packets discarded by the port because it is down or congested",
                bits);
    }
    else if (strstr(input_str, "port_xmit_constraint_errors")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of packets not transmitted from the switch physical port",
                bits);
    }
    else if (strstr(input_str, "port_rcv_remote_physical_errors")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of packets marked with EBP (End of Bad Packet) delimiter received on the port",
                bits);
    }
    else if (strstr(input_str, "symbol_error")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of minor link errors detected on one or more physical lanes",
                bits);
    }
    else if (strstr(input_str, "VL15_dropped")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Number of incoming VL15 packets (can include management packets) dropped due to resource limitations of the port",
                bits);
    }
    else if (strstr(input_str, "link_error_recovery")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of times the Port Training state machine has successfully completed the link error recovery process",
                bits);
    }
    else if (strstr(input_str, "link_downed")) {
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%d-bit).",
                "Total number of times the Port Training state machine has failed link error recovery process and downed the link",
                bits);
    }
    else {
        // default event description: re-format the filename
        ret = snprintf(desc, PAPI_MAX_STR_LEN, "%s (%s).",
                input_str, (extended ? "free-running 64bit counter" :
                    "overflowing, auto-resetting counter"));
        desc[0] = toupper(desc[0]);
        for (i=0 ; i<len ; ++i)
            if (desc[i] == '_')
                desc[i] = ' ';
    }
    if (ret <= 0 || PAPI_MAX_STR_LEN <= ret) {
        // nothing smart to be done about it.
    }
    return desc;
}

    static ib_device_t*
add_ib_device(const char* name, int port)
{
    ib_device_t *new_dev = (ib_device_t*) papi_calloc(sizeof(ib_device_t), 1);
    if (new_dev == 0) {
        PAPIERROR("cannot allocate memory for new IB device structure");
        return (0);
    }

    new_dev->dev_name = papi_strdup(name);
    new_dev->dev_port = port;
    if (new_dev->dev_name==0)
    {
        PAPIERROR("cannot allocate memory for device internal fields");
        papi_free(new_dev->dev_name);
        papi_free(new_dev);
        return (0);
    }

    // prepend the new device to the device list
    new_dev->next = root_device;
    root_device = new_dev;

    return (new_dev);
}

    static ib_counter_t*
add_ib_counter(const char* name, const char* file_name, int extended, ib_device_t *device)
{
    ib_counter_t *new_cnt = (ib_counter_t*) papi_calloc(sizeof(ib_counter_t), 1);
    if (new_cnt == 0) {
        PAPIERROR("cannot allocate memory for new IB counter structure");
        return (0);
    }

    new_cnt->ev_name = papi_strdup(name);
    new_cnt->ev_file_name = papi_strdup(file_name);
    new_cnt->extended = extended;
    new_cnt->ev_device = device;
    if (new_cnt->ev_name==0 || new_cnt->ev_file_name==0)
    {
        PAPIERROR("cannot allocate memory for counter internal fields");
        /* free(NULL) is allowed and means no operation is performed */
        papi_free(new_cnt->ev_name);
        papi_free(new_cnt->ev_file_name);
        papi_free(new_cnt);
        return (0);
    }

    // prepend the new counter to the counter list
    new_cnt->next = root_counter;
    root_counter = new_cnt;

    return (new_cnt);
}


    static int
find_ib_device_events(ib_device_t *dev, int extended)
{
    int nevents = 0;
    DIR *cnt_dir = NULL;
    char counters_path[128];

    if ( extended ) {
        /* mofed driver version <4.0 */
        snprintf(counters_path, sizeof(counters_path), "%s/%s/ports/%d/counters_ext", 
                ib_dir_path, dev->dev_name, dev->dev_port);

        cnt_dir = opendir(counters_path);
        if (cnt_dir == NULL) {
            /* directory counters_ext in sysfs fs has changed to hw_counters */
            /* in 4.0 version of mofed driver */
            SUBDBG("cannot open counters directory `%s'\n", counters_path);

            snprintf(counters_path, sizeof(counters_path), "%s/%s/ports/%d/%scounters", 
                    ib_dir_path, dev->dev_name, dev->dev_port, "hw_");

            cnt_dir = opendir(counters_path);
        }
    }
    else {
        snprintf(counters_path, sizeof(counters_path), "%s/%s/ports/%d/counters", 
                ib_dir_path, dev->dev_name, dev->dev_port);
        cnt_dir = opendir(counters_path);
    }

    if (cnt_dir == NULL) {
        SUBDBG("cannot open counters directory `%s'\n", counters_path);
        goto out;
    }

    struct dirent *ev_ent;
    /* iterate over all the events */
    while ((ev_ent = readdir(cnt_dir)) != NULL) {
        char *ev_name = ev_ent->d_name;
        long long value = -1;
        char event_path[FILENAME_MAX];
        char counter_name[PAPI_HUGE_STR_LEN];

        if (ev_name[0] == '.')
            continue;

        /* Check that we can read an integer from the counter file */
        snprintf(event_path, sizeof(event_path), "%s/%s", counters_path, ev_name);
        if (pscanf(event_path, "%lld", &value) != 1) {
            SUBDBG("cannot read value for event '%s'\n", ev_name);
            continue;
        }

        /* Adding 3 exceptions to the counter size
         * https://community.mellanox.com/s/article/understanding-mlx5-linux-counters-and-status-parameters
         * port_rcv_data, port_rcv_packets, and port_xmit_data are listed as 64 bits counters,
         * located in the 32 bits directory */
        int fixed_extended = extended;
        if (   !strcmp("port_rcv_data", ev_name)
            || !strcmp("port_rcv_packets", ev_name)
            || !strcmp("port_xmit_data", ev_name))
            fixed_extended = 3-extended; // higher bit, location is wrong, mark it, and flip the size, 2 + (1-extended)

        /* Create new counter */
        snprintf(counter_name, sizeof(counter_name), "%s_%d%s:%s", 
                dev->dev_name, dev->dev_port, (fixed_extended?"_ext":""), ev_name);
        if (add_ib_counter(counter_name, ev_name, fixed_extended, dev))
        {
            SUBDBG("Added new counter `%s'\n", counter_name);
            nevents += 1;
        }
    }

out:
    if (cnt_dir != NULL)
        closedir(cnt_dir);

    return (nevents);
}

    static int 
find_ib_devices() 
{
    DIR *ib_dir = NULL;
    int result = PAPI_OK;
    num_events = 0;

    ib_dir = opendir(ib_dir_path);
    if (ib_dir == NULL) {
        SUBDBG("cannot open `%s'\n", ib_dir_path);
        strncpy(_infiniband_vector.cmp_info.disabled_reason,
                "Infiniband sysfs interface not found", PAPI_MAX_STR_LEN);
        result = PAPI_ENOSUPP;
        goto out;
    }

    struct dirent *hca_ent;
    while ((hca_ent = readdir(ib_dir)) != NULL) {
        char *hca = hca_ent->d_name;
        char ports_path[FILENAME_MAX];
        DIR *ports_dir = NULL;

        if (hca[0] == '.')
            goto next_hca;

        snprintf(ports_path, sizeof(ports_path), "%s/%s/ports", ib_dir_path, hca);
        ports_dir = opendir(ports_path);
        if (ports_dir == NULL) {
            SUBDBG("cannot open `%s'\n", ports_path);
            goto next_hca;
        }

        struct dirent *port_ent;
        while ((port_ent = readdir(ports_dir)) != NULL) {
            int port = atoi(port_ent->d_name);
            if (port <= 0)
                continue;

            /* Check that port is active. .../HCA/ports/PORT/state should read "4: ACTIVE." */
            int state = -1;
            char state_path[FILENAME_MAX];
            snprintf(state_path, sizeof(state_path), "%s/%s/ports/%d/state", ib_dir_path, hca, port);
            if (pscanf(state_path, "%d", &state) != 1) {
                SUBDBG("cannot read state of IB HCA `%s' port %d\n", hca, port);
                continue;
            }

            if (state != 4) {
                SUBDBG("skipping inactive IB HCA `%s', port %d, state %d\n", hca, port, state);
                continue;
            }

            /* Create dev name (HCA/PORT) and get stats for dev. */
            SUBDBG("Found IB device `%s', port %d\n", hca, port);
            ib_device_t *dev = add_ib_device(hca, port);
            if (!dev)
                continue;
            // do we want to check for short counters only if no extended counters found?
            num_events += find_ib_device_events(dev, 1);  // check if we have extended (64bit) counters
            num_events += find_ib_device_events(dev, 0);  // check also for short counters
        }

next_hca:
        if (ports_dir != NULL)
            closedir(ports_dir);
    }

    if (root_device == 0)  // no active devices found
    {
        strncpy(_infiniband_vector.cmp_info.disabled_reason,
                "No active Infiniband ports found", PAPI_MAX_STR_LEN);
        result = PAPI_ENOIMPL;
    } else if (num_events == 0)
    {
        strncpy(_infiniband_vector.cmp_info.disabled_reason,
                "No supported Infiniband events found", PAPI_MAX_STR_LEN);
        result = PAPI_ENOIMPL;
    } else
    {
        // Events are stored in a linked list, in reverse order than how I found them
        // Revert them again, so that they are in finding order, not that it matters.
        int i = num_events - 1;
        // now allocate memory to store the counters into the native table
        infiniband_native_events = (infiniband_native_event_entry_t*)
            papi_calloc(num_events, sizeof(infiniband_native_event_entry_t));
        ib_counter_t *iter = root_counter;
        while (iter != 0)
        {
            infiniband_native_events[i].name = iter->ev_name;
            infiniband_native_events[i].file_name = iter->ev_file_name;
            infiniband_native_events[i].device = iter->ev_device;
            infiniband_native_events[i].extended = iter->extended;
            infiniband_native_events[i].resources.selector = i + 1;
            infiniband_native_events[i].description = 
                make_ib_event_description(iter->ev_file_name, iter->extended);

            ib_counter_t *tmp = iter;
            iter = iter->next;
            papi_free(tmp);
            -- i;
        }
        root_counter = 0;
    }

out:
    if (ib_dir != NULL)
        closedir(ib_dir);

    return (result);
}

    static long long
read_ib_counter_value(int index)
{
    char ev_file[FILENAME_MAX];
    char counters_path[FILENAME_MAX];
    DIR *cnt_dir = NULL;
    long long value = 0ll;
    infiniband_native_event_entry_t *iter = &infiniband_native_events[index];

    if ( iter->extended == 1 || iter->extended == 2 ) {
        /* extended == 1, counter is 32b, in the 32b location || 2, counter is 64b in the 64b dir */
        /* mofed driver version <4.0 */
        snprintf(counters_path, sizeof(counters_path), "%s/%s/ports/%d/counters%s",
                ib_dir_path, iter->device->dev_name, iter->device->dev_port, "_ext");

        cnt_dir = opendir(counters_path);
        if (cnt_dir == NULL) {
            /* directory counters_ext in sysfs fs has changed to hw_counters */
            /* in 4.0 version of mofed driver */
            snprintf(counters_path, sizeof(counters_path), "%s/%s/ports/%d/%scounters",
                    ib_dir_path, iter->device->dev_name, iter->device->dev_port, "hw_");

            cnt_dir = opendir(counters_path);
        }
    }
    else {
        /* extended == 0, counter is 32b, in the 64b location || 3, counter is 64b in the 32b dir */
        snprintf(counters_path, sizeof(counters_path), "%s/%s/ports/%d/counters",
                ib_dir_path, iter->device->dev_name, iter->device->dev_port );
        cnt_dir = opendir(counters_path);
    }


    if (cnt_dir != NULL)
        closedir(cnt_dir);


    snprintf(ev_file, strlen(counters_path) + strlen(iter->file_name) + 2, "%s/%s",
            counters_path, iter->file_name);

    if (pscanf(ev_file, "%lld", &value) != 1) {
        PAPIERROR("cannot read value for counter '%s'\n", iter->name);
    } else
    {
        SUBDBG("Counter '%s': %lld\n", iter->name, value);
    }
    return (value);
}

    static void
deallocate_infiniband_resources()
{
    int i;

    if (infiniband_native_events)
    {
        for (i=0 ; i<num_events ; ++i) {
            if (infiniband_native_events[i].name)
                papi_free(infiniband_native_events[i].name);
            if (infiniband_native_events[i].file_name)
                papi_free(infiniband_native_events[i].file_name);
            if (infiniband_native_events[i].description)
                papi_free(infiniband_native_events[i].description);
        }
        papi_free(infiniband_native_events);
    }

    ib_device_t *iter = root_device;
    while (iter != 0) 
    {
        if (iter->dev_name)
            papi_free(iter->dev_name);

        ib_device_t *tmp = iter;
        iter = iter->next;
        papi_free(tmp);
    }
    root_device = 0;
}

/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * This is called whenever a thread is initialized
 */
    static int
_infiniband_init_thread( hwd_context_t *ctx )
{
    (void) ctx;
    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
    static int
_infiniband_init_component( int cidx )
{
    /* discover Infiniband devices and available events */
    int result = find_ib_devices();

    if (result != PAPI_OK)  // we couldn't initialize the component
    {
        // deallocate any eventually allocated memory
        deallocate_infiniband_resources();
    }

    _infiniband_vector.cmp_info.num_native_events = num_events;

    _infiniband_vector.cmp_info.num_cntrs = num_events;
    _infiniband_vector.cmp_info.num_mpx_cntrs = num_events;


    /* Export the component id */
    _infiniband_vector.cmp_info.CmpIdx = cidx;
    _papi_hwd[cidx]->cmp_info.disabled = result;

    return (result);
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
    static int
_infiniband_init_control_state( hwd_control_state_t *ctl )
{
    infiniband_control_state_t* control = (infiniband_control_state_t*) ctl;
    int i;

    for (i=0 ; i<INFINIBAND_MAX_COUNTERS ; ++i) {
        control->being_measured[i] = 0;
    }

    return PAPI_OK;
}

/*
 *
 */
    static int
_infiniband_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    infiniband_context_t* context = (infiniband_context_t*) ctx;
    infiniband_control_state_t* control = (infiniband_control_state_t*) ctl;
    long long now = PAPI_get_real_usec();
    int i;

    for (i=0 ; i<INFINIBAND_MAX_COUNTERS ; ++i) {
        if (control->being_measured[i] && control->need_difference[i]) {
            context->start_value[i] = read_ib_counter_value(i);
        }
    }
    control->lastupdate = now;

    return PAPI_OK;
}


/*
 *
 */
    static int
_infiniband_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    infiniband_context_t* context = (infiniband_context_t*) ctx;
    infiniband_control_state_t* control = (infiniband_control_state_t*) ctl;
    long long now = PAPI_get_real_usec();
    int i;
    long long temp;

    for (i=0 ; i<INFINIBAND_MAX_COUNTERS ; ++i) {
        if (control->being_measured[i])
        {
            temp = read_ib_counter_value(i);
            if (context->start_value[i] && control->need_difference[i]) {
                /* Must subtract values, but check for wraparound. 
                 * We cannot even detect all wraparound cases. Using the short,
                 * auto-resetting IB counters is error prone.
                 */
                if (temp < context->start_value[i]) {
                    SUBDBG("Wraparound!\nstart:\t%#016x\ttemp:\t%#016x",
                            (unsigned)context->start_value[i], (unsigned)temp);
                    /* The counters auto-reset. I cannot even adjust them to 
                     * account for a simple wraparound. 
                     * Just use the current reading of the counter, which is useless.
                     */
                } else
                    temp -= context->start_value[i];
            }
            control->counts[i] = temp;
        }
    }
    control->lastupdate = now;

    return PAPI_OK;
}


/*
 *
 */
    static int
_infiniband_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
        long_long ** events, int flags )
{
    ( void ) flags;

    _infiniband_stop(ctx, ctl);  /* we cannot actually stop the counters */
    /* Pass back a pointer to our results */
    *events = ((infiniband_control_state_t*) ctl)->counts;

    return PAPI_OK;
}


    static int
_infiniband_shutdown_component( void )
{
    /* Cleanup resources used by this component before leaving */
    deallocate_infiniband_resources();

    return PAPI_OK;
}

    static int
_infiniband_shutdown_thread( hwd_context_t *ctx )
{
    ( void ) ctx;

    return PAPI_OK;
}



/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
    static int
_infiniband_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    ( void ) ctx;
    ( void ) code;
    ( void ) option;
    return PAPI_OK;
}


    static int
_infiniband_update_control_state( hwd_control_state_t *ctl,
        NativeInfo_t * native, 
        int count,
        hwd_context_t *ctx )
{
    int i, index;
    ( void ) ctx;

    infiniband_control_state_t* control = (infiniband_control_state_t*) ctl;

    for (i=0 ; i<INFINIBAND_MAX_COUNTERS ; ++i) {
        control->being_measured[i] = 0;
    }

    for (i=0 ; i<count ; ++i) {
        index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
        native[i].ni_position =
            infiniband_native_events[index].resources.selector - 1;
        control->being_measured[index] = 1;
        control->need_difference[index] = 1;
    }
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
_infiniband_set_domain( hwd_control_state_t *ctl, int domain )
{
    int found = 0;
    (void) ctl;

    if (PAPI_DOM_USER & domain)
        found = 1;

    if (PAPI_DOM_KERNEL & domain)
        found = 1;

    if (PAPI_DOM_OTHER & domain)
        found = 1;

    if (!found)
        return (PAPI_EINVAL);

    return (PAPI_OK);
}


/*
 * Cannot reset the counters using the sysfs interface.
 */
    static int
_infiniband_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    (void) ctl;
    return PAPI_OK;
}


/*
 * Native Event functions
 */
    static int
_infiniband_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    switch (modifier) {
        case PAPI_ENUM_FIRST:
            if (num_events == 0)
                return (PAPI_ENOEVNT);

            *EventCode = 0;
            return PAPI_OK;

        case PAPI_ENUM_EVENTS:
            {
                int index = *EventCode & PAPI_NATIVE_AND_MASK;

                if (index < num_events - 1) {
                    *EventCode = *EventCode + 1;
                    return PAPI_OK;
                } else
                    return PAPI_ENOEVNT;

                break;
            }
        default:
            return PAPI_EINVAL;
    }
    return PAPI_EINVAL;
}

/*
 *
 */
    static int
_infiniband_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode;

    if (index>=0 && index<num_events) {
        strncpy( name, infiniband_native_events[index].name, len );
    }

    return PAPI_OK;
}

/*
 *
 */
    static int
_infiniband_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
    int index = EventCode;

    if (index>=0 && index<num_events) {
        strncpy(name, infiniband_native_events[index].description, len);
    }
    return PAPI_OK;
}

    static int
_infiniband_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info)
{
    int index = EventCode;

    if ( ( index < 0) || (index >= num_events )) return PAPI_ENOEVNT; 

    if (infiniband_native_events[index].name)
    {
        strncpy(info->symbol, infiniband_native_events[index].name, PAPI_HUGE_STR_LEN);
        info->symbol[PAPI_HUGE_STR_LEN-1] = '\0';
    }
    if (infiniband_native_events[index].description)
    {
        strncpy(info->long_descr, infiniband_native_events[index].description, PAPI_MAX_STR_LEN);
        info->long_descr[PAPI_MAX_STR_LEN-1] = '\0';
    }

    strncpy(info->units, "\0", 1);
    /* infiniband_native_events[index].units, sizeof(info->units)); */

    /*   info->data_type = infiniband_native_events[index].return_type;
    */
    return PAPI_OK;
}


/*
 *
 */
papi_vector_t _infiniband_vector = {
    .cmp_info = {
        /* component information (unspecified values are initialized to 0) */
        .name = "infiniband",
        .short_name = "infiniband",
        .version = "5.3.0",
        .description = "Linux Infiniband statistics using the sysfs interface",
        .num_mpx_cntrs = INFINIBAND_MAX_COUNTERS,
        .num_cntrs = INFINIBAND_MAX_COUNTERS,
        .default_domain = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .default_granularity = PAPI_GRN_SYS,
        .available_granularities = PAPI_GRN_SYS,
        .hardware_intr_sig = PAPI_INT_SIGNAL,

        /* component specific cmp_info initializations */
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context = sizeof (infiniband_context_t),
        .control_state = sizeof (infiniband_control_state_t),
        .reg_value = sizeof (infiniband_register_t),
        .reg_alloc = 1 /* unused */
    },
    /* function pointers in this component */
    .init_thread =          _infiniband_init_thread,
    .init_component =       _infiniband_init_component,
    .init_control_state =   _infiniband_init_control_state,
    .start =                _infiniband_start,
    .stop =                 _infiniband_stop,
    .read =                 _infiniband_read,
    .shutdown_thread =      _infiniband_shutdown_thread,
    .shutdown_component =   _infiniband_shutdown_component,
    .ctl =                  _infiniband_ctl,
    .update_control_state = _infiniband_update_control_state,
    .set_domain =           _infiniband_set_domain,
    .reset =                _infiniband_reset,

    .ntv_enum_events =      _infiniband_ntv_enum_events,
    .ntv_code_to_name =     _infiniband_ntv_code_to_name,
    .ntv_code_to_descr =    _infiniband_ntv_code_to_descr,
    .ntv_code_to_info =     _infiniband_ntv_code_to_info,
};
