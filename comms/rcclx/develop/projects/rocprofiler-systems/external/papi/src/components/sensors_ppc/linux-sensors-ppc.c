/**
 * @file    linux-sensors_ppc.c
 * @author  Philip Vaccaro
 * @ingroup papi_components
 * @brief sensors_ppc component
 *
 * To work, the sensors_ppc kernel module must be loaded.
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
#include <pthread.h>
#include <inttypes.h>

#include "linux-sensors-ppc.h"

// The following macro exit if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

papi_vector_t _sensors_ppc_vector;

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

#define DESC_LINE_SIZE_ALLOWED 66
static void
_space_padding(char *buf, size_t max)
{
    size_t len = strlen(buf);
    /* 80 columns - 12 header - 2 footer*/
    size_t nlines = 1+ len / DESC_LINE_SIZE_ALLOWED, c = len;
    /* space_padding */
    for (; c < nlines * DESC_LINE_SIZE_ALLOWED && c < max-1; ++c) buf[c] = ' ';
    buf[c] = '\0';
}

/** @brief Refresh_data locks in write and update ping and pong at
 *  the same time for OCC occ_id.
 *  The occ_names array contains constant memory and doesn't
 *  need to be updated.
 *  Ping and Pong are read outside of the critical path, and
 *  only the swap needs to be protected.
 * */
static void
refresh_data(int occ_id, int forced)
{
    long long now = PAPI_get_real_nsec();
    if (forced || now > last_refresh[occ_id] + OCC_REFRESH_TIME) {
        void *buf = double_ping[occ_id];

        uint32_t ping_off = be32toh(occ_hdr[occ_id]->reading_ping_offset);
        uint32_t pong_off = be32toh(occ_hdr[occ_id]->reading_pong_offset);

        lseek (event_fd, occ_id * OCC_SENSOR_DATA_BLOCK_SIZE + ping_off, SEEK_SET);

        /* To limit risks of begin desynchronized, we read one chunk */
        /* In memory, ping and pong are 40kB, with a 4kB buffer
         * of nothingness in between */
        int to_read = pong_off - ping_off + OCC_PING_DATA_BLOCK_SIZE;

        int rc, bytes;
        /* copy memory iteratively until the full chunk is saved */
        for (rc = bytes = 0; bytes < to_read; bytes += rc) {
            rc = read(event_fd, buf + bytes, to_read - bytes);
            if (!rc || rc < 0) /* done */ break;
        }

        papi_sensors_ppc_lock();
        double_ping[occ_id] = ping[occ_id];
        ping[occ_id] = buf;
        pong[occ_id] = ping[occ_id] + (pong_off - ping_off);
        last_refresh[occ_id] = now;
        papi_sensors_ppc_unlock();
    }
}
static double
_pow(int x, int y)
{
    if (0 == y)   return 1.;
    if (0 == x)   return 0.;
    if (0  > y)   return 1. / _pow(x, -y);
    if (1 == y)   return 1. * x;
    if (0 == y%2) return _pow(x, y/2) * _pow(x, y/2);
    else          return _pow(x, y/2) * _pow(x, y/2) * x;
}

#define TO_FP(f)    ((f >> 8) * _pow(10, ((int8_t)(f & 0xFF))))

static long long
read_sensors_ppc_record(int s, int gidx, int midx)
{
    uint64_t value = 41;
    uint32_t offset = be32toh(occ_names[s][gidx].reading_offset);
    uint32_t scale  = be32toh(occ_names[s][gidx].scale_factor);
    uint32_t freq   = be32toh(occ_names[s][gidx].freq);

    occ_sensor_record_t *record = NULL;
    /* Let's see if the data segment needs a refresh */
    refresh_data(s, 0);

    papi_sensors_ppc_lock();
    occ_sensor_record_t *sping = (occ_sensor_record_t *)((uint64_t)ping[s] + offset);
    occ_sensor_record_t *spong = (occ_sensor_record_t *)((uint64_t)pong[s] + offset);

    if (*ping && *pong) {
        if (be64toh(sping->timestamp) > be64toh(spong->timestamp))
            record = sping;
        else
            record = spong;
    } else if (*ping && !*pong) {
        record = sping;
    } else if (!*ping && *pong) {
        record = spong;
    } else if (!*ping && !*pong) {
        return value;
    }

    switch (midx) {
        case OCC_SENSORS_ACCUMULATOR_TAG:
            /* freq, per sensor, contains freq sampling for the last 500us of accumulation */
            value = (uint64_t)(be64toh(record->accumulator) / TO_FP(freq));
            break;
        default:
            /* That one might upset people
             * All the entries below sample (including it) are uint16_t packed */
            value = (uint64_t)(be16toh((&record->sample)[midx]) * TO_FP(scale));
            break;
    }
    papi_sensors_ppc_unlock();

    return value;
}

static long long
read_sensors_ppc_counter(int s, int gidx)
{
    uint32_t offset = be32toh(occ_names[s][gidx].reading_offset);
    uint32_t scale  = be32toh(occ_names[s][gidx].scale_factor);

    occ_sensor_counter_t *counter = NULL;

    refresh_data(s, 0);

    papi_sensors_ppc_lock();
    occ_sensor_counter_t *sping = (occ_sensor_counter_t *)((uint64_t)ping[s] + offset);
    occ_sensor_counter_t *spong = (occ_sensor_counter_t *)((uint64_t)pong[s] + offset);

    if (*ping && *pong) {
        if (be64toh(sping->timestamp) > be64toh(spong->timestamp))
            counter = sping;
        else
            counter = spong;
    } else if (*ping && !*pong) {
        counter = sping;
    } else if (!*ping && *pong) {
        counter = spong;
    } else if (!*ping && !*pong) {
        return 40;
    }

    uint64_t value = be64toh(counter->accumulator) * TO_FP(scale);
    papi_sensors_ppc_unlock();

    return value;
}

static int
_sensors_ppc_is_counter(int index)
{
    int s = 0;
    /* get OCC s from index */
    for (; index > occ_num_events[s+1] && s < MAX_OCCS; ++s);

    int ridx = index - occ_num_events[s];
    int gidx = ridx / OCC_SENSORS_MASKS;
    return (OCC_SENSOR_READING_COUNTER == occ_names[s][gidx].structure_type);
}

static long long
read_sensors_ppc_value( int index )
{
    int s = 0;
    /* get OCC s from index */
    for (; index > occ_num_events[s+1] && s < MAX_OCCS; ++s);

    int ridx = index - occ_num_events[s];
    int gidx = ridx / OCC_SENSORS_MASKS;
    int midx = ridx % OCC_SENSORS_MASKS;
    uint8_t structure_type = occ_names[s][gidx].structure_type;

    switch (structure_type) {
        case OCC_SENSOR_READING_FULL:
            return read_sensors_ppc_record(s, gidx, midx);
        case OCC_SENSOR_READING_COUNTER:
            if (OCC_SENSORS_ACCUMULATOR_TAG == midx)
                return read_sensors_ppc_counter(s, gidx);
            /* fall through */
            /* counters only return the accumulator */
        default:
            return 42;
    }
}


/************************* PAPI Functions **********************************/

/*
 * This is called whenever a thread is initialized
 */
static int
_sensors_ppc_init_thread( hwd_context_t *ctx )
{
    (void) ctx;

    return PAPI_OK;
}

/*
 * Called when PAPI process is initialized (i.e. PAPI_library_init)
 */
static int
_sensors_ppc_init_component( int cidx )
{
    int retval = PAPI_OK;
    int s = -1;
    int strErr;
    char events_dir[128];
    char event_path[128];
    char *strCpy;
    DIR *events;

    const PAPI_hw_info_t *hw_info;
    hw_info=&( _papi_hwi_system_info.hw_info );

    if ( PAPI_VENDOR_IBM != hw_info->vendor ) {
        strCpy=strncpy(_sensors_ppc_vector.cmp_info.disabled_reason, "Not an IBM processor", PAPI_MAX_STR_LEN);
        _sensors_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        retval = PAPI_ENOSUPP;
        goto fn_fail;
    }

    int ret = snprintf(events_dir, sizeof(events_dir), "/sys/firmware/opal/exports/");
    if (ret <= 0 || (int)(sizeof(events_dir)) <= ret) HANDLE_STRING_ERROR;
    if (NULL == (events = opendir(events_dir))) {
        strErr=snprintf(_sensors_ppc_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
          "%s:%i Could not open events_dir='%s'.", __FILE__, __LINE__, events_dir);
        _sensors_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        retval = PAPI_ENOSUPP;
        goto fn_fail;
    }

    ret = snprintf(event_path, sizeof(event_path), "%s%s", events_dir, pkg_sys_name);
    if (ret <= 0 || (int)(sizeof(event_path)) <= ret) HANDLE_STRING_ERROR;
    if (-1 == access(event_path, F_OK)) {
        strErr=snprintf(_sensors_ppc_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
          "%s:%i Could not access event_path='%s'.", __FILE__, __LINE__, event_path);
        _sensors_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        retval = PAPI_ENOSUPP;
        goto fn_fail;
    }

    event_fd = open(event_path, pkg_sys_flag);
    if (event_fd < 0) {
        strErr=snprintf(_sensors_ppc_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
          "%s:%i Could not open event_path='%s'.", __FILE__, __LINE__, event_path);
        _sensors_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        retval = PAPI_ENOSUPP;
        goto fn_fail;
    }

    memset(occ_num_events, 0, (MAX_OCCS+1)*sizeof(int));
    num_events = 0;
    for ( s = 0; s < MAX_OCCS; ++s ) {
        void *buf = NULL;
        if (NULL == (buf = malloc(OCC_SENSOR_DATA_BLOCK_SIZE))) {
            strErr=snprintf(_sensors_ppc_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "%s:%i Failed to alloc %i bytes for buf.", __FILE__, __LINE__, OCC_SENSOR_DATA_BLOCK_SIZE);
            _sensors_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            retval = PAPI_ENOMEM;
            goto fn_fail;
        }

        occ_hdr[s] = (struct occ_sensor_data_header_s*)buf;

        lseek (event_fd, s * OCC_SENSOR_DATA_BLOCK_SIZE, SEEK_SET);

        int rc, bytes;
        /* copy memory iteratively until the full chunk is saved */
        for (rc = bytes = 0; bytes < OCC_SENSOR_DATA_BLOCK_SIZE; bytes += rc) {
            rc = read(event_fd, buf + bytes, OCC_SENSOR_DATA_BLOCK_SIZE - bytes);
            if (!rc || rc < 0) /* done */ break;
        }

        if (OCC_SENSOR_DATA_BLOCK_SIZE != bytes) {
            /* We are running out of OCCs, let's stop there */
            free(buf);
            num_occs = s;
            s = MAX_OCCS;
            continue;
        }

        occ_sensor_name_t *names = (occ_sensor_name_t*)((uint64_t)buf + be32toh(occ_hdr[s]->names_offset));
        int n_sensors = be16toh(occ_hdr[s]->nr_sensors);

        /* Prepare the double buffering for the ping/pong buffers */
        int ping_off = be32toh(occ_hdr[s]->reading_ping_offset);
        int pong_off = be32toh(occ_hdr[s]->reading_pong_offset);
        /* Ping and pong are both 40kB, and we have a 4kB separator.
         * In theory, the distance between the beginnings of ping and pong is (40+4) kB.
         * But they expose an offset for the pong buffer.
         * So I won't trust the 4kB distance between buffers, and compute the buffer size
         * based on on both offsets ans the size of pong */
        int buff_size = pong_off - ping_off + OCC_PING_DATA_BLOCK_SIZE;

        ping[s] = (uint32_t*)malloc(buff_size);
        if (ping[s] == NULL) {
            strErr=snprintf(_sensors_ppc_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "%s:%i Failed to alloc %i bytes for ping[%i].", __FILE__, __LINE__, buff_size, s);
            _sensors_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            retval = PAPI_ENOMEM;
            goto fn_fail;
        }

        double_ping[s] = (uint32_t*)malloc(buff_size);
        if (double_ping[s] == NULL) {
            strErr=snprintf(_sensors_ppc_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
            "%s:%i Failed to alloc %i bytes for double_ping[%i].", __FILE__, __LINE__, buff_size, s);
            _sensors_ppc_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            retval = PAPI_ENOMEM;
            goto fn_fail;
        }

        double_pong[s] = double_ping[s];

        refresh_data(s, 1);

        /* Not all events will exist, counter-based evens only have an accumulator to report */
        occ_num_events[s+1] = occ_num_events[s] + (n_sensors * OCC_SENSORS_MASKS);

        num_events += (n_sensors * OCC_SENSORS_MASKS);

        /* occ_names map to read-only information that change only after reboot */
        occ_names[s] = names;
    }

    /* Export the total number of events available */
    _sensors_ppc_vector.cmp_info.num_native_events = num_events;
    _sensors_ppc_vector.cmp_info.num_cntrs = num_events;
    _sensors_ppc_vector.cmp_info.num_mpx_cntrs = num_events;

    /* 0 active events */
    num_events = 0;

    /* Export the component id */
    _sensors_ppc_vector.cmp_info.CmpIdx = cidx;

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
_sensors_ppc_init_control_state( hwd_control_state_t *ctl )
{
    _sensors_ppc_control_state_t* control = ( _sensors_ppc_control_state_t* ) ctl;

    memset( control, 0, sizeof ( _sensors_ppc_control_state_t ) );

    return PAPI_OK;
}

static int
_sensors_ppc_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    SUBDBG("Enter _sensors_ppc_start\n");

    _sensors_ppc_context_t*       context = ( _sensors_ppc_context_t* ) ctx;
    _sensors_ppc_control_state_t* control = ( _sensors_ppc_control_state_t* ) ctl;

    memset( context->start_value, 0, sizeof(long long) * SENSORS_PPC_MAX_COUNTERS);

    int c, i;
    for( c = 0; c < num_events; c++ ) {
        i = control->which_counter[c];
        if (_sensors_ppc_is_counter(i))
            context->start_value[c] = read_sensors_ppc_value(i);
    }

    /* At the end, ctx->start if full of 0s, except for counter-type sensors */
    return PAPI_OK;
}

static int
_sensors_ppc_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    (void) ctl;

    /* not sure what the side effect of stop is supposed to be, do a read? */
    return PAPI_OK;
}

/* Shutdown a thread */
static int
_sensors_ppc_shutdown_thread( hwd_context_t *ctx )
{
    (void) ctx;

    return PAPI_OK;
}


static int
_sensors_ppc_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
                   long long **events, int flags )
{
    SUBDBG("Enter _sensors_ppc_read\n");

    (void) flags;
    _sensors_ppc_control_state_t* control = ( _sensors_ppc_control_state_t* ) ctl;
    _sensors_ppc_context_t* context = ( _sensors_ppc_context_t* ) ctx;

    long long start_val = 0;
    long long curr_val = 0;
    int c, i;

    /* c is the index in the dense array of selected counters */
    /* using control->which_counters[c], fetch actual indices in i */
    /* all subsequent methods use "global" indices i */
    for ( c = 0; c < num_events; c++ ) {
        i = control->which_counter[c];
        start_val = context->start_value[c];
        curr_val = read_sensors_ppc_value(i);

        if (start_val) {
            /* Make sure an event is a counter. */
            if (_sensors_ppc_is_counter(i)) {
                /* Wraparound. */
                if(start_val > curr_val) {
                    curr_val += (0x100000000 - start_val);
                }
                /* Normal subtraction. */
                else if (start_val < curr_val) {
                    curr_val -= start_val;
                }
            }
        }
        control->count[c]=curr_val;
    }

    *events = ( ( _sensors_ppc_control_state_t* ) ctl )->count;
    return PAPI_OK;
}

/*
 * Clean up what was setup in sensors_ppc_init_component().
 */
static int
_sensors_ppc_shutdown_component( void )
{
    close(event_fd);

    int s;
    papi_sensors_ppc_lock();
    for (s = 0; s < num_occs; ++s) {
        free(occ_hdr[s]);
        if (ping[s] != NULL) free(ping[s]);
        if (double_ping[s] != NULL) free(double_ping[s]);
    }
    papi_sensors_ppc_unlock();
    return PAPI_OK;
}

/* This function sets various options in the component. The valid
 * codes being passed in are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN,
 * PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 */
static int
_sensors_ppc_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    SUBDBG( "Enter: ctx: %p\n", ctx );
    (void) ctx;
    (void) code;
    (void) option;

    return PAPI_OK;
}


static int
_sensors_ppc_update_control_state( hwd_control_state_t *ctl,
                                   NativeInfo_t *native, int count,
                                   hwd_context_t *ctx )
{
    (void) ctx;
    int i, index;
    num_events = count;
    _sensors_ppc_control_state_t* control = ( _sensors_ppc_control_state_t* ) ctl;
    if (count == 0) return PAPI_OK;

    /* control contains a dense array of unsorted events */
    for ( i = 0; i < count; i++ ) {
        index = native[i].ni_event;
        control->which_counter[i]=index;
        native[i].ni_position = i;
    }

    return PAPI_OK;
}

static int
_sensors_ppc_set_domain( hwd_control_state_t *ctl, int domain )
{
    (void) ctl;
    if ( PAPI_DOM_ALL != domain )
        return PAPI_EINVAL;
    return PAPI_OK;
}

static int
_sensors_ppc_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    (void) ctl;

    return PAPI_OK;
}

/*
 * Iterator function. Given an Eventcode, returns the next valid Eventcode to consider
 * returning anything but PAPI_OK will stop lookups and ignore next events.
 */
static int
_sensors_ppc_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    int index;
    switch (modifier) {

        case PAPI_ENUM_FIRST:
            *EventCode = 0;
            return PAPI_OK;

        case PAPI_ENUM_EVENTS:
            index = *EventCode & PAPI_NATIVE_AND_MASK;
            if (index < occ_num_events[num_occs] - 1) {
                if (_sensors_ppc_is_counter(index+1))
                    /* For counters, exposing only the accumulator,
                     * skips ghost events from _sample to _job_sched_max */
                    *EventCode = *EventCode + OCC_SENSORS_MASKS;
                else
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
_sensors_ppc_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    int index = EventCode & PAPI_NATIVE_AND_MASK;

    if ( index < 0 && index >= occ_num_events[num_occs] )
        return PAPI_ENOEVNT;

    int s = 0;
    /* get OCC s from index */
    for (; index > occ_num_events[s+1] && s < MAX_OCCS; ++s);

    int ridx = index - occ_num_events[s];
    int gidx = ridx / OCC_SENSORS_MASKS;
    int midx = ridx % OCC_SENSORS_MASKS;

    /* EventCode maps to a counter */
    /* Counters only expose their accumulator */
    if (_sensors_ppc_is_counter(index) && midx != OCC_SENSORS_ACCUMULATOR_TAG)
        return PAPI_ENOEVNT;

    char buf[512];
    int ret = snprintf(buf, 512, "%s:occ=%d%s", occ_names[s][gidx].name, s, sensors_ppc_fake_qualifiers[midx]);
    if (ret <= 0 || 512 <= ret)
        return PAPI_ENOSUPP;
    _local_strlcpy( name, buf, len);

    return PAPI_OK;
}

/* This is the optional function used by utils/papi_*_avail.
 * Not providing it will force the tools to forge a description using
 * ntv_code_to_desc, ntv_code_to_*.
 */
static int
_sensors_ppc_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t *info )
{
    int index = EventCode;

    if ( index < 0 || index >= occ_num_events[num_occs])
        return PAPI_ENOEVNT;

    int s = 0;
    /* get OCC s from index */
    for (; index > occ_num_events[s+1] && s < MAX_OCCS; ++s);

    int ridx = index - occ_num_events[s];
    int gidx = ridx / OCC_SENSORS_MASKS;
    int midx = ridx % OCC_SENSORS_MASKS;

    /* EventCode maps to a counter */
    /* Counters only expose their accumulator */
    if (_sensors_ppc_is_counter(index) && midx != OCC_SENSORS_ACCUMULATOR_TAG)
        return PAPI_ENOEVNT;

    char buf[512];
    int ret = snprintf(buf, 512, "%s:occ=%d%s", occ_names[s][gidx].name, s, sensors_ppc_fake_qualifiers[midx]);
    if (ret <= 0 || 512 <= ret)
        return PAPI_ENOSUPP;
    _local_strlcpy( info->symbol, buf, sizeof( info->symbol ));
    _local_strlcpy( info->units, occ_names[s][gidx].units, sizeof( info->units ) );
    /* If it ends with:
     * Qw: w-th Quad unit        [0-5]
     * Cxx: xx-th core           [0-23]
     * My: y-th memory channel   [0-8]
     * CHvv: vv-th memory module [0-15]
     * or starts with:
     * GPUz: z-th GPU            [0-2]
     * TEMPGPUz: z-th GPU        [0-2]
     * */
    uint16_t type = be16toh(occ_names[s][gidx].type);
    char *name = strdup(occ_names[s][gidx].name);
    uint32_t freq = be32toh(occ_names[s][gidx].freq);
    int tgt = -1;
    switch(type) {
        /* IPS, STOPDEEPACTCxx, STOPDEEPREQCxx, IPSCxx, NOTBZECxx, NOTFINCxx,
         *   MRDMy, MWRMy, PROCPWRTHROT, PROCOTTHROT, MEMPWRTHROT, MEMOTTHROT,
         *   GPUzHWTHROT, GPUzSWTHROT, GPUzSWOTTHROT, GPUzSWPWRTHROT */
        case OCC_SENSOR_TYPE_PERFORMANCE:
            if (!strncmp(name, "GPU", 3)) {
                char z[] = {name[3], '\0'};
                tgt = atoi(z);
                name[3] = 'z';
                if (!strncmp(name, "GPUzHWTHROT", 11))
                    ret = snprintf(buf, 512, "Total time GPU %d has been throttled by hardware (thermal or power brake)", tgt);
                else if (!strncmp(name, "GPUzSWTHROT", 11))
                    ret = snprintf(buf, 512, "Total time GPU %d has been throttled by software for any reason", tgt);
                else if (!strncmp(name, "GPUzSWOTTHROT", 13))
                    ret = snprintf(buf, 512, "Total time GPU %d has been throttled by software due to thermal", tgt);
                else if (!strncmp(name, "GPUzSWPWRTHROT", 14))
                    ret = snprintf(buf, 512, "Total time GPU %d has been throttled by software due to power", tgt);
                else
                    ret = snprintf(buf, 512, "[PERFORMANCE] Unexpected: GPU-%d %s", tgt, name);
            }
            else if (!strncmp(name, "IPSCxx", 4)) {
                tgt = atoi(name+4);
                ret = snprintf(buf, 512, "Instructions per second for core %d on this Processor", tgt);
            }
            else if (!strncmp(name, "IPS", 3))
                ret = snprintf(buf, 512, "Vector sensor that takes the average of all the cores this Processor");
            else if (!strncmp(name, "STOPDEEPACTCxx", 12)) {
                tgt = atoi(name+12);
                ret = snprintf(buf, 512, "Deepest actual stop state that was fully entered during sample time for core %d", tgt);
            }
            else if (!strncmp(name, "STOPDEEPREQCxx", 12)) {
                tgt = atoi(name+12);
                ret = snprintf(buf, 512, "Deepest stop state that has been requested during sample time for core %d", tgt);
            }
            else if (!strncmp(name, "MEMPWRTHROT", 11))
                ret = snprintf(buf, 512, "Count of memory throttled due to power");
            else if (!strncmp(name, "MEMOTTHROT", 10))
                ret = snprintf(buf, 512, "Count of memory throttled due to memory Over temperature");
            else if (!strncmp(name, "PROCOTTHROT", 11))
                ret = snprintf(buf, 512, "Count of processor throttled for temperature");
            else if (!strncmp(name, "PROCPWRTHROT", 12))
                ret = snprintf(buf, 512, "Count of processor throttled due to power");
            else if (!strncmp(name, "MWRM", 4)) {
                tgt = atoi(name+4);
                ret = snprintf(buf, 512, "Memory write requests per sec for MC %d", tgt);
            }
            else if (!strncmp(name, "MRDM", 4)) {
                tgt = atoi(name+4);
                ret = snprintf(buf, 512, "Memory read requests per sec for MC %d", tgt);
            }
            else
                ret = snprintf(buf, 512, "[PERFORMANCE] Unexpected: %s", name);
            break;

        /* PWRSYS, PWRGPU, PWRAPSSCHvv, PWRPROC, PWRVDD, PWRVDN, PWRMEM */
        case OCC_SENSOR_TYPE_POWER:
            if (!strncmp(name, "PWRSYS", 6))
                ret = snprintf(buf, 512, "Bulk power of the system/node");
            else if (!strncmp(name, "PWRGPU", 6))
                ret = snprintf(buf, 512, "Power consumption for GPUs per socket (OCC) read from APSS");
            else if (!strncmp(name, "PWRPROC", 7))
                ret = snprintf(buf, 512, "Power consumption for this Processor");
            else if (!strncmp(name, "PWRVDD", 6))
                ret = snprintf(buf, 512, "Power consumption for this Processor's Vdd (calculated from AVSBus readings)");
            else if (!strncmp(name, "PWRVDN", 6))
                ret = snprintf(buf, 512, "Power consumption for this Processor's Vdn (nest) (calculated from AVSBus readings)");
            else if (!strncmp(name, "PWRMEM", 6))
                ret = snprintf(buf, 512, "Power consumption for Memory for this Processor read from APSS");
            else if (!strncmp(name, "PWRAPSSCH", 9)) {
                tgt = atoi(name+9);
                ret = snprintf(buf, 512, "Power Provided by APSS channel %d", tgt);
            }
            else
                ret = snprintf(buf, 512, "[POWER] Unexpected: %s", name);
            break;

        /* FREQA, FREQACxx */
        case OCC_SENSOR_TYPE_FREQUENCY:
            if (!strncmp(name, "FREQACxx", 6)) {
                tgt = atoi(name+6);
                ret = snprintf(buf, 512, "Average/actual frequency for this processor, Core %d based on OCA data", tgt);
            }
            else if (!strncmp(name, "FREQA", 5))
                ret = snprintf(buf, 512, "Average of all core frequencies for Processor");
            else
                ret = snprintf(buf, 512, "[FREQUENCY] Unexpected: %s", name);
            break;

        case OCC_SENSOR_TYPE_TIME:
            ret = snprintf(buf, 512, "[TIME] Unexpected: %s", name);
            break;

        /* UTILCxx, UTIL, NUTILCxx, MEMSPSTATMy, MEMSPMy */
        case OCC_SENSOR_TYPE_UTILIZATION:
            if (!strncmp(name, "MEMSPSTATM", 10)) {
                tgt = atoi(name+10);
                ret = snprintf(buf, 512, "Static Memory throttle level setting for MCA %d when not in a memory throttle condition", tgt);
            }
            else if (!strncmp(name, "MEMSPM", 6)) {
                tgt = atoi(name+6);
                ret = snprintf(buf, 512, "Current Memory throttle level setting for MCA %d", tgt);
            }
            else if (!strncmp(name, "NUTILC", 6)) {
                tgt = atoi(name+6);
                ret = snprintf(buf, 512, "Normalized average utilization, rolling average of this Processor's Core %d", tgt);
            }
            else if (!strncmp(name, "UTILC", 5)) {
                tgt = atoi(name+5);
                ret = snprintf(buf, 512, "Utilization of this Processor's Core %d (where 100%% means fully utilized): NOTE: per thread HW counters are combined as appropriate to give this core level utilization sensor", tgt);
            }
            else if (!strncmp(name, "UTIL", 4))
                ret = snprintf(buf, 512, "Average of all Cores UTILC[yy] sensor");
            else
                ret = snprintf(buf, 512, "[UTILIZATION] Unexpected: %s", name);
            break;

        /* TEMPNEST, TEMPPROCTHRMCxx, TEMPVDD, TEMPDIMMvv, TEMPGPUz, TEMPGPUzMEM*/
        case OCC_SENSOR_TYPE_TEMPERATURE:
            if (!strncmp(name, "TEMPNEST", 8))
                ret = snprintf(buf, 512, "Average temperature of nest DTS sensors");
            else if (!strncmp(name, "TEMPVDD", 7))
                ret = snprintf(buf, 512, "VRM Vdd temperature");
            else if (!strncmp(name, "TEMPPROCTHRMCxx", 13)) {
                tgt = atoi(name+13);
                ret = snprintf(buf, 512, "The combined weighted core/quad temperature for processor core %d", tgt);
            }
            else if (!strncmp(name, "TEMPDIMMvv", 8)) {
                tgt = atoi(name+8);
                ret = snprintf(buf, 512, "DIMM temperature for DIMM %d", tgt);
            }
            else if (!strncmp(name, "TEMPGPUz", 7)) {
                char z[] = {name[7], '\0'};
                tgt = atoi(z);
                name[7] = 'z';
                if (!strncmp(name, "TEMPGPUzMEM", 11))
                    ret = snprintf(buf, 512, "GPU %d hottest HBM temperature (individual memory temperatures are not available)", tgt);
                else if (!strncmp(name, "TEMPGPUz", 8))
                    ret = snprintf(buf, 512, "GPU %d board temperature", tgt);
                else
                    ret = snprintf(buf, 512, "[TEMPERATURE] Unexpected: GPU-%d %s", tgt, name);
            }
            else
                ret = snprintf(buf, 512, "[TEMPERATURE] Unexpected: %s", name);
            break;

        /* VOLTVDD, VOLTVDDSENSE, VOLTVDN, VOLTVDNSENSE, VOLTDROOPCNTCx, VOLTDROOPCNTQw */
        case OCC_SENSOR_TYPE_VOLTAGE:
            if (!strncmp(name, "VOLTVDDS", 8))
                ret = snprintf(buf, 512, "Vdn Voltage at the remote sense. (AVS reading adjusted for loadline)");
            else if (!strncmp(name, "VOLTVDNS", 8))
                ret = snprintf(buf, 512, "Vdd Voltage at the remote sense. (AVS reading adjusted for loadline)");
            else if (!strncmp(name, "VOLTVDD", 7))
                ret = snprintf(buf, 512, "Processor Vdd Voltage (read from AVSBus)");
            else if (!strncmp(name, "VOLTVDN", 7))
                ret = snprintf(buf, 512, "Processor Vdn Voltage (read from AVSBus)");
            else if (!strncmp(name, "VOLTDROOPCNTC", 13)) {
                tgt = atoi(name+13);
                ret = snprintf(buf, 512, "Small voltage droop count for core %d", tgt);
            }
            else if (!strncmp(name, "VOLTDROOPCNTQ", 13)) {
                tgt = atoi(name+13);
                ret = snprintf(buf, 512, "Small voltage droop count for core %d", tgt);
            }
            else
                ret = snprintf(buf, 512, "[VOLTAGE] Unexpected: %s", name);
            break;

    /* CURVDD, CURVDN */
        case OCC_SENSOR_TYPE_CURRENT:
            if (!strncmp(name, "CURVDN", 6))
                ret = snprintf(buf, 512, "Processor Vdn Current (read from AVSBus)");
            else if (!strncmp(name, "CURVDD", 6))
                ret = snprintf(buf, 512, "Processor Vdd Current (read from AVSBus)");
            else
                ret = snprintf(buf, 512, "[CURRENT] Unexpected: %s", name);
            break;

        case OCC_SENSOR_TYPE_GENERIC:
        default:
            ret = snprintf(buf, 512, "[GENERIC] Unexpected: %s", name);
            break;
    }

    if (ret <= 0 || 512 <= ret)
        return PAPI_ENOSUPP;
    _space_padding(buf, sizeof(buf));
    ret = snprintf(buf+strlen(buf), 512, "%s", sensors_ppc_fake_qualif_desc[midx]);
    if (ret <= 0 || 512 <= ret)
        return PAPI_ENOSUPP;
    _space_padding(buf, sizeof(buf));
    ret = snprintf(buf+strlen(buf), 512, "Sampling period: %lfs", 1./freq);
    if (ret <= 0 || 512 <= ret)
        return PAPI_ENOSUPP;

    _local_strlcpy( info->long_descr, buf, sizeof(info->long_descr));
    info->data_type = PAPI_DATATYPE_INT64;

    return PAPI_OK;
}

papi_vector_t _sensors_ppc_vector = {
    .cmp_info = { /* (unspecified values are initialized to 0) */
        .name = "sensors_ppc",
        .short_name = "sensors_ppc",
        .description = "Linux sensors_ppc energy measurements",
        .version = "5.3.0",
        .default_domain = PAPI_DOM_ALL,
        .default_granularity = PAPI_GRN_SYS,
        .available_granularities = PAPI_GRN_SYS,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        .available_domains = PAPI_DOM_ALL,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        .context = sizeof ( _sensors_ppc_context_t ),
        .control_state = sizeof ( _sensors_ppc_control_state_t ),
        .reg_value = sizeof ( _sensors_ppc_register_t ),
        .reg_alloc = sizeof ( _sensors_ppc_reg_alloc_t ),
    },
    /* function pointers in this component */
    .init_thread =          _sensors_ppc_init_thread,
    .init_component =       _sensors_ppc_init_component,
    .init_control_state =   _sensors_ppc_init_control_state,
    .update_control_state = _sensors_ppc_update_control_state,
    .start =                _sensors_ppc_start,
    .stop =                 _sensors_ppc_stop,
    .read =                 _sensors_ppc_read,
    .shutdown_thread =      _sensors_ppc_shutdown_thread,
    .shutdown_component =   _sensors_ppc_shutdown_component,
    .ctl =                  _sensors_ppc_ctl,

    .set_domain =           _sensors_ppc_set_domain,
    .reset =                _sensors_ppc_reset,

    .ntv_enum_events =      _sensors_ppc_ntv_enum_events,
    .ntv_code_to_name =     _sensors_ppc_ntv_code_to_name,
    .ntv_code_to_info =     _sensors_ppc_ntv_code_to_info,
};
