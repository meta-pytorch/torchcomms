/* include for rand() */
#include <stdlib.h>
#include <stdint.h>
#include "vendor_common.h"
#include "vendor_profiler_v1.h"

/**
  * Event identifier encoding format:
  * +--------------------+-------+-+--+--+
  * |       unused       |  dev  | |  |id|
  * +--------------------+-------+-+--+--+
  *
  * unused    : 18 bits
  * device    :  7 bits ([0 - 127] devices)
  * function  :  1 bits (exponential or sum)
  * qlmask    :  2 bits (qualifier mask)
  * nameid    :  2 bits ([0 - 3] event names)
  */
#define EVENTS_WIDTH (sizeof(uint32_t) * 8)
#define DEVICE_WIDTH (7)
#define OPCODE_WIDTH (1)
#define QLMASK_WIDTH (2)
#define NAMEID_WIDTH (2)
#define UNUSED_WIDTH (EVENTS_WIDTH - DEVICE_WIDTH - OPCODE_WIDTH - QLMASK_WIDTH - NAMEID_WIDTH)
#define DEVICE_SHIFT (EVENTS_WIDTH - UNUSED_WIDTH - DEVICE_WIDTH)
#define OPCODE_SHIFT (DEVICE_SHIFT - OPCODE_WIDTH)
#define QLMASK_SHIFT (OPCODE_SHIFT - QLMASK_WIDTH)
#define NAMEID_SHIFT (QLMASK_SHIFT - NAMEID_WIDTH)
#define DEVICE_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - DEVICE_WIDTH)) << DEVICE_SHIFT)
#define OPCODE_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - OPCODE_WIDTH)) << OPCODE_SHIFT)
#define QLMASK_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - QLMASK_WIDTH)) << QLMASK_SHIFT)
#define NAMEID_MASK  ((0xFFFFFFFF >> (EVENTS_WIDTH - NAMEID_WIDTH)) << NAMEID_SHIFT)
#define DEVICE_FLAG  (0x2)
#define OPCODE_FLAG  (0x1)
#define OPCODE_EXP   (0x0)
#define OPCODE_SUM   (0x1)

typedef struct {
    char name[PAPI_MAX_STR_LEN];
    char descr[PAPI_2MAX_STR_LEN];
} ntv_event_t;

typedef struct {
    ntv_event_t *events;
    int num_events;
} ntv_event_table_t;

struct vendord_ctx {
    int state;
    unsigned int *events_id;
    long long *counters;
    int num_events;
};

static struct {
    char *name;
    char *descr;
} vendor_events[] = {
    { "TEMPLATE_ZERO"    , "This is a template counter, that always returns 0" },
    { "TEMPLATE_CONSTANT", "This is a template counter, that always returns a constant value of 42" },
    { "TEMPLATE_FUNCTION", "This is a template counter, that allows for different functions" },
    { NULL, NULL }
};

static ntv_event_table_t ntv_table;
static ntv_event_table_t *ntv_table_p;

int
vendorp1_init_pre(void)
{
    return PAPI_OK;
}

static int load_profiler_v1_symbols(void);
static int unload_profiler_v1_symbols(void);
static int initialize_event_table(void);
static int finalize_event_table(void);

typedef struct {
    int device;
    int opcode;
    int flags;
    int nameid;
} event_info_t;

static int evt_id_create(event_info_t *info, uint32_t *event_id);
static int evt_id_to_info(uint32_t event_id, event_info_t *info);
static int evt_name_to_device(const char *name, int *device);
static int evt_name_to_opcode(const char *name, int *opcode);
static int evt_name_to_basename(const char *name, char *base, int len);

int
vendorp1_init(void)
{
    int papi_errno;

    papi_errno = load_profiler_v1_symbols();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = initialize_event_table();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ntv_table_p = &ntv_table;

  fn_exit:
    return papi_errno;
  fn_fail:
    finalize_event_table();
    unload_profiler_v1_symbols();
    goto fn_exit;
}

int
vendorp1_shutdown(void)
{
    finalize_event_table();
    ntv_table_p = NULL;
    unload_profiler_v1_symbols();
    return PAPI_OK;
}

static int init_ctx(unsigned int *events_id, int num_events, vendorp_ctx_t ctx);
static int open_ctx(vendorp_ctx_t ctx);
static int close_ctx(vendorp_ctx_t ctx);
static int finalize_ctx(vendorp_ctx_t ctx);

int
vendorp1_ctx_open(unsigned int *events_id, int num_events, vendorp_ctx_t *ctx)
{
    int papi_errno;

    *ctx = papi_calloc(1, sizeof(struct vendord_ctx));
    if (NULL == *ctx) {
        return PAPI_ENOMEM;
    }

    _papi_hwi_lock(_templ_lock);

    papi_errno = init_ctx(events_id, num_events, *ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = open_ctx(*ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    (*ctx)->state |= TEMPL_CTX_OPENED;

  fn_exit:
    _papi_hwi_unlock(_templ_lock);
    return papi_errno;
  fn_fail:
    close_ctx(*ctx);
    finalize_ctx(*ctx);
    goto fn_exit;
}

int
vendorp1_ctx_start(vendorp_ctx_t ctx)
{
    ctx->state |= TEMPL_CTX_RUNNING;
    return PAPI_OK;
}

int
vendorp1_ctx_read(vendorp_ctx_t ctx, long long **counters)
{
    int papi_errno;

    int i;
    for (i = 0; i < ctx->num_events; ++i) {
        event_info_t info;
        papi_errno = evt_id_to_info(ctx->events_id[i], &info);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        if (0 == strcmp(ntv_table_p->events[info.nameid].name, "TEMPLATE_ZERO")) {
            ctx->counters[i] = (long long) 0;
        } else if (0 == strcmp(ntv_table_p->events[info.nameid].name, "TEMPLATE_CONSTANT")) {
            ctx->counters[i] = (long long) 42;
        } else if (0 == strcmp(ntv_table_p->events[info.nameid].name, "TEMPLATE_FUNCTION")) {
            if (info.opcode == OPCODE_EXP) {
                ctx->counters[i] = (ctx->counters[i]) ? ctx->counters[i] * 2 : 2;
            } else {
                ctx->counters[i] = (ctx->counters[i]) ? ctx->counters[i] + 1 : 1;
            }
        }
    }
    *counters = ctx->counters;
    return PAPI_OK;
}

int
vendorp1_ctx_stop(vendorp_ctx_t ctx)
{
    ctx->state &= ~TEMPL_CTX_RUNNING;
    return PAPI_OK;
}

int
vendorp1_ctx_reset(vendorp_ctx_t ctx)
{
    memset(ctx->counters, 0, sizeof(ctx->counters) * ctx->num_events);
    return PAPI_OK;
}

int
vendorp1_ctx_close(vendorp_ctx_t ctx)
{
    int papi_errno;

    _papi_hwi_lock(_templ_lock);

    papi_errno = close_ctx(ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ctx->state &= ~TEMPL_CTX_OPENED;

    papi_errno = finalize_ctx(ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_free(ctx);

  fn_exit:
    _papi_hwi_unlock(_templ_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;

}

int
vendorp1_evt_enum(unsigned int *event_code, int modifier)
{
    int papi_errno;

    event_info_t info;
    papi_errno = evt_id_to_info(*event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch(modifier) {
        case PAPI_ENUM_FIRST:
            if (ntv_table_p->num_events == 0) {
                papi_errno = PAPI_ENOEVNT;
            }
            info.device = 0;
            info.opcode = 0;
            info.flags = 0;
            info.nameid = 0;
            papi_errno = evt_id_create(&info, event_code);
            break;
        case PAPI_ENUM_EVENTS:
            if (info.nameid + 1 >= ntv_table_p->num_events) {
                papi_errno = PAPI_ENOEVNT;
                break;
            }
            ++info.nameid;
            papi_errno = evt_id_create(&info, event_code);
            break;
        case PAPI_NTV_ENUM_UMASKS:
            if (info.flags == 0) {
                info.flags = DEVICE_FLAG;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }
            if (info.flags & DEVICE_FLAG && info.nameid == 2) {
                info.flags = OPCODE_FLAG;
                papi_errno = evt_id_create(&info, event_code);
                break;
            }
            papi_errno = PAPI_END;
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
}

int
vendorp1_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno;

    event_info_t info;
    papi_errno = evt_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch (info.flags) {
        case DEVICE_FLAG | OPCODE_FLAG:
            snprintf(name, len, "%s:device=%i:function=%s",
                     ntv_table_p->events[info.nameid].name,
                     info.device, (info.opcode == OPCODE_EXP) ? "exp" : "sum");
            break;
        case DEVICE_FLAG:
            snprintf(name, len, "%s:device=%i",
                     ntv_table_p->events[info.nameid].name,
                     info.device);
            break;
        case OPCODE_FLAG:
            snprintf(name, len, "%s:function=%s",
                     ntv_table_p->events[info.nameid].name,
                     (info.opcode == OPCODE_EXP) ? "exp" : "sum");
            break;
        default:
            papi_errno = PAPI_ENOEVNT;
    }

    snprintf(name, len, "%s", ntv_table_p->events[info.nameid].name);
    return papi_errno;
}

int
vendorp1_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
    int papi_errno;

    event_info_t info;
    papi_errno = evt_id_to_info(event_code, &info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    snprintf(descr, len, "%s", ntv_table_p->events[info.nameid].descr);
    return PAPI_OK;
}

int
vendorp1_evt_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    int papi_errno;

    event_info_t code_info;
    papi_errno = evt_id_to_info(event_code, &code_info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch (code_info.flags) {
        case 0:
            sprintf(info->symbol, "%s", ntv_table_p->events[code_info.nameid].name);
            sprintf(info->long_descr, "%s", ntv_table_p->events[code_info.nameid].descr);
            break;
        case DEVICE_FLAG | OPCODE_FLAG:
            sprintf(info->symbol, "%s:device=%i:function=%s",
                    ntv_table_p->events[code_info.nameid].name,
                    code_info.device,
                    (code_info.opcode == OPCODE_EXP) ? "exp" : "sum");
            sprintf(info->long_descr, "%s", ntv_table_p->events[code_info.nameid].descr);
            break;
        case DEVICE_FLAG:
        {
            int i;
            char devices[PAPI_MAX_STR_LEN] = { 0 };
            for (i = 0; i < device_table_p->num_devices; ++i) {
                sprintf(devices + strlen(devices), "%i,", i);
            }
            *(devices + strlen(devices) - 1) = 0;
            sprintf(info->symbol, "%s:device=%i", ntv_table_p->events[code_info.nameid].name, code_info.device);
            sprintf(info->long_descr, "%s masks:Device qualifier [%s]",
                    ntv_table_p->events[code_info.nameid].descr, devices);
            break;
        }
        case OPCODE_FLAG:
            sprintf(info->symbol, "%s:function=%s",
                    ntv_table_p->events[code_info.nameid].name,
                    (code_info.opcode == OPCODE_EXP) ? "exp" : "sum");
            sprintf(info->long_descr, "%s masks:Mandatory function qualifier (exp,sum)",
                    ntv_table_p->events[code_info.nameid].descr);
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
}

int
vendorp1_evt_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno;

    char basename[PAPI_MAX_STR_LEN] = { 0 };
    papi_errno = evt_name_to_basename(name, basename, PAPI_MAX_STR_LEN);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int device;
    papi_errno = evt_name_to_device(name, &device);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int opcode = 0;
    papi_errno = evt_name_to_opcode(name, &opcode);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int i, nameid = 0;
    for (i = 0; i < ntv_table_p->num_events; ++i) {
        if (0 == strcmp(ntv_table_p->events[i].name, basename)) {
            nameid = i;
            break;
        }
    }

    event_info_t info = { 0, 0, 0, 0 };
    if (0 == strcmp(ntv_table_p->events[nameid].name, "TEMPLATE_FUNCTION")) {
        info.device = device;
        info.opcode = opcode;
        info.flags = (DEVICE_FLAG | OPCODE_FLAG);
        info.nameid = nameid;
        papi_errno = evt_id_create(&info, event_code);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    } else {
        info.device = device;
        info.opcode = 0;
        info.flags = DEVICE_FLAG;
        info.nameid = nameid;
        papi_errno = evt_id_create(&info, event_code);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    papi_errno = evt_id_to_info(*event_code, &info);

    return papi_errno;
}

int
load_profiler_v1_symbols(void)
{
    return PAPI_OK;
}

int
unload_profiler_v1_symbols(void)
{
    return PAPI_OK;
}

static int get_events_count(int *num_events);
static int get_events(ntv_event_t *events, int num_events);

int
initialize_event_table(void)
{
    int papi_errno, num_events;

    papi_errno = get_events_count(&num_events);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    ntv_event_t *events = papi_calloc(num_events, sizeof(ntv_event_t));
    if (NULL == events) {
        return PAPI_ENOMEM;
    }

    papi_errno = get_events(events, num_events);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ntv_table.events = events;
    ntv_table.num_events = num_events;

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_free(events);
    goto fn_exit;
}

int
finalize_event_table(void)
{
    papi_free(ntv_table_p->events);
    ntv_table_p->num_events = 0;
    ntv_table_p = NULL;
    return PAPI_OK;
}

int
init_ctx(unsigned int *events_id, int num_events, vendorp_ctx_t ctx)
{
    ctx->events_id = events_id;
    ctx->num_events = num_events;
    ctx->counters = papi_calloc(num_events, sizeof(long long));
    if (NULL == ctx->counters) {
        return PAPI_ENOMEM;
    }
    return PAPI_OK;
}

int
open_ctx(vendorp_ctx_t ctx __attribute__((unused)))
{
    return PAPI_OK;
}

int
close_ctx(vendorp_ctx_t ctx __attribute__((unused)))
{
    return PAPI_OK;
}

int
finalize_ctx(vendorp_ctx_t ctx)
{
    ctx->events_id = NULL;
    ctx->num_events = 0;
    papi_free(ctx->counters);
    return PAPI_OK;
}

int
get_events_count(int *num_events)
{
    int i = 0;
    while (vendor_events[i++].name != NULL);
    *num_events = i - 1;
    return PAPI_OK;
}

int
get_events(ntv_event_t *events, int num_events)
{
    int i = 0;
    while (vendor_events[i].name != NULL) {
        snprintf(events[i].name, PAPI_MAX_STR_LEN, "%s", vendor_events[i].name);
        snprintf(events[i].descr, PAPI_2MAX_STR_LEN, "%s", vendor_events[i].descr);
        ++i;
    }
    return (num_events - i) ? PAPI_EMISC : PAPI_OK;
}

int
evt_id_create(event_info_t *info, uint32_t *event_id)
{
    *event_id  = (uint32_t)(info->device << DEVICE_SHIFT);
    *event_id |= (uint32_t)(info->opcode << OPCODE_SHIFT);
    *event_id |= (uint32_t)(info->flags  << QLMASK_SHIFT);
    *event_id |= (uint32_t)(info->nameid << NAMEID_SHIFT);
    return PAPI_OK;
}

int
evt_id_to_info(uint32_t event_id, event_info_t *info)
{
    info->device = (int)((event_id & DEVICE_MASK) >> DEVICE_SHIFT);
    info->opcode = (int)((event_id & OPCODE_MASK) >> OPCODE_SHIFT);
    info->flags  = (int)((event_id & QLMASK_MASK) >> QLMASK_SHIFT);
    info->nameid = (int)((event_id & NAMEID_MASK) >> NAMEID_SHIFT);

    if (info->device >= device_table_p->num_devices) {
        return PAPI_ENOEVNT;
    }

    if (info->nameid >= ntv_table_p->num_events) {
        return PAPI_ENOEVNT;
    }

    if (0 == strcmp(ntv_table_p->events[info->nameid].name, "TEMPLATE_FUNCTION") && 0 == info->flags) {
        return PAPI_ENOEVNT;
    }

    return PAPI_OK;
}

int
evt_name_to_device(const char *name, int *device)
{
    *device = 0;
    char *p = strstr(name, ":device=");
    if (p) {
        *device = (int) strtol(p + strlen(":device="), NULL, 10);
    }
    return PAPI_OK;
}

int
evt_name_to_opcode(const char *name, int *opcode)
{
    char basename[PAPI_MAX_STR_LEN] = { 0 };
    evt_name_to_basename(name, basename, PAPI_MAX_STR_LEN);
    if (0 == strcmp(basename, "TEMPLATE_FUNCTION")) {
        char *p = strstr(name, ":function=");
        if (p) {
            if (strncmp(p + strlen(":function="), "exp", strlen("exp")) == 0) {
                *opcode = OPCODE_EXP;
            } else if (strncmp(p + strlen(":function="), "sum", strlen("sum")) == 0) {
                *opcode = OPCODE_SUM;
            } else {
                return PAPI_ENOEVNT;
            }
        } else {
            return PAPI_ENOEVNT;
        }
    }
    return PAPI_OK;
}

int
evt_name_to_basename(const char *name, char *base, int len)
{
    char *p = strstr(name, ":");
    if (p) {
        if (len < (int)(p - name)) {
            return PAPI_EBUF;
        }
        strncpy(base, name, (size_t)(p - name));
    } else {
        if (len < (int) strlen(name)) {
            return PAPI_EBUF;
        }
        strncpy(base, name, (size_t) len);
    }
    return PAPI_OK;
}
