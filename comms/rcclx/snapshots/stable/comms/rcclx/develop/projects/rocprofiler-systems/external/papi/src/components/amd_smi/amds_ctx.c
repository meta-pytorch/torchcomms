/**
 * @file    amds_ctx.c
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#include "amds.h"
#include "amds_priv.h"
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include <stdint.h>   // for uint64_t, INT types

unsigned int _amd_smi_lock;

/* Use a 64-bit global device mask to support up to 64 devices */
static uint64_t device_mask = 0;

typedef struct {
  unsigned int code;
  amds_event_info_t info;
  native_event_t event;
} ctx_event_t;

static int first_device_from_map(uint64_t map) {
  if (!map)
    return -1;
  int limit = amds_get_device_count();
  if (limit <= 0 || limit > 64)
    limit = 64;
  int d;
  for (d = 0; d < limit; ++d) {
    if (amds_dev_check(map, d))
      return d;
  }
  return -1;
}

static int acquire_devices(ctx_event_t *events, int num_events, uint64_t *bitmask) {
  if (!bitmask) return PAPI_EINVAL;
  if (num_events < 0) return PAPI_EINVAL;
  if (num_events > 0 && !events) return PAPI_EINVAL;

  uint64_t mask_acq = 0;
  int i;
  for (i = 0; i < num_events; ++i) {
    if (!(events[i].info.flags & AMDS_DEVICE_FLAG))
      continue;
    int dev_id = events[i].info.device;
    if (dev_id < 0) continue;
    if (dev_id >= 64) return PAPI_EINVAL;
    mask_acq |= (UINT64_C(1) << dev_id);
  }

  _papi_hwi_lock(_amd_smi_lock);
  if (mask_acq & device_mask) {
    _papi_hwi_unlock(_amd_smi_lock);
    return PAPI_ECNFLCT;
  }
  device_mask |= mask_acq;
  _papi_hwi_unlock(_amd_smi_lock);

  *bitmask = mask_acq;
  return PAPI_OK;
}

static int release_devices(uint64_t *bitmask) {
  if (!bitmask) return PAPI_EINVAL;
  uint64_t mask_rel = *bitmask;

  _papi_hwi_lock(_amd_smi_lock);
  if ((mask_rel & device_mask) != mask_rel) {
    _papi_hwi_unlock(_amd_smi_lock);
    return PAPI_EMISC;
  }
  /* Clear with &= ~mask for clarity and robustness */
  device_mask &= ~mask_rel;
  _papi_hwi_unlock(_amd_smi_lock);

  *bitmask = 0;
  return PAPI_OK;
}

/* Context management: open/close, start/stop, read/write, reset */
struct amds_ctx {
  int state;
  unsigned int *events_id;
  int num_events;
  ctx_event_t *events;
  long long *counters;
  uint64_t device_mask;       /* was int32_t: now 64-bit to match global */
};

int amds_ctx_open(unsigned int *event_ids, int num_events, amds_ctx_t *ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (num_events < 0) return PAPI_EINVAL;
  if (num_events > 0 && !event_ids) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  amds_ctx_t new_ctx = (amds_ctx_t)papi_calloc(1, sizeof(struct amds_ctx));
  if (new_ctx == NULL) {
    return PAPI_ENOMEM;
  }
  new_ctx->events_id = event_ids;
  new_ctx->num_events = num_events;
  new_ctx->counters = (long long *)papi_calloc((size_t)num_events, sizeof(long long));
  if (new_ctx->counters == NULL) {
    papi_free(new_ctx);
    return PAPI_ENOMEM;
  }

  if (num_events > 0) {
    new_ctx->events = (ctx_event_t *)papi_calloc((size_t)num_events, sizeof(ctx_event_t));
    if (new_ctx->events == NULL) {
      papi_free(new_ctx->counters);
      papi_free(new_ctx);
      return PAPI_ENOMEM;
    }
  }

  int papi_errno = PAPI_OK;
  int i;
  for (i = 0; i < num_events; ++i) {
    new_ctx->events[i].code = event_ids[i];
    papi_errno = amds_evt_id_to_info(event_ids[i], &new_ctx->events[i].info);
    if (papi_errno != PAPI_OK)
      goto fail;
    native_event_t *base = &ntv_table_p->events[new_ctx->events[i].info.nameid];
    new_ctx->events[i].event = *base;
    new_ctx->events[i].event.priv = NULL;
    new_ctx->events[i].event.value = 0;
    if (new_ctx->events[i].info.flags & AMDS_DEVICE_FLAG) {
      new_ctx->events[i].event.device = new_ctx->events[i].info.device;
    } else if (base->device_map) {
      int dev = new_ctx->events[i].info.device;
      if (!amds_dev_check(base->device_map, dev))
        dev = first_device_from_map(base->device_map);
      if (dev < 0) {
        papi_errno = PAPI_ENOEVNT;
        goto fail;
      }
      new_ctx->events[i].info.flags |= AMDS_DEVICE_FLAG;
      new_ctx->events[i].info.device = dev;
      new_ctx->events[i].event.device = dev;
    }
  }

  papi_errno = acquire_devices(new_ctx->events, num_events, &new_ctx->device_mask);
  if (papi_errno != PAPI_OK)
    goto fail;

  for (i = 0; i < num_events; ++i) {
    native_event_t *ev = &new_ctx->events[i].event;
    if (ev->open_func) {
      papi_errno = ev->open_func(ev);
      if (papi_errno != PAPI_OK) {
        int j;
        for (j = 0; j < i; ++j) {
          native_event_t *prev = &new_ctx->events[j].event;
          if (prev->close_func)
            prev->close_func(prev);
        }
        release_devices(&new_ctx->device_mask);
        goto fail;
      }
    }
  }

  *ctx = new_ctx;
  return PAPI_OK;

fail:
  if (new_ctx->events) {
    for (i = 0; i < num_events; ++i) {
      new_ctx->events[i].event.priv = NULL;
    }
    papi_free(new_ctx->events);
  }
  papi_free(new_ctx->counters);
  papi_free(new_ctx);
  return papi_errno;
}

int amds_ctx_close(amds_ctx_t ctx) {
  if (!ctx)
    return PAPI_OK;
  if (!ntv_table_p) {
    /* Best effort: release devices and free even if table is gone */
    (void)release_devices(&ctx->device_mask);
    if (ctx->events)
      papi_free(ctx->events);
    papi_free(ctx->counters);
    papi_free(ctx);
    return PAPI_OK;
  }
  int i;
  if (!ctx->events)
    return PAPI_EMISC;

  for (i = 0; i < ctx->num_events; ++i) {
    native_event_t *ev = &ctx->events[i].event;
    if (ev->close_func)
      ev->close_func(ev);
  }
  /* release device usage */
  (void)release_devices(&ctx->device_mask);
  if (ctx->events)
    papi_free(ctx->events);
  papi_free(ctx->counters);
  papi_free(ctx);
  return PAPI_OK;
}

int amds_ctx_start(amds_ctx_t ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  int papi_errno = PAPI_OK;
  int i;
  for (i = 0; i < ctx->num_events; ++i) {
    if (!ctx->events)
      break;
    native_event_t *ev = &ctx->events[i].event;
    if (ev->start_func) {
      papi_errno = ev->start_func(ev);
      if (papi_errno != PAPI_OK)
        return papi_errno;
    }
  }
  ctx->state |= AMDS_EVENTS_RUNNING;
  return papi_errno;
}

int amds_ctx_stop(amds_ctx_t ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (!(ctx->state & AMDS_EVENTS_RUNNING)) {
    return PAPI_OK;
  }
  if (!ntv_table_p) return PAPI_ECMP;
  if (!ctx->events)
    return PAPI_EMISC;

  int papi_errno = PAPI_OK;
  int i;
  for (i = 0; i < ctx->num_events; ++i) {
    native_event_t *ev = &ctx->events[i].event;
    if (ev->stop_func) {
      int papi_errno_stop = ev->stop_func(ev);
      if (papi_errno == PAPI_OK)
        papi_errno = papi_errno_stop;
    }
  }
  ctx->state &= ~AMDS_EVENTS_RUNNING;
  return papi_errno;
}

int amds_ctx_read(amds_ctx_t ctx, long long **counts) {
  if (!ctx || !counts) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  /* Always produce a fully defined buffer */
  int i;
  for (i = 0; i < ctx->num_events; ++i) {
    ctx->counters[i] = 0;  /* overwritten below */
  }

  /* Optional: track first error, but don't bail early */
  int papi_errno = PAPI_OK;

  for (i = 0; i < ctx->num_events; ++i) {
    if (!ctx->events)
      break;
    native_event_t *ev = &ctx->events[i].event;
    int papi_errno_access = PAPI_OK;
    if (ev->access_func) {
      papi_errno_access = ev->access_func(PAPI_MODE_READ, ev);
    } else {
      papi_errno_access = PAPI_ECMP;
    }

    if (papi_errno_access == PAPI_OK) {
      ctx->counters[i] = (long long)ev->value;
    } else {
      ctx->counters[i] = (long long)papi_errno_access;  /* surface failure per-event */
      if (papi_errno == PAPI_OK) {
        papi_errno = papi_errno_access;  /* remember, but keep going */
      }
    }
  }

  *counts = ctx->counters;

  /* return PAPI_OK so callers can inspect per-event errors in the counters */
  return PAPI_OK;
}

int amds_ctx_write(amds_ctx_t ctx, long long *counts) {
  if (!ctx || !counts) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  int papi_errno = PAPI_OK;
  int i;
  for (i = 0; i < ctx->num_events; ++i) {
    if (!ctx->events)
      break;
    native_event_t *ev = &ctx->events[i].event;
    if (!ev->access_func) return PAPI_ECMP;
    ev->value = counts[i];
    papi_errno = ev->access_func(PAPI_MODE_WRITE, ev);
    if (papi_errno != PAPI_OK) {
      return papi_errno;
    }
  }
  return papi_errno;
}

int amds_ctx_reset(amds_ctx_t ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  int i;
  for (i = 0; i < ctx->num_events; ++i) {
    if (!ctx->events)
      break;
    native_event_t *ev = &ctx->events[i].event;
    ev->value = 0;
    if (ctx->counters) ctx->counters[i] = 0;
  }
  return PAPI_OK;
}
