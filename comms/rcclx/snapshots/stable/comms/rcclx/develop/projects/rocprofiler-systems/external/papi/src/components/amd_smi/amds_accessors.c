/**
 * @file    amds_accessors.c
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#include "amds_priv.h"
#include "papi.h"
#include "papi_memory.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
/* -------- Helpers and new accessors (GPU read-only additions) -------- */
static uint64_t _str_to_u64_hash(const char *s) {
  /* djb2 64-bit */
  uint64_t hash = 5381;
  if (!s)
    return 0;
  int c;
  while ((c = *s++)) {
    hash = ((hash << 5) + hash) + (uint8_t)c;
  }
  return hash;
}

int access_amdsmi_lib_version(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_lib_version_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  amdsmi_version_t vinfo;
  memset(&vinfo, 0, sizeof(vinfo));
  amdsmi_status_t st = amdsmi_get_lib_version_p(&vinfo);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = (int64_t)vinfo.major;
    break;
  case 1:
    event->value = (int64_t)vinfo.minor;
    break;
  case 2:
    event->value = (int64_t)vinfo.release;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
int access_amdsmi_num_devices(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  event->value = (int64_t)gpu_count;
  return PAPI_OK;
}
int access_amdsmi_uuid_hash(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_device_uuid_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  char buf[PAPI_MAX_STR_LEN] = {0};
  unsigned int len = sizeof(buf);
  amdsmi_status_t st = amdsmi_get_gpu_device_uuid_p(device_handles[event->device], &len, buf);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0: /* hash */
    event->value = (int64_t)_str_to_u64_hash(buf);
    break;
  case 1: /* length */
    event->value = (int64_t)len;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
int access_amdsmi_gpu_string_hash(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  char buf[PAPI_2MAX_STR_LEN] = {0};
  amdsmi_status_t st = AMDSMI_STATUS_NOT_SUPPORTED;
  switch (event->variant) {
  case 0: /* vendor name */
    if (!amdsmi_get_gpu_vendor_name_p)
      return PAPI_ENOSUPP;
    st = amdsmi_get_gpu_vendor_name_p(device_handles[event->device], buf, sizeof(buf));
    break;
  case 1: /* vram vendor */
    if (!amdsmi_get_gpu_vram_vendor_p)
      return PAPI_ENOSUPP;
    st = amdsmi_get_gpu_vram_vendor_p(device_handles[event->device], buf, sizeof(buf));
    break;
  case 2: /* subsystem name */
    if (!amdsmi_get_gpu_subsystem_name_p)
      return PAPI_ENOSUPP;
    st = amdsmi_get_gpu_subsystem_name_p(device_handles[event->device], buf, sizeof(buf));
    break;
  case 3: /* driver name */
  case 4: /* driver date */
  case 8: /* driver version */
    if (!amdsmi_get_gpu_driver_info_p)
      return PAPI_ENOSUPP;
    {
      amdsmi_driver_info_t dinfo;
      memset(&dinfo, 0, sizeof(dinfo));
      st = amdsmi_get_gpu_driver_info_p(device_handles[event->device], &dinfo);
      if (st == AMDSMI_STATUS_SUCCESS) {
        if (event->variant == 3)
          CHECK_SNPRINTF(buf, sizeof(buf), "%s", dinfo.driver_name);
        else if (event->variant == 4)
          CHECK_SNPRINTF(buf, sizeof(buf), "%s", dinfo.driver_date);
        else
          CHECK_SNPRINTF(buf, sizeof(buf), "%s", dinfo.driver_version);
      }
    }
    break;
  case 5: /* vbios version */
  case 6: /* vbios part number */
  case 7: /* vbios build date */
    if (!amdsmi_get_gpu_vbios_info_p)
      return PAPI_ENOSUPP;
    {
      amdsmi_vbios_info_t vb;
      memset(&vb, 0, sizeof(vb));
      st = amdsmi_get_gpu_vbios_info_p(device_handles[event->device], &vb);
      if (st == AMDSMI_STATUS_SUCCESS) {
        if (event->variant == 5)
          CHECK_SNPRINTF(buf, sizeof(buf), "%s", vb.version);
        else if (event->variant == 6)
          CHECK_SNPRINTF(buf, sizeof(buf), "%s", vb.part_number);
        else
          CHECK_SNPRINTF(buf, sizeof(buf), "%s", vb.build_date);
      }
    }
    break;
  default:
    return PAPI_ENOSUPP;
  }
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)_str_to_u64_hash(buf);
  return PAPI_OK;
}
#if AMDSMI_LIB_VERSION_MAJOR >= 25
int access_amdsmi_enumeration_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_enumeration_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_enumeration_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_enumeration_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = (int64_t)info.drm_render;
    break;
  case 1:
    event->value = (int64_t)info.drm_card;
    break;
  case 2:
    event->value = (int64_t)info.hsa_id;
    break;
  case 3:
    event->value = (int64_t)info.hip_id;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
#endif
int access_amdsmi_asic_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_asic_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_asic_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_asic_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = (int64_t)info.vendor_id;
    break;
  case 1:
    event->value = (int64_t)info.device_id;
    break;
  case 2:
    event->value = (int64_t)info.subvendor_id;
    break;
  case 3:
    event->value = (int64_t)0 /* not provided in amdsmi_asic_info_t */;
    break;
  case 4:
    event->value = (int64_t)info.rev_id;
    break;
  case 5:
    event->value = (int64_t)info.num_of_compute_units;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}

static int link_metrics_variant_value(const amdsmi_link_metrics_t *lm, uint32_t index,
                                      int variant, uint64_t *out)
{
  if (!lm || !out)
    return PAPI_EINVAL;

  switch (variant) {
  case 0:
    *out = lm->links[index].read; /* KB */
    break;
  case 1:
    *out = lm->links[index].write; /* KB */
    break;
  case 2:
    *out = lm->links[index].bit_rate; /* Gb/s */
    break;
  case 3:
    *out = lm->links[index].max_bandwidth; /* Gb/s */
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_link_metrics(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_link_metrics_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_link_metrics_t lm;
  memset(&lm, 0, sizeof(lm));
  if (amdsmi_get_link_metrics_p(device_handles[event->device], &lm) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  uint32_t count = lm.num_links;
  if (count > AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK)
    count = AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK;

  uint32_t enc = event->subvariant;
  uint32_t link_type = enc >> 16;
  uint32_t link_index = enc & 0xFFFF; /* 0xFFFF aggregates all links */

  uint64_t total = 0;
  if (link_index == 0xFFFF) {
    uint32_t i;
    for (i = 0; i < count; ++i) {
      uint64_t value = 0;
      if (link_type && lm.links[i].link_type != link_type)
        continue;
      if (link_metrics_variant_value(&lm, i, event->variant, &value) != PAPI_OK)
        return PAPI_ENOSUPP;
      total += value;
    }
  } else {
    uint64_t value = 0;
    if (link_index >= count)
      return PAPI_EMISC;
    if (link_type && lm.links[link_index].link_type != link_type)
      return PAPI_EMISC;
    if (link_metrics_variant_value(&lm, link_index, event->variant, &value) != PAPI_OK)
      return PAPI_ENOSUPP;
    total = value;
  }

  if (total > (uint64_t)INT64_MAX)
    total = (uint64_t)INT64_MAX;
  event->value = (int64_t)total;
  return PAPI_OK;
}

#if AMDSMI_LIB_VERSION_MAJOR >= 25
int access_amdsmi_xgmi_link_status(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_xgmi_link_status_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_xgmi_link_status_t st;
  memset(&st, 0, sizeof(st));
  if (amdsmi_get_gpu_xgmi_link_status_p(device_handles[event->device], &st) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  uint32_t li = (uint32_t)event->subvariant;
  if (li >= st.total_links || li >= AMDSMI_MAX_NUM_XGMI_LINKS)
    return PAPI_EMISC;
  event->value = (int64_t)st.status[li];
  return PAPI_OK;
}
#endif

int access_amdsmi_xgmi_error_status(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_gpu_xgmi_error_status_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_xgmi_status_t st;
  if (amdsmi_gpu_xgmi_error_status_p(device_handles[event->device], &st) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)st;
  return PAPI_OK;
}

int access_amdsmi_link_weight(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_topo_get_link_weight_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  int src = event->device;
  int dst = (int)event->subvariant;
  if (src < 0 || src >= device_count || dst < 0 || dst >= device_count ||
      !device_handles[src] || !device_handles[dst] || src == dst)
    return PAPI_EMISC;
  uint64_t weight = 0;
  if (amdsmi_topo_get_link_weight_p(device_handles[src],
                                    device_handles[dst], &weight) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  if (weight > (uint64_t)INT64_MAX)
    weight = (uint64_t)INT64_MAX;
  event->value = (int64_t)weight;
  return PAPI_OK;
}

int access_amdsmi_link_type(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_topo_get_link_type_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  int src = event->device;
  int dst = (int)event->subvariant;
  if (src < 0 || src >= device_count || dst < 0 || dst >= device_count ||
      !device_handles[src] || !device_handles[dst] || src == dst)
    return PAPI_EMISC;
  uint64_t hops = 0;
  amdsmi_link_type_compat_t type;
  if (amdsmi_topo_get_link_type_p(device_handles[src], device_handles[dst],
                                  &hops, &type) != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  if (event->variant == 0) {
    if (hops > (uint64_t)INT64_MAX)
      hops = (uint64_t)INT64_MAX;
    event->value = (int64_t)hops;
  } else if (event->variant == 1) {
    event->value = (int64_t)type;
  } else {
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_p2p_status(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_topo_get_p2p_status_p)
    return PAPI_ENOSUPP;

  native_event_t *event = (native_event_t *)arg;
  const int src = event->device;
  const int dst = (int)event->subvariant;

  if (src < 0 || src >= device_count || dst < 0 || dst >= device_count ||
      !device_handles[src] || !device_handles[dst] || src == dst)
    return PAPI_EMISC;

  // 1) Prefer the cheap predicate to avoid the buggy slow path:
  bool accessible = false;
  if (amdsmi_is_P2P_accessible_p &&
      amdsmi_is_P2P_accessible_p(device_handles[src], device_handles[dst],
                                 &accessible) == AMDSMI_STATUS_SUCCESS &&
      accessible) {
    // 2) Only for accessible pairs, ask for detailed capabilities:
    amdsmi_link_type_compat_t type = 0;
    amdsmi_p2p_capability_t cap = {0};
    if (amdsmi_topo_get_p2p_status_p(device_handles[src], device_handles[dst],
                                     &type, &cap) != AMDSMI_STATUS_SUCCESS)
      return PAPI_EMISC;  // unexpected for accessible pairs

    switch (event->variant) {
      case 0: event->value = (int64_t)type; break;
      case 1: event->value = cap.is_iolink_coherent; break;
      case 2: event->value = cap.is_iolink_atomics_32bit; break;
      case 3: event->value = cap.is_iolink_atomics_64bit; break;
      case 4: event->value = cap.is_iolink_dma; break;
      case 5: event->value = cap.is_iolink_bi_directional; break;
      default: return PAPI_ENOSUPP;
    }
    return PAPI_OK;
  }

  // 3) Non-accessible or predicate missing: report a sensible value without
  // touching the buggy call. Type (variant 0) can still be queried safely via
  // amdsmi_topo_get_link_type; the rest are false by definition.
  if (event->variant == 0 && amdsmi_topo_get_link_type_p) {
    uint64_t hops = 0;
    amdsmi_link_type_compat_t type = 0; // UNKNOWN/PCIE/XGMI per platform
    if (amdsmi_topo_get_link_type_p(device_handles[src], device_handles[dst],
                                    &hops, &type) == AMDSMI_STATUS_SUCCESS) {
      event->value = (int64_t)type;
      return PAPI_OK;
    }
    // If link_type also fails, fall through to no data.
  }

  // For non-accessible pairs, the capability booleans are zero.
  event->value = 0;
  return PAPI_OK;
}


int access_amdsmi_p2p_accessible(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_is_P2P_accessible_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  int src = event->device;
  int dst = (int)event->subvariant;
  if (src < 0 || src >= device_count || dst < 0 || dst >= device_count ||
      !device_handles[src] || !device_handles[dst] || src == dst)
    return PAPI_EMISC;
  bool accessible = false;
  if (amdsmi_is_P2P_accessible_p(device_handles[src], device_handles[dst],
                                 &accessible) != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = accessible ? 1 : 0;
  return PAPI_OK;
}

int access_amdsmi_link_topology_nearest(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_link_topology_nearest_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_topology_nearest_t info;
  memset(&info, 0, sizeof(info));
  if (amdsmi_get_link_topology_nearest_p(
          device_handles[event->device], (amdsmi_link_type_t)event->variant,
          &info) != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)info.count;
  return PAPI_OK;
}

int access_amdsmi_topo_numa(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_topo_get_numa_node_number_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  uint32_t node = 0;
  if (amdsmi_topo_get_numa_node_number_p(device_handles[event->device], &node) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)node;
  return PAPI_OK;
}

int access_amdsmi_device_bdf(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_device_bdf_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_bdf_t bdf;
  memset(&bdf, 0, sizeof(bdf));
  if (amdsmi_get_gpu_device_bdf_p(device_handles[event->device], &bdf) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = (int64_t)bdf.domain_number;
    break;
  case 1:
    event->value = (int64_t)bdf.bus_number;
    break;
  case 2:
    event->value = (int64_t)bdf.device_number;
    break;
  case 3:
    event->value = (int64_t)bdf.function_number;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_kfd_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_kfd_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_kfd_info_t info;
  memset(&info, 0, sizeof(info));
  if (amdsmi_get_gpu_kfd_info_p(device_handles[event->device], &info) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = (int64_t)info.kfd_id;
    break;
  case 1:
    event->value = (int64_t)info.node_id;
    break;
  case 2:
    event->value = (int64_t)info.current_partition_id;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_xgmi_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_xgmi_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_xgmi_info_t info;
  memset(&info, 0, sizeof(info));
  if (amdsmi_get_xgmi_info_p(device_handles[event->device], &info) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = (uint64_t)info.xgmi_lanes;
    break;
  case 1:
    event->value = (uint64_t)info.xgmi_hive_id;
    break;
  case 2:
    event->value = (uint64_t)info.xgmi_node_id;
    break;
  case 3:
    event->value = (uint64_t)info.index;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_process_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_process_list_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_proc_info_t list[16];
  uint32_t count = 16;
  amdsmi_status_t st =
      amdsmi_get_gpu_process_list_p(device_handles[event->device], &count, list);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  uint32_t proc = event->subvariant;
  amdsmi_proc_info_t *p = NULL;

  if (event->variant != 8) {
    if (proc >= count) {
      event->value = 0;
      return PAPI_OK;
    }
    p = &list[proc];
  }

  switch (event->variant) {
  case 0:
    event->value = (int64_t)p->pid;
    break;
  case 1:
    event->value = (int64_t)p->mem;
    break;
  case 2:
    event->value = (int64_t)p->engine_usage.gfx;
    break;
  case 3:
    event->value = (int64_t)p->engine_usage.enc;
    break;
  case 4:
    event->value = (int64_t)p->memory_usage.gtt_mem;
    break;
  case 5:
    event->value = (int64_t)p->memory_usage.cpu_mem;
    break;
  case 6:
    event->value = (int64_t)p->memory_usage.vram_mem;
    break;
  case 7:
    /* cu_occupancy added in AMD SMI 6.4.3 and kept in later majors. */
#if AMDSMI_LIB_VERSION_MAJOR > 25
    event->value = (int64_t)p->cu_occupancy;
#elif defined(AMDSMI_LIB_VERSION_MINOR) && AMDSMI_LIB_VERSION_MINOR >= 5
    event->value = (int64_t)p->cu_occupancy;
#else
    event->value = (int64_t)p->reserved[0];
#endif
    break;
  case 8:
    event->value = (int64_t)count;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}
int access_amdsmi_ecc_total(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_total_ecc_count_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_error_count_t ec;
  memset(&ec, 0, sizeof(ec));
  if (amdsmi_get_gpu_total_ecc_count_p(device_handles[event->device], &ec) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  uint64_t val;
  switch (event->variant) {
  case 0:
    val = ec.correctable_count;
    break;
  case 1:
    val = ec.uncorrectable_count;
    break;
  case 2:
    val = ec.deferred_count;
    break;
  default:
    return PAPI_ENOSUPP;
  }

  if (val > (uint64_t)INT64_MAX)
    val = (uint64_t)INT64_MAX;
  event->value = (int64_t)val;
  return PAPI_OK;
}

int access_amdsmi_ecc_block(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_ecc_count_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_error_count_t ec;
  memset(&ec, 0, sizeof(ec));
  if (amdsmi_get_gpu_ecc_count_p(device_handles[event->device],
                                 (amdsmi_gpu_block_t)event->subvariant, &ec) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  uint64_t val;
  switch (event->variant) {
  case 0:
    val = ec.correctable_count;
    break;
  case 1:
    val = ec.uncorrectable_count;
    break;
  case 2:
    val = ec.deferred_count;
    break;
  default:
    return PAPI_ENOSUPP;
  }

  if (val > (uint64_t)INT64_MAX)
    val = (uint64_t)INT64_MAX;
  event->value = (int64_t)val;
  return PAPI_OK;
}

int access_amdsmi_ecc_status(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_ecc_status_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_ras_err_state_t st;
  if (amdsmi_get_gpu_ecc_status_p(device_handles[event->device],
                                  (amdsmi_gpu_block_t)event->subvariant, &st) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)st;
  return PAPI_OK;
}

int access_amdsmi_ecc_enabled_mask(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_ecc_enabled_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  uint64_t mask = 0;
  if (amdsmi_get_gpu_ecc_enabled_p(device_handles[event->device], &mask) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)mask;
  return PAPI_OK;
}
int access_amdsmi_compute_partition_hash(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_compute_partition_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  char buf[PAPI_MAX_STR_LEN] = {0};
  if (amdsmi_get_gpu_compute_partition_p(device_handles[event->device], buf,
                                         sizeof(buf)) != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)_str_to_u64_hash(buf);
  return PAPI_OK;
}
int access_amdsmi_memory_partition_hash(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_memory_partition_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  if (amdsmi_is_gpu_memory_partition_supported_p) {
    bool supported = false;
    if (amdsmi_is_gpu_memory_partition_supported_p(device_handles[event->device],
                                                   &supported) !=
            AMDSMI_STATUS_SUCCESS ||
        !supported)
      return PAPI_ENOSUPP;
  }
  char buf[PAPI_MAX_STR_LEN] = {0};
  if (amdsmi_get_gpu_memory_partition_p(device_handles[event->device], buf,
                                        sizeof(buf)) != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  buf[sizeof(buf) - 1] = '\0';
  event->value = (int64_t)_str_to_u64_hash(buf);
  return PAPI_OK;
}

#if AMDSMI_LIB_VERSION_MAJOR >= 25
int access_amdsmi_memory_partition_config(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_memory_partition_config_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  if (amdsmi_is_gpu_memory_partition_supported_p) {
    bool supported = false;
    if (amdsmi_is_gpu_memory_partition_supported_p(device_handles[event->device],
                                                   &supported) !=
            AMDSMI_STATUS_SUCCESS ||
        !supported)
      return PAPI_ENOSUPP;
  }
  amdsmi_memory_partition_config_t cfg = {0};
  if (amdsmi_get_gpu_memory_partition_config_p(device_handles[event->device],
                                               &cfg) != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    /* Union holds bit flags; expose the mask value */
    event->value = (int64_t)cfg.partition_caps.nps_cap_mask;
    break;
  case 1:
    event->value = (int64_t)cfg.mp_mode;
    break;
  case 2:
    event->value = (int64_t)cfg.num_numa_ranges;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}
#endif
int access_amdsmi_accelerator_num_partitions(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_accelerator_partition_profile_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_accelerator_partition_profile_t prof = {0};
  uint32_t ids[AMDSMI_MAX_ACCELERATOR_PARTITIONS] = {0};
  if (amdsmi_get_gpu_accelerator_partition_profile_p(device_handles[event->device],
                                                     &prof, ids) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)prof.num_partitions;
  return PAPI_OK;
}
/* Access function implementations (read/write operations for each event) */
int access_amdsmi_temp_metric(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC; /* ensure device handle is valid */
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  int64_t tmp = 0;
  amdsmi_status_t status =
      amdsmi_get_temp_metric_p(device_handles[event->device],
                               (amdsmi_temperature_type_t)event->subvariant,
                               (amdsmi_temperature_metric_t)event->variant,
                               &tmp);
  if (status == AMDSMI_STATUS_SUCCESS) {
    event->value = (uint64_t)tmp;
    return PAPI_OK;
  }
  return PAPI_EMISC;
}
int access_amdsmi_fan_rpms(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  int64_t speed = 0;
  amdsmi_status_t status = amdsmi_get_gpu_fan_rpms_p(device_handles[event->device], event->subvariant, &speed);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = speed;
  return PAPI_OK;
}
int access_amdsmi_fan_speed(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }

  int64_t current = 0;
  amdsmi_status_t status =
      amdsmi_get_gpu_fan_speed_p(device_handles[event->device], event->subvariant,
                                 &current);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  if (mode == PAPI_MODE_READ) {
    event->value = current;
    return PAPI_OK;
  }

  if (mode == PAPI_MODE_WRITE) {

    if (event->value < 0)
      return PAPI_EINVAL;

    uint64_t new_speed = (uint64_t)event->value;
    uint64_t max_speed = 255;
    if (amdsmi_get_gpu_fan_speed_max_p) {
      int64_t max_query = 0;
      status = amdsmi_get_gpu_fan_speed_max_p(device_handles[event->device],  event->subvariant, &max_query);
      if (status == AMDSMI_STATUS_SUCCESS && max_query > 0)
        max_speed = (uint64_t)max_query;
    }
    if (new_speed > max_speed)
      return PAPI_EINVAL;

    status = amdsmi_set_gpu_fan_speed_p(device_handles[event->device], event->subvariant, new_speed);
    if (status != AMDSMI_STATUS_SUCCESS)
      return PAPI_EMISC;

    int64_t verify = 0;
    status = amdsmi_get_gpu_fan_speed_p(device_handles[event->device],  event->subvariant, &verify);
    event->value = (status == AMDSMI_STATUS_SUCCESS) ? verify : (int64_t)new_speed;
    return PAPI_OK;
  }

  return PAPI_ENOSUPP;
}
int access_amdsmi_mem_total(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint64_t data = 0;
  amdsmi_status_t status = amds_query_gpu_memory_total(
      device_handles[event->device], (amdsmi_memory_type_t)event->variant,
      &data);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED)
    return PAPI_ENOSUPP;
  if (status == AMDSMI_STATUS_INVAL)
    return PAPI_EINVAL;
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)data;
  return PAPI_OK;
}
int access_amdsmi_mem_usage(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint64_t data = 0;
  amdsmi_status_t status = amds_query_gpu_memory_usage(
      device_handles[event->device], (amdsmi_memory_type_t)event->variant,
      &data);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED)
    return PAPI_ENOSUPP;
  if (status == AMDSMI_STATUS_INVAL)
    return PAPI_EINVAL;
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)data;
  return PAPI_OK;
}
int access_amdsmi_power_cap(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (!amdsmi_get_power_cap_info_p) {
    return PAPI_ENOSUPP;
  }
  if (mode == PAPI_MODE_READ) {
    // Read current power cap
    amdsmi_power_cap_info_t info;
    memset(&info, 0, sizeof(info));
    amdsmi_status_t status = amdsmi_get_power_cap_info_p(device_handles[event->device], 0, &info); // sensor index 0
    if (status != AMDSMI_STATUS_SUCCESS) {
      return PAPI_EMISC;
    }
    event->value = (int64_t)info.power_cap;
    return PAPI_OK;
  } else if (mode == PAPI_MODE_WRITE) {
    // Set new power cap (value expected in microWatts if API uses uW)
    amdsmi_power_cap_info_t limits;
    memset(&limits, 0, sizeof(limits));
    if (amdsmi_get_power_cap_info_p(device_handles[event->device], 0, &limits) != AMDSMI_STATUS_SUCCESS) {
      return PAPI_EMISC;
    }
    uint64_t new_cap = (uint64_t)event->value;
    if (limits.min_power_cap && new_cap < limits.min_power_cap)
      return PAPI_EINVAL;
    if (limits.max_power_cap && new_cap > limits.max_power_cap)
      return PAPI_EINVAL;
    amdsmi_status_t status = amdsmi_set_power_cap_p(device_handles[event->device], 0, new_cap);
    if (status != AMDSMI_STATUS_SUCCESS)
      return PAPI_EMISC;
    event->value = (int64_t)new_cap;
    return PAPI_OK;
  }
  return PAPI_ENOSUPP;
}
int access_amdsmi_power_cap_range(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  amdsmi_power_cap_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t status = amdsmi_get_power_cap_info_p(device_handles[event->device], 0, &info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  if (event->variant == 1) {
    event->value = (int64_t)info.min_power_cap;
  } else if (event->variant == 2) {
    event->value = (int64_t)info.max_power_cap;
  } else if (event->variant == 3) {
    event->value = (int64_t)info.default_power_cap;
  } else if (event->variant == 4) {
    event->value = (int64_t)info.dpm_cap;
  } else {
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
int access_amdsmi_power_average(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  amdsmi_power_info_t power;
  memset(&power, 0, sizeof(power));
  amdsmi_status_t status = amdsmi_get_power_info_p(device_handles[event->device], &power);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  switch (event->variant) {
  case 0:
    event->value = (int64_t)power.average_socket_power;
    break;
  case 1:
    event->value = (int64_t)power.current_socket_power;
    break;
  case 2:
    event->value = (int64_t)power.power_limit;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}
int access_amdsmi_pci_throughput(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint64_t sent = 0, received = 0, max_pkt = 0;
  amdsmi_status_t status = amdsmi_get_gpu_pci_throughput_p(device_handles[event->device], &sent, &received, &max_pkt);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  switch (event->variant) {
  case 0:
    event->value = (int64_t)sent;
    break;
  case 1:
    event->value = (int64_t)received;
    break;
  case 2:
    event->value = (int64_t)max_pkt;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
int access_amdsmi_pci_replay_counter(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint64_t counter = 0;
  amdsmi_status_t status = amdsmi_get_gpu_pci_replay_counter_p(device_handles[event->device], &counter);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = (int64_t)counter;
  return PAPI_OK;
}
int access_amdsmi_clk_freq(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;

  amdsmi_frequencies_t freq_info;
  memset(&freq_info, 0, sizeof(freq_info));  /* critical */

  static const amdsmi_clk_type_t clk_variant_map[] = {
      AMDSMI_CLK_TYPE_SYS,  AMDSMI_CLK_TYPE_GFX,  AMDSMI_CLK_TYPE_DF,
      AMDSMI_CLK_TYPE_DCEF, AMDSMI_CLK_TYPE_SOC,  AMDSMI_CLK_TYPE_MEM,
      AMDSMI_CLK_TYPE_PCIE, AMDSMI_CLK_TYPE_VCLK0, AMDSMI_CLK_TYPE_VCLK1,
      AMDSMI_CLK_TYPE_DCLK0, AMDSMI_CLK_TYPE_DCLK1};
  const size_t clk_variant_count =
      sizeof(clk_variant_map) / sizeof(clk_variant_map[0]);
  if (event->variant >= clk_variant_count)
    return PAPI_ENOSUPP;

  amdsmi_clk_type_t clk_type = clk_variant_map[event->variant];

  amdsmi_status_t status =
      amdsmi_get_clk_freq_p(device_handles[event->device], clk_type, &freq_info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }

  if (event->subvariant == 0) {
    event->value = freq_info.num_supported;
  } else if (event->subvariant == 1) {
    event->value = (freq_info.num_supported > 0) ? freq_info.frequency[0] : 0;
  } else {
    int idx = event->subvariant - 2;
    if (idx >= 0 && (uint32_t)idx < freq_info.num_supported) {
      event->value = freq_info.frequency[idx];
    } else {
      event->value = 0;
    }
  }
  return PAPI_OK;
}


int access_amdsmi_clock_info(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;

  amdsmi_clk_type_t clk_types[] = {AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_MEM};
  if (event->variant < 0 || event->variant >= 2)
    return PAPI_EMISC;

  amdsmi_clk_info_t info;
  memset(&info, 0, sizeof(info));  /* critical */

  amdsmi_status_t status =
      amdsmi_get_clock_info_p(device_handles[event->device],
                              clk_types[event->variant], &info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }

  switch (event->subvariant) {
    case 0: event->value = info.clk;           break;
    case 1: event->value = info.min_clk;       break;
    case 2: event->value = info.max_clk;       break;
    case 3: event->value = info.clk_locked;    break;
    case 4: event->value = info.clk_deep_sleep;break;
    default: return PAPI_EMISC;
  }
  return PAPI_OK;
}


int access_amdsmi_metrics_header_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_metrics_header_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amd_metrics_table_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  if (amdsmi_get_gpu_metrics_header_info_p(device_handles[event->device], &hdr) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = hdr.structure_size;
    break;
  case 1:
    event->value = hdr.format_revision;
    break;
  case 2:
    event->value = hdr.content_revision;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}
int access_amdsmi_gpu_metrics(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  amdsmi_gpu_metrics_t metrics;
  memset(&metrics, 0, sizeof(metrics));
  amdsmi_status_t status = amdsmi_get_gpu_metrics_info_p(device_handles[event->device], &metrics);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  switch (event->variant) {
  case 0:
    event->value = metrics.throttle_status;
    break;
  case 1:
    event->value = (int64_t)metrics.indep_throttle_status;
    break;
  case 2:
    event->value = metrics.pcie_link_width;
    break;
  case 3:
    event->value = metrics.pcie_link_speed;
    break;
  case 4:
    event->value = (int64_t)metrics.pcie_bandwidth_acc;
    break;
  case 5:
    event->value = (int64_t)metrics.pcie_bandwidth_inst;
    break;
  case 6:
    event->value = (int64_t)metrics.pcie_l0_to_recov_count_acc;
    break;
  case 7:
    event->value = (int64_t)metrics.pcie_replay_count_acc;
    break;
  case 8:
    event->value = (int64_t)metrics.pcie_replay_rover_count_acc;
    break;
  case 9:
    event->value = metrics.pcie_nak_sent_count_acc;
    break;
  case 10:
    event->value = metrics.pcie_nak_rcvd_count_acc;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}
int access_amdsmi_gpu_info(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  amdsmi_status_t status;
  switch (event->variant) {
  case 0: {
    uint16_t id = 0;
    status = amdsmi_get_gpu_id_p(device_handles[event->device], &id);
    if (status == AMDSMI_STATUS_SUCCESS) {
      event->value = id;
    }
    break;
  }
  case 1: {
    uint16_t rev = 0;
    status = amdsmi_get_gpu_revision_p(device_handles[event->device], &rev);
    if (status == AMDSMI_STATUS_SUCCESS) {
      event->value = rev;
    }
    break;
  }
  case 2: {
    uint16_t subid = 0;
    status = amdsmi_get_gpu_subsystem_id_p(device_handles[event->device], &subid);
    if (status == AMDSMI_STATUS_SUCCESS) {
      event->value = subid;
    }
    break;
  }
  case 3: {
    uint64_t bdfid = 0;
    status = amdsmi_get_gpu_bdf_id_p(device_handles[event->device], &bdfid);
    if (status == AMDSMI_STATUS_SUCCESS) {
      event->value = (int64_t)bdfid;
    }
    break;
  }
#if AMDSMI_LIB_VERSION_MAJOR >= 25
  case 4: {
    if (amdsmi_lib_major < 25 || !amdsmi_get_gpu_virtualization_mode_p)
      return PAPI_ENOSUPP;
    amdsmi_virtualization_mode_t mode_val;
    status = amdsmi_get_gpu_virtualization_mode_p(device_handles[event->device], &mode_val);
    if (status == AMDSMI_STATUS_SUCCESS) {
      event->value = mode_val;
    }
    break;
  }
#endif
  case 5: {
    int32_t numa_node = -1;
    status = amdsmi_get_gpu_topo_numa_affinity_p(device_handles[event->device], &numa_node);
    if (status == AMDSMI_STATUS_SUCCESS) {
      event->value = numa_node;
    }
    break;
  }
  default:
    return PAPI_EMISC;
  }
  return (status == AMDSMI_STATUS_SUCCESS ? PAPI_OK : PAPI_EMISC);
}
int access_amdsmi_gpu_activity(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  amdsmi_engine_usage_t usage;
  memset(&usage, 0, sizeof(usage));
  amdsmi_status_t status = amdsmi_get_gpu_activity_p(device_handles[event->device], &usage);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  switch (event->variant) {
  case 0:
    event->value = usage.gfx_activity;
    break;
  case 1:
    event->value = usage.umc_activity;
    break;
  case 2:
    event->value = usage.mm_activity;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
int access_amdsmi_fan_speed_max(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  int64_t max_speed = 0;
  amdsmi_status_t status = amdsmi_get_gpu_fan_speed_max_p(device_handles[event->device], event->subvariant, &max_speed);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = max_speed;
  return PAPI_OK;
}
int access_amdsmi_pci_bandwidth(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_pci_bandwidth_p)
    return PAPI_ENOSUPP;

  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles || !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_pcie_bandwidth_t bw;
  memset(&bw, 0, sizeof(bw));  /* critical */

  if (amdsmi_get_gpu_pci_bandwidth_p(device_handles[event->device], &bw) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  uint32_t cur = bw.transfer_rate.current;
  if (cur >= bw.transfer_rate.num_supported) {
    event->value = 0;
    return PAPI_OK;
  }

  switch (event->variant) {
    case 0: event->value = bw.transfer_rate.num_supported;    break;
    case 1: event->value = (int64_t)bw.transfer_rate.frequency[cur]; break;
    case 2: event->value = bw.lanes[cur];                     break;
    default: return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}


int access_amdsmi_energy_count(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint64_t energy = 0;
  float resolution = 0.0;
  uint64_t timestamp = 0;
  amdsmi_status_t status = amdsmi_get_energy_count_p(device_handles[event->device], &energy, &resolution, &timestamp);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  switch (event->variant) {
  case 0: {
    // Convert accumulated energy count to microJoules
    double total_uJ = (double)energy * (double)resolution;
    if (total_uJ > INT64_MAX) {
      /* clamp to avoid overflow */
      total_uJ = INT64_MAX;
    }
    event->value = total_uJ;
    break;
    }
  case 1:
    // Resolution microJoules per count
    event->value = (int64_t)(resolution);
    break;
  case 2:
    // Raw timestamp returned by the SMI library (nanoseconds)
    event->value = (int64_t)timestamp;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}

int access_amdsmi_xgmi_bandwidth(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_minmax_bandwidth_between_processors_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= gpu_count || !device_handles ||
      !device_handles[event->device])
    return PAPI_EMISC;
  if (event->subvariant < 0 || event->subvariant >= gpu_count ||
      !device_handles[event->subvariant])
    return PAPI_EMISC;

  amdsmi_processor_handle src = device_handles[event->device];
  amdsmi_processor_handle dst = device_handles[event->subvariant];
  uint64_t min_bw = 0, max_bw = 0;
  if (amdsmi_get_minmax_bandwidth_between_processors_p(src, dst, &min_bw,
                                                       &max_bw) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  event->value = (event->variant == 0) ? (int64_t)min_bw : (int64_t)max_bw;
  return PAPI_OK;
}
int access_amdsmi_power_profile_status(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  amdsmi_power_profile_status_t status_info;
  amdsmi_status_t status = amdsmi_get_gpu_power_profile_presets_p(device_handles[event->device], 0, &status_info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  if (event->variant == 0) {
    event->value = status_info.num_profiles;
  } else if (event->variant == 1) {
    event->value = (int64_t)status_info.current;
  } else {
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
#ifndef AMDSMI_DISABLE_ESMI
/* The functions below implement CPU metrics access */
int access_amdsmi_cpu_socket_power(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint32_t power = 0;
  amdsmi_status_t status = amdsmi_get_cpu_socket_power_p(device_handles[event->device], &power);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = (int64_t)power;
  return PAPI_OK;
}
int access_amdsmi_cpu_socket_energy(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint64_t energy = 0;
  amdsmi_status_t status = amdsmi_get_cpu_socket_energy_p(device_handles[event->device], &energy);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = (int64_t)energy;
  return PAPI_OK;
}
int access_amdsmi_cpu_socket_freq_limit(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint16_t freq = 0;
  char *src = NULL;
  amdsmi_status_t status = amdsmi_get_cpu_socket_current_active_freq_limit_p(device_handles[event->device], &freq, &src);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  if (src)
    free(src);
  event->value = freq;
  return PAPI_OK;
}
int access_amdsmi_cpu_socket_freq_range(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint16_t fmax = 0, fmin = 0;
  amdsmi_status_t status = amdsmi_get_cpu_socket_freq_range_p(device_handles[event->device], &fmax, &fmin);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  if (event->variant == 0) {
    event->value = fmin;
  } else {
    event->value = fmax;
  }
  return PAPI_OK;
}
int access_amdsmi_cpu_power_cap(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint32_t cap_value = 0;
  amdsmi_status_t status;
  if (event->variant == 0) {
    status = amdsmi_get_cpu_socket_power_cap_p(device_handles[event->device], &cap_value);
  } else {
    status = amdsmi_get_cpu_socket_power_cap_max_p(device_handles[event->device], &cap_value);
  }
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = (int64_t)cap_value;
  return PAPI_OK;
}
int access_amdsmi_cpu_core_energy(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  int s_index = event->device - gpu_count;
  if (s_index < 0 || s_index >= cpu_count) {
    return PAPI_EMISC;
  }
  uint64_t energy = 0;
  amdsmi_status_t status = amdsmi_get_cpu_core_energy_p(cpu_core_handles[s_index][event->subvariant], &energy);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = (int64_t)energy;
  return PAPI_OK;
}
int access_amdsmi_cpu_core_freq_limit(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  int s_index = event->device - gpu_count;
  if (s_index < 0 || s_index >= cpu_count) {
    return PAPI_EMISC;
  }
  uint32_t freq = 0;
  amdsmi_status_t status = amdsmi_get_cpu_core_current_freq_limit_p(cpu_core_handles[s_index][event->subvariant], &freq);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = freq;
  return PAPI_OK;
}
int access_amdsmi_cpu_core_boostlimit(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  int s_index = event->device - gpu_count;
  if (s_index < 0 || s_index >= cpu_count) {
    return PAPI_EMISC;
  }
  uint32_t boost = 0;
  amdsmi_status_t status = amdsmi_get_cpu_core_boostlimit_p(cpu_core_handles[s_index][event->subvariant], &boost);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = boost;
  return PAPI_OK;
}
int access_amdsmi_cpu_cclk_limit(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint32_t cclk = 0;
  amdsmi_status_t status =
      amdsmi_get_cpu_cclk_limit_p(device_handles[event->device], &cclk);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = cclk;
  return PAPI_OK;
}
int access_amdsmi_cpu_io_bw(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  const char *links[] = {"P0", "P1", "P2", "P3", "P4"};
  amdsmi_io_bw_encoding_t bw_types[] = {AGG_BW0, RD_BW0, WR_BW0};
  if (event->variant < 0 || event->variant >= 5 || event->subvariant < 0 ||
      event->subvariant >= 3)
    return PAPI_EMISC;
  amdsmi_link_id_bw_type_t link = {bw_types[event->subvariant],
                                   (char *)links[event->variant]};
  uint32_t bw = 0;
  amdsmi_status_t status = amdsmi_get_cpu_current_io_bandwidth_p(
      device_handles[event->device], link, &bw);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = bw;
  return PAPI_OK;
}
int access_amdsmi_cpu_xgmi_bw(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  const char *links[] = {"G0", "G1", "G2", "G3",
                         "G4", "G5", "G6", "G7"};
  amdsmi_io_bw_encoding_t bw_types[] = {AGG_BW0, RD_BW0, WR_BW0};
  if (event->variant < 0 || event->variant >= 8 || event->subvariant < 0 ||
      event->subvariant >= 3)
    return PAPI_EMISC;
  amdsmi_link_id_bw_type_t link = {bw_types[event->subvariant],
                                   (char *)links[event->variant]};
  uint32_t bw = 0;
  amdsmi_status_t status = amdsmi_get_cpu_current_xgmi_bw_p(
      device_handles[event->device], link, &bw);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = bw;
  return PAPI_OK;
}
int access_amdsmi_cpu_ddr_bw(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  amdsmi_ddr_bw_metrics_t bw;
  memset(&bw, 0, sizeof(bw));
  amdsmi_status_t status =
      amdsmi_get_cpu_ddr_bw_p(device_handles[event->device], &bw);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = bw.max_bw;
    break;
  case 1:
    event->value = bw.utilized_bw;
    break;
  case 2:
    event->value = bw.utilized_pct;
    break;
  default:
    return PAPI_EMISC;
  }
  return PAPI_OK;
}
int access_amdsmi_cpu_fclk_mclk(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint32_t fclk = 0, mclk = 0;
  amdsmi_status_t status = amdsmi_get_cpu_fclk_mclk_p(
      device_handles[event->device], &fclk, &mclk);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  if (event->variant == 0)
    event->value = fclk;
  else if (event->variant == 1)
    event->value = mclk;
  else
    return PAPI_EMISC;
  return PAPI_OK;
}
int access_amdsmi_cpu_hsmp_driver_version(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  amdsmi_hsmp_driver_version_t ver;
  memset(&ver, 0, sizeof(ver));
  amdsmi_status_t status = amdsmi_get_cpu_hsmp_driver_version_p(
      device_handles[event->device], &ver);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  if (event->variant == 0)
    event->value = ver.major;
  else if (event->variant == 1)
    event->value = ver.minor;
  else
    return PAPI_EMISC;
  return PAPI_OK;
}
int access_amdsmi_cpu_hsmp_proto_ver(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint32_t ver = 0;
  amdsmi_status_t status =
      amdsmi_get_cpu_hsmp_proto_ver_p(device_handles[event->device], &ver);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = ver;
  return PAPI_OK;
}
int access_amdsmi_cpu_prochot_status(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint32_t status = 0;
  amdsmi_status_t smi_status = amdsmi_get_cpu_prochot_status_p(
      device_handles[event->device], &status);
  if (smi_status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = status;
  return PAPI_OK;
}
int access_amdsmi_cpu_svi_power(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  uint32_t power = 0;
  amdsmi_status_t status = amdsmi_get_cpu_pwr_svi_telemetry_all_rails_p(
      device_handles[event->device], &power);
  if (status != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = power;
  return PAPI_OK;
}
int access_amdsmi_dimm_temp(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  amdsmi_dimm_thermal_t dimm_temp;
  memset(&dimm_temp, 0, sizeof(dimm_temp));
  amdsmi_status_t status = amdsmi_get_cpu_dimm_thermal_sensor_p(device_handles[event->device], (uint8_t)event->subvariant, &dimm_temp);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = (int64_t)dimm_temp.temp;
  return PAPI_OK;
}
int access_amdsmi_dimm_power(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  amdsmi_dimm_power_t dimm_pow;
  memset(&dimm_pow, 0, sizeof(dimm_pow));
  amdsmi_status_t status = amdsmi_get_cpu_dimm_power_consumption_p(device_handles[event->device], (uint8_t)event->subvariant, &dimm_pow);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = dimm_pow.power;
  return PAPI_OK;
}
int access_amdsmi_dimm_range_refresh(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  amdsmi_temp_range_refresh_rate_t rate;
  memset(&rate, 0, sizeof(rate));
  amdsmi_status_t status =
      amdsmi_get_cpu_dimm_temp_range_and_refresh_rate_p(device_handles[event->device], (uint8_t)event->subvariant, &rate);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  if (event->variant == 0) {
    event->value = rate.range;
  } else {
    event->value = rate.ref_rate;
  }
  return PAPI_OK;
}
int access_amdsmi_threads_per_core(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint32_t threads = 0;
  amdsmi_status_t status = amdsmi_get_threads_per_core_p(&threads);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = threads;
  return PAPI_OK;
}
int access_amdsmi_cpu_family(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint32_t family = 0;
  amdsmi_status_t status = amdsmi_get_cpu_family_p(&family);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = family;
  return PAPI_OK;
}
int access_amdsmi_cpu_model(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  uint32_t model = 0;
  amdsmi_status_t status = amdsmi_get_cpu_model_p(&model);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  event->value = model;
  return PAPI_OK;
}
int access_amdsmi_smu_fw_version(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  amdsmi_smu_fw_version_t fw;
  memset(&fw, 0, sizeof(fw));
  amdsmi_status_t status = amdsmi_get_cpu_smu_fw_version_p(device_handles[event->device], &fw);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return PAPI_EMISC;
  }
  int encoded = ((int)fw.major << 16) | ((int)fw.minor << 8) | fw.debug;
  event->value = encoded;
  return PAPI_OK;
}
#endif

int access_amdsmi_cache_stat(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ) {
    return PAPI_ENOSUPP;
  }
  if (!amdsmi_get_gpu_cache_info_p)
    return PAPI_ENOSUPP;

  amdsmi_gpu_cache_info_t info;
  amdsmi_status_t st = amdsmi_get_gpu_cache_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  /* subvariant = cache index chosen during registration */
  if (event->subvariant >= info.num_cache_types)
    return PAPI_EMISC;

  uint64_t val = 0;
  switch (event->variant) {
  case 0: /* size in bytes (reported in KB) */
    val = (uint64_t)info.cache[event->subvariant].cache_size * 1024ULL;
    break;
  case 1: /* maximum number of CUs sharing this cache */
    val = (uint64_t)info.cache[event->subvariant].max_num_cu_shared;
    break;
  case 2: /* number of cache instances */
    val = (uint64_t)info.cache[event->subvariant].num_cache_instance;
    break;
  default:
    return PAPI_EINVAL;
  }
  event->value = val;
  return PAPI_OK;
}

int access_amdsmi_overdrive_level(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_overdrive_level_p)
    return PAPI_ENOSUPP;

  uint32_t od = 0;
  amdsmi_status_t st = amdsmi_get_gpu_overdrive_level_p(device_handles[event->device], &od);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)od;
  return PAPI_OK;
}

int access_amdsmi_mem_overdrive_level(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_mem_overdrive_level_p)
    return PAPI_ENOSUPP;

  uint32_t od = 0;
  amdsmi_status_t st = amdsmi_get_gpu_mem_overdrive_level_p(device_handles[event->device], &od);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)od;
  return PAPI_OK;
}

int access_amdsmi_od_volt_regions_count(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_od_volt_curve_regions_p)
    return PAPI_ENOSUPP;

  /* Probe to get count; API requires a buffer, so do a two-call pattern */
  uint32_t num = 0;
  amdsmi_freq_volt_region_t *buf = NULL;

  /* First call: ask for 0 (expect MORE_DATA/INSUFFICIENT_SIZE with num set) */
  amdsmi_status_t st = amdsmi_get_gpu_od_volt_curve_regions_p(device_handles[event->device], &num, buf);
  if (st == AMDSMI_STATUS_INSUFFICIENT_SIZE || st == AMDSMI_STATUS_NO_DATA) {
    if (num == 0)
      return PAPI_EMISC;
    buf = (amdsmi_freq_volt_region_t *)papi_calloc(num, sizeof(amdsmi_freq_volt_region_t));
    if (!buf)
      return PAPI_ENOMEM;
    st = amdsmi_get_gpu_od_volt_curve_regions_p(device_handles[event->device], &num, buf);
  }
  if (st != AMDSMI_STATUS_SUCCESS) {
    if (buf)
      papi_free(buf);
    return PAPI_EMISC;
  }
  event->value = (uint64_t)num;
  if (buf)
    papi_free(buf);
  return PAPI_OK;
}

int access_amdsmi_od_volt_curve_range(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_od_volt_curve_regions_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }

  uint32_t num = 0;
  amdsmi_status_t st = amdsmi_get_gpu_od_volt_curve_regions_p(device_handles[event->device], &num, NULL);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  if (event->subvariant >= num)
    return PAPI_EMISC;

  amdsmi_freq_volt_region_t *regs = (amdsmi_freq_volt_region_t *)papi_calloc(num, sizeof(amdsmi_freq_volt_region_t));
  if (!regs)
    return PAPI_ENOMEM;
  st = amdsmi_get_gpu_od_volt_curve_regions_p(device_handles[event->device], &num, regs);
  if (st != AMDSMI_STATUS_SUCCESS) {
    papi_free(regs);
    return PAPI_EMISC;
  }

  amdsmi_freq_volt_region_t r = regs[event->subvariant];
  papi_free(regs);

  switch (event->variant) {
  case 0:
    event->value = (int64_t)r.freq_range.lower_bound;
    break;
  case 1:
    event->value = (int64_t)r.freq_range.upper_bound;
    break;
  case 2:
    event->value = (int64_t)r.volt_range.lower_bound;
    break;
  case 3:
    event->value = (int64_t)r.volt_range.upper_bound;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_od_volt_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_od_volt_info_p)
    return PAPI_ENOSUPP;

  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }

  amdsmi_od_volt_freq_data_t info;
  memset(&info, 0, sizeof(info));  

  amdsmi_status_t st =
      amdsmi_get_gpu_od_volt_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  switch (event->variant) {
    case 0: event->value = (int64_t)info.curr_sclk_range.lower_bound; break;
    case 1: event->value = (int64_t)info.curr_sclk_range.upper_bound; break;
    case 2: event->value = (int64_t)info.curr_mclk_range.lower_bound; break;
    case 3: event->value = (int64_t)info.curr_mclk_range.upper_bound; break;
    case 4: event->value = (int64_t)info.sclk_freq_limits.lower_bound; break;
    case 5: event->value = (int64_t)info.sclk_freq_limits.upper_bound; break;
    case 6: event->value = (int64_t)info.mclk_freq_limits.lower_bound; break;
    case 7: event->value = (int64_t)info.mclk_freq_limits.upper_bound; break;
    case 8:
      if (event->subvariant >= AMDSMI_NUM_VOLTAGE_CURVE_POINTS) return PAPI_EMISC;
      event->value = (int64_t)info.curve.vc_points[event->subvariant].frequency;
      break;
    case 9:
      if (event->subvariant >= AMDSMI_NUM_VOLTAGE_CURVE_POINTS) return PAPI_EMISC;
      event->value = (int64_t)info.curve.vc_points[event->subvariant].voltage;
      break;
    default:
      return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}


int access_amdsmi_perf_level(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_perf_level_p)
    return PAPI_ENOSUPP;

  amdsmi_dev_perf_level_t perf = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;
  amdsmi_status_t st = amdsmi_get_gpu_perf_level_p(device_handles[event->device], &perf);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)perf;
  return PAPI_OK;
}

int access_amdsmi_pm_metrics_count(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (amdsmi_lib_major < 25 || !amdsmi_get_gpu_pm_metrics_info_p)
    return PAPI_ENOSUPP;

  amdsmi_name_value_t *metrics = NULL;
  uint32_t count = 0;
  amdsmi_status_t st = amdsmi_get_gpu_pm_metrics_info_p(device_handles[event->device], &metrics, &count);
  if (metrics)
    free(metrics); /* library allocates */
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)count;
  return PAPI_OK;
}

int access_amdsmi_pm_metric_value(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (amdsmi_lib_major < 25 || !amdsmi_get_gpu_pm_metrics_info_p)
    return PAPI_ENOSUPP;

  amdsmi_name_value_t *metrics = NULL;
  uint32_t count = 0;
  amdsmi_status_t st = amdsmi_get_gpu_pm_metrics_info_p(device_handles[event->device], &metrics, &count);
  if (st != AMDSMI_STATUS_SUCCESS || event->variant >= count) {
    if (metrics)
      free(metrics);
    return PAPI_EMISC;
  }
  event->value = (int64_t)metrics[event->variant].value;
  free(metrics);
  return PAPI_OK;
}

int access_amdsmi_pm_enabled(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_is_gpu_power_management_enabled_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  bool enabled = false;
  if (amdsmi_is_gpu_power_management_enabled_p(device_handles[event->device],
                                               &enabled) !=
      AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = enabled ? 1 : 0;
  return PAPI_OK;
}

int access_amdsmi_ras_ecc_schema(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_ras_feature_info_p)
    return PAPI_ENOSUPP;

  amdsmi_ras_feature_t ras = {0};
  amdsmi_status_t st = amdsmi_get_gpu_ras_feature_info_p(device_handles[event->device], &ras);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)ras.ecc_correction_schema_flag;
  return PAPI_OK;
}

int access_amdsmi_ras_eeprom_version(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_ras_feature_info_p)
    return PAPI_ENOSUPP;

  amdsmi_ras_feature_t ras = {0};
  amdsmi_status_t st = amdsmi_get_gpu_ras_feature_info_p(device_handles[event->device], &ras);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)ras.ras_eeprom_version;
  return PAPI_OK;
}

int access_amdsmi_ras_eeprom_validate(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_gpu_validate_ras_eeprom_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_status_t st =
      amdsmi_gpu_validate_ras_eeprom_p(device_handles[event->device]);
  if (st == AMDSMI_STATUS_SUCCESS) {
    event->value = 0;              /* checksum OK */
  } else if (st == AMDSMI_STATUS_CORRUPTED_EEPROM) {
    event->value = 1;              /* EEPROM corrupted */
  } else {
    event->value = 2;              /* unexpected failure */
  }
  return PAPI_OK;
}

int access_amdsmi_ras_block_state(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_ras_block_features_enabled_p)
    return PAPI_ENOSUPP;

  amdsmi_ras_err_state_t state;
  amdsmi_status_t st =
      amdsmi_get_gpu_ras_block_features_enabled_p(device_handles[event->device], (amdsmi_gpu_block_t)event->variant, &state);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)state;
  return PAPI_OK;
}

int access_amdsmi_reg_count(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (amdsmi_lib_major < 25 || !amdsmi_get_gpu_reg_table_info_p)
    return PAPI_ENOSUPP;

  amdsmi_reg_type_t reg_type = (amdsmi_reg_type_t)event->variant; /* set at registration */
  amdsmi_name_value_t *regs = NULL;
  uint32_t num = 0;
  amdsmi_status_t st = amdsmi_get_gpu_reg_table_info_p(device_handles[event->device], reg_type, &regs, &num);
  if (regs)
    free(regs);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)num;
  return PAPI_OK;
}

int access_amdsmi_reg_value(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (amdsmi_lib_major < 25 || !amdsmi_get_gpu_reg_table_info_p)
    return PAPI_ENOSUPP;

  amdsmi_reg_type_t reg_type = (amdsmi_reg_type_t)event->variant;
  amdsmi_name_value_t *regs = NULL;
  uint32_t num = 0;
  amdsmi_status_t st = amdsmi_get_gpu_reg_table_info_p(device_handles[event->device], reg_type, &regs, &num);
  if (st != AMDSMI_STATUS_SUCCESS || event->subvariant >= num) {
    if (regs)
      free(regs);
    return PAPI_EMISC;
  }
  event->value = regs[event->subvariant].value;
  free(regs);
  return PAPI_OK;
}

int access_amdsmi_voltage(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_volt_metric_p)
    return PAPI_ENOSUPP;

  amdsmi_voltage_type_t sensor = (amdsmi_voltage_type_t)event->subvariant;  /* set at registration */
  amdsmi_voltage_metric_t metric = (amdsmi_voltage_metric_t)event->variant; /* e.g., AMDSMI_VOLT_CURRENT */
  int64_t mv = 0;
  amdsmi_status_t st = amdsmi_get_gpu_volt_metric_p(device_handles[event->device], sensor, metric, &mv);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)mv; /* API reports mV */
  return PAPI_OK;
}

int access_amdsmi_vram_width(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_vram_info_p)
    return PAPI_ENOSUPP;

  amdsmi_vram_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_vram_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)info.vram_bit_width;
  return PAPI_OK;
}

int access_amdsmi_vram_size(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_vram_info_p)
    return PAPI_ENOSUPP;

  amdsmi_vram_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_vram_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  /* vram_size reported in MB */
  event->value = (uint64_t)info.vram_size * 1024ULL * 1024ULL;
  return PAPI_OK;
}

int access_amdsmi_vram_type(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_vram_info_p)
    return PAPI_ENOSUPP;

  amdsmi_vram_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_vram_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)info.vram_type;
  return PAPI_OK;
}

int access_amdsmi_vram_vendor(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_vram_info_p)
    return PAPI_ENOSUPP;

  amdsmi_vram_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_vram_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)info.vram_vendor;
  return PAPI_OK;
}

int access_amdsmi_vram_usage(int mode, void *arg) {
  if (mode != PAPI_MODE_READ) return PAPI_ENOSUPP;

  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }

  /* variant: 0 = total MB, 1 = used MB */
  if (event->variant == 0) {
    /* TOTAL: prefer vram_info to avoid the buggy usage path */
    if (!amdsmi_get_gpu_vram_info_p) return PAPI_ENOSUPP;

    amdsmi_vram_info_t vinf;
    memset(&vinf, 0, sizeof(vinf));
    if (amdsmi_get_gpu_vram_info_p(device_handles[event->device], &vinf)
        != AMDSMI_STATUS_SUCCESS)
      return PAPI_EMISC;
    /* vinf.vram_size is reported in MB by AMD SMI */
    event->value = (uint64_t)vinf.vram_size;
    return PAPI_OK;
  }

  /* USED: keep using vram_usage for the used number */
  if (!amdsmi_get_gpu_vram_usage_p) return PAPI_ENOSUPP;

  amdsmi_vram_usage_t u;
  memset(&u, 0, sizeof(u));
  if (amdsmi_get_gpu_vram_usage_p(device_handles[event->device], &u)
      != AMDSMI_STATUS_SUCCESS) {
    event->value = 0;
    return PAPI_OK;
  }
  event->value = (uint64_t)u.vram_used;  /* MB */
  return PAPI_OK;
}


int access_amdsmi_soc_pstate_id(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_soc_pstate_p)
    return PAPI_ENOSUPP;

  amdsmi_dpm_policy_t pol = {0};
  amdsmi_status_t st = amdsmi_get_soc_pstate_p(device_handles[event->device], &pol);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)pol.current;
  return PAPI_OK;
}

int access_amdsmi_soc_pstate_supported(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_soc_pstate_p)
    return PAPI_ENOSUPP;

  amdsmi_dpm_policy_t pol = {0};
  amdsmi_status_t st = amdsmi_get_soc_pstate_p(device_handles[event->device], &pol);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)pol.num_supported;
  return PAPI_OK;
}

int access_amdsmi_xgmi_plpd_id(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_xgmi_plpd_p)
    return PAPI_ENOSUPP;

  amdsmi_dpm_policy_t pol = {0};
  amdsmi_status_t st = amdsmi_get_xgmi_plpd_p(device_handles[event->device], &pol);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)pol.current;
  return PAPI_OK;
}

int access_amdsmi_xgmi_plpd_supported(int mode, void *arg) {
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device]) {
    return PAPI_EMISC;
  }
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_xgmi_plpd_p)
    return PAPI_ENOSUPP;

  amdsmi_dpm_policy_t pol = {0};
  amdsmi_status_t st = amdsmi_get_xgmi_plpd_p(device_handles[event->device], &pol);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (uint64_t)pol.num_supported;
  return PAPI_OK;
}

int access_amdsmi_process_isolation(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_process_isolation_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  uint32_t val = 0;
  amdsmi_status_t st = amdsmi_get_gpu_process_isolation_p(device_handles[event->device], &val);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)val;
  return PAPI_OK;
}

int access_amdsmi_xcd_counter(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_xcd_counter_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  uint16_t cnt = 0;
  amdsmi_status_t st = amdsmi_get_gpu_xcd_counter_p(device_handles[event->device], &cnt);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = cnt;
  return PAPI_OK;
}

int access_amdsmi_board_info_hash(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_board_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_board_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_board_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  const char *field = NULL;
  switch (event->variant) {
  case 0:
    field = info.product_serial;
    break;
  case 1:
    field = info.model_number;
    break;
  case 2:
    field = info.fru_id;
    break;
  case 3:
    field = info.product_name;
    break;
  case 4:
    field = info.manufacturer_name;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  if (!field || !field[0])
    return PAPI_ENOSUPP;
  event->value = (int64_t)_str_to_u64_hash(field);
  return PAPI_OK;
}

int access_amdsmi_fw_version(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_fw_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles || !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_fw_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_fw_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;

  amdsmi_fw_block_t id = (amdsmi_fw_block_t)event->variant;
  uint8_t n = info.num_fw_info;
  if (n > AMDSMI_FW_ID__MAX)
    n = AMDSMI_FW_ID__MAX;
  for (uint8_t i = 0; i < n; ++i) {
    if (info.fw_info_list[i].fw_id == id) {
      event->value = (int64_t)info.fw_info_list[i].fw_version;
      return PAPI_OK;
    }
  }
  return PAPI_EMISC;
}

#if AMDSMI_LIB_VERSION_MAJOR >= 25
int access_amdsmi_vram_max_bandwidth(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (amdsmi_lib_major < 25 || !amdsmi_get_gpu_vram_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_vram_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_gpu_vram_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)info.vram_max_bandwidth; /* GB/s */
  return PAPI_OK;
}
#endif

int access_amdsmi_memory_reserved_pages(int mode, void *arg) {
  if (mode != PAPI_MODE_READ || !amdsmi_get_gpu_memory_reserved_pages_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;
  uint32_t num = 0;
  if (amdsmi_get_gpu_memory_reserved_pages_p(device_handles[event->device], &num,
                                             NULL) != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)num;
  return PAPI_OK;
}

int access_amdsmi_bad_page_count(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_bad_page_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  uint32_t num = 0;
  amdsmi_status_t st = amdsmi_get_gpu_bad_page_info_p(device_handles[event->device], &num, NULL);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)num;
  return PAPI_OK;
}

int access_amdsmi_bad_page_threshold(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_bad_page_threshold_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  uint32_t thr = 0;
  amdsmi_status_t st = amdsmi_get_gpu_bad_page_threshold_p(device_handles[event->device], &thr);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)thr;
  return PAPI_OK;
}

int access_amdsmi_bad_page_record(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_gpu_bad_page_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  uint32_t num = 0;
  amdsmi_status_t st = amdsmi_get_gpu_bad_page_info_p(device_handles[event->device], &num, NULL);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  if (event->subvariant >= num)
    return PAPI_EMISC;
  amdsmi_retired_page_record_t *recs = (amdsmi_retired_page_record_t *)papi_calloc(num, sizeof(amdsmi_retired_page_record_t));
  if (!recs)
    return PAPI_ENOMEM;
  st = amdsmi_get_gpu_bad_page_info_p(device_handles[event->device], &num, recs);
  if (st != AMDSMI_STATUS_SUCCESS) {
    papi_free(recs);
    return PAPI_EMISC;
  }
  amdsmi_retired_page_record_t rec = recs[event->subvariant];
  papi_free(recs);
  switch (event->variant) {
  case 0:
    event->value = (int64_t)rec.page_address;
    break;
  case 1:
    event->value = (int64_t)rec.page_size;
    break;
  case 2:
    event->value = (int64_t)rec.status;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_power_sensor(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_power_info_v2_p)
    return PAPI_ENOSUPP;

  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count ||
      !device_handles[event->device])
    return PAPI_EMISC;

  amdsmi_power_info_t info;
  memset(&info, 0, sizeof(info));  /* critical: avoid uninitialised fields */

  amdsmi_status_t st =
      amdsmi_get_power_info_v2_p(device_handles[event->device],
                                 (uint32_t)event->subvariant, &info);
  if (st != AMDSMI_STATUS_SUCCESS) {
    event->value = 0;
    return PAPI_OK;
  }

  switch (event->variant) {
    case 0: event->value = (int64_t)info.current_socket_power; break; /* W */
    case 1: event->value = (int64_t)info.average_socket_power; break; /* W */
#if AMDSMI_LIB_VERSION_MAJOR >= 25
    case 2: event->value = (int64_t)info.socket_power; break;         /* uW */
#endif
    case 3: event->value = (int64_t)info.gfx_voltage; break;          /* mV */
    case 4: event->value = (int64_t)info.soc_voltage; break;          /* mV */
    case 5: event->value = (int64_t)info.mem_voltage; break;          /* mV */
    case 6: event->value = (int64_t)info.power_limit; break;          /* W */
    default: return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}


int access_amdsmi_pcie_info(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_pcie_info_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_pcie_info_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st = amdsmi_get_pcie_info_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  // Variant mapping:
  // 0 max width, 1 max speed, 2 interface version, 3 slot type,
  // 4 max interface version (lib >=25),
  // 5 current width, 6 current speed, 7 bandwidth,
  // 8 replay count, 9 L0->recovery count, 10 replay rollover count,
  // 11 NAK sent count, 12 NAK received count,
  // 13 other-end recovery count
  switch (event->variant) {
  case 0:
    event->value = info.pcie_static.max_pcie_width;
    break;
  case 1:
    event->value = (int64_t)info.pcie_static.max_pcie_speed;
    break;
  case 2:
    event->value = (int64_t)info.pcie_static.pcie_interface_version;
    break;
  case 3:
    event->value = (int64_t)info.pcie_static.slot_type;
    break;
#if AMDSMI_LIB_VERSION_MAJOR >= 25
  case 4:
    if (amdsmi_lib_major < 25)
      return PAPI_ENOSUPP;
    event->value = (int64_t)info.pcie_static.max_pcie_interface_version;
    break;
#endif
  case 5:
    event->value = info.pcie_metric.pcie_width;
    break;
  case 6:
    event->value = (int64_t)info.pcie_metric.pcie_speed;
    break;
  case 7:
    event->value = (int64_t)info.pcie_metric.pcie_bandwidth;
    break;
  case 8:
    event->value = (int64_t)info.pcie_metric.pcie_replay_count;
    break;
  case 9:
    event->value = (int64_t)info.pcie_metric.pcie_l0_to_recovery_count;
    break;
  case 10:
    event->value = (int64_t)info.pcie_metric.pcie_replay_roll_over_count;
    break;
  case 11:
    event->value = (int64_t)info.pcie_metric.pcie_nak_sent_count;
    break;
  case 12:
    event->value = (int64_t)info.pcie_metric.pcie_nak_received_count;
    break;
  case 13:
    event->value = (int64_t)info.pcie_metric.pcie_lc_perf_other_end_recovery_count;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}

int access_amdsmi_event_notification(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_init_gpu_event_notification_p || !amdsmi_set_gpu_event_notification_mask_p || !amdsmi_get_gpu_event_notification_p ||
      !amdsmi_stop_gpu_event_notification_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_status_t st = amdsmi_init_gpu_event_notification_p(device_handles[event->device]);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  uint64_t mask = AMDSMI_EVENT_MASK_FROM_INDEX(event->variant);
  st = amdsmi_set_gpu_event_notification_mask_p(device_handles[event->device], mask);
  if (st != AMDSMI_STATUS_SUCCESS) {
    amdsmi_stop_gpu_event_notification_p(device_handles[event->device]);
    return PAPI_EMISC;
  }
  amdsmi_evt_notification_data_t data[8];
  uint32_t num = 8;
  st = amdsmi_get_gpu_event_notification_p(0, &num, data);
  uint32_t cnt = 0;
  if (st == AMDSMI_STATUS_SUCCESS) {
    for (uint32_t i = 0; i < num; ++i)
      if (data[i].event == (amdsmi_evt_notification_type_t)event->variant)
        cnt++;
  }
  amdsmi_stop_gpu_event_notification_p(device_handles[event->device]);
  event->value = (int64_t)cnt;
  return PAPI_OK;
}

int access_amdsmi_utilization_count(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_utilization_count_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_utilization_counter_t cnt;
  memset(&cnt, 0, sizeof(cnt));
  cnt.type = (amdsmi_utilization_counter_type_t)event->variant;
  uint64_t ts = 0;
  amdsmi_status_t st =
      amdsmi_get_utilization_count_p(device_handles[event->device], &cnt, 1, &ts);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  event->value = (int64_t)cnt.value;
  return PAPI_OK;
}

int access_amdsmi_violation_status(int mode, void *arg) {
  if (mode != PAPI_MODE_READ)
    return PAPI_ENOSUPP;
  if (!amdsmi_get_violation_status_p)
    return PAPI_ENOSUPP;
  native_event_t *event = (native_event_t *)arg;
  if (event->device < 0 || event->device >= device_count || !device_handles ||
      !device_handles[event->device])
    return PAPI_EMISC;
  amdsmi_violation_status_t info;
  memset(&info, 0, sizeof(info));
  amdsmi_status_t st =
      amdsmi_get_violation_status_p(device_handles[event->device], &info);
  if (st != AMDSMI_STATUS_SUCCESS)
    return PAPI_EMISC;
  switch (event->variant) {
  case 0:
    event->value = (int64_t)info.acc_ppt_pwr;
    break;
  case 1:
    event->value = (int64_t)info.acc_socket_thrm;
    break;
  case 2:
    event->value = (int64_t)info.acc_vr_thrm;
    break;
  case 3:
    event->value = (int64_t)info.per_ppt_pwr;
    break;
  case 4:
    event->value = (int64_t)info.per_socket_thrm;
    break;
  case 5:
    event->value = (int64_t)info.per_vr_thrm;
    break;
  case 6:
    event->value = (int64_t)info.active_ppt_pwr;
    break;
  case 7:
    event->value = (int64_t)info.active_socket_thrm;
    break;
  case 8:
    event->value = (int64_t)info.active_vr_thrm;
    break;
  default:
    return PAPI_ENOSUPP;
  }
  return PAPI_OK;
}
