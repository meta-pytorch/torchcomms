// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <limits.h>

#include "comms/utils/cvars/nccl_cvars.h"

namespace {
constexpr int MAX_FILE_LEN = 255;

// TODO: We follow the instructions from AMD
// (https://ontrack.amd.com/browse/FBA-621). Revisit this function and file
// once AMD supports this feature in their official release.
//
// Adopted from
// https://github.com/ROCm/rccl/blob/275fdd43c104de684b2db5c0147e560bbde0a2e1/src/graph/xml.cc#L446
void ncclTopoGetStrFromSysHelper(
    const char* path,
    const char* fileName,
    char* strValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE* file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_FILE_LEN) {
      int len = fread(strValue + offset, 1, MAX_FILE_LEN - offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
  } else {
    strValue[offset - 1] = '\0';
  }
}
} // namespace

inline bool getGpuDirectRDMASupported() {
  // TODO: We follow the instructions from AMD
  // (https://ontrack.amd.com/browse/FBA-621). Revisit this function and file
  // once AMD supports this feature in their official release.
  //
  // Adopted from
  // https://github.com/ROCm/rccl/blob/275fdd43c104de684b2db5c0147e560bbde0a2e1/src/transport/net_ib.cc#L734

  int ncclIbGdrModuleLoaded = 0;
  // Check for `memory_peers` directory containing `amdkfd/version`
  // This `memory_peers` directory is created by NIC-GPU driver interaction
  // On Linux kernel 5.15.0 (e.g. Ubuntu 22.04), `memory_peers` is created under
  // `/sys/kernel/mm/` However, on newer kernels like Ubuntu 24.04.1 (Linux
  // kernel 6.8.0) or Ubuntu 22.04.4 HWE (Linux kernel 6.5.0), this
  // `memory_peers` directory is either not created (go to else-if condition) or
  // created under a different path like `/sys/kernel/` or `/sys/` (depending on
  // your ib_peer_mem module)
  const char* memory_peers_paths[] = {
      "/sys/kernel/mm/memory_peers/amdkfd/version",
      "/sys/kernel/memory_peers/amdkfd/version",
      "/sys/memory_peers/amdkfd/version",
      NULL};
  int i = 0;

  while (memory_peers_paths[i]) {
    if (access(memory_peers_paths[i], F_OK) == 0) {
      ncclIbGdrModuleLoaded = 1;
      break;
    } else {
      ncclIbGdrModuleLoaded = 0;
    }
    ++i;
  }

  char strValue[MAX_FILE_LEN];
  ncclTopoGetStrFromSysHelper(
      "/sys/devices/virtual/dmi/id", "bios_version", strValue);
  if (strncmp("Hyper-V UEFI Release", strValue, 20) == 0) {
    int roMode = NCCL_IB_PCI_RELAXED_ORDERING;
    ncclTopoGetStrFromSysHelper("/proc/sys/kernel", "numa_balancing", strValue);
    if (strcmp(strValue, "1") == 0 && roMode == 0)
      ncclIbGdrModuleLoaded = 0;
  }

  if (ncclIbGdrModuleLoaded == 0) {
    // Check for `ib_register_peer_memory_client` symbol in `/proc/kallsyms`
    // if your system uses native OS ib_peer module
    char buf[256];
    FILE* fp = NULL;
    fp = fopen("/proc/kallsyms", "r");

    if (fp == NULL) {
    } else {
      while (fgets(buf, sizeof(buf), fp) != NULL) {
        if (strstr(buf, "t ib_register_peer_memory_client") != NULL ||
            strstr(buf, "T ib_register_peer_memory_client") != NULL) {
          ncclIbGdrModuleLoaded = 1;
        }
      }
    }
  }

  return ncclIbGdrModuleLoaded != 0;
}
#endif
