// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/HostDeviceAdapter.h"

#include <unistd.h>
#include <cstdlib>
#include <system_error>

namespace uniflow {

Result<void*> HostDeviceAdapter::pinnedHostAlloc(size_t size) {
  if (size == 0) {
    return Err(ErrCode::InvalidArgument, "pinnedHostAlloc: size must be > 0");
  }
  static const size_t pageSize = [] {
    const long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? static_cast<size_t>(value) : size_t{4096};
  }();
  void* ptr = nullptr;
  const int rc = ::posix_memalign(&ptr, pageSize, size);
  if (rc != 0) {
    return Err(
        ErrCode::DriverError,
        "posix_memalign failed: " + std::generic_category().message(rc));
  }
  return ptr;
}

Status HostDeviceAdapter::pinnedHostFree(void* ptr) {
  ::free(ptr);
  return Ok();
}

Result<void*> HostDeviceAdapter::hostGetDevicePointer(void* hostPtr) {
  return hostPtr;
}

Result<void*> HostDeviceAdapter::hostGetDevicePointer(
    int deviceId,
    void* hostPtr) {
  if (deviceId >= 0) {
    return Err(
        ErrCode::NotImplemented, "host storage has no GPU device mapping");
  }
  return hostGetDevicePointer(hostPtr);
}

Result<bool> HostDeviceAdapter::isDmaBuffSupported(int deviceId) {
  if (deviceId >= 0) {
    return Err(
        ErrCode::NotImplemented, "host adapter cannot register local VRAM");
  }
  return false;
}

Result<DmaBuff> HostDeviceAdapter::exportDmaBuff(
    int /* deviceId */,
    void* /* ptr */,
    size_t /* len */,
    DmaBufMapping /* mapping */) {
  return Err(ErrCode::NotImplemented, "host adapter cannot export GPU DMA-BUF");
}

Status HostDeviceAdapter::closeDmaBuff(DmaBuff& buff) {
  if (buff.fd >= 0) {
    return Err(
        ErrCode::InvalidArgument, "host adapter does not own GPU DMA-BUF");
  }
  return Ok();
}

std::shared_ptr<DeviceAdapter> createHostDeviceAdapter() {
  return std::make_shared<HostDeviceAdapter>();
}

} // namespace uniflow
