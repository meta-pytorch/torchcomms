#pragma once

#include <gmock/gmock.h>
#include "comms/torchcomms/device/xpu/XpuApi.hpp"

namespace torch::comms::test {

class XpuMock : public XpuApi {
 public:
  ~XpuMock() override = default;

  MOCK_METHOD(xpu_result_t, setDevice, (int device), (override));
  MOCK_METHOD(xpu_result_t, getDeviceProperties, (xpuDeviceProp* prop, int device), (override));
  MOCK_METHOD(xpu_result_t, memGetInfo, (size_t* free, size_t* total), (override));
  MOCK_METHOD(xpu_result_t, getDeviceCount, (int* count), (override));

  MOCK_METHOD(xpu_result_t, streamCreateWithPriority, (xpuStream_t& stream, unsigned int flags, int priority), (override));
  MOCK_METHOD(xpu_result_t, streamDestroy, (const xpuStream_t& stream), (override));
  MOCK_METHOD(xpu_result_t, streamWaitEvent, (const xpuStream_t& stream, xpuEvent_t& event, unsigned int flags), (override));
  MOCK_METHOD(xpuStream_t, getCurrentXPUStream, (int device_index), (override));
  MOCK_METHOD(xpu_result_t, streamSynchronize, (const xpuStream_t& stream), (override));
  MOCK_METHOD(xpu_result_t, streamIsCapturing, (const xpuStream_t& stream, xpuStreamCaptureStatus* pCaptureStatus), (override));
  MOCK_METHOD(xpu_result_t, streamGetCaptureInfo, (const xpuStream_t& stream, xpuStreamCaptureStatus* pCaptureStatus, unsigned long long* pId), (override));

  MOCK_METHOD(xpu_result_t, malloc, (void** devPtr, size_t size), (override));
  MOCK_METHOD(xpu_result_t, free, (void* devPtr), (override));
  MOCK_METHOD(xpu_result_t, memcpyAsync, (void* dst, const void* src, size_t count, const xpuStream_t& stream), (override));

  MOCK_METHOD(xpu_result_t, eventCreate, (xpuEvent_t& event), (override));
  MOCK_METHOD(xpu_result_t, eventCreateWithFlags, (xpuEvent_t& event, unsigned int flags), (override));
  MOCK_METHOD(xpu_result_t, eventDestroy, (const xpuEvent_t& event), (override));
  MOCK_METHOD(xpu_result_t, eventRecord, (xpuEvent_t& event, const xpuStream_t& stream), (override));
  MOCK_METHOD(xpu_result_t, eventQuery, (const xpuEvent_t& event), (override));

  MOCK_METHOD(xpu_result_t, userObjectCreate, (xpuUserObject_t* object_out, void* ptr, xpuHostFn_t destroy, unsigned int initialRefcount, unsigned int flags), (override));
  MOCK_METHOD(xpu_result_t, graphRetainUserObject, (xpuGraph_t graph, xpuUserObject_t object, unsigned int count, unsigned int flags), (override));
  MOCK_METHOD(xpu_result_t, streamGetCaptureInfo_v2, (const xpuStream_t& stream, xpuStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, xpuGraph_t* graph_out, const xpuGraphNode_t** dependencies_out, size_t* numDependencies_out), (override));

  MOCK_METHOD(const char*, getErrorString, (xpu_result_t error), (override));

  void setupDefaultBehaviors() {
    using ::testing::_;
    using ::testing::Return;
    
    ON_CALL(*this, setDevice(_)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, getDeviceProperties(_, _)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, memGetInfo(_, _)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, getDeviceCount(_)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, streamCreateWithPriority(_, _, _)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, streamDestroy(_)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, streamWaitEvent(_, _, _)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, eventCreateWithFlags(_, _)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, eventDestroy(_)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, eventRecord(_, _)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, eventQuery(_)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, malloc(_, _)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, free(_)).WillByDefault(Return(XPU_SUCCESS));
    ON_CALL(*this, getErrorString(_)).WillByDefault(Return("Mock XPU Error"));
    
    ON_CALL(*this, getCurrentXPUStream(_)).WillByDefault([](int device_index) {
        return c10::xpu::XPUStream::unpack3(0, device_index, c10::DeviceType::XPU);
    });
  }
};

} // namespace torch::comms::test