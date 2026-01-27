// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "TorchCommNCCLXTestBase.hpp"

#include <gmock/gmock.h>

#include <chrono>
#include <cstdlib>
#include <memory>

#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXBootstrap.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CachingAllocatorHookMock.hpp"

namespace torch {
namespace comms {
namespace test {

// ============================================================================
// 1. INITIALIZATION TESTS
// ============================================================================

TEST_F(TorchCommNCCLXTest, TestOptionsEnvironmentVariables) {
  setupCCAExpectations(0, 0, 1);

  setOptionsEnvironmentVariables(false, 1.0); // false abort

  CommOptions options1;
  EXPECT_EQ(options1.abort_process_on_timeout_or_error, false);
  EXPECT_EQ(options1.timeout, std::chrono::milliseconds(1000));

  setOptionsEnvironmentVariables(true, 1.5); // true abort
  CommOptions options2;
  EXPECT_EQ(options2.abort_process_on_timeout_or_error, true);
  EXPECT_EQ(options2.timeout, std::chrono::milliseconds(1500));
}

TEST_F(TorchCommNCCLXTest, UniqueCommDesc) {
  setupRankAndSize(0, 4); // rank 0, size 4
  cuda_mock_->setupDefaultBehaviors();
  mock_hook_->setupDefaultBehaviors();

  ON_CALL(*nccl_mock_, getUniqueId(_))
      .WillByDefault(
          DoAll(SetArgPointee<0>(ncclUniqueId{}), Return(ncclSuccess)));

  ON_CALL(*nccl_mock_, commInitRankConfig(_, _, _, 0, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ON_CALL(*nccl_mock_, commSplit(_, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<3>(reinterpret_cast<ncclComm_t>(0x4000)),
          Return(ncclSuccess)));

  auto validateCommNameUnique =
      [](std::vector<std::shared_ptr<TorchCommBackend>> comms) {
        for (size_t i = 0; i < comms.size(); i++) {
          for (size_t j = i + 1; j < comms.size(); j++) {
            EXPECT_NE(comms[i]->getCommName(), comms[j]->getCommName());
          }
        }
      };

  auto comm1 = createMockedTorchComm();
  auto comm2 = createMockedTorchComm();

  EXPECT_NO_THROW(comm1->init(at::kCUDA, "test_name1", default_options_));
  EXPECT_NO_THROW(comm2->init(at::kCUDA, "test_name2", default_options_));

  auto split_comm1 = comm1->split({0, 1, 2, 3}, "split_comm");
  EXPECT_TRUE(split_comm1 != nullptr);
  auto split_comm2 = comm2->split({0, 1, 2, 3}, "split_comm");
  EXPECT_TRUE(split_comm2 != nullptr);

  auto split_comm1_2 = split_comm1->split({0, 1}, "split_comm");
  EXPECT_TRUE(split_comm1_2 != nullptr);
  auto split_comm1_3 = split_comm1->split({0, 1, 2}, "split_comm");
  EXPECT_TRUE(split_comm1_3 != nullptr);

  auto split_comm2_2 = split_comm2->split({0, 1}, "split_comm");
  EXPECT_TRUE(split_comm2_2 != nullptr);
  auto split_comm2_3 = split_comm2->split({0, 1, 2}, "split_comm");
  EXPECT_TRUE(split_comm2_3 != nullptr);

  std::vector<std::shared_ptr<TorchCommBackend>> comms = {
      comm1,
      comm2,
      split_comm1,
      split_comm2,
      split_comm1_2,
      split_comm1_3,
      split_comm2_2,
      split_comm2_3};

  validateCommNameUnique(comms);
  for (auto& comm : comms) {
    comm->finalize();
  }
}

TEST_F(TorchCommNCCLXTest, InitializationRank0GetUniqueId) {
  // Test: if node is rank 0, it will try to get a unique id and store it in the
  // store
  setupRankAndSize(0, 2); // rank 0, size 2

  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  cuda_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();

  // Expect rank 0 to get unique ID and store it
  ncclUniqueId expected_id{};
  // NOLINTNEXTLINE(facebook-hte-BadMemset)
  memset(&expected_id, 0x42, sizeof(expected_id)); // Fill with test pattern

  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .WillOnce(DoAll(SetArgPointee<0>(expected_id), Return(ncclSuccess)));

  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  auto bootstrap = new TorchCommNCCLXBootstrap(
      store_, *device_, nccl_mock_, cuda_mock_, std::chrono::seconds(60));
  auto store_key = bootstrap->getNCCLXStoreKeyPrefix() +
      std::to_string(bootstrap->getNCCLXStoreKeyCounter() - 1);
  delete bootstrap;

  // Verify the unique ID was stored in the store
  auto stored_vec = store_->get(store_key);
  ncclUniqueId stored_id;
  memcpy(&stored_id, stored_vec.data(), sizeof(stored_id));

  EXPECT_EQ(memcmp(&stored_id, &expected_id, sizeof(ncclUniqueId)), 0);
  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, InitializationNonRank0ReadUniqueId) {
  // Test: if node is not rank 0, it will not try to get a unique id, but will
  // get it from store
  setupRankAndSize(1, 2); // rank 1, size 2

  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  auto bootstrap = new TorchCommNCCLXBootstrap(
      store_, *device_, nccl_mock_, cuda_mock_, std::chrono::seconds(60));
  auto store_key = bootstrap->getNCCLXStoreKeyPrefix() +
      std::to_string(bootstrap->getNCCLXStoreKeyCounter());
  delete bootstrap;

  // Pre-populate store with unique ID (as if rank 0 already stored it)
  ncclUniqueId expected_id{};
  // NOLINTNEXTLINE(facebook-hte-BadMemset)
  memset(&expected_id, 0x42, sizeof(expected_id)); // Fill with test pattern
  std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
  memcpy(id_vec.data(), &expected_id, sizeof(expected_id));
  store_->set(store_key, id_vec);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();

  // Expect rank 1 to NOT call getUniqueId, but to use the stored ID
  EXPECT_CALL(*nccl_mock_, getUniqueId(_))
      .Times(0); // Should not be called for non-rank 0

  EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 1, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  auto options = default_options_;
  options.store = store_;
  EXPECT_NO_THROW(comm->init(*device_, "test_name", options));
  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, InitializationFailsWithInvalidDeviceId) {
  // Test: TorchComm creation should fail when device ID is invalid
  setupRankAndSize(0, 2); // rank 0, size 2

  // Setup CCA expectations - first init (device -1) should succeed
  setupCCAExpectations(1, 2, 0);

  cuda_mock_->setupDefaultBehaviors();

  // Test with negative device ID
  {
    at::Device invalid_device(at::DeviceType::CUDA, -1);
    auto comm = std::make_shared<TestTorchCommNCCLX>();
    comm->setCudaApi(cuda_mock_);
    comm->setNcclApi(nccl_mock_);

    // Mock getDeviceCount to return a valid device count (needed for rank %
    // device_count)
    EXPECT_CALL(*cuda_mock_, getDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    // Mock malloc for barrier buffer allocation in bootstrap constructor
    EXPECT_CALL(*cuda_mock_, malloc(_, sizeof(float)))
        .Times(2)
        .WillRepeatedly(DoAll(
            SetArgPointee<0>(reinterpret_cast<void*>(0x1000)),
            Return(cudaSuccess)));

    // Mock free for barrier buffer deallocation in bootstrap destructor
    EXPECT_CALL(*cuda_mock_, free(reinterpret_cast<void*>(0x1000)))
        .Times(2)
        .WillRepeatedly(Return(cudaSuccess));

    // Mock CUDA API to be called with device ID 0, since the boostrap
    // logic will assign a device ID in this case based on the rank.
    EXPECT_CALL(*cuda_mock_, setDevice(0))
        .Times(2)
        .WillRepeatedly(Return(cudaSuccess));
    EXPECT_CALL(*nccl_mock_, commInitRankConfig(_, 2, _, 0, _))
        .WillOnce(DoAll(
            SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
            Return(ncclSuccess)));

    // Initialization should NOT throw
    EXPECT_NO_THROW(comm->init(invalid_device, "test_name", default_options_));

    comm->finalize();
  }

  // Reset mocks for next test
  ::testing::Mock::VerifyAndClear(cuda_mock_.get());
  ::testing::Mock::VerifyAndClear(nccl_mock_.get());
  ::testing::Mock::VerifyAndClear(mock_hook_);

  // Setup CCA expectations - no init should succeed, so only destructor calls
  setupCCAExpectations(0, 1, 1);

  // Test with device ID larger than available devices
  {
    at::Device invalid_device(at::DeviceType::CUDA, 127);
    auto comm = std::make_shared<TestTorchCommNCCLX>();
    comm->setCudaApi(cuda_mock_);
    comm->setNcclApi(nccl_mock_);

    // Mock CUDA API to return error for device ID that's too large
    EXPECT_CALL(*cuda_mock_, setDevice(127))
        .WillOnce(Return(cudaErrorInvalidDevice));
    EXPECT_CALL(*cuda_mock_, getErrorString(cudaErrorInvalidDevice))
        .WillRepeatedly(Return("set device error"));

    // Initialization should throw due to invalid device
    EXPECT_THROW(
        comm->init(invalid_device, "test_name", default_options_),
        std::runtime_error);

    comm->finalize();
  }
}

// ============================================================================
// 2. FINALIZE HANDLING TESTS
// ============================================================================

TEST_F(TorchCommNCCLXTest, FinalizeNoJobsScheduled) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  // Test: if no jobs are scheduled, finalize() returns immediately
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  // Expect finalize to clean up resources
  setupNormalDestruction(*comm);

  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(TorchCommNCCLXTest, FinalizeWorkNotFinishedWaitsForCompletion) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  // Test: if work is scheduled but not finished, finalize waits until work
  // completes
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  // Set up expectations for send operation
  auto tensor = createTestTensor({10, 10});

  setupEventsForWork(*comm, 1);

  EXPECT_CALL(*nccl_mock_, send(_, _, _, 1, _, _))
      .WillOnce(Return(ncclSuccess));

  // Set up expectations for finalize - work is initially not ready, then
  // becomes ready

  InSequence seq;
  auto& work_event = work_events_[0];
  // First few queries return not ready
  EXPECT_CALL(
      *cuda_mock_,
      eventQuery(reinterpret_cast<cudaEvent_t>(work_event.start_event)))
      .Times(2)
      .WillRepeatedly(Return(cudaErrorNotReady));

  // Then start event becomes ready
  EXPECT_CALL(
      *cuda_mock_,
      eventQuery(reinterpret_cast<cudaEvent_t>(work_event.start_event)))
      .WillOnce(Return(cudaSuccess));

  // End event queries - first not ready, then ready
  EXPECT_CALL(
      *cuda_mock_,
      eventQuery(reinterpret_cast<cudaEvent_t>(work_event.end_event)))
      .Times(2)
      .WillRepeatedly(Return(cudaErrorNotReady));

  EXPECT_CALL(
      *cuda_mock_,
      eventQuery(reinterpret_cast<cudaEvent_t>(work_event.end_event)))
      .WillRepeatedly(Return(cudaSuccess));

  // Schedule work
  auto work = comm->send(tensor, 1, true); // async send

  setupNormalDestruction(*comm);

  // Finalize should wait for work to complete
  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(TorchCommNCCLXTest, FinalizeWorkErrorThrowsNCCLXException) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  // Test: if work errors because cudaEventQuery returns error, finalize throws
  // NCCLXException
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  auto tensor = createTestTensor({10, 10});

  setupEventsForWork(*comm, 1);

  // Schedule work
  auto work = comm->send(tensor, 1, true); // async send

  auto& work_event = work_events_[0];
  setupWorkToError(work_event);

  // Should throw NCCLXException
  EXPECT_THROW(
      {
        try {
          comm->finalize();
        } catch (const NCCLXException& e) {
          EXPECT_EQ(e.getResult(), ncclInternalError);
          throw;
        }
      },
      NCCLXException);
}

TEST_F(TorchCommNCCLXTest, FinalizeWorkTimeoutThrowsRuntimeError) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  // Test: if work times out, finalize throws std::runtime_error
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  // Set up expectations for send operation
  auto tensor = createTestTensor({10, 10});

  setupEventsForWork(*comm, 1);
  // Schedule work
  auto work = comm->send(tensor, 1, true);

  auto& work_event = work_events_[0];
  setupWorkToTimeout(work_event);

  // Should throw runtime_error due to timeout
  EXPECT_THROW(
      {
        try {
          comm->finalize();
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(error_msg.find("timed out") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// ============================================================================
// 3. COLLECTIVE CALLING BEHAVIOR TESTS
// ============================================================================
// TODO: add more tests for other collectives
TEST_F(TorchCommNCCLXTest, WorkErrorCausesAbortDuringCollective) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 3 (abort, finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 3, 1);

  // Test: if work errors, calling TorchCommNCCLX method calls commAbort and
  // throws NCCLXException
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();
  comm->init(*device_, "test_name", default_options_);

  auto tensor = createTestTensor({10, 10});

  setupEventsForWork(*comm, 1);

  // Work starts but encounters an error
  WorkEvent& work_event = work_events_[0];
  setupWorkToError(work_event);

  auto work1 = comm->send(tensor, 1, true);

  comm->waitTillError();

  // Should throw NCCLXException due to error
  EXPECT_THROW(
      {
        try {
          auto work2 = comm->send(tensor, 1, true);
        } catch (const NCCLXException& e) {
          EXPECT_EQ(e.getResult(), ncclInternalError);
          throw;
        }
      },
      NCCLXException);

  // commDestroy should not be called since comm was aborted (set to nullptr)
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).Times(0);

  EXPECT_THROW(comm->finalize(), NCCLXException);
}

TEST_F(TorchCommNCCLXTest, WorkDefaultTimeoutCausesAbortDuringCollective) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 3 (abort, finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 3, 1);

  // Test: if work errors, calling TorchCommNCCLX method calls commAbort and
  // throws NCCLXException
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();
  comm->init(*device_, "test_name", default_options_);

  auto tensor = createTestTensor({10, 10});

  setupEventsForWork(*comm, 1);

  // Work starts but encounters an error
  WorkEvent& work_event = work_events_[0];
  setupWorkToTimeout(work_event);

  auto work1 = comm->send(tensor, 1, true);
  auto ncclx_work = reinterpret_cast<TorchWorkNCCLX*>(work1.get());
  EXPECT_EQ(getWorkTimeout(ncclx_work), default_options_.timeout);

  comm->waitTillTimeout();

  // Should throw NCCLXException due to error
  EXPECT_THROW(comm->send(tensor, 1, true), std::runtime_error);

  // commDestroy should not be called since comm was aborted (set to nullptr)
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).Times(0);

  EXPECT_THROW(comm->finalize(), std::runtime_error);
}

TEST_F(TorchCommNCCLXTest, WorkOperationTimeoutCausesAbortDuringCollective) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 3 (abort, finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 3, 1);

  // Test: if work errors, calling TorchCommNCCLX method calls commAbort and
  // throws NCCLXException
  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();
  comm->init(*device_, "test_name", default_options_);

  auto tensor = createTestTensor({10, 10});

  setupEventsForWork(*comm, 1);

  // Work starts but encounters an error
  WorkEvent& work_event = work_events_[0];
  setupWorkToTimeout(work_event);

  SendOptions send_options4;
  send_options4.timeout =
      std::chrono::milliseconds(3000); /* 3 seconds timeout */
  auto work1 = comm->send(tensor, 1, true, send_options4);

  auto ncclx_work = reinterpret_cast<TorchWorkNCCLX*>(work1.get());
  EXPECT_EQ(getWorkTimeout(ncclx_work), std::chrono::milliseconds(3000));

  comm->waitTillTimeout();

  // Should throw NCCLXException due to error
  SendOptions send_options3;
  EXPECT_THROW(comm->send(tensor, 1, true, send_options3), std::runtime_error);

  // commDestroy should not be called since comm was aborted (set to nullptr)
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).Times(0);

  EXPECT_THROW(comm->finalize(), std::runtime_error);
}

TEST_F(TorchCommNCCLXTest, AbortProcessOnTimeoutCausesProcessDeath) {
  // Setup CCA expectations
  // constructor, init, destructor are in the EXPECT_DEATH macro, which don't
  // count towards the expectations.  We only count the clear call in the
  // teardown.
  setupCCAExpectations(0, 0, 1);

  // Test: when abort_process_on_timeout_or_error is true, timeout should cause
  // process death
  EXPECT_DEATH(
      {
        // Create TorchComm with abort_process_on_timeout_or_error = true and
        // short timeout
        cuda_mock_->setupDefaultBehaviors();
        nccl_mock_->setupDefaultBehaviors();

        CommOptions options;
        options.abort_process_on_timeout_or_error = true;
        options.timeout = std::chrono::milliseconds(1000); // 1 second timeout
        options.store = store_;

        auto comm = createMockedTorchComm();

        cuda_mock_->setupDefaultBehaviors();
        nccl_mock_->setupDefaultBehaviors();
        comm->init(*device_, "test_name", options);

        auto tensor = createTestTensor({10, 10});
        setupEventsForWork(*comm, 1);

        // Work starts but never completes (simulating timeout)
        WorkEvent& work_event = work_events_[0];
        setupWorkToTimeout(work_event);
        // Expect commAbort to be called due to timeout with abort option
        // enabled
        EXPECT_CALL(
            *nccl_mock_, commAbort(reinterpret_cast<ncclComm_t>(0x3000)))
            .WillOnce(Return(ncclSuccess));

        auto work1 = comm->send(tensor, 1, true);

        // Wait for timeout to occur - this should trigger process abort
        comm->waitTillTimeout();

        // Try to perform another operation - this should trigger the abort
        auto work2 = comm->send(tensor, 1, true);
      },
      ".*"); // Match any death message
}

TEST_F(
    TorchCommNCCLXTest,
    CachingAllocatorHookRegistersAndUnregistersOnCreateAndDestroy) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 1 (destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 1, 1);

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  EXPECT_FALSE(mock_hook_->isCommRegistered(comm.get()));
  // fake registration of the comm.
  mock_hook_->registerComm(comm.get());
  // Destroy the comm
  comm.reset();
  EXPECT_FALSE(mock_hook_->isCommRegistered(comm.get()));
}

TEST_F(
    TorchCommNCCLXTest,
    CachingAllocatorHookRegistersAndUnregistersOnCreateAndFinalize) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  EXPECT_FALSE(mock_hook_->isCommRegistered(comm.get()));
  comm->init(*device_, "test_name", default_options_);
  // Check that the comm is registered
  EXPECT_TRUE(mock_hook_->isCommRegistered(comm.get()));
  // Finalize the comm
  comm->finalize();
  EXPECT_FALSE(mock_hook_->isCommRegistered(comm.get()));
}

TEST_F(
    TorchCommNCCLXTest,
    CachingAllocatorHookUnregistersOnTimeoutDuringFinalize) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();

  comm->init(*device_, "test_name", default_options_);
  EXPECT_TRUE(mock_hook_->isCommRegistered(comm.get()));

  setupEventsForWork(*comm, 1);
  comm->barrier(true);

  auto& work_event = work_events_[0];
  setupWorkToTimeout(work_event);
  comm->waitTillTimeout();

  // Finalize should cause a timeout
  EXPECT_THROW(comm->finalize(), std::runtime_error);
  EXPECT_FALSE(mock_hook_->isCommRegistered(comm.get()));
}

TEST_F(
    TorchCommNCCLXTest,
    CachingAllocatorHookUnregistersOnErrorDuringFinalize) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();

  comm->init(*device_, "test_name", default_options_);

  setupEventsForWork(*comm, 1);
  comm->barrier(true);

  auto& work_event = work_events_[0];
  setupWorkToError(work_event);
  comm->waitTillError();

  // Finalize should cause an error
  EXPECT_THROW(comm->finalize(), NCCLXException);
  EXPECT_FALSE(mock_hook_->isCommRegistered(comm.get()));
}

TEST_F(
    TorchCommNCCLXTest,
    CachingAllocatorHookUnregistersOnErrorOrTimeoutDuringCollective) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 3 (abort, finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 3, 1);

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();

  comm->init(*device_, "test_name", default_options_);

  setupEventsForWork(*comm, 1);

  comm->barrier(true);

  auto& work_event = work_events_[0];
  setupWorkToTimeout(work_event);

  comm->waitTillTimeout();
  EXPECT_THROW(comm->barrier(true), std::runtime_error);
  EXPECT_FALSE(mock_hook_->isCommRegistered(comm.get()));
  EXPECT_THROW(comm->finalize(), std::runtime_error);
}

TEST_F(
    TorchCommNCCLXTest,
    CachingAllocatorHookMemoryRegistrationWithMultipleComms) {
  CachingAllocatorHook::setInstance(
      std::make_unique<CachingAllocatorHookImpl>());
  auto& allocator = CachingAllocatorHook::getInstance();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // Create and initialize two communicators
  auto comm1 = createMockedTorchComm();
  auto comm2 = createMockedTorchComm();

  comm1->init(*device_, "test_name", default_options_);
  comm2->init(*device_, "test_name", default_options_);

  // Create memory registration trace entry
  auto alloc_entry = createAllocation(0x1000);

  // Set up expectations for register_address on both comms
  EXPECT_CALL(
      *nccl_mock_, commRegister(_, reinterpret_cast<void*>(0x1000), 1024, _))
      .Times(2)
      .WillRepeatedly(DoAll(
          SetArgPointee<3>(reinterpret_cast<void*>(0x2000)),
          Return(ncclSuccess)));

  // Simulate memory allocation
  allocator.regDeregMem(alloc_entry);

  // Create a third communicator after memory registration
  auto comm3 = createMockedTorchComm();

  // Set up expectations for register_address on the new comm for previously
  // registered memory
  EXPECT_CALL(
      *nccl_mock_, commRegister(_, reinterpret_cast<void*>(0x1000), 1024, _))
      .WillOnce(DoAll(
          SetArgPointee<3>(reinterpret_cast<void*>(0x2000)),
          Return(ncclSuccess)));

  // Initialize the third comm - this should register all previously registered
  // addresses
  comm3->init(*device_, "test_name", default_options_);
  EXPECT_TRUE(allocator.isCommRegistered(comm3.get()));

  // Create memory deregistration trace entry
  c10::cuda::CUDACachingAllocator::TraceEntry dealloc_entry =
      createDeallocation(0x1000);
  EXPECT_CALL(*nccl_mock_, commDeregister(_, reinterpret_cast<void*>(0x2000)))
      .Times(1)
      .WillRepeatedly(Return(ncclSuccess));

  comm1->finalize();

  // Set up expectations for deregister_address on all three comms
  EXPECT_CALL(*nccl_mock_, commDeregister(_, reinterpret_cast<void*>(0x2000)))
      .Times(2)
      .WillRepeatedly(Return(ncclSuccess));

  // Simulate memory deallocation
  allocator.regDeregMem(dealloc_entry);

  // Clean up
  setupNormalDestruction(*comm2);
  comm2->finalize();
  setupNormalDestruction(*comm3);
  comm3->finalize();
}

TEST_F(
    TorchCommNCCLXTest,
    CachingAllocatorHookMemoryRegistrationErrorHandling) {
  // Test: Verify error handling during memory registration and deregistration
  CachingAllocatorHook::setInstance(
      std::make_unique<CachingAllocatorHookImpl>());
  auto& allocator = CachingAllocatorHook::getInstance();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // Create and initialize a communicator
  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_name", default_options_);

  // Create memory registration trace entry
  c10::cuda::CUDACachingAllocator::TraceEntry alloc_entry =
      createAllocation(0x1000);
  // Set up expectations for register_address to fail
  EXPECT_CALL(
      *nccl_mock_, commRegister(_, reinterpret_cast<void*>(0x1000), 1024, _))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInvalidArgument))
      .WillOnce(Return("Invalid argument"));

  // Simulate memory allocation - should throw due to registration failure
  EXPECT_THROW(allocator.regDeregMem(alloc_entry), std::runtime_error);

  // Try again with successful registration
  EXPECT_CALL(
      *nccl_mock_, commRegister(_, reinterpret_cast<void*>(0x2000), 1024, _))
      .WillOnce(DoAll(
          SetArgPointee<3>(reinterpret_cast<void*>(0x2000)),
          Return(ncclSuccess)));

  // Simulate successful memory allocation
  auto alloc_entry2 = createAllocation(0x2000);
  allocator.regDeregMem(alloc_entry2);

  // Create memory deregistration trace entry
  auto dealloc_entry = createDeallocation(0x2000);

  // Set up expectations for deregister_address to fail
  EXPECT_CALL(*nccl_mock_, commDeregister(_, reinterpret_cast<void*>(0x2000)))
      .WillOnce(Return(ncclInvalidArgument));

  // Simulate memory deallocation - should throw due to deregistration failure
  EXPECT_THROW(allocator.regDeregMem(dealloc_entry), std::runtime_error);

  // Clean up
  setupNormalDestruction(*comm);
  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, Getters) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = CommOptions();
  options.timeout = std::chrono::milliseconds(2000);
  options.abort_process_on_timeout_or_error = false;
  options.store = store_;
  comm->init(*device_, "test_name", options);

  EXPECT_EQ(comm->getOptions(), options);
  EXPECT_EQ(comm->getDevice(), device_);

  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, HighPriorityStreamCreation) {
  // Default priority for stream creation.
  {
    setupRankAndSize(0, 2); // rank 0, size 2
    // we don't call teardown in this test, so no clear
    setupCCAExpectations(1, 2, 0);

    auto comm = createMockedTorchComm();

    int priority_arg_call;
    EXPECT_CALL(*cuda_mock_, getStreamPriorityRange(_, _)).Times(0);
    EXPECT_CALL(*cuda_mock_, streamCreateWithPriority(_, _, _))
        .WillOnce(DoAll(
            SetArgPointee<0>(reinterpret_cast<cudaStream_t>(0x1)),
            SaveArg<2>(&priority_arg_call),
            Return(cudaSuccess)));
    nccl_mock_->setupDefaultBehaviors();
    EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

    // Default priority is zero
    EXPECT_EQ(priority_arg_call, 0);

    comm->finalize();
  }

  // Reset mocks for next subtest below.
  //
  // This needs to happen after the scope above is done, so objects have
  // been destroyed.
  ::testing::Mock::VerifyAndClear(cuda_mock_.get());
  ::testing::Mock::VerifyAndClear(nccl_mock_.get());
  ::testing::Mock::VerifyAndClear(mock_hook_);

  // High priority for stream creation.
  {
    setupRankAndSize(0, 2); // rank 0, size 2
    setupCCAExpectations(1, 2, 1);

    auto options = CommOptions();
    options.hints["torchcomm::ncclx::high_priority_stream"] = "true";
    options.store = store_;
    auto comm = createMockedTorchComm();

    int priority_arg_call;
    EXPECT_CALL(*cuda_mock_, getStreamPriorityRange(_, _))
        .WillOnce(DoAll(
            SetArgPointee<0>(10), SetArgPointee<1>(-10), Return(cudaSuccess)));
    EXPECT_CALL(*cuda_mock_, streamCreateWithPriority(_, _, _))
        .WillOnce(DoAll(
            SetArgPointee<0>(reinterpret_cast<cudaStream_t>(0x1)),
            SaveArg<2>(&priority_arg_call),
            Return(cudaSuccess)));

    nccl_mock_->setupDefaultBehaviors();
    EXPECT_NO_THROW(comm->init(*device_, "test_name", options));

    // Highest priority must match the second value returned by the CUDA API
    // call getStreamPriorityRange.
    EXPECT_EQ(priority_arg_call, -10);

    comm->finalize();
  }
}

// ============================================================================
// INITIALIZATION STATE TESTS
// ============================================================================
TEST_F(TorchCommNCCLXTest, InitialStateIsUninitialized) {
  // Setup CCA expectations - no init/finalize calls
  setupCCAExpectations(0, 1, 1);

  auto comm = createMockedTorchComm();

  // Access the initialization state through a test-specific method
  // Since init_state_ is private, we'll test the behavior indirectly

  // Attempting to finalize without initialization should throw
  EXPECT_THROW(
      {
        try {
          comm->finalize();
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(error_msg.find("not initialized") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommNCCLXTest, InitializationStateTransitionsCorrectly) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // After init, should be in INITIALIZED state
  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // Should be able to finalize after initialization
  setupNormalDestruction(*comm);
  EXPECT_NO_THROW(comm->finalize());
}

TEST_F(TorchCommNCCLXTest, DoubleInitializationThrowsException) {
  // Setup CCA expectations
  // Register - 1 (first init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // First initialization should succeed
  EXPECT_NO_THROW(comm->init(*device_, "test_name", default_options_));

  // Second initialization should throw
  EXPECT_THROW(
      {
        try {
          comm->init(*device_, "test_name", default_options_);
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(
              error_msg.find("already initialized") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);

  setupNormalDestruction(*comm);
  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, DoubleFinalizeThrowsException) {
  // Setup CCA expectations
  // Register - 1 (init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // Initialize first
  comm->init(*device_, "test_name", default_options_);

  // First finalize should succeed
  setupNormalDestruction(*comm);
  EXPECT_NO_THROW(comm->finalize());

  // Second finalize should throw
  EXPECT_THROW(
      {
        try {
          comm->finalize();
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(error_msg.find("already finalized") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommNCCLXTest, InitializeAfterFinalizeThrowsException) {
  // Setup CCA expectations
  // Register - 1 (first init)
  // Deregister - 2 (finalize, destructor)
  // Clear - 1 (destructor)
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // Initialize and finalize
  comm->init(*device_, "test_name", default_options_);
  setupNormalDestruction(*comm);
  comm->finalize();

  // Attempting to initialize after finalize should throw
  EXPECT_THROW(
      {
        try {
          comm->init(*device_, "test_name", default_options_);
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(error_msg.find("already finalized") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(TorchCommNCCLXTest, FinalizeWithoutInitializeThrowsException) {
  // Setup CCA expectations - no init/finalize calls
  setupCCAExpectations(0, 1, 1);

  auto comm = createMockedTorchComm();

  // Attempting to finalize without initialization should throw
  EXPECT_THROW(
      {
        try {
          comm->finalize();
        } catch (const std::runtime_error& e) {
          std::string error_msg = e.what();
          EXPECT_TRUE(error_msg.find("not initialized") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(
    TorchCommNCCLXTest,
    CollectiveOperationsWithoutInitializationThrowException) {
  // Setup CCA expectations - no init calls
  setupCCAExpectations(0, 1, 1);

  auto comm = createMockedTorchComm();

  // Initialize and then finalize the communicator
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // Create test tensors for various operations
  auto tensor = createTestTensor({10, 10});
  auto input_tensor = createTestTensor({10, 10});
  auto output_tensor = createTestTensor({20, 10}); // 2x size for 2 ranks
  auto large_input_tensor =
      createTestTensor({20, 10}); // Divisible by comm_size (2)
  auto large_output_tensor = createTestTensor({20, 10});

  std::vector<at::Tensor> tensor_list = {
      createTestTensor({10, 10}), createTestTensor({10, 10})};
  std::vector<at::Tensor> input_list = {
      createTestTensor({10, 10}), createTestTensor({10, 10})};
  std::vector<at::Tensor> output_list = {
      createTestTensor({10, 10}), createTestTensor({10, 10})};

  // Helper lambda to test that operations throw "not initialized" exception
  auto testOperation = [](const std::function<void()>& operation) {
    EXPECT_THROW(
        {
          try {
            operation();
          } catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_TRUE(error_msg.find("not initialized") != std::string::npos);
            throw;
          }
        },
        std::runtime_error);
  };

  // Test point-to-point operations
  testOperation([&]() { comm->send(tensor, 1, false); });
  testOperation([&]() { comm->recv(tensor, 0, false); });

  // Test collective operations
  testOperation([&]() { comm->broadcast(tensor, 0, false); });
  testOperation([&]() { comm->all_reduce(tensor, ReduceOp::SUM, false); });
  testOperation([&]() { comm->reduce(tensor, 0, ReduceOp::SUM, false); });
  testOperation([&]() { comm->all_gather(tensor_list, tensor, false); });
  testOperation([&]() { comm->all_gather_v(tensor_list, tensor, false); });
  testOperation(
      [&]() { comm->all_gather_single(output_tensor, input_tensor, false); });
  testOperation([&]() {
    comm->reduce_scatter(tensor, input_list, ReduceOp::SUM, false);
  });
  testOperation([&]() {
    comm->reduce_scatter_v(tensor, input_list, ReduceOp::SUM, false);
  });
  testOperation([&]() {
    comm->reduce_scatter_single(
        tensor, large_input_tensor, ReduceOp::SUM, false);
  });
  testOperation([&]() {
    comm->all_to_all_single(large_output_tensor, large_input_tensor, false);
  });
  testOperation([&]() { comm->all_to_all(output_list, input_list, false); });
  testOperation([&]() { comm->barrier(false); });
  testOperation([&]() { comm->scatter(tensor, input_list, 0, false); });
  testOperation([&]() { comm->gather(output_list, input_tensor, 0, false); });

  // Test async versions of some operations
  testOperation([&]() { comm->send(tensor, 1, true); });
  testOperation([&]() { comm->broadcast(tensor, 0, true); });
  testOperation([&]() { comm->barrier(true); });
}

TEST_F(TorchCommNCCLXTest, CollectiveOperationsAfterFinalizeThrowException) {
  // Setup CCA expectations - init and finalize calls
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  // Initialize and then finalize the communicator
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);
  setupNormalDestruction(*comm);
  comm->finalize();

  // Create test tensors for various operations
  auto tensor = createTestTensor({10, 10});
  auto input_tensor = createTestTensor({10, 10});
  auto output_tensor = createTestTensor({20, 10}); // 2x size for 2 ranks
  auto large_input_tensor =
      createTestTensor({20, 10}); // Divisible by comm_size (2)
  auto large_output_tensor = createTestTensor({20, 10});

  std::vector<at::Tensor> tensor_list = {
      createTestTensor({10, 10}), createTestTensor({10, 10})};
  std::vector<at::Tensor> input_list = {
      createTestTensor({10, 10}), createTestTensor({10, 10})};
  std::vector<at::Tensor> output_list = {
      createTestTensor({10, 10}), createTestTensor({10, 10})};

  // Helper lambda to test that operations throw "not initialized" exception
  auto testOperation = [](const std::function<void()>& operation) {
    EXPECT_THROW(
        {
          try {
            operation();
          } catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_TRUE(error_msg.find("not initialized") != std::string::npos);
            throw;
          }
        },
        std::runtime_error);
  };

  // Test point-to-point operations after finalize
  testOperation([&]() { comm->send(tensor, 1, false); });
  testOperation([&]() { comm->recv(tensor, 0, false); });

  // Test collective operations after finalize
  testOperation([&]() { comm->broadcast(tensor, 0, false); });
  testOperation([&]() { comm->all_reduce(tensor, ReduceOp::SUM, false); });
  testOperation([&]() { comm->reduce(tensor, 0, ReduceOp::SUM, false); });
  testOperation([&]() { comm->all_gather(tensor_list, tensor, false); });
  testOperation([&]() { comm->all_gather_v(tensor_list, tensor, false); });
  testOperation(
      [&]() { comm->all_gather_single(output_tensor, input_tensor, false); });
  testOperation([&]() {
    comm->reduce_scatter(tensor, input_list, ReduceOp::SUM, false);
  });
  testOperation([&]() {
    comm->reduce_scatter_single(
        tensor, large_input_tensor, ReduceOp::SUM, false);
  });
  testOperation([&]() {
    comm->all_to_all_single(large_output_tensor, large_input_tensor, false);
  });
  testOperation([&]() { comm->all_to_all(output_list, input_list, false); });
  testOperation([&]() { comm->barrier(false); });
  testOperation([&]() { comm->scatter(tensor, input_list, 0, false); });
  testOperation([&]() { comm->gather(output_list, input_tensor, 0, false); });

  // Test async versions of some operations after finalize
  testOperation([&]() { comm->send(tensor, 1, true); });
  testOperation([&]() { comm->broadcast(tensor, 0, true); });
  testOperation([&]() { comm->barrier(true); });
}

// Test class with pre-hook memory allocation
class TorchCommNCCLXPreHookTest : public TorchCommNCCLXTest {
 protected:
  void SetUp() override {
    // Set up device. make it the cpu device because we're mocking cuda.
    device_ = at::Device(at::DeviceType::CPU, 0);

    // Create tensors before setting up the CachingAllocatorHookMock
    // This simulates memory being allocated before the hook is attached
    createTestTensor({10, 10});

    // Now call the parent SetUp which will create the hook mock
    TorchCommNCCLXTest::SetUp();
  }
};

TEST_F(TorchCommNCCLXPreHookTest, MemAllocatedBeforeCommRegistered) {
  // Setup CCA expectations - init and finalize calls
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  // Initialize and then finalize the communicator
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);
  EXPECT_TRUE(mock_hook_->isMemRegisteredCalled());
  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, AlltoallvDynamicDispatchCombine) {
  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  // Create test data
  auto input_tensor = createTestTensor({100});
  std::vector<at::Tensor> output_tensor_list = {
      createTestTensor({50}), createTestTensor({50})};
  auto input_chunk_sizes =
      at::ones({2}, at::TensorOptions().device(*device_).dtype(at::kLong));
  auto input_chunk_indices =
      at::ones({2}, at::TensorOptions().device(*device_).dtype(at::kLong));
  auto input_chunk_count_per_rank =
      at::ones({2}, at::TensorOptions().device(*device_).dtype(at::kLong));
  auto output_chunk_sizes_per_rank =
      at::ones({4}, at::TensorOptions().device(*device_).dtype(at::kLong));
  auto output_tensor = createTestTensor({100});

  // Test alltoallv_dynamic_dispatch
  EXPECT_CALL(
      *nccl_mock_, alltoallvDynamicDispatch(_, _, _, _, _, _, _, _, _, _, _, _))
      .WillOnce(Return(ncclSuccess));

  auto work1 = comm->alltoallv_dynamic_dispatch(
      output_tensor_list,
      output_chunk_sizes_per_rank,
      input_tensor,
      input_chunk_sizes,
      input_chunk_indices,
      input_chunk_count_per_rank,
      false);

  EXPECT_NE(work1, nullptr);

  // Test alltoallv_dynamic_combine
  EXPECT_CALL(
      *nccl_mock_, alltoallvDynamicCombine(_, _, _, _, _, _, _, _, _, _, _))
      .WillOnce(Return(ncclSuccess));

  auto work2 = comm->alltoallv_dynamic_combine(
      output_tensor,
      input_tensor,
      input_chunk_sizes,
      input_chunk_indices,
      input_chunk_count_per_rank,
      false);

  EXPECT_NE(work2, nullptr);

  setupNormalDestruction(*comm);
  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, AlltoallvDedupExecCombine) {
  setupRankAndSize(0, 4);
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  const int nnodes = 2;
  const int nranks = 4;
  const int nlocalranks = 2;
  const int num_send_blocks = 16;
  const int block_count = 64;
  const int block_num_recv_buckets = 2;
  const int num_recv_buckets = 2;
  const at::ScalarType dtype = at::ScalarType::Float;
  const bool async_op = true;

  // Create test tensors
  auto input_tensor = createTestTensor({100}, dtype);
  auto output_tensor = createTestTensor({100}, dtype);
  auto recv_block_ids = createTestTensor({100}, at::kInt);
  auto send_indices = createTestTensor({nnodes * num_send_blocks}, at::kInt);
  auto forward_indices =
      createTestTensor({nnodes * nlocalranks * num_send_blocks}, at::kInt);
  auto recv_indices =
      createTestTensor({num_recv_buckets * nranks * num_send_blocks}, at::kInt);

  EXPECT_CALL(*nccl_mock_, alltoallvDedupInit(_, _, _, _, _, _, _, _))
      .WillOnce(Return(ncclSuccess));

  EXPECT_CALL(*nccl_mock_, pFree(_)).WillOnce(Return(ncclSuccess));

  EXPECT_CALL(*nccl_mock_, alltoallvDedupExec(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclSuccess));

  auto pReq = comm->alltoallv_dedup_init(
      num_send_blocks,
      block_count,
      block_num_recv_buckets,
      num_recv_buckets,
      dtype,
      async_op);
  EXPECT_NE(pReq, nullptr);

  auto work1 = comm->alltoallv_dedup_exec(
      output_tensor,
      recv_block_ids,
      input_tensor,
      send_indices,
      forward_indices,
      recv_indices,
      pReq);
  EXPECT_NE(work1, nullptr);

  auto work2 = comm->alltoallv_dedup_combine(
      output_tensor,
      input_tensor,
      send_indices,
      forward_indices,
      recv_indices,
      pReq);
  EXPECT_NE(work2, nullptr);

  setupNormalDestruction(*comm);
  comm->finalize();
}

// ============================================================================
// NCCLXException TESTS
// ============================================================================

TEST_F(TorchCommNCCLXTest, NCCLXExceptionIncludesLastErrorString) {
  // Test that NCCLXException message includes the NCCL last error string
  // from getLastError() API

  nccl_mock_->setupDefaultBehaviors();

  // Set up specific return values for error strings
  const std::string error_string = "internal error";
  const std::string last_error_string = "Detailed NCCL error: rank 0 timed out";

  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInternalError))
      .WillOnce(Return(error_string.c_str()));
  EXPECT_CALL(*nccl_mock_, getLastError(_)).WillOnce(Return(last_error_string));

  // Create the exception
  ncclComm_t mock_comm = reinterpret_cast<ncclComm_t>(0x3000);
  NCCLXException exception(
      *nccl_mock_, "Test operation failed", ncclInternalError, mock_comm);

  // Verify the exception message contains both the error string and last error
  std::string what_message = exception.what();
  EXPECT_TRUE(what_message.find("Test operation failed") != std::string::npos)
      << "Exception message should contain the operation message";
  EXPECT_TRUE(what_message.find(error_string) != std::string::npos)
      << "Exception message should contain the NCCL error string";
  EXPECT_TRUE(what_message.find(last_error_string) != std::string::npos)
      << "Exception message should contain the NCCL last error string";
  EXPECT_TRUE(what_message.find("NCCL Last Error:") != std::string::npos)
      << "Exception message should contain 'NCCL Last Error:' label";

  // Verify the result code is preserved
  EXPECT_EQ(exception.getResult(), ncclInternalError);
}

TEST_F(TorchCommNCCLXTest, NCCLXExceptionFromFailedSendIncludesLastError) {
  // Test that when send() fails, the thrown NCCLXException includes
  // the NCCL last error string
  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  auto tensor = createTestTensor({10, 10});

  // Set up send to fail with ncclInternalError
  const std::string last_error_detail =
      "Connection to peer 1 failed: timeout after 30s";
  EXPECT_CALL(*nccl_mock_, send(_, _, _, _, _, _))
      .WillOnce(Return(ncclInternalError));
  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInternalError))
      .WillRepeatedly(Return("internal error"));
  EXPECT_CALL(*nccl_mock_, getLastError(_)).WillOnce(Return(last_error_detail));

  // Attempt send and verify the exception message
  EXPECT_THROW(
      {
        try {
          comm->send(tensor, 1, false);
        } catch (const NCCLXException& e) {
          std::string what_message = e.what();
          EXPECT_TRUE(
              what_message.find("NCCLX Send failed") != std::string::npos)
              << "Exception should mention the failed operation";
          EXPECT_TRUE(what_message.find(last_error_detail) != std::string::npos)
              << "Exception should include the NCCL last error detail: "
              << what_message;
          EXPECT_EQ(e.getResult(), ncclInternalError);
          throw;
        }
      },
      NCCLXException);

  comm->finalize();
}

TEST_F(TorchCommNCCLXTest, NCCLXExceptionFromFailedAllReduceIncludesLastError) {
  // Test that when all_reduce() fails, the thrown NCCLXException includes
  // the NCCL last error string
  setupRankAndSize(0, 2);
  setupCCAExpectations(1, 2, 1);

  auto comm = createMockedTorchComm();

  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm->init(*device_, "test_name", default_options_);

  auto tensor = createTestTensor({10, 10});

  // Set up allReduce to fail with ncclSystemError
  const std::string last_error_detail = "CUDA error: out of memory on device 0";
  EXPECT_CALL(*nccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclSystemError));
  EXPECT_CALL(*nccl_mock_, getErrorString(ncclSystemError))
      .WillRepeatedly(Return("system error"));
  EXPECT_CALL(*nccl_mock_, getLastError(_)).WillOnce(Return(last_error_detail));

  // Attempt all_reduce and verify the exception message
  EXPECT_THROW(
      {
        try {
          comm->all_reduce(tensor, ReduceOp::SUM, false);
        } catch (const NCCLXException& e) {
          std::string what_message = e.what();
          EXPECT_TRUE(
              what_message.find("NCCLX AllReduce failed") != std::string::npos)
              << "Exception should mention the failed operation";
          EXPECT_TRUE(what_message.find(last_error_detail) != std::string::npos)
              << "Exception should include the NCCL last error detail: "
              << what_message;
          EXPECT_EQ(e.getResult(), ncclSystemError);
          throw;
        }
      },
      NCCLXException);

  comm->finalize();
}

} // namespace test
} // namespace comms
} // namespace torch
