#include "comms/ctran/tests/CtranCclxIntegrationTestUtils.h"

#define NCCLCHECK_TEST(cmd)                  \
  do {                                       \
    ncclResult_t r = cmd;                    \
    if (r != ncclSuccess) {                  \
      printf(                                \
          "Failed, NCCL error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          ncclGetErrorString(r));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

void CtranCclxIntegrationTestUtils::SetUp() {
  // required by RCCL
  setenv("HSA_NO_SCRATCH_RECLAIM", "1", 0);
  CtranDistTest::SetUp();
  ncclComm = createNcclComm();
  ctranComm = commRAII->ctranComm;
}

void CtranCclxIntegrationTestUtils::TearDown() {
  NCCLCHECK_TEST(ncclCommDestroy(ncclComm));
  CtranDistTest::TearDown();
}

ncclComm_t CtranCclxIntegrationTestUtils::createNcclComm() {
  ncclComm_t ncclComm;
  ncclUniqueId id;
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&id));
  }

  std::vector<uint8_t> idStorage;
  if (getInitEnvType() == InitEnvType::TCP_STORE && tcpStore_ != nullptr) {
    auto storeKey =
        fmt::format("{}_ncclCommId", getTcpStoreKey(TcpStorePhase::INIT));

    // use TCPStore to broadcast NCCL unique ID
    if (globalRank == 0) {
      idStorage.resize(sizeof(id));
      ::memcpy(idStorage.data(), &id, sizeof(id));
      tcpStore_->set(storeKey, idStorage);
    } else {
      // Wait for commid to be set by rank 0
      tcpStore_->wait({storeKey});
      if (tcpStore_->check({storeKey})) {
        idStorage.resize(sizeof(id));
        idStorage = tcpStore_->get(storeKey);
        ::memcpy(&id, idStorage.data(), sizeof(id));
      } else {
        LOG(FATAL) << storeKey << " check failed";
      }
    }
  } else {
    // use MPI to broadcast NCCL unique ID
    MPICHECK_TEST(
        MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  }

  CUDACHECK_TEST(cudaSetDevice(localRank));
  NCCLCHECK_TEST(ncclCommInitRank(&ncclComm, numRanks, id, globalRank));
  return ncclComm;
}

void CtranCclxIntegrationTestUtils::ncclOrCtranRegister(
    void* buff,
    size_t size,
    void** handle) {
  if (testCclxAPI_) {
    NCCLCHECK_TEST(ncclCommRegister(ncclComm, buff, size, handle));
  } else {
    COMMCHECK_TEST(ctranComm->ctran_->commRegister(buff, size, handle));
  }
}

void CtranCclxIntegrationTestUtils::ncclOrCtranDeregister(void* handle) {
  if (testCclxAPI_) {
    NCCLCHECK_TEST(ncclCommDeregister(ncclComm, handle));
  } else {
    COMMCHECK_TEST(ctranComm->ctran_->commDeregister(handle));
  }
}
