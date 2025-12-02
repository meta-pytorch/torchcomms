// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranTestUtils.h"

#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/mccl/utils/Utils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

namespace ctran {

void CtranEnvironmentBase::SetUp() {
  // support MPI for now
  // TODO can also support TCPStore
  MPI_CHECK(MPI_Init(nullptr, nullptr));

  // set up default envs for CTRAN tests

  // default logging level = INFO
  // indiviudal test can override the logging level
  setenv("NCCL_DEBUG", "INFO", 0);

  // Disable FBWHOAMI Topology failure for tests
  setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "0", 1);
  setenv("NCCL_CTRAN_PROFILING", "none", 1);
  setenv("NCCL_CTRAN_ENABLE", "1", 0);
  setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
}

void CtranEnvironmentBase::TearDown() {
  MPI_CHECK(MPI_Finalize());
}

void CtranDistTestFixture::SetUp() {
  // get my rank info via MPI
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  MPI_Comm localComm{MPI_COMM_NULL};
  MPI_CHECK(MPI_Comm_split_type(
      MPI_COMM_WORLD,
      OMPI_COMM_TYPE_HOST,
      globalRank,
      MPI_INFO_NULL,
      &localComm));
  MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
  MPI_CHECK(MPI_Comm_size(localComm, &numLocalRanks_));
  MPI_CHECK(MPI_Comm_free(&localComm));

  // initialize ctran settings
  CUDACHECK_TEST(cudaSetDevice(localRank));
  ncclCvarInit();
  ctran::logging::initCtranLogging(true /*alwaysInit*/);
  COMMCHECK_TEST(ctran::utils::commCudaLibraryInit());
}

void CtranDistTestFixture::TearDown() {}

std::unique_ptr<CtranComm> CtranDistTestFixture::makeCtranComm() {
  const std::string uuid{"0"};
  uint64_t commHash =
      ctran::utils::getHash(uuid.data(), static_cast<int>(uuid.size()));
  std::string commDesc = fmt::format("CtranTestComm-{}", globalRank);

  auto comm =
      std::make_unique<CtranComm>(ctran::utils::createAbort(/*enabled=*/false));
  comm->logMetaData_.commId = 0;
  comm->logMetaData_.commHash = commHash;
  comm->logMetaData_.commDesc = commDesc;
  comm->logMetaData_.rank = globalRank;
  comm->logMetaData_.nRanks = numRanks;

  comm->bootstrap_ = std::make_unique<meta::comms::MpiBootstrap>();

  // Initialize StateX
  int cudaDev;
  CUDACHECK_TEST(cudaGetDevice(&cudaDev));
  const int cudaArch = ctran::utils::getCudaArch(cudaDev).value_or(-1);
  const int64_t busId = ctran::utils::BusId::makeFrom(cudaDev).toInt64();

  std::vector<ncclx::RankTopology> rankTopologies{};
  std::vector<int> commRanksToWorldRanks{};
  comm->statex_ = std::make_unique<ncclx::CommStateX>(
      globalRank,
      numRanks,
      cudaDev,
      cudaArch,
      busId,
      commHash,
      rankTopologies,
      commRanksToWorldRanks,
      commDesc);

  // Initialize StateX with topology using helper function
  mccl::utils::initRankTopology(comm->statex_.get(), comm->bootstrap_.get());

  // TODO: add memCache if enabled

  // Initialize Ctran
  comm->config_.commDesc = comm->statex_->commDesc().c_str();

  COMMCHECK_TEST(ctranInit(comm.get()));
  CHECK(ctranInitialized(comm.get())) << "Ctran not initialized";
  return comm;
}

} // namespace ctran
