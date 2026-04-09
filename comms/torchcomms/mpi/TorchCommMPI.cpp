// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/mpi/TorchCommMPI.hpp"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <set>
#include <string>

#include <fmt/core.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/utils/Logging.hpp"
#include "comms/torchcomms/utils/StoreManager.hpp"
#include "comms/torchcomms/utils/TracingGuard.hpp"
#include "comms/torchcomms/utils/Utils.hpp"

namespace torch::comms {

namespace {

void ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for MPI operations");
  }
}

} // namespace

// ---- Helpers ----
void TorchCommMPI::mpiCheck(int result, const char* msg) {
  if (result != MPI_SUCCESS) {
    char errStr[MPI_MAX_ERROR_STRING];
    int errLen;
    MPI_Error_string(result, errStr, &errLen);
    throw std::runtime_error(
        fmt::format("MPI error: {} - {}", msg, std::string(errStr, errLen)));
  }
}

MPI_Datatype TorchCommMPI::toMPIDatatype(at::ScalarType type) {
  switch (type) {
    case at::ScalarType::Float:
      return MPI_FLOAT;
    case at::ScalarType::Double:
      return MPI_DOUBLE;
    case at::ScalarType::Int:
      return MPI_INT;
    case at::ScalarType::Long:
      return MPI_LONG_LONG;
    case at::ScalarType::Char:
      return MPI_SIGNED_CHAR;
    case at::ScalarType::Byte:
    case at::ScalarType::Bool:
      return MPI_UNSIGNED_CHAR;
    case at::ScalarType::Short:
      return MPI_SHORT;
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
      // MPI has no native half/bfloat16 — use byte-level ops
      return MPI_BYTE;
    default:
      TORCH_CHECK(false, "Unsupported scalar type for MPI");
  }
}

MPI_Op TorchCommMPI::toMPIReduceOp(const ReduceOp& op) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:
    case ReduceOp::RedOpType::AVG:
    case ReduceOp::RedOpType::PREMUL_SUM:
      return MPI_SUM;
    case ReduceOp::RedOpType::PRODUCT:
      return MPI_PROD;
    case ReduceOp::RedOpType::MIN:
      return MPI_MIN;
    case ReduceOp::RedOpType::MAX:
      return MPI_MAX;
    case ReduceOp::RedOpType::BAND:
      return MPI_BAND;
    case ReduceOp::RedOpType::BOR:
      return MPI_BOR;
    case ReduceOp::RedOpType::BXOR:
      return MPI_BXOR;
    default:
      TORCH_CHECK(false, "Unsupported reduce op for MPI");
  }
}

// ---- Construction / Destruction ----
TorchCommMPI::TorchCommMPI() : device_(at::kCPU) {}

TorchCommMPI::~TorchCommMPI() {
  if (comm_ != MPI_COMM_NULL) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Comm_free(&comm_);
    }
    comm_ = MPI_COMM_NULL;
  }
}

// ---- Lifecycle ----
void TorchCommMPI::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  TC_LOG(INFO, this) << "Initializing TorchCommMPI for device: " << device;
  device_ = device;
  name_ = name;
  options_ = options;
  options_.store = nullptr;

  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommMPI already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommMPI already finalized");
  }

  // Initialize MPI if not already done
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    mpiCheck(MPI_Init(nullptr, nullptr), "MPI_Init");
  }

  // Duplicate COMM_WORLD for isolation (unless split() pre-set comm_)
  if (comm_ == MPI_COMM_NULL) {
    mpiCheck(MPI_Comm_dup(MPI_COMM_WORLD, &comm_), "MPI_Comm_dup");
  }

  mpiCheck(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
  mpiCheck(MPI_Comm_size(comm_, &comm_size_), "MPI_Comm_size");

  // Store setup (needed for split() and internal plumbing).
  // When launched via mpirun (no torchrun), MASTER_ADDR/MASTER_PORT may
  // not be set.  In that case, use MPI to bootstrap a TCPStore: rank 0
  // creates a server on an OS-assigned port, broadcasts the port via MPI,
  // and all ranks connect.
  bool persistentStore = false;
  if (options.hints.contains("persistent_store")) {
    persistentStore = string_to_bool(options.hints.at("persistent_store"));
  }

  if (options.store) {
    if (persistentStore) {
      store_ = options.store;
    } else {
      // If MASTER_ADDR is set, use the standard path.
      // Otherwise bootstrap via MPI.
      if (std::getenv("MASTER_ADDR")) {
        store_ = dupPrefixStore(name_, options.store, options.timeout);
      } else {
        store_ = options.store;
      }
    }
  } else {
    // Bootstrap TCPStore via MPI when no env vars are available.
    // Rank 0 creates a server, broadcasts the port.
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    // Broadcast hostname from rank 0
    mpiCheck(MPI_Bcast(hostname, 256, MPI_CHAR, 0, comm_), "MPI_Bcast host");

    c10d::TCPStoreOptions storeOpts;
    storeOpts.isServer = (rank_ == 0);
    storeOpts.port = 0; // OS-assigned
    storeOpts.waitWorkers = false;
    storeOpts.useLibUV = true;
    storeOpts.timeout = options.timeout;

    c10::intrusive_ptr<c10d::TCPStore> tcpStore;
    if (rank_ == 0) {
      tcpStore =
          c10::make_intrusive<c10d::TCPStore>(std::string(hostname), storeOpts);
      int port = static_cast<int>(tcpStore->getPort());
      mpiCheck(MPI_Bcast(&port, 1, MPI_INT, 0, comm_), "MPI_Bcast port");
    } else {
      int port;
      mpiCheck(MPI_Bcast(&port, 1, MPI_INT, 0, comm_), "MPI_Bcast port");
      storeOpts.port = port;
      tcpStore =
          c10::make_intrusive<c10d::TCPStore>(std::string(hostname), storeOpts);
    }

    store_ = c10::make_intrusive<c10d::PrefixStore>(name_, tcpStore);
  }

  init_state_ = InitializationState::INITIALIZED;
  TracingGuard tracingGuard(name_, comm_size_, "init", rank_);
  TC_LOG(INFO, this) << "TorchCommMPI initialized for rank: " << rank_;
}

void TorchCommMPI::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommMPI not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommMPI already finalized");
  }
  init_state_ = InitializationState::FINALIZED;

  // Free the communicator; destructor handles it if finalize() isn't called
  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;
  }
}

int TorchCommMPI::getRank() const {
  return rank_;
}

int TorchCommMPI::getSize() const {
  return comm_size_;
}

std::string_view TorchCommMPI::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommMPI::getCommName() const {
  return name_;
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::createWork(
    std::function<void()> fn,
    bool async_op) {
  if (async_op) {
    return c10::make_intrusive<TorchWorkThread>(std::move(fn));
  }
  fn();
  return c10::make_intrusive<TorchWorkCompleted>();
}

void TorchCommMPI::checkInitialized() {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommMPI not initialized");
  }
}

// ---- Point-to-Point Operations ----
c10::intrusive_ptr<TorchWork> TorchCommMPI::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);

  TracingGuard tracingGuard(name_, comm_size_, "send", dst, tensor, tensor);
  auto tensorCPU = tensor.to(at::kCPU).contiguous();

  return createWork(
      [this, tensorCPU, dst]() {
        auto nbytes = tensorCPU.numel() * tensorCPU.element_size();
        mpiCheck(
            MPI_Send(
                tensorCPU.data_ptr(),
                static_cast<int>(nbytes),
                MPI_BYTE,
                dst,
                0,
                comm_),
            "MPI_Send");
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);

  TracingGuard tracingGuard(name_, comm_size_, "recv", src, tensor, tensor);
  auto tensorCPU = tensor.to(at::kCPU).contiguous();

  return createWork(
      [this, tensor, tensorCPU, src]() mutable {
        auto nbytes = tensorCPU.numel() * tensorCPU.element_size();
        mpiCheck(
            MPI_Recv(
                tensorCPU.data_ptr(),
                static_cast<int>(nbytes),
                MPI_BYTE,
                src,
                0,
                comm_,
                MPI_STATUS_IGNORE),
            "MPI_Recv");
        if (tensorCPU.device() != tensor.device()) {
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  checkInitialized();

  return createWork(
      [this, ops]() {
        for (const auto& op : ops) {
          if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
            send(op.tensor, op.peer, false)->wait();
          } else {
            at::Tensor t = op.tensor;
            recv(t, op.peer, false)->wait();
          }
        }
      },
      async_op);
}

// ---- Collective Operations ----
c10::intrusive_ptr<TorchWork> TorchCommMPI::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "broadcast", root, tensor, tensor);

  auto tensorCPU = tensor.to(at::kCPU).contiguous();

  return createWork(
      [this, tensor, tensorCPU, root]() mutable {
        mpiCheck(
            MPI_Bcast(
                tensorCPU.data_ptr(),
                static_cast<int>(tensorCPU.numel()),
                toMPIDatatype(tensorCPU.scalar_type()),
                root,
                comm_),
            "MPI_Bcast");

        if (tensorCPU.device() != tensor.device()) {
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, tensor, tensor);

  auto tensorCPU = tensor.to(at::kCPU).contiguous();
  bool isAvg = op.type() == ReduceOp::RedOpType::AVG;
  bool isPremul = op.type() == ReduceOp::RedOpType::PREMUL_SUM;

  return createWork(
      [this, tensor, tensorCPU, op, isAvg, isPremul]() mutable {
        if (isPremul) {
          auto factor = *op.factor();
          try {
            tensorCPU *= std::get<double>(factor);
          } catch (const std::bad_variant_access&) {
            tensorCPU *= std::get<at::Tensor>(factor).to(at::kCPU);
          }
        }

        mpiCheck(
            MPI_Allreduce(
                MPI_IN_PLACE,
                tensorCPU.data_ptr(),
                static_cast<int>(tensorCPU.numel()),
                toMPIDatatype(tensorCPU.scalar_type()),
                toMPIReduceOp(op),
                comm_),
            "MPI_Allreduce");

        if (isAvg) {
          if (at::isIntegralType(
                  tensorCPU.scalar_type(), /*includeBool=*/false)) {
            tensorCPU.div_(comm_size_, "trunc");
          } else {
            tensorCPU /= comm_size_;
          }
        }

        if (tensorCPU.device() != tensor.device()) {
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);

  TracingGuard tracingGuard(name_, comm_size_, "reduce", root, tensor, tensor);

  auto tensorCPU = tensor.to(at::kCPU).contiguous();

  return createWork(
      [this, tensor, tensorCPU, root, op]() mutable {
        if (rank_ == root) {
          mpiCheck(
              MPI_Reduce(
                  MPI_IN_PLACE,
                  tensorCPU.data_ptr(),
                  static_cast<int>(tensorCPU.numel()),
                  toMPIDatatype(tensorCPU.scalar_type()),
                  toMPIReduceOp(op),
                  root,
                  comm_),
              "MPI_Reduce");
        } else {
          mpiCheck(
              MPI_Reduce(
                  tensorCPU.data_ptr(),
                  nullptr,
                  static_cast<int>(tensorCPU.numel()),
                  toMPIDatatype(tensorCPU.scalar_type()),
                  toMPIReduceOp(op),
                  root,
                  comm_),
              "MPI_Reduce");
        }

        if (tensorCPU.device() != tensor.device()) {
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "all_gather", rank_, tensor_list, {tensor});

  auto inputCPU = tensor.to(at::kCPU).contiguous();

  return createWork(
      [this, tensor_list, inputCPU]() mutable {
        auto totalElements = inputCPU.numel() * comm_size_;
        auto outputCPU = at::empty({totalElements}, inputCPU.options());

        mpiCheck(
            MPI_Allgather(
                inputCPU.data_ptr(),
                static_cast<int>(inputCPU.numel()),
                toMPIDatatype(inputCPU.scalar_type()),
                outputCPU.data_ptr(),
                static_cast<int>(inputCPU.numel()),
                toMPIDatatype(inputCPU.scalar_type()),
                comm_),
            "MPI_Allgather");

        auto chunkSize = inputCPU.numel();
        for (int i = 0; i < comm_size_; ++i) {
          auto chunk = outputCPU.narrow(0, i * chunkSize, chunkSize);
          tensor_list[i].copy_(chunk);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "all_gather_v", rank_, tensor_list, {tensor});

  // Implement via broadcast rounds (same as UCC/Gloo pattern)
  auto inputCPU = tensor.to(at::kCPU).contiguous();
  std::vector<at::Tensor> tensorListCPU;
  tensorListCPU.reserve(tensor_list.size());
  for (const auto& t : tensor_list) {
    tensorListCPU.push_back(t.to(at::kCPU).contiguous());
  }

  return createWork(
      [this, tensor_list, inputCPU, tensorListCPU]() mutable {
        for (int i = 0; i < comm_size_; ++i) {
          at::Tensor buf;
          if (i == rank_) {
            buf = inputCPU.clone();
          } else {
            buf = tensorListCPU[i];
          }

          mpiCheck(
              MPI_Bcast(
                  buf.data_ptr(),
                  static_cast<int>(buf.numel()),
                  toMPIDatatype(buf.scalar_type()),
                  i,
                  comm_),
              "MPI_Bcast (all_gather_v)");

          if (i == rank_) {
            tensorListCPU[i].copy_(buf);
          }
        }

        for (size_t i = 0; i < tensor_list.size(); ++i) {
          if (tensorListCPU[i].device() != tensor_list[i].device()) {
            tensor_list[i].copy_(tensorListCPU[i]);
          }
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_gather_single", rank_, input, output);

  auto inputCPU = input.to(at::kCPU).contiguous();
  auto outputCPU = output.to(at::kCPU).contiguous();

  return createWork(
      [this, output, inputCPU, outputCPU]() mutable {
        mpiCheck(
            MPI_Allgather(
                inputCPU.data_ptr(),
                static_cast<int>(inputCPU.numel()),
                toMPIDatatype(inputCPU.scalar_type()),
                outputCPU.data_ptr(),
                static_cast<int>(inputCPU.numel()),
                toMPIDatatype(outputCPU.scalar_type()),
                comm_),
            "MPI_Allgather (single)");

        if (outputCPU.device() != output.device()) {
          output.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output);
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter", rank_, input_list, {output});

  auto input = at::cat(input_list, 0);
  ReduceScatterSingleOptions singleOptions;
  singleOptions.timeout = options.timeout;
  singleOptions.hints = options.hints;
  return reduce_scatter_single(output, input, op, async_op, singleOptions);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output);
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter_v", rank_, input_list, {output});

  // Implement via allreduce + extract (same pattern as UCC)
  std::vector<at::Tensor> inputListCPU;
  inputListCPU.reserve(input_list.size());
  for (const auto& t : input_list) {
    inputListCPU.push_back(t.to(at::kCPU).contiguous());
  }
  auto outputCPU = output.to(at::kCPU).contiguous();

  return createWork(
      [this, output, input_list, inputListCPU, outputCPU, op]() mutable {
        for (int i = 0; i < comm_size_; ++i) {
          auto& inputTensor = inputListCPU[i];

          mpiCheck(
              MPI_Allreduce(
                  MPI_IN_PLACE,
                  inputTensor.data_ptr(),
                  static_cast<int>(inputTensor.numel()),
                  toMPIDatatype(inputTensor.scalar_type()),
                  toMPIReduceOp(op),
                  comm_),
              "MPI_Allreduce (reduce_scatter_v)");

          if (i == rank_) {
            outputCPU.copy_(inputTensor);
          }
        }

        if (outputCPU.device() != output.device()) {
          output.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  TracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter_single", rank_, input, output);

  auto inputCPU = input.to(at::kCPU).contiguous();

  return createWork(
      [this, output, inputCPU, op]() mutable {
        // Use MPI_Reduce_scatter_block for equal-sized chunks
        auto chunkSize = output.numel();
        auto outputCPU = at::empty({chunkSize}, inputCPU.options());

        mpiCheck(
            MPI_Reduce_scatter_block(
                inputCPU.data_ptr(),
                outputCPU.data_ptr(),
                static_cast<int>(chunkSize),
                toMPIDatatype(inputCPU.scalar_type()),
                toMPIReduceOp(op),
                comm_),
            "MPI_Reduce_scatter_block");

        if (op.type() == ReduceOp::RedOpType::AVG) {
          if (at::isIntegralType(
                  outputCPU.scalar_type(), /*includeBool=*/false)) {
            outputCPU.div_(comm_size_, "trunc");
          } else {
            outputCPU /= comm_size_;
          }
        }

        output.copy_(outputCPU);
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_single", rank_, input, output);

  auto inputCPU = input.to(at::kCPU).contiguous();
  auto outputCPU = output.to(at::kCPU).contiguous();

  return createWork(
      [this, output, inputCPU, outputCPU]() mutable {
        auto chunkSize = inputCPU.numel() / comm_size_;
        mpiCheck(
            MPI_Alltoall(
                inputCPU.data_ptr(),
                static_cast<int>(chunkSize),
                toMPIDatatype(inputCPU.scalar_type()),
                outputCPU.data_ptr(),
                static_cast<int>(chunkSize),
                toMPIDatatype(outputCPU.scalar_type()),
                comm_),
            "MPI_Alltoall");

        if (outputCPU.device() != output.device()) {
          output.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_v_single", rank_, input, output);

  auto inputCPU = input.to(at::kCPU).contiguous();
  auto outputCPU = output.to(at::kCPU).contiguous();

  return createWork(
      [this,
       output,
       inputCPU,
       outputCPU,
       input_split_sizes,
       output_split_sizes]() mutable {
        auto inputDim0Numel = inputCPU.numel() /
            std::max(inputCPU.size(0), static_cast<int64_t>(1));
        auto outputDim0Numel = outputCPU.numel() /
            std::max(outputCPU.size(0), static_cast<int64_t>(1));

        std::vector<int> sendcounts(comm_size_);
        std::vector<int> sdispls(comm_size_);
        std::vector<int> recvcounts(comm_size_);
        std::vector<int> rdispls(comm_size_);

        int send_offset = 0;
        int recv_offset = 0;
        for (int i = 0; i < comm_size_; ++i) {
          sendcounts[i] =
              static_cast<int>(input_split_sizes[i] * inputDim0Numel);
          sdispls[i] = send_offset;
          send_offset += sendcounts[i];

          recvcounts[i] =
              static_cast<int>(output_split_sizes[i] * outputDim0Numel);
          rdispls[i] = recv_offset;
          recv_offset += recvcounts[i];
        }

        mpiCheck(
            MPI_Alltoallv(
                inputCPU.data_ptr(),
                sendcounts.data(),
                sdispls.data(),
                toMPIDatatype(inputCPU.scalar_type()),
                outputCPU.data_ptr(),
                recvcounts.data(),
                rdispls.data(),
                toMPIDatatype(outputCPU.scalar_type()),
                comm_),
            "MPI_Alltoallv");

        if (outputCPU.device() != output.device()) {
          output.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  checkInitialized();
  for (const auto& t : input_tensor_list) {
    ensureTensorContiguous(t);
  }
  for (const auto& t : output_tensor_list) {
    ensureTensorContiguous(t);
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_to_all",
      rank_,
      input_tensor_list,
      output_tensor_list);

  auto tensorSize = input_tensor_list.at(0).numel();
  auto totalElements = tensorSize * comm_size_;
  auto inputConcatCPU = at::empty(
      {totalElements}, input_tensor_list.at(0).options().device(at::kCPU));
  auto outputConcatCPU = at::empty(
      {totalElements}, output_tensor_list.at(0).options().device(at::kCPU));

  for (int i = 0; i < comm_size_; ++i) {
    inputConcatCPU.narrow(0, i * tensorSize, tensorSize)
        .copy_(input_tensor_list[i]);
  }

  return createWork(
      [this,
       output_tensor_list,
       inputConcatCPU,
       outputConcatCPU,
       tensorSize]() mutable {
        mpiCheck(
            MPI_Alltoall(
                inputConcatCPU.data_ptr(),
                static_cast<int>(tensorSize),
                toMPIDatatype(inputConcatCPU.scalar_type()),
                outputConcatCPU.data_ptr(),
                static_cast<int>(tensorSize),
                toMPIDatatype(outputConcatCPU.scalar_type()),
                comm_),
            "MPI_Alltoall (list)");

        for (int i = 0; i < comm_size_; ++i) {
          output_tensor_list[i].copy_(
              outputConcatCPU.narrow(0, i * tensorSize, tensorSize));
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();

  TracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);

  return createWork(
      [this]() { mpiCheck(MPI_Barrier(comm_), "MPI_Barrier"); }, async_op);
}

// ---- Scatter and Gather ----
c10::intrusive_ptr<TorchWork> TorchCommMPI::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output_tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "scatter", root, input_tensor_list, {output_tensor});

  auto outputCPU = output_tensor.to(at::kCPU).contiguous();

  // Prepare contiguous send buffer on root
  at::Tensor sendBufCPU;
  if (rank_ == root) {
    auto totalElements = outputCPU.numel() * comm_size_;
    sendBufCPU =
        at::empty({totalElements}, outputCPU.options().device(at::kCPU));
    for (int i = 0; i < comm_size_; ++i) {
      sendBufCPU.narrow(0, i * outputCPU.numel(), outputCPU.numel())
          .copy_(input_tensor_list[i]);
    }
  }

  return createWork(
      [this, output_tensor, outputCPU, sendBufCPU, root]() mutable {
        mpiCheck(
            MPI_Scatter(
                rank_ == root ? sendBufCPU.data_ptr() : nullptr,
                static_cast<int>(outputCPU.numel()),
                toMPIDatatype(outputCPU.scalar_type()),
                outputCPU.data_ptr(),
                static_cast<int>(outputCPU.numel()),
                toMPIDatatype(outputCPU.scalar_type()),
                root,
                comm_),
            "MPI_Scatter");

        if (outputCPU.device() != output_tensor.device()) {
          output_tensor.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommMPI::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  checkInitialized();
  ensureTensorContiguous(input_tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "gather", root, {input_tensor}, output_tensor_list);

  auto inputCPU = input_tensor.to(at::kCPU).contiguous();

  return createWork(
      [this, output_tensor_list, inputCPU, root]() mutable {
        at::Tensor recvBufCPU;
        if (rank_ == root) {
          auto totalElements = inputCPU.numel() * comm_size_;
          recvBufCPU = at::empty({totalElements}, inputCPU.options());
        }

        mpiCheck(
            MPI_Gather(
                inputCPU.data_ptr(),
                static_cast<int>(inputCPU.numel()),
                toMPIDatatype(inputCPU.scalar_type()),
                rank_ == root ? recvBufCPU.data_ptr() : nullptr,
                static_cast<int>(inputCPU.numel()),
                toMPIDatatype(inputCPU.scalar_type()),
                root,
                comm_),
            "MPI_Gather");

        if (rank_ == root) {
          auto chunkSize = inputCPU.numel();
          for (int i = 0; i < comm_size_; ++i) {
            output_tensor_list[i].copy_(
                recvBufCPU.narrow(0, i * chunkSize, chunkSize));
          }
        }
      },
      async_op);
}

// ---- Communicator Management ----
std::shared_ptr<TorchCommBackend> TorchCommMPI::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  for (int rank : ranks) {
    if (rank < 0 || rank >= comm_size_) {
      throw std::runtime_error(fmt::format(
          "Invalid rank {} in ranks. Valid ranks are 0 to {}",
          rank,
          comm_size_ - 1));
    }
  }

  std::set<int> unique_ranks(ranks.begin(), ranks.end());
  if (unique_ranks.size() != ranks.size()) {
    throw std::runtime_error("Duplicate ranks found in ranks list");
  }

  if (ranks.empty()) {
    return nullptr;
  }

  auto it = std::find(ranks.begin(), ranks.end(), rank_);
  if (it == ranks.end()) {
    throw std::runtime_error(fmt::format(
        "Current rank {} is not included in the provided ranks list", rank_));
  }

  auto new_torchcomm = std::make_shared<TorchCommMPI>();

  int color = *std::min_element(ranks.begin(), ranks.end());

  // Use MPI_Comm_split to create sub-communicator
  MPI_Comm new_mpi_comm;
  int key = static_cast<int>(std::distance(ranks.begin(), it));
  mpiCheck(
      MPI_Comm_split(comm_, color, key, &new_mpi_comm), "MPI_Comm_split");
  new_torchcomm->comm_ = new_mpi_comm;

  auto new_name = fmt::format("{}_{}", name, color);
  auto split_id = splitCounter_++;
  auto store_prefix = fmt::format("{}/{}_{}", split_id, name, color);
  auto new_store = c10::make_intrusive<c10d::PrefixStore>(store_prefix, store_);

  CommOptions new_options = options;
  new_options.store = new_store;
  new_options.hints["persistent_store"] = "true";

  new_torchcomm->init(device_, new_name, new_options);

  return new_torchcomm;
}

// ---- Static Registration ----
namespace {
class MPIRegistration {
 public:
  MPIRegistration() {
    TorchCommFactory::get().register_backend(
        "mpi", []() { return std::make_shared<TorchCommMPI>(); });
  }
};

static const MPIRegistration registration{};
} // namespace

} // namespace torch::comms
