// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ucc/TorchCommUCC.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <set>
#include <string>

#include <fmt/core.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/utils/Logging.hpp"
#include "comms/torchcomms/utils/StoreManager.hpp"
#include "comms/torchcomms/utils/TracingGuard.hpp"
#include "comms/torchcomms/utils/Utils.hpp"

namespace {
// Set UCX_MODULE_DIR so UCX can find its transport modules installed
// alongside the _comms_ucc extension.  UCC itself locates its plugins
// via dladdr(libucc.so) + UCC_MODULE_SUBDIR, so no UCC_HOME is needed.
void setUCCModulesPath() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&setUCCModulesPath), &info) ||
      !info.dli_fname) {
    return;
  }
  std::string libPath(info.dli_fname);
  auto lastSlash = libPath.rfind('/');
  if (lastSlash == std::string::npos) {
    return;
  }
  std::string baseDir = libPath.substr(0, lastSlash);
  std::string uccDir = baseDir + "/ucc";

  // UCX needs to know where its transport modules are
  std::string ucxDir = uccDir + "/ucx";
  if (!std::getenv("UCX_MODULE_DIR")) {
    setenv("UCX_MODULE_DIR", ucxDir.c_str(), 0);
  }
}

// Auto-initialize on library load
struct UCCPathInitializer {
  UCCPathInitializer() {
    setUCCModulesPath();
  }
};
static UCCPathInitializer uccPathInit;
} // namespace

namespace torch::comms {

namespace {

void ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for UCC operations");
  }
}

} // namespace

// ---- OOB collinfo passed through UCC's void* coll_info ----
struct TorchCommUCC::OOBCollInfo {
  c10::intrusive_ptr<c10d::Store> store;
  int rank;
  int size;
  std::atomic<uint32_t> seq{0}; // incremented per OOB allgather call
};

struct OOBRequest {
  std::vector<uint8_t> result;
  bool done{true};
};

// ---- OOB callbacks for UCC team creation ----
ucc_status_t TorchCommUCC::oobAllgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req) {
  auto* info = static_cast<OOBCollInfo*>(coll_info);
  uint32_t seq = info->seq.fetch_add(1);
  auto prefix = fmt::format("ucc_oob_{}", seq);

  // Publish this rank's data
  std::vector<uint8_t> data(
      static_cast<uint8_t*>(sbuf),
      static_cast<uint8_t*>(sbuf) + msglen);
  info->store->set(fmt::format("{}_{}", prefix, info->rank), data);

  // Collect from all ranks
  for (int i = 0; i < info->size; ++i) {
    auto key = fmt::format("{}_{}", prefix, i);
    auto val = info->store->get(key);
    std::memcpy(
        static_cast<uint8_t*>(rbuf) + i * msglen, val.data(), msglen);
  }

  auto* oob_req = new OOBRequest();
  oob_req->done = true;
  *req = oob_req;
  return UCC_OK;
}

ucc_status_t TorchCommUCC::oobAllgatherTest(void* req) {
  auto* oob_req = static_cast<OOBRequest*>(req);
  return oob_req->done ? UCC_OK : UCC_INPROGRESS;
}

ucc_status_t TorchCommUCC::oobAllgatherFree(void* req) {
  delete static_cast<OOBRequest*>(req);
  return UCC_OK;
}

// ---- Helpers ----
void TorchCommUCC::uccCheck(ucc_status_t status, const char* msg) {
  if (status != UCC_OK) {
    throw std::runtime_error(
        fmt::format("UCC error: {} - {}", msg, ucc_status_string(status)));
  }
}

ucc_datatype_t TorchCommUCC::toUCCDatatype(at::ScalarType type) {
  switch (type) {
    case at::ScalarType::Float:
      return UCC_DT_FLOAT32;
    case at::ScalarType::Double:
      return UCC_DT_FLOAT64;
    case at::ScalarType::Half:
      return UCC_DT_FLOAT16;
    case at::ScalarType::BFloat16:
      return UCC_DT_BFLOAT16;
    case at::ScalarType::Int:
      return UCC_DT_INT32;
    case at::ScalarType::Long:
      return UCC_DT_INT64;
    case at::ScalarType::Char:
      return UCC_DT_INT8;
    case at::ScalarType::Byte:
    case at::ScalarType::Bool:
      return UCC_DT_UINT8;
    case at::ScalarType::Short:
      return UCC_DT_INT16;
    default:
      TORCH_CHECK(false, "Unsupported scalar type for UCC");
  }
}

ucc_reduction_op_t TorchCommUCC::toUCCReduceOp(const ReduceOp& op) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:
    case ReduceOp::RedOpType::AVG:
    case ReduceOp::RedOpType::PREMUL_SUM:
      return UCC_OP_SUM;
    case ReduceOp::RedOpType::PRODUCT:
      return UCC_OP_PROD;
    case ReduceOp::RedOpType::MIN:
      return UCC_OP_MIN;
    case ReduceOp::RedOpType::MAX:
      return UCC_OP_MAX;
    case ReduceOp::RedOpType::BAND:
      return UCC_OP_BAND;
    case ReduceOp::RedOpType::BOR:
      return UCC_OP_BOR;
    case ReduceOp::RedOpType::BXOR:
      return UCC_OP_BXOR;
    default:
      TORCH_CHECK(false, "Unsupported reduce op for UCC");
  }
}

// ---- Construction / Destruction ----
TorchCommUCC::TorchCommUCC() : device_(at::kCPU) {}

TorchCommUCC::~TorchCommUCC() {
  if (team_) {
    // Wait for any outstanding work before destroying team
    while (ucc_team_destroy(team_) == UCC_INPROGRESS) {
      ucc_context_progress(ctx_);
    }
    team_ = nullptr;
  }
  if (ctx_) {
    ucc_context_destroy(ctx_);
    ctx_ = nullptr;
  }
  if (lib_) {
    ucc_finalize(lib_);
    lib_ = nullptr;
  }
}

// ---- Lifecycle ----
void TorchCommUCC::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  TC_LOG(INFO, this) << "Initializing TorchCommUCC for device: " << device;
  device_ = device;
  name_ = name;
  options_ = options;
  options_.store = nullptr;

  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommUCC already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommUCC already finalized");
  }
  init_state_ = InitializationState::INITIALIZED;

  if (rank_ == -1 || comm_size_ == -1) {
    auto [rank, comm_size] = query_ranksize();
    rank_ = rank;
    comm_size_ = comm_size;
  }

  // Store setup
  bool persistentStore = false;
  if (options.hints.contains("persistent_store")) {
    persistentStore = string_to_bool(options.hints.at("persistent_store"));
  }
  c10::intrusive_ptr<c10d::Store> bootstrapStore;

  if (options.store) {
    if (persistentStore) {
      store_ = options.store;
    } else {
      bootstrapStore = options.store;
      store_ = dupPrefixStore(name_, bootstrapStore, options.timeout);
    }
  } else {
    bootstrapStore = createPrefixStore(name_, options.timeout);
    store_ = dupPrefixStore(name_, bootstrapStore, options.timeout);
  }

  // Initialize UCC library
  ucc_lib_config_h lib_config;
  ucc_lib_params_t lib_params;
  std::memset(&lib_params, 0, sizeof(lib_params));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_SINGLE;

  uccCheck(ucc_lib_config_read(nullptr, nullptr, &lib_config), "lib_config_read");
  uccCheck(ucc_init(&lib_params, lib_config, &lib_), "ucc_init");
  ucc_lib_config_release(lib_config);

  // Create UCC context
  ucc_context_config_h ctx_config;
  uccCheck(
      ucc_context_config_read(lib_, nullptr, &ctx_config),
      "context_config_read");

  // OOB info lives as a member — UCC may call OOB callbacks during
  // context/team cleanup, so it must outlive ctx_ and team_.
  oob_info_ = std::make_unique<OOBCollInfo>();
  oob_info_->store = store_;
  oob_info_->rank = rank_;
  oob_info_->size = comm_size_;

  ucc_context_params_t ctx_params;
  std::memset(&ctx_params, 0, sizeof(ctx_params));
  ctx_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  ctx_params.type = UCC_CONTEXT_SHARED;
  ctx_params.oob.allgather = oobAllgather;
  ctx_params.oob.req_test = oobAllgatherTest;
  ctx_params.oob.req_free = oobAllgatherFree;
  ctx_params.oob.coll_info = oob_info_.get();
  ctx_params.oob.n_oob_eps = comm_size_;
  ctx_params.oob.oob_ep = rank_;

  uccCheck(
      ucc_context_create(lib_, &ctx_params, ctx_config, &ctx_),
      "context_create");
  ucc_context_config_release(ctx_config);

  // Create UCC team (reuse oob_info — seq keeps incrementing)

  ucc_team_params_t team_params;
  std::memset(&team_params, 0, sizeof(team_params));
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP |
      UCC_TEAM_PARAM_FIELD_EP_RANGE | UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oobAllgather;
  team_params.oob.req_test = oobAllgatherTest;
  team_params.oob.req_free = oobAllgatherFree;
  team_params.oob.coll_info = oob_info_.get();
  team_params.oob.n_oob_eps = comm_size_;
  team_params.oob.oob_ep = rank_;
  team_params.ep = static_cast<uint64_t>(rank_);
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;

  uccCheck(ucc_team_create_post(&ctx_, 1, &team_params, &team_), "team_create");

  // Progress team creation to completion
  while (ucc_team_create_test(team_) == UCC_INPROGRESS) {
    ucc_context_progress(ctx_);
  }

  bootstrapStore.reset();

  TracingGuard tracingGuard(name_, comm_size_, "init", rank_);
  TC_LOG(INFO, this) << "TorchCommUCC initialized for rank: " << rank_;
}

void TorchCommUCC::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommUCC not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommUCC already finalized");
  }
  init_state_ = InitializationState::FINALIZED;
}

int TorchCommUCC::getRank() const {
  return rank_;
}

int TorchCommUCC::getSize() const {
  return comm_size_;
}

std::string_view TorchCommUCC::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommUCC::getCommName() const {
  return name_;
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::createWork(
    std::function<void()> fn,
    bool async_op) {
  if (async_op) {
    return c10::make_intrusive<TorchWorkThread>(std::move(fn));
  }
  fn();
  return c10::make_intrusive<TorchWorkCompleted>();
}

void TorchCommUCC::checkInitialized() {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommUCC not initialized");
  }
}

// ---- Point-to-Point Operations ----
c10::intrusive_ptr<TorchWork> TorchCommUCC::send(
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
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        // UCC doesn't have native send/recv in collective API.
        // Use a barrier-synchronized store-based exchange.
        auto key = fmt::format(
            "ucc_p2p_{}_{}_{}",
            p2pCounter_++,
            rank_,
            dst);
        auto nbytes = tensorCPU.numel() * tensorCPU.element_size();
        std::vector<uint8_t> data(
            static_cast<uint8_t*>(tensorCPU.data_ptr()),
            static_cast<uint8_t*>(tensorCPU.data_ptr()) + nbytes);
        store_->set(key, data);
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::recv(
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
        auto key = fmt::format(
            "ucc_p2p_{}_{}_{}",
            p2pCounter_++,
            src,
            rank_);
        auto val = store_->get(key);
        std::memcpy(
            tensorCPU.data_ptr(), val.data(), val.size());
        if (tensorCPU.device() != tensor.device()) {
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::batch_op_issue(
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

// ---- Helper to run a UCC collective synchronously ----
namespace {
void runUCCCollective(
    ucc_team_h team,
    ucc_context_h ctx,
    ucc_coll_args_t& args) {
  ucc_coll_req_h req;
  auto st = ucc_collective_init(&args, &req, team);
  if (st != UCC_OK) {
    throw std::runtime_error(
        fmt::format("ucc_collective_init failed: {}", ucc_status_string(st)));
  }
  st = ucc_collective_post(req);
  if (st != UCC_OK) {
    ucc_collective_finalize(req);
    throw std::runtime_error(
        fmt::format("ucc_collective_post failed: {}", ucc_status_string(st)));
  }
  while ((st = ucc_collective_test(req)) == UCC_INPROGRESS) {
    ucc_context_progress(ctx);
  }
  if (st != UCC_OK) {
    ucc_collective_finalize(req);
    throw std::runtime_error(
        fmt::format(
            "ucc_collective_test failed: {}", ucc_status_string(st)));
  }
  ucc_collective_finalize(req);
}
} // namespace

// ---- Collective Operations ----
c10::intrusive_ptr<TorchWork> TorchCommUCC::broadcast(
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
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_BCAST;
        args.src.info.buffer = tensorCPU.data_ptr();
        args.src.info.count = static_cast<ucc_count_t>(tensorCPU.numel());
        args.src.info.datatype = toUCCDatatype(tensorCPU.scalar_type());
        args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.root = root;

        runUCCCollective(team_, ctx_, args);

        if (tensorCPU.device() != tensor.device()) {
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::all_reduce(
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

        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.dst.info.buffer = tensorCPU.data_ptr();
        args.dst.info.count = static_cast<ucc_count_t>(tensorCPU.numel());
        args.dst.info.datatype = toUCCDatatype(tensorCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.op = toUCCReduceOp(op);

        runUCCCollective(team_, ctx_, args);

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

c10::intrusive_ptr<TorchWork> TorchCommUCC::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  checkInitialized();
  ensureTensorContiguous(tensor);

  TracingGuard tracingGuard(name_, comm_size_, "reduce", root, tensor, tensor);

  // UCC doesn't have a direct reduce-to-root. Implement via allreduce.
  auto tensorCPU = tensor.to(at::kCPU).contiguous();

  return createWork(
      [this, tensor, tensorCPU, root, op]() mutable {
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.dst.info.buffer = tensorCPU.data_ptr();
        args.dst.info.count = static_cast<ucc_count_t>(tensorCPU.numel());
        args.dst.info.datatype = toUCCDatatype(tensorCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.op = toUCCReduceOp(op);

        runUCCCollective(team_, ctx_, args);

        if (tensorCPU.device() != tensor.device()) {
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::all_gather(
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

        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLGATHER;
        args.src.info.buffer = inputCPU.data_ptr();
        args.src.info.count = static_cast<ucc_count_t>(inputCPU.numel());
        args.src.info.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.dst.info.buffer = outputCPU.data_ptr();
        args.dst.info.count = static_cast<ucc_count_t>(totalElements);
        args.dst.info.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

        runUCCCollective(team_, ctx_, args);

        auto chunkSize = inputCPU.numel();
        for (int i = 0; i < comm_size_; ++i) {
          auto chunk = outputCPU.narrow(0, i * chunkSize, chunkSize);
          tensor_list[i].copy_(chunk);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::all_gather_v(
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

  // Implement via allgatherv using broadcast rounds (same as Gloo pattern)
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

          ucc_coll_args_t args;
          std::memset(&args, 0, sizeof(args));
          args.coll_type = UCC_COLL_TYPE_BCAST;
          args.src.info.buffer = buf.data_ptr();
          args.src.info.count = static_cast<ucc_count_t>(buf.numel());
          args.src.info.datatype = toUCCDatatype(buf.scalar_type());
          args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
          args.root = i;

          runUCCCollective(team_, ctx_, args);

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

c10::intrusive_ptr<TorchWork> TorchCommUCC::all_gather_single(
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
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLGATHER;
        args.src.info.buffer = inputCPU.data_ptr();
        args.src.info.count = static_cast<ucc_count_t>(inputCPU.numel());
        args.src.info.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.dst.info.buffer = outputCPU.data_ptr();
        args.dst.info.count = static_cast<ucc_count_t>(outputCPU.numel());
        args.dst.info.datatype = toUCCDatatype(outputCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

        runUCCCollective(team_, ctx_, args);

        if (outputCPU.device() != output.device()) {
          output.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::reduce_scatter(
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

c10::intrusive_ptr<TorchWork> TorchCommUCC::reduce_scatter_v(
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

  // Implement via allreduce + extract (same pattern as Gloo)
  std::vector<at::Tensor> inputListCPU;
  inputListCPU.reserve(input_list.size());
  for (const auto& t : input_list) {
    inputListCPU.push_back(t.to(at::kCPU).contiguous());
  }
  auto outputCPU = output.to(at::kCPU).contiguous();

  return createWork(
      [this, output, input_list, inputListCPU, outputCPU, op]() mutable {
        // Use multiple allreduce rounds, one per rank's portion
        for (int i = 0; i < comm_size_; ++i) {
          auto& inputTensor = inputListCPU[i];

          ucc_coll_args_t args;
          std::memset(&args, 0, sizeof(args));
          args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
          args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
          args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
          args.dst.info.buffer = inputTensor.data_ptr();
          args.dst.info.count =
              static_cast<ucc_count_t>(inputTensor.numel());
          args.dst.info.datatype =
              toUCCDatatype(inputTensor.scalar_type());
          args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
          args.op = toUCCReduceOp(op);

          runUCCCollective(team_, ctx_, args);

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

c10::intrusive_ptr<TorchWork> TorchCommUCC::reduce_scatter_single(
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
        // allreduce the input, then extract our portion
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.dst.info.buffer = inputCPU.data_ptr();
        args.dst.info.count = static_cast<ucc_count_t>(inputCPU.numel());
        args.dst.info.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.op = toUCCReduceOp(op);

        runUCCCollective(team_, ctx_, args);

        if (op.type() == ReduceOp::RedOpType::AVG) {
          if (at::isIntegralType(
                  inputCPU.scalar_type(), /*includeBool=*/false)) {
            inputCPU.div_(comm_size_, "trunc");
          } else {
            inputCPU /= comm_size_;
          }
        }

        auto chunkSize = output.numel();
        auto start = rank_ * chunkSize;
        auto chunk = inputCPU.narrow(0, start, chunkSize);
        output.copy_(chunk);
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::all_to_all_single(
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
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLTOALL;
        args.src.info.buffer = inputCPU.data_ptr();
        args.src.info.count = static_cast<ucc_count_t>(inputCPU.numel());
        args.src.info.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.dst.info.buffer = outputCPU.data_ptr();
        args.dst.info.count = static_cast<ucc_count_t>(outputCPU.numel());
        args.dst.info.datatype = toUCCDatatype(outputCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

        runUCCCollective(team_, ctx_, args);

        if (outputCPU.device() != output.device()) {
          output.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::all_to_all_v_single(
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

        std::vector<ucc_count_t> send_counts(comm_size_);
        std::vector<ucc_aint_t> send_displacements(comm_size_);
        std::vector<ucc_count_t> recv_counts(comm_size_);
        std::vector<ucc_aint_t> recv_displacements(comm_size_);

        ucc_aint_t send_offset = 0;
        ucc_aint_t recv_offset = 0;
        for (int i = 0; i < comm_size_; ++i) {
          send_counts[i] = static_cast<ucc_count_t>(
              input_split_sizes[i] * inputDim0Numel);
          send_displacements[i] = send_offset;
          send_offset += send_counts[i];

          recv_counts[i] = static_cast<ucc_count_t>(
              output_split_sizes[i] * outputDim0Numel);
          recv_displacements[i] = recv_offset;
          recv_offset += recv_counts[i];
        }

        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLTOALLV;
        args.src.info_v.buffer = inputCPU.data_ptr();
        args.src.info_v.counts = send_counts.data();
        args.src.info_v.displacements = send_displacements.data();
        args.src.info_v.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.src.info_v.mem_type = UCC_MEMORY_TYPE_HOST;
        args.dst.info_v.buffer = outputCPU.data_ptr();
        args.dst.info_v.counts = recv_counts.data();
        args.dst.info_v.displacements = recv_displacements.data();
        args.dst.info_v.datatype = toUCCDatatype(outputCPU.scalar_type());
        args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

        runUCCCollective(team_, ctx_, args);

        if (outputCPU.device() != output.device()) {
          output.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::all_to_all(
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
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLTOALL;
        args.src.info.buffer = inputConcatCPU.data_ptr();
        args.src.info.count =
            static_cast<ucc_count_t>(inputConcatCPU.numel());
        args.src.info.datatype =
            toUCCDatatype(inputConcatCPU.scalar_type());
        args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.dst.info.buffer = outputConcatCPU.data_ptr();
        args.dst.info.count =
            static_cast<ucc_count_t>(outputConcatCPU.numel());
        args.dst.info.datatype =
            toUCCDatatype(outputConcatCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

        runUCCCollective(team_, ctx_, args);

        for (int i = 0; i < comm_size_; ++i) {
          output_tensor_list[i].copy_(
              outputConcatCPU.narrow(0, i * tensorSize, tensorSize));
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();

  TracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);

  return createWork(
      [this]() {
        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_BARRIER;

        runUCCCollective(team_, ctx_, args);
      },
      async_op);
}

// ---- Scatter and Gather ----
c10::intrusive_ptr<TorchWork> TorchCommUCC::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  checkInitialized();
  ensureTensorContiguous(output_tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "scatter", root, input_tensor_list, {output_tensor});

  // Use broadcast rounds: root broadcasts chunk[i] to rank i
  auto outputCPU = output_tensor.to(at::kCPU).contiguous();
  std::vector<at::Tensor> inputListCPU;
  if (rank_ == root) {
    inputListCPU.reserve(input_tensor_list.size());
    for (const auto& t : input_tensor_list) {
      inputListCPU.push_back(t.to(at::kCPU).contiguous());
    }
  }

  return createWork(
      [this, output_tensor, outputCPU, inputListCPU, root]() mutable {
        // Root sends each chunk via broadcast
        at::Tensor buf = outputCPU.clone();
        for (int i = 0; i < comm_size_; ++i) {
          if (rank_ == root) {
            buf.copy_(inputListCPU[i]);
          }

          ucc_coll_args_t args;
          std::memset(&args, 0, sizeof(args));
          args.coll_type = UCC_COLL_TYPE_BCAST;
          args.src.info.buffer = buf.data_ptr();
          args.src.info.count = static_cast<ucc_count_t>(buf.numel());
          args.src.info.datatype = toUCCDatatype(buf.scalar_type());
          args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
          args.root = root;

          runUCCCollective(team_, ctx_, args);

          if (i == rank_) {
            outputCPU.copy_(buf);
          }
        }

        if (outputCPU.device() != output_tensor.device()) {
          output_tensor.copy_(outputCPU);
        }
      },
      async_op);
}

c10::intrusive_ptr<TorchWork> TorchCommUCC::gather(
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

  // Use allgather, then root extracts
  return createWork(
      [this, output_tensor_list, inputCPU, root]() mutable {
        auto totalElements = inputCPU.numel() * comm_size_;
        auto gatheredCPU = at::empty({totalElements}, inputCPU.options());

        ucc_coll_args_t args;
        std::memset(&args, 0, sizeof(args));
        args.coll_type = UCC_COLL_TYPE_ALLGATHER;
        args.src.info.buffer = inputCPU.data_ptr();
        args.src.info.count = static_cast<ucc_count_t>(inputCPU.numel());
        args.src.info.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.dst.info.buffer = gatheredCPU.data_ptr();
        args.dst.info.count = static_cast<ucc_count_t>(totalElements);
        args.dst.info.datatype = toUCCDatatype(inputCPU.scalar_type());
        args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

        runUCCCollective(team_, ctx_, args);

        if (rank_ == root) {
          auto chunkSize = inputCPU.numel();
          for (int i = 0; i < comm_size_; ++i) {
            output_tensor_list[i].copy_(
                gatheredCPU.narrow(0, i * chunkSize, chunkSize));
          }
        }
      },
      async_op);
}

// ---- Communicator Management ----
std::shared_ptr<TorchCommBackend> TorchCommUCC::split(
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

  auto new_torchcomm = std::make_shared<TorchCommUCC>();

  int color = *std::min_element(ranks.begin(), ranks.end());
  int new_rank = static_cast<int>(std::distance(ranks.begin(), it));
  int new_size = static_cast<int>(ranks.size());

  new_torchcomm->rank_ = new_rank;
  new_torchcomm->comm_size_ = new_size;

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
class UCCRegistration {
 public:
  UCCRegistration() {
    TorchCommFactory::get().register_backend(
        "ucc", []() { return std::make_shared<TorchCommUCC>(); });
  }
};

static const UCCRegistration registration{};
} // namespace

} // namespace torch::comms
