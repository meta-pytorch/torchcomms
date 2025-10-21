// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/gloo/TorchCommGloo.hpp"

#include <set>
#include <string>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual

#include <gloo/algorithm.h>
#include <gloo/allgather.h>
#include <gloo/allreduce.h>
#include <gloo/alltoall.h>
#include <gloo/alltoallv.h>
#include <gloo/barrier.h>
#include <gloo/broadcast.h>
#include <gloo/context.h>
#include <gloo/gather.h>
#include <gloo/reduce.h>
#include <gloo/reduce_scatter.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/scatter.h>
#include <gloo/transport/device.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/transport/unbound_buffer.h>

#include "comms/torchcomms/StoreManager.hpp"
#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommUtils.hpp"
#include "comms/torchcomms/gloo/GlooStore.hpp"

namespace torch {
namespace comms {

template <typename T>
inline T* getDataPointer(const at::Tensor& tensor) {
  // This method is only used in ProcessGroupGloo for now. Call sites must make
  // sure that the input tensor is contiguous. It is OK if the tensor does not
  // start from the beginning of the storage. For example, it could come from
  // chunk(..., dim=0)[1]. Hence, we need to use data_ptr() instead of
  // tensor.storage().data()
  // NB: not using tensor.data<T>() because tensor is not aware of gloo::TYPE
  return static_cast<T*>(tensor.data_ptr());
}

#define GENERATE_ALL_TYPES(type, func, args...)  \
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(args);                         \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(args);                        \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<c10::Half>(args);                     \
      break;                                     \
    case ::at::ScalarType::BFloat16:             \
      func<c10::BFloat16>(args);                 \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(args);                        \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
    case ::at::ScalarType::Bool:                 \
      func<uint8_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(args);                       \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }

namespace {
void ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for NCCL operations");
  }
}

using ReduceFunc = void (*)(void*, const void*, const void*, size_t);

template <typename T, std::enable_if_t<!std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r.type()) {
    case ReduceOp::RedOpType::SUM:
    case ReduceOp::RedOpType::AVG:
    case ReduceOp::RedOpType::PREMUL_SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::RedOpType::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::RedOpType::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::RedOpType::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::RedOpType::BAND:
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with non-integral dtype");
      break;
    case ReduceOp::RedOpType::BOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with non-integral dtype");
      break;
    case ReduceOp::RedOpType::BXOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with non-integral dtype");
      break;
    default:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

void preReduce(at::Tensor& tensor, const ReduceOp& r) {
  // PREMUL_SUM is a special case where we need to multiply the tensor by the
  // factor before the reduce operation.
  if (r.type() == ReduceOp::RedOpType::PREMUL_SUM) {
    auto factor = *r.factor();
    try {
      tensor *= std::get<double>(factor);
    } catch (const std::bad_variant_access&) {
      tensor *= std::get<at::Tensor>(factor);
    }
  }
}

void postReduce(at::Tensor& tensor, const ReduceOp& r) {
  // Gloo doesn't support AVG so we use SUM + division.
  if (r.type() == ReduceOp::RedOpType::AVG) {
    tensor /= static_cast<float>(tensor.numel());
  }
}

// Bitwise AND with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void band(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] & tb[i];
  }
}

// Bitwise OR with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] | tb[i];
  }
}

// Bitwise XOR with SFINAE guard for integral types.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void bxor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] ^ tb[i];
  }
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r.type()) {
    case ReduceOp::RedOpType::SUM:
    case ReduceOp::RedOpType::AVG:
    case ReduceOp::RedOpType::PREMUL_SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::RedOpType::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::RedOpType::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::RedOpType::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::RedOpType::BAND:
      return ReduceFunc(&band<T>);
    case ReduceOp::RedOpType::BOR:
      return ReduceFunc(&bor<T>);
    case ReduceOp::RedOpType::BXOR:
      return ReduceFunc(&bxor<T>);
    default:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

template <typename T>
void getFunction(gloo::AllreduceOptions::Func& fn, const ReduceOp& op) {
  fn = toFunction<T>(op);
}

gloo::AllreduceOptions::Func getFunction(
    const at::ScalarType& dtype,
    const ReduceOp& op) {
  gloo::AllreduceOptions::Func fn;
  GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
  return fn;
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setScatterInputs(O& opts, std::vector<at::Tensor>& inputTensors) {
  std::vector<T*> ptrs;
  ptrs.reserve(inputTensors.size());
  for (auto& tensor : inputTensors) {
    ptrs.push_back(getDataPointer<T>(tensor));
  }
  opts.setInputs(ptrs, inputTensors.at(0).numel());
}

template <typename T, typename O>
void setAlltoallvInput(
    O& opts,
    at::Tensor& inputTensor,
    std::vector<int64_t>& elementsPerRank) {
  opts.template setInput<T>(getDataPointer<T>(inputTensor), elementsPerRank);
}

template <typename T, typename O>
void setAlltoallvOutput(
    O& opts,
    at::Tensor& outputTensor,
    std::vector<int64_t>& elementsPerRank) {
  opts.template setOutput<T>(getDataPointer<T>(outputTensor), elementsPerRank);
}

template <typename T>
void sendTensor(
    std::shared_ptr<gloo::Context> context,
    const at::Tensor& tensor,
    int dst,
    uint64_t tag,
    std::chrono::milliseconds timeout) {
  auto buffer = context->createUnboundBuffer(
      const_cast<void*>(static_cast<const void*>(getDataPointer<T>(tensor))),
      tensor.numel() * sizeof(T));
  buffer->send(dst, tag);
  if (timeout == kNoTimeout) {
    buffer->waitSend();
  } else {
    buffer->waitSend(timeout);
  }
}

template <typename T>
void recvTensor(
    std::shared_ptr<gloo::Context> context,
    at::Tensor& tensor,
    int src,
    uint64_t tag,
    std::chrono::milliseconds timeout) {
  auto buffer = context->createUnboundBuffer(
      static_cast<void*>(getDataPointer<T>(tensor)),
      tensor.numel() * sizeof(T));
  buffer->recv(src, tag);
  if (timeout == kNoTimeout) {
    buffer->waitRecv();
  } else {
    buffer->waitRecv(timeout);
  }
}

} // namespace

TorchCommGloo::TorchCommGloo() : device_(at::kCPU) {}

TorchCommGloo::~TorchCommGloo() {}

void TorchCommGloo::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  TC_LOG(INFO) << "Initializing TorchCommGloo for device: " << device;
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommGloo already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommGloo already finalized");
  }
  init_state_ = InitializationState::INITIALIZED;

  if (rank_ == -1 || comm_size_ == -1) {
    auto ranksize = query_ranksize();
    rank_ = ranksize.first;
    comm_size_ = ranksize.second;
  }

  ::gloo::transport::tcp::attr attr;
  attr.hostname = env_to_value<std::string>("TORCHCOMM_GLOO_HOSTNAME", "");
  attr.iface = env_to_value<std::string>("TORCHCOMM_GLOO_INTERFACE", "");
  auto& hints = options.hints;
  if (hints.contains("hostname")) {
    attr.hostname = hints.at("hostname");
  }
  if (hints.contains("interface")) {
    attr.iface = hints.at("interface");
  }
  auto gloo_device = hints.contains("lazy")
      ? ::gloo::transport::tcp::CreateLazyDevice(attr)
      : ::gloo::transport::tcp::CreateDevice(attr);
  auto context =
      std::make_shared<::gloo::rendezvous::Context>(rank_, comm_size_);
  context->setTimeout(options.timeout);

  auto store = options.store;
  if (!store) {
    store = StoreManager::get().getStore(
        TorchCommGloo::kBackendName, name, options.timeout);
  }

  if (rank_ == 0) {
    int64_t initCount = store->add("init_count", 1);
    TORCH_INTERNAL_ASSERT(
        initCount == 1, "detected multiple communicators on same store!");
  }

  auto connectStore = std::make_shared<GlooStore>(store);
  context->connectFullMesh(connectStore, gloo_device);

  context_ = std::move(context);
  store_ = std::move(store);

  tracing_ = std::make_shared<TorchCommTracing>(name, comm_size_, rank_);
  tracing_->recordEvent("init");

  TC_LOG(INFO) << "TorchCommGloo initialized for rank: " << rank_;
}

void TorchCommGloo::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommGloo not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommGloo already finalized");
  }
  init_state_ = InitializationState::FINALIZED;
}

int TorchCommGloo::getRank() const {
  return rank_;
}

int TorchCommGloo::getSize() const {
  return comm_size_;
}

std::string_view TorchCommGloo::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommGloo::getCommName() const {
  return name_;
}
std::shared_ptr<TorchWork> TorchCommGloo::createWork(
    std::function<void()> fn,
    bool async_op) {
  if (async_op) {
    return std::make_shared<TorchWorkThread>(std::move(fn));
  }

  fn();
  return std::make_shared<TorchWorkCompleted>();
}

// Point-to-Point Operations
std::shared_ptr<TorchWork> TorchCommGloo::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  if (dst < 0 || dst >= comm_size_) {
    throw std::runtime_error("Invalid destination rank for send operation");
  }

  if (dst == rank_) {
    throw std::runtime_error("Cannot send to self");
  }

  tracing_->recordEventWithInputOutput("send", dst, {tensor}, {tensor});

  // Convert tensor to CPU for Gloo compatibility
  auto tensorCPU = tensor.to(at::kCPU);

  return createWork(
      [tensorCPU, dst, options, context = context_, tag = nextTag()]() {
        const auto& scalarType = tensorCPU.scalar_type();

        // Use type dispatch to send tensor
        GENERATE_ALL_TYPES(
            scalarType,
            sendTensor,
            context,
            tensorCPU,
            dst,
            tag,
            options.timeout);
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  if (src < 0 || src >= comm_size_) {
    throw std::runtime_error("Invalid source rank for recv operation");
  }

  if (src == rank_) {
    throw std::runtime_error("Cannot recv from self");
  }

  tracing_->recordEventWithInputOutput("recv", src, {tensor}, {tensor});

  // Convert tensor to CPU for Gloo compatibility
  auto tensorCPU = tensor.to(at::kCPU);

  return createWork(
      [tensor,
       tensorCPU,
       src,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        const auto& scalarType = tensorCPU.scalar_type();

        // Use type dispatch to receive tensor
        GENERATE_ALL_TYPES(
            scalarType,
            recvTensor,
            context,
            tensorCPU,
            src,
            tag,
            options.timeout);

        if (tensorCPU.device() != tensor.device()) {
          // Copy back to original device if needed
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

// Batch P2P Operations
std::shared_ptr<TorchWork> TorchCommGloo::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool /*async_op*/,
    const BatchP2POptions& /*options*/) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  if (ops.empty()) {
    throw std::runtime_error("Cannot issue empty batch operation");
  }

  // Collect input and output tensors for work tracking
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;

  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      input_tensors.push_back(tensor);
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      output_tensors.push_back(tensor);
    } else {
      throw std::runtime_error("Unknown op type");
    }
  }

  tracing_->recordEventWithInputOutput(
      "batch_op_issue", rank_, input_tensors, output_tensors);

  throw std::runtime_error("Not implemented yet");
}

// Collective Operations
std::shared_ptr<TorchWork> TorchCommGloo::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("broadcast", root, {tensor}, {tensor});

  // This will synchronize the stream.
  auto tensorCPU = tensor.to(at::kCPU);

  return createWork(
      [tensor,
       tensorCPU,
       root,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::BroadcastOptions opts(context);
        const auto& scalarType = tensor.scalar_type();
        opts.setRoot(root);
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensorCPU);

        gloo::broadcast(opts);

        if (tensorCPU.device() != tensor.device()) {
          // This will block the CPU thread so we don't need to synchronize the
          // streams.
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::all_reduce(
    at::Tensor& tensor,
    ReduceOp op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("all_reduce", rank_, {tensor}, {tensor});

  // This will synchronize the stream.
  auto tensorCPU = tensor.to(at::kCPU);

  return createWork(
      [tensor,
       tensorCPU,
       op,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::AllreduceOptions opts(context);
        const auto& scalarType = tensor.scalar_type();
        opts.setReduceFunction(getFunction(scalarType, op));
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensorCPU);

        preReduce(tensorCPU, op);
        gloo::allreduce(opts);
        postReduce(tensorCPU, op);

        if (tensorCPU.device() != tensor.device()) {
          // This will block the CPU thread so we don't need to synchronize the
          // streams.
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::reduce(
    const at::Tensor& tensor,
    int root,
    ReduceOp op,
    bool async_op,
    const ReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("reduce", root, {tensor}, {tensor});

  // This will synchronize the stream.
  auto tensorCPU = tensor.to(at::kCPU);

  return createWork(
      [tensor,
       tensorCPU,
       root,
       op,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::ReduceOptions opts(context);
        const auto& scalarType = tensor.scalar_type();
        opts.setReduceFunction(getFunction(scalarType, op));
        opts.setRoot(root);
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensorCPU);

        preReduce(tensorCPU, op);
        gloo::reduce(opts);
        postReduce(tensorCPU, op);

        if (tensorCPU.device() != tensor.device()) {
          // This will block the CPU thread so we don't need to synchronize the
          // streams.
          tensor.copy_(tensorCPU);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather");
  }

  // Ensure input tensor is contiguous
  ensureTensorContiguous(tensor);

  // Check that all output tensors are contiguous and have correct size
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
    if (t.numel() != tensor.numel()) {
      throw std::runtime_error(
          "All tensors in tensor_list must have same size as input tensor");
    }
  }

  tracing_->recordEventWithInputOutput(
      "all_gather", rank_, tensor_list, {tensor});

  // Convert tensors to CPU
  auto tensorCPU = tensor.to(at::kCPU);
  std::vector<at::Tensor> tensorListCPU;
  tensorListCPU.reserve(tensor_list.size());
  for (const auto& t : tensor_list) {
    tensorListCPU.push_back(t.to(at::kCPU));
  }

  return createWork(
      [tensor,
       tensor_list,
       tensorCPU,
       tensorListCPU,
       options,
       size = comm_size_,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::AllgatherOptions opts(context);
        const auto& scalarType = tensor.scalar_type();
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // Create concatenated output buffer
        auto totalElements = tensorCPU.numel() * size;
        auto concatOutput = at::empty({totalElements}, tensorCPU.options());

        // Use type dispatch to set input and output
        GENERATE_ALL_TYPES(scalarType, setInput, opts, tensorCPU);
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, concatOutput);

        gloo::allgather(opts);

        // Split concatenated output back to individual tensors
        auto chunkSize = tensorCPU.numel();
        for (int i = 0; i < size; ++i) {
          auto start = i * chunkSize;
          auto chunk = concatOutput.narrow(0, start, chunkSize);
          tensorListCPU[i].copy_(chunk);
        }

        // Copy results back to original device if needed
        for (size_t i = 0; i < tensorListCPU.size(); ++i) {
          if (tensorListCPU[i].device() != tensor_list[i].device()) {
            tensor_list[i].copy_(tensorListCPU[i]);
          }
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (output.numel() != input.numel() * comm_size_) {
    throw std::runtime_error(
        "Output tensor size must be input_size * comm_size for all_gather_single");
  }

  tracing_->recordEventWithInputOutput(
      "all_gather_single", rank_, {input}, {output});

  // Convert tensors to CPU
  auto inputCPU = input.to(at::kCPU);
  auto outputCPU = output.to(at::kCPU);

  return createWork(
      [input,
       output,
       inputCPU,
       outputCPU,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::AllgatherOptions opts(context);
        const auto& scalarType = input.scalar_type();
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // Use type dispatch to set input and output
        GENERATE_ALL_TYPES(scalarType, setInput, opts, inputCPU);
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputCPU);

        gloo::allgather(opts);

        if (outputCPU.device() != output.device()) {
          // This will block the CPU thread so we don't need to synchronize the
          // streams.
          output.copy_(outputCPU);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    ReduceOp op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);

  if (input_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_list size must equal comm_size for reduce_scatter");
  }

  // Check that all input tensors are contiguous and have correct size
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
    if (t.numel() != output.numel()) {
      throw std::runtime_error(
          "All input tensors must have same size as output tensor");
    }
  }

  tracing_->recordEventWithInputOutput(
      "reduce_scatter", rank_, input_list, {output});

  // Concatenate input tensors
  auto input = at::cat(input_list, 0);

  ReduceScatterSingleOptions singleOptions;
  singleOptions.timeout = options.timeout;
  singleOptions.hints = options.hints;
  return reduce_scatter_single(output, input, op, async_op, singleOptions);
}

std::shared_ptr<TorchWork> TorchCommGloo::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    ReduceOp op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (input.numel() != output.numel() * comm_size_) {
    throw std::runtime_error(
        "Input tensor size must be output_size * comm_size for reduce_scatter_single");
  }

  tracing_->recordEventWithInputOutput(
      "reduce_scatter_single", rank_, {input}, {output});

  // Convert tensors to CPU (noop if already on CPU)
  auto inputCPU = input.to(at::kCPU);

  return createWork(
      [input,
       output,
       inputCPU,
       op,
       options,
       rank = rank_,
       context = context_,
       tag = nextTag()]() mutable {
        // For reduce_scatter_single, we can simulate it by:
        // 1. All-reduce the input tensor
        // 2. Each rank takes its portion from the result

        gloo::AllreduceOptions opts(context);
        const auto& scalarType = input.scalar_type();
        opts.setReduceFunction(getFunction(scalarType, op));
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // Use type dispatch to set input/output for all-reduce
        GENERATE_ALL_TYPES(scalarType, setInput, opts, inputCPU);
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, inputCPU);

        preReduce(inputCPU, op);
        gloo::allreduce(opts);
        postReduce(inputCPU, op);

        // Extract this rank's portion from the reduced result
        auto chunkSize = output.numel();
        auto start = rank * chunkSize;
        auto chunk = inputCPU.narrow(0, start, chunkSize);
        output.copy_(chunk);
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (input.numel() != output.numel()) {
    throw std::runtime_error(
        "Input and output tensors must have same size for all_to_all_single");
  }

  if (input.numel() % comm_size_ != 0) {
    throw std::runtime_error(
        "Tensor size must be divisible by comm_size for all_to_all_single");
  }

  tracing_->recordEventWithInputOutput(
      "all_to_all_single", rank_, {input}, {output});

  // Convert tensors to CPU
  auto inputCPU = input.to(at::kCPU);
  auto outputCPU = output.to(at::kCPU);

  return createWork(
      [input,
       output,
       inputCPU,
       outputCPU,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::AlltoallOptions opts(context);
        const auto& scalarType = input.scalar_type();
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // Use type dispatch to set input and output
        GENERATE_ALL_TYPES(scalarType, setInput, opts, inputCPU);
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputCPU);

        gloo::alltoall(opts);

        if (outputCPU.device() != output.device()) {
          // This will block the CPU thread so we don't need to synchronize the
          // streams.
          output.copy_(outputCPU);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  // Validate split sizes vectors
  if (input_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  if (output_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "output_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  // Validate that split sizes sum to tensor sizes
  uint64_t input_total = 0;
  uint64_t output_total = 0;
  for (int i = 0; i < comm_size_; ++i) {
    input_total += input_split_sizes[i];
    output_total += output_split_sizes[i];
  }

  if (input_total != static_cast<uint64_t>(input.size(0))) {
    throw std::runtime_error(
        "Sum of input_split_sizes must equal input tensor size for all_to_all_v_single");
  }

  if (output_total != static_cast<uint64_t>(output.size(0))) {
    throw std::runtime_error(
        "Sum of output_split_sizes must equal output tensor size for all_to_all_v_single");
  }

  tracing_->recordEventWithInputOutput(
      "all_to_all_v_single", rank_, {input}, {output});

  // Convert tensors to CPU (noop if already on CPU)
  auto inputCPU = input.to(at::kCPU);
  auto outputCPU = output.to(at::kCPU);

  return createWork(
      [input,
       output,
       inputCPU,
       outputCPU,
       input_split_sizes,
       output_split_sizes,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::AlltoallvOptions opts(context);
        const auto& scalarType = input.scalar_type();
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // Convert split sizes to int64_t vectors for Gloo API
        std::vector<int64_t> inputElements, outputElements;
        inputElements.reserve(input_split_sizes.size());
        outputElements.reserve(output_split_sizes.size());

        // Calculate number of elements in each dim 0 chunk.
        auto dim0Numel =
            input.numel() / std::max(input.size(0), static_cast<int64_t>(1));
        for (auto size : input_split_sizes) {
          inputElements.push_back(static_cast<int64_t>(size) * dim0Numel);
        }
        for (auto size : output_split_sizes) {
          outputElements.push_back(static_cast<int64_t>(size) * dim0Numel);
        }

        // Use type dispatch to set input and output with split sizes
        GENERATE_ALL_TYPES(
            scalarType, setAlltoallvInput, opts, inputCPU, inputElements);
        GENERATE_ALL_TYPES(
            scalarType, setAlltoallvOutput, opts, outputCPU, outputElements);

        gloo::alltoallv(opts);

        if (outputCPU.device() != output.device()) {
          // This will block the CPU thread so we don't need to synchronize the
          // streams.
          output.copy_(outputCPU);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (output_tensor_list.size() != static_cast<size_t>(comm_size_) ||
      input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "Tensor list sizes must equal comm_size for all_to_all");
  }

  // Validate all tensors
  for (int i = 0; i < comm_size_; ++i) {
    ensureTensorContiguous(input_tensor_list[i]);
    ensureTensorContiguous(output_tensor_list[i]);
  }

  tracing_->recordEventWithInputOutput(
      "all_to_all", rank_, input_tensor_list, output_tensor_list);

  // Get tensor size (all tensors should be same size)
  auto tensorSize = input_tensor_list.at(0).numel();
  for (const auto& t : input_tensor_list) {
    if (t.numel() != tensorSize) {
      throw std::runtime_error(
          "All input tensors must have same size for all_to_all");
    }
  }
  for (const auto& t : output_tensor_list) {
    if (t.numel() != tensorSize) {
      throw std::runtime_error(
          "All output tensors must have same size for all_to_all");
    }
  }

  // Create concatenated input and output buffers
  auto totalElements = tensorSize * comm_size_;
  auto inputConcatCPU = at::empty(
      {totalElements}, input_tensor_list.at(0).options().device(at::kCPU));
  auto outputConcatCPU = at::empty(
      {totalElements}, output_tensor_list.at(0).options().device(at::kCPU));

  // Copy input tensors to concatenated buffer
  for (int i = 0; i < comm_size_; ++i) {
    auto start = i * tensorSize;
    auto chunk = inputConcatCPU.narrow(0, start, tensorSize);
    chunk.copy_(input_tensor_list[i]);
  }

  return createWork(
      [output_tensor_list,
       inputConcatCPU,
       outputConcatCPU,
       tensorSize,
       comm_size = comm_size_,
       options,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::AlltoallOptions opts(context);
        const auto& scalarType = inputConcatCPU.scalar_type();
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // Use type dispatch to set input and output
        GENERATE_ALL_TYPES(scalarType, setInput, opts, inputConcatCPU);
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputConcatCPU);

        gloo::alltoall(opts);

        // Copy results back to individual output tensors
        for (int i = 0; i < comm_size; ++i) {
          auto start = i * tensorSize;
          auto chunk = outputConcatCPU.narrow(0, start, tensorSize);
          output_tensor_list[i].copy_(chunk);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  tracing_->recordEvent("barrier");

  return createWork(
      [options, context = context_, tag = nextTag()]() {
        gloo::BarrierOptions opts(context);
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }
        gloo::barrier(opts);
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);

  // Only the root rank needs valid input tensors
  if (rank_ == root) {
    if (input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "input_tensor_list size must equal comm_size for scatter");
    }

    for (const auto& t : input_tensor_list) {
      ensureTensorContiguous(t);
      if (t.numel() != output_tensor.numel()) {
        throw std::runtime_error(
            "All input tensors must have same size as output tensor");
      }
    }
  }

  tracing_->recordEventWithInputOutput(
      "scatter", root, input_tensor_list, {output_tensor});

  // Convert tensors to CPU
  auto outputCPU = output_tensor.to(at::kCPU);

  // Only root rank needs to prepare input list
  std::vector<at::Tensor> inputListCPU;
  if (rank_ == root) {
    inputListCPU.reserve(input_tensor_list.size());
    for (const auto& t : input_tensor_list) {
      inputListCPU.push_back(t.to(at::kCPU));
    }
  }

  return createWork(
      [output_tensor,
       outputCPU,
       inputListCPU,
       root,
       options,
       rank = rank_,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::ScatterOptions opts(context);
        const auto& scalarType = output_tensor.scalar_type();
        opts.setRoot(root);
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // Use type dispatch to set output
        GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputCPU);

        // Root rank sets inputs using the correct Gloo scatter API
        if (rank == root && !inputListCPU.empty()) {
          GENERATE_ALL_TYPES(scalarType, setScatterInputs, opts, inputListCPU);
        }

        gloo::scatter(opts);

        if (outputCPU.device() != output_tensor.device()) {
          // This will block the CPU thread so we don't need to synchronize the
          // streams.
          output_tensor.copy_(outputCPU);
        }
      },
      async_op);
}

std::shared_ptr<TorchWork> TorchCommGloo::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);

  // Only the root rank needs valid output tensors
  if (rank_ == root) {
    if (output_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "output_tensor_list size must equal comm_size for gather");
    }

    for (const auto& t : output_tensor_list) {
      ensureTensorContiguous(t);
      if (t.numel() != input_tensor.numel()) {
        throw std::runtime_error(
            "All output tensors must have same size as input tensor");
      }
    }
  }

  tracing_->recordEventWithInputOutput(
      "gather", root, {input_tensor}, output_tensor_list);

  // Convert tensors to CPU
  auto inputCPU = input_tensor.to(at::kCPU);

  // Only root rank needs to prepare output
  at::Tensor outputConcatCPU;
  std::vector<at::Tensor> outputListCPU;
  if (rank_ == root) {
    // Create concatenated output buffer
    auto totalElements = input_tensor.numel() * comm_size_;
    outputConcatCPU = at::empty({totalElements}, inputCPU.options());

    // Also convert individual output tensors to CPU for final copy
    outputListCPU.reserve(output_tensor_list.size());
    for (const auto& t : output_tensor_list) {
      outputListCPU.push_back(t.to(at::kCPU));
    }
  }

  return createWork(
      [input_tensor,
       output_tensor_list,
       inputCPU,
       outputConcatCPU,
       outputListCPU,
       root,
       options,
       rank = rank_,
       size = comm_size_,
       context = context_,
       tag = nextTag()]() mutable {
        gloo::GatherOptions opts(context);
        const auto& scalarType = input_tensor.scalar_type();
        opts.setRoot(root);
        opts.setTag(tag);
        if (options.timeout != kNoTimeout) {
          opts.setTimeout(options.timeout);
        }

        // All ranks set input
        GENERATE_ALL_TYPES(scalarType, setInput, opts, inputCPU);

        // Only root sets output
        if (rank == root) {
          GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputConcatCPU);
        }

        gloo::gather(opts);

        // Root rank splits concatenated output back to individual tensors
        if (rank == root) {
          auto chunkSize = inputCPU.numel();
          for (int i = 0; i < size; ++i) {
            auto start = i * chunkSize;
            auto chunk = outputConcatCPU.narrow(0, start, chunkSize);
            outputListCPU[i].copy_(chunk);
          }

          // Copy results back to original device if needed
          for (size_t i = 0; i < outputListCPU.size(); ++i) {
            if (outputListCPU[i].device() != output_tensor_list[i].device()) {
              output_tensor_list[i].copy_(outputListCPU[i]);
            }
          }
        }
      },
      async_op);
}

std::shared_ptr<TorchCommBackend> TorchCommGloo::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  // Validate that all ranks are valid
  for (int rank : ranks) {
    if (rank < 0 || rank >= comm_size_) {
      throw std::runtime_error(
          "Invalid rank " + std::to_string(rank) +
          " in ranks. Valid ranks are 0 to " + std::to_string(comm_size_ - 1));
    }
  }

  // Check for duplicate ranks
  std::set<int> unique_ranks(ranks.begin(), ranks.end());
  if (unique_ranks.size() != ranks.size()) {
    throw std::runtime_error("Duplicate ranks found in ranks list");
  }

  if (ranks.empty()) {
    // Empty list means exclude all ranks - return nullptr
    return nullptr;
  }

  // Check if current rank is in the non-empty list
  auto it = std::find(ranks.begin(), ranks.end(), rank_);
  if (it == ranks.end()) {
    // Current rank is not in the non-empty list - this is an error
    throw std::runtime_error(
        "Current rank " + std::to_string(rank_) +
        " is not included in the provided ranks list");
  }

  auto new_torchcomm = std::make_shared<TorchCommGloo>();

  // Set color to the lowest rank in the group and calculate new rank
  int color = *std::min_element(ranks.begin(), ranks.end());
  int new_rank = static_cast<int>(std::distance(ranks.begin(), it));
  int new_size = static_cast<int>(ranks.size());

  new_torchcomm->rank_ = new_rank;
  new_torchcomm->comm_size_ = new_size;

  auto new_name = fmt::format("{}_{}", name, color);
  auto new_store = c10::make_intrusive<c10d::PrefixStore>(new_name, store_);

  CommOptions new_options = options;
  new_options.store = new_store;

  new_torchcomm->init(device_, new_name, new_options);

  return new_torchcomm;
}

void TorchCommGloo::checkInitialized() {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommGloo not initialized");
  }
}

void TorchCommGloo::checkAndAbortIfTimedOutOrError() {}

namespace {
class GlooRegistration {
 public:
  GlooRegistration() {
    TorchCommFactory::get().register_backend(
        "gloo", []() { return std::make_shared<TorchCommGloo>(); });
  }
};

static GlooRegistration registration{};
} // namespace

} // namespace comms
} // namespace torch
