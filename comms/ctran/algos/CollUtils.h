// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
// Ctran compiles collectives for bf16 and fbp8 types if
// __CUDA_BF16_TYPES_EXIST__ defined but it's defined in correcponding cuda
// headers. So we need to include them here to make sure Ctran compiles
// collectives for bf16 and fbp8 types.
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

// Use "static const" definitions here.  constness helps disallow []
// usage of the unordered map, which is risky and causes the key to be
// automatically created if it doesn't already exist.  Declaring this
// as const forces access to be protected using .at().
#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__) && \
    defined(NCCL_ENABLE_FP8)
#define CTRAN_DATATYPE_TO_FUNC_MAPPER(var, fn)                            \
  static const std::unordered_map<commDataType_t, const void*> var = {    \
      {commInt8, reinterpret_cast<const void*>(fn<int8_t>)},              \
      {commUint8, reinterpret_cast<const void*>(fn<uint8_t>)},            \
      {commInt32, reinterpret_cast<const void*>(fn<int32_t>)},            \
      {commUint32, reinterpret_cast<const void*>(fn<uint32_t>)},          \
      {commInt64, reinterpret_cast<const void*>(fn<int64_t>)},            \
      {commUint64, reinterpret_cast<const void*>(fn<uint64_t>)},          \
      {commFloat16, reinterpret_cast<const void*>(fn<half>)},             \
      {commFloat32, reinterpret_cast<const void*>(fn<float>)},            \
      {commFloat64, reinterpret_cast<const void*>(fn<double>)},           \
      {commBfloat16, reinterpret_cast<const void*>(fn<__nv_bfloat16>)},   \
      {commFloat8e5m2, reinterpret_cast<const void*>(fn<__nv_fp8_e5m2>)}, \
      {commFloat8e4m3, reinterpret_cast<const void*>(fn<__nv_fp8_e4m3>)}, \
  };
#elif defined(__CUDA_BF16_TYPES_EXIST__)
#define CTRAN_DATATYPE_TO_FUNC_MAPPER(var, fn)                          \
  static const std::unordered_map<commDataType_t, const void*> var = {  \
      {commInt8, reinterpret_cast<const void*>(fn<int8_t>)},            \
      {commUint8, reinterpret_cast<const void*>(fn<uint8_t>)},          \
      {commInt32, reinterpret_cast<const void*>(fn<int32_t>)},          \
      {commUint32, reinterpret_cast<const void*>(fn<uint32_t>)},        \
      {commInt64, reinterpret_cast<const void*>(fn<int64_t>)},          \
      {commUint64, reinterpret_cast<const void*>(fn<uint64_t>)},        \
      {commFloat16, reinterpret_cast<const void*>(fn<half>)},           \
      {commFloat32, reinterpret_cast<const void*>(fn<float>)},          \
      {commFloat64, reinterpret_cast<const void*>(fn<double>)},         \
      {commBfloat16, reinterpret_cast<const void*>(fn<__nv_bfloat16>)}, \
  };
#else
#define CTRAN_DATATYPE_TO_FUNC_MAPPER(var, fn)                         \
  static const std::unordered_map<commDataType_t, const void*> var = { \
      {commInt8, reinterpret_cast<const void*>(fn<int8_t>)},           \
      {commUint8, reinterpret_cast<const void*>(fn<uint8_t>)},         \
      {commInt32, reinterpret_cast<const void*>(fn<int32_t>)},         \
      {commUint32, reinterpret_cast<const void*>(fn<uint32_t>)},       \
      {commInt64, reinterpret_cast<const void*>(fn<int64_t>)},         \
      {commUint64, reinterpret_cast<const void*>(fn<uint64_t>)},       \
      {commFloat16, reinterpret_cast<const void*>(fn<half>)},          \
      {commFloat32, reinterpret_cast<const void*>(fn<float>)},         \
      {commFloat64, reinterpret_cast<const void*>(fn<double>)},        \
  };
#endif

struct CtranPairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

#define CTRAN_REDOP_FUNCMAP(dtype, type, fn)                \
  {std::make_pair(dtype, commSum),                          \
   reinterpret_cast<const void*>(&fn<type, commSum>)},      \
      {std::make_pair(dtype, commProd),                     \
       reinterpret_cast<const void*>(&fn<type, commProd>)}, \
      {std::make_pair(dtype, commAvg),                      \
       reinterpret_cast<const void*>(&fn<type, commAvg>)},  \
      {std::make_pair(dtype, commMax),                      \
       reinterpret_cast<const void*>(&fn<type, commMax>)},  \
  {                                                         \
    std::make_pair(dtype, commMin),                         \
        reinterpret_cast<const void*>(&fn<type, commMin>)   \
  }

#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__) && \
    defined(NCCL_ENABLE_FP8)
#define CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(var, fn)              \
  static const std::unordered_map<                                \
      std::pair<commDataType_t, commRedOp_t>,                     \
      const void*,                                                \
      CtranPairHash>                                              \
      var = {                                                     \
          CTRAN_REDOP_FUNCMAP(commInt8, int8_t, fn),              \
          CTRAN_REDOP_FUNCMAP(commUint8, uint8_t, fn),            \
          CTRAN_REDOP_FUNCMAP(commInt32, int32_t, fn),            \
          CTRAN_REDOP_FUNCMAP(commUint32, uint32_t, fn),          \
          CTRAN_REDOP_FUNCMAP(commInt64, int64_t, fn),            \
          CTRAN_REDOP_FUNCMAP(commUint64, uint64_t, fn),          \
          CTRAN_REDOP_FUNCMAP(commFloat16, half, fn),             \
          CTRAN_REDOP_FUNCMAP(commFloat32, float, fn),            \
          CTRAN_REDOP_FUNCMAP(commFloat64, double, fn),           \
          CTRAN_REDOP_FUNCMAP(commBfloat16, __nv_bfloat16, fn),   \
          CTRAN_REDOP_FUNCMAP(commFloat8e5m2, __nv_fp8_e5m2, fn), \
          CTRAN_REDOP_FUNCMAP(commFloat8e4m3, __nv_fp8_e4m3, fn), \
  };
#elif defined(__CUDA_BF16_TYPES_EXIST__)
#define CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(var, fn)            \
  static const std::unordered_map<                              \
      std::pair<commDataType_t, commRedOp_t>,                   \
      const void*,                                              \
      CtranPairHash>                                            \
      var = {                                                   \
          CTRAN_REDOP_FUNCMAP(commInt8, int8_t, fn),            \
          CTRAN_REDOP_FUNCMAP(commUint8, uint8_t, fn),          \
          CTRAN_REDOP_FUNCMAP(commInt32, int32_t, fn),          \
          CTRAN_REDOP_FUNCMAP(commUint32, uint32_t, fn),        \
          CTRAN_REDOP_FUNCMAP(commInt64, int64_t, fn),          \
          CTRAN_REDOP_FUNCMAP(commUint64, uint64_t, fn),        \
          CTRAN_REDOP_FUNCMAP(commFloat16, half, fn),           \
          CTRAN_REDOP_FUNCMAP(commFloat32, float, fn),          \
          CTRAN_REDOP_FUNCMAP(commFloat64, double, fn),         \
          CTRAN_REDOP_FUNCMAP(commBfloat16, __nv_bfloat16, fn), \
  };
#else
#define CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(var, fn)     \
  static const std::unordered_map<                       \
      std::pair<commDataType_t, commRedOp_t>,            \
      const void*,                                       \
      CtranPairHash>                                     \
      var = {                                            \
          CTRAN_REDOP_FUNCMAP(commInt8, int8_t, fn),     \
          CTRAN_REDOP_FUNCMAP(commUint8, uint8_t, fn),   \
          CTRAN_REDOP_FUNCMAP(commInt32, int32_t, fn),   \
          CTRAN_REDOP_FUNCMAP(commUint32, uint32_t, fn), \
          CTRAN_REDOP_FUNCMAP(commInt64, int64_t, fn),   \
          CTRAN_REDOP_FUNCMAP(commUint64, uint64_t, fn), \
          CTRAN_REDOP_FUNCMAP(commFloat16, half, fn),    \
          CTRAN_REDOP_FUNCMAP(commFloat32, float, fn),   \
          CTRAN_REDOP_FUNCMAP(commFloat64, double, fn),  \
  };
#endif

struct CtranTupleHash {
  template <class T1, class T2, class T3>
  std::size_t operator()(const std::tuple<T1, T2, T3>& p) const {
    auto h1 = std::hash<T1>{}(std::get<0>(p));
    auto h2 = std::hash<T2>{}(std::get<1>(p));
    auto h3 = std::hash<T3>{}(std::get<2>(p));
    return h1 ^ h2 ^ h3;
  }
};

#define CTRAN_COLL_INFO(                                                         \
    algoStr, sendbuff, recvbuff, count, datatype, peer, comm, stream)            \
  do {                                                                           \
    CLOGF_SUBSYS(                                                                \
        INFO,                                                                    \
        COLL,                                                                    \
        "{}: opCount {} sendbuff {} recvbuff {} count {} datatype {} peer {} "   \
        "comm {} Ctran {} commHash {:x} commDesc {} [nranks={}, localRanks={}] " \
        "stream={}",                                                             \
        algoStr,                                                                 \
        comm->ctran_->getOpCount(),                                              \
        (void*)sendbuff,                                                         \
        (void*)recvbuff,                                                         \
        count,                                                                   \
        datatype,                                                                \
        peer,                                                                    \
        (void*)comm,                                                             \
        (void*)comm->ctran_.get(),                                               \
        comm->statex_->commHash(),                                               \
        comm->statex_->commDesc(),                                               \
        comm->statex_->nRanks(),                                                 \
        comm->statex_->nLocalRanks(),                                            \
        (void*)stream);                                                          \
  } while (0)

#define CTRAN_HOST_COLL_INFO(                                                    \
    algoStr, sendbuff, recvbuff, count, datatype, peer, comm, ctran, req)        \
  do {                                                                           \
    CLOGF_SUBSYS(                                                                \
        INFO,                                                                    \
        COLL,                                                                    \
        "{}: opCount {} sendbuff {} recvbuff {} count {} datatype {} peer {} "   \
        "comm {} Ctran {} commHash {:x} commDesc {} [nranks={}, localRanks={}] " \
        "req={}",                                                                \
        algoStr,                                                                 \
        ctran->getOpCount(),                                                     \
        (void*)sendbuff,                                                         \
        (void*)recvbuff,                                                         \
        count,                                                                   \
        datatype,                                                                \
        peer,                                                                    \
        (void*)comm,                                                             \
        (void*)ctran,                                                            \
        comm->statex_->commHash(),                                               \
        comm->statex_->commDesc(),                                               \
        comm->statex_->nRanks(),                                                 \
        comm->statex_->nLocalRanks(),                                            \
        (void*)&req);                                                            \
  } while (0)

#define CTRAN_REDCOLL_INFO(                                                  \
    algoStr, sendbuff, recvbuff, count, datatype, redOp, peer, comm, stream) \
  do {                                                                       \
    CLOGF_SUBSYS(                                                            \
        INFO,                                                                \
        COLL,                                                                \
        "{}: opCount {} sendbuff {} recvbuff {} count {} datatype {} "       \
        "redOp {} peer {} comm {} commHash {:x} commDesc {} [nranks={}, "    \
        "localRanks={}] stream={}",                                          \
        algoStr,                                                             \
        comm->ctran_->getOpCount(),                                          \
        (void*)sendbuff,                                                     \
        (void*)recvbuff,                                                     \
        count,                                                               \
        datatype,                                                            \
        redOp,                                                               \
        peer,                                                                \
        (void*)comm,                                                         \
        comm->statex_->commHash(),                                           \
        comm->statex_->commDesc(),                                           \
        comm->statex_->nRanks(),                                             \
        comm->statex_->nLocalRanks(),                                        \
        (void*)stream);                                                      \
  } while (0)

#define CTRAN_RMA_INFO(                                                                                 \
    algoStr,                                                                                            \
    opCount,                                                                                            \
    winOpCount,                                                                                         \
    originBuff,                                                                                         \
    targetDisp,                                                                                         \
    count,                                                                                              \
    datatype,                                                                                           \
    rank,                                                                                               \
    peer,                                                                                               \
    win,                                                                                                \
    comm,                                                                                               \
    stream)                                                                                             \
  do {                                                                                                  \
    CLOGF_SUBSYS(                                                                                       \
        INFO,                                                                                           \
        COLL,                                                                                           \
        "CTRAN-RMA {}: opCount {} winOpCount {} originBuff {} targetDisp {} count {} "                  \
        "datatype {} rank {} peer {} win {} winBase {} comm {} commHash {:x} [nranks={} localRanks={}]" \
        "stream={}",                                                                                    \
        algoStr,                                                                                        \
        opCount,                                                                                        \
        winOpCount,                                                                                     \
        (void*)originBuff,                                                                              \
        targetDisp,                                                                                     \
        count,                                                                                          \
        datatype,                                                                                       \
        rank,                                                                                           \
        peer,                                                                                           \
        (void*)win,                                                                                     \
        win->winBasePtr,                                                                                \
        (void*)comm,                                                                                    \
        comm->statex_->commHash(),                                                                      \
        comm->statex_->nRanks(),                                                                        \
        comm->statex_->nLocalRanks(),                                                                   \
        (void*)stream);                                                                                 \
  } while (0)
