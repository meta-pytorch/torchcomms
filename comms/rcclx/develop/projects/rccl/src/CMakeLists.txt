# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#
# RCCL Source Directory CMakeLists.txt
# This file handles source collection, hipification, and library creation
#==================================================================================================

# When RCCL is included as a subproject (e.g. rocm-systems), CMAKE_SOURCE_DIR points to the
# super-repo root. Use PROJECT_SOURCE_DIR for RCCL-relative paths.
set(RCCL_SOURCE_DIR "${PROJECT_SOURCE_DIR}")

# net_ib_cast: populate NET_IB_CAST_SOURCES and NET_IB_CAST_INCLUDE_DIRS
add_subdirectory(transport/net_ib_cast)

# devcomm: populate DEVCOMM_SOURCES and DEVCOMM_INCLUDE_DIRS
add_subdirectory(devcomm)

# Collect list of all source files
#==================================================================================================
# E.g: find src -type f \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sort
set(SRC_FILES
  allocator.cc
  bootstrap.cc
  ce_coll.cc
  channel.cc
  collectives.cc
  commDump.cc
  mem_manager.cc
  debug.cc
  dev_runtime.cc
  dda_all_reduce_ipc.cu
  dda_reduce_scatter_ipc.cu
  dda_all_gather_ipc.cu
  dda_alltoall_ipc.cu
  device_buffer.cc
  enqueue.cc
  group.cc
  ipc_gpu_barrier.cu
  ipc_init.cu
  ipc_mem_handler.cc
  init.cc
  init_nvtx.cc
  mnnvl.cc
  proxy.cc
  rccl_wrap.cc
  sym_kernels.cc
  transport.cc
  device/all_gather.h
  device/all_reduce.h
  device/alltoall_pivot.h
  device/alltoall_gda.h
  device/alltoallv_gda.h
  device/broadcast.h
  device/common.h
  device/common_kernel.h
  device/op128.h
  device/primitives.h
  device/prims_ll128.h
  device/prims_ll.h
  device/prims_simple.h
  device/reduce.h
  device/reduce_kernel.h
  device/reduce_scatter.h
  device/rccl_metadata.h
  device/sendrecv.h
  device/hierarchical_ag_shuffle.h
  device/common.cu
  device/onerank.cu
  device/network/unpack/unpack_defs.h
  device/network/unpack/unpack.h
  device/symmetric/all_gather.cuh
  device/symmetric/all_gather_gin.cuh
  device/symmetric/all_reduce.cuh
  device/symmetric/data_ops.cuh
  device/symmetric/gin_scratch.h
  device/symmetric/gin_scratch__funcs.h
  device/symmetric/gin_scratch__types.h
  device/symmetric/kernel.cuh
  device/symmetric/primitives.cuh
  device/symmetric/reduce_scatter.cuh
  device/symmetric/reduce_scatter_gin.cuh
  device/symmetric/tma_ptx.cuh
  graph/connect.cc
  graph/paths.cc
  graph/rings.cc
  graph/rings.h
  graph/rome_models.cc
  graph/rome_models.h
  graph/rome_topo_consensus.cc
  graph/rome_topo_consensus.h
  graph/search.cc
  graph/topo.cc
  graph/topo.h
  graph/trees.cc
  graph/tuning.cc
  graph/xml.cc
  graph/xml.h
  include/alloc.h
  include/mem_manager.h
  include/allocator.h
  include/algorithms/CollCommon.h
  include/algorithms/all_reduce/all_reduce_dda.h
  include/algorithms/reduce_scatter/reduce_scatter_dda.h
  include/algorithms/all_gather/all_gather_dda.h
  include/algorithms/alltoall/alltoall_dda.h
  include/alt_rsmi.h
  include/archinfo.h
  include/api_trace.h
  include/argcheck.h
  include/BfdBacktrace.hpp
  include/bitops.h
  include/bootstrap.h
  include/ce_coll.h
  include/ce_fault_inject.h
  include/channel.h
  include/checks.h
  include/collectives.h
  include/coll_net.h
  include/cudawrap.h
  include/comm.h
  include/compiler.h
  include/compiler/gcc.h
  include/compiler/msvc.h
  include/core.h
  include/cpuset.h
# include/cudawrap.h
  include/dda_all_reduce_ipc.h
  include/dda_reduce_scatter_ipc.h
  include/dda_all_gather_ipc.h
  include/dda_alltoall_ipc.h
  include/debug.h
  include/dev_runtime.h
  include/device.h
  include/env.h
  include/device_buffer.h
  include/enqueue.h
  include/gdrwrap.h
  include/git_version.h
  include/graph.h
  include/group.h
  include/gin.h
  include/gin/gin_host.h
  include/gin/gin_host_proxy.h
  include/hip_rocm_version_info.h
  include/ibvcore.h
  include/ibvsymbols.h
  include/ibvwrap.h
  include/info.h
  include/ipc_gpu_barrier.h
  include/ipc_init.h
  include/ipc_init_detail.h
  include/ipc_mem_handler.h
  include/ipcsocket.h
  include/mem_manager.h
  include/mnnvl.h
  include/nccl_common.h
  include/nccl_device.h
  include/nccl_device/net_device.h
  include/net.h
  include/nvmlwrap.h
  include/nvtx.h
  include/nvtx_payload_schemas.h
  include/nvtx_stub.h
  include/p2p.h
  include/param.h
  include/profiler.h
  include/proxy.h
  include/ras.h
  include/rccl_common.h
  include/rccl_vars.h
  include/register.h
  include/register_inline.h
  include/rccl_float8.h
  include/rocmwrap.h
  include/roctx.h
  include/recorder.h
  include/scheduler.h
  include/shm.h
  include/shmutils.h
  include/signals.h
  include/socket.h
  include/strongstream.h
  include/sym_kernels.h
  include/timer.h
  include/transport.h
  include/trees.h
  include/tuner.h
  include/utils.h
  include/mlx5/mlx5dvcore.h
  include/mlx5/mlx5dvsymbols.h
  include/mlx5/mlx5dvwrap.h
  include/ionic/ionicdvcore.h
  include/ionic/ionicdvsymbols.h
  include/ionic/ionicdvwrap.h
  include/nccl_device/comm.h
  include/nccl_device/comm_tmp.h
  include/nccl_device/coop.h
  include/nccl_device/core.h
  include/nccl_device/core_tmp.h
  include/nccl_device/barrier.h
  include/nccl_device/gin.h
  include/nccl_device/gin_tmp.h
  include/nccl_device/gin_barrier.h
  include/nccl_device/gin/gdaki/gin_gdaki.h
  include/nccl_device/gin/gdaki/gin_gdaki_device_host_common.h
  include/nccl_device/gin/gin_device_api.h
  include/nccl_device/gin/gin_device_common.h
  include/nccl_device/gin/gin_device_host_common.h
  include/nccl_device/gin/proxy/gin_proxy.h
  include/nccl_device/gin/proxy/gin_proxy_device_host_common.h
  include/nccl_device/ll_a2a.h
  include/nccl_device/lsa_barrier.h
  include/nccl_device/ptr.h
  include/nccl_device/rccl_ptr.h
  include/nccl_device/utility.h
  include/nccl_device/impl/comm__funcs.h
  include/nccl_device/impl/comm__types.h
  include/nccl_device/impl/core__funcs.h
  include/nccl_device/impl/core__types.h
  include/nccl_device/impl/barrier__funcs.h
  include/nccl_device/impl/barrier__types.h
  include/nccl_device/impl/gin__funcs.h
  include/nccl_device/impl/gin__types.h
  include/nccl_device/impl/gin_barrier__funcs.h
  include/nccl_device/impl/gin_barrier__types.h
  include/nccl_device/impl/ll_a2a__funcs.h
  include/nccl_device/impl/ll_a2a__types.h
  include/nccl_device/impl/lsa_barrier__funcs.h
  include/nccl_device/impl/lsa_barrier__types.h
  include/nccl_device/impl/ptr__funcs.h
  include/nccl_device/impl/ptr__types.h
  include/nccl_device/impl/multimem__funcs.h
  include/nccl_device/impl/reduce_copy__funcs.h
  include/nccl_device/impl/reduce_copy__impl.h
  include/nccl_device/impl/reduce_copy__types.h
  include/nccl_device/impl/vector__funcs.h
  include/nccl_device/impl/vector__types.h
  include/nccl_device/reduce_copy.h
  include/nvtx3/nvToolsExt.h
  include/nvtx3/nvToolsExtCounters.h
  include/nvtx3/nvToolsExtCuda.h
  include/nvtx3/nvToolsExtCudaRt.h
  include/nvtx3/nvToolsExtMem.h
  include/nvtx3/nvToolsExtMemCudaRt.h
  include/nvtx3/nvToolsExtOpenCL.h
  include/nvtx3/nvToolsExtPayload.h
  include/nvtx3/nvToolsExtPayloadHelper.h
  include/nvtx3/nvToolsExtSemanticsCounters.h
  include/nvtx3/nvToolsExtSemanticsScope.h
  include/nvtx3/nvToolsExtSync.h
  include/nvtx3/nvtx3.hpp
  include/nvtx3/nvtxDetail/nvtxExtHelperMacros.h
  include/nvtx3/nvtxDetail/nvtxExtImpl.h
  include/nvtx3/nvtxDetail/nvtxExtImplCounters_v1.h
  include/nvtx3/nvtxDetail/nvtxExtImplMem_v1.h
  include/nvtx3/nvtxDetail/nvtxExtImplMemCudaRt_v1.h
  include/nvtx3/nvtxDetail/nvtxExtImplPayload_v1.h
  include/nvtx3/nvtxDetail/nvtxExtInit.h
  include/nvtx3/nvtxDetail/nvtxExtPayloadHelperInternal.h
  include/nvtx3/nvtxDetail/nvtxExtPayloadTypeInfo.h
  include/nvtx3/nvtxDetail/nvtxExtTypes.h
  include/nvtx3/nvtxDetail/nvtxImpl.h
  include/nvtx3/nvtxDetail/nvtxImplCore.h
  include/nvtx3/nvtxDetail/nvtxImplCuda_v3.h
  include/nvtx3/nvtxDetail/nvtxImplCudaRt_v3.h
  include/nvtx3/nvtxDetail/nvtxImplOpenCL_v3.h
  include/nvtx3/nvtxDetail/nvtxImplSync_v3.h
  include/nvtx3/nvtxDetail/nvtxInit.h
  include/nvtx3/nvtxDetail/nvtxInitDecls.h
  include/nvtx3/nvtxDetail/nvtxInitDefs.h
  include/nvtx3/nvtxDetail/nvtxLinkOnce.h
  include/nvtx3/nvtxDetail/nvtxTypes.h
  include/proxy_trace/proxy_trace.h
  include/plugin/nccl_net.h
  include/plugin/nccl_profiler.h
  include/plugin/nccl_tuner.h
  include/plugin/nccl_env.h
  include/plugin/env/env_v1.h
  include/plugin/plugin.h
  include/plugin/net/net_v6.h
  include/plugin/net/net_v7.h
  include/plugin/net/net_v8.h
  include/plugin/net/net_v9.h
  include/plugin/net/net_v10.h
  include/plugin/net/net_v11.h
  include/plugin/net/net_v12.h
  include/plugin/profiler/net_ib_v1.h
  include/plugin/profiler/net_ib.h
  include/plugin/profiler/net_socket_v1.h
  include/plugin/profiler/net_socket.h
  include/plugin/profiler/profiler_v1.h
  include/plugin/profiler/profiler_v2.h
  include/plugin/profiler/profiler_v3.h
  include/plugin/profiler/profiler_v4.h
  include/plugin/profiler/profiler_v5.h
  include/plugin/profiler/profiler_v6.h
  include/plugin/tuner/tuner_v2.h
  include/plugin/tuner/tuner_v3.h
  include/plugin/tuner/tuner_v4.h
  include/plugin/tuner/tuner_v5.h
  include/plugin/tuner/tuner_v6.h
  include/plugin/nccl_gin.h
  include/plugin/gin/gin_v11.h
  include/plugin/gin/gin_v12.h
  include/plugin/gin/gin_v13.h
  misc/alt_rsmi.cc
  misc/archinfo.cc
  misc/argcheck.cc
  misc/api_trace.c
  misc/api_trace.cc
# misc/cudawrap.cc
# misc/gdrwrap.cc
  misc/ibvsymbols.cc
  misc/ibvwrap.cc
  os/linux_ipcsocket.cc
  misc/mlx5dvsymbols.cc
  misc/mlx5dvwrap.cc
  misc/ionicdvsymbols.cc
  misc/ionicdvwrap.cc
# misc/nvmlwrap.cc
  misc/nvmlwrap_stub.cc
  misc/param.cc
  misc/rocmwrap.cc
  misc/roctx.cc
  misc/recorder.cc
  misc/shmutils.cc
  misc/signals.cc
  misc/socket.cc
  misc/strongstream.cc
  misc/utils.cc
  misc/proxy_trace/proxy_trace.cc
  nccl_device/core.cc
  nccl_device/gin_barrier.cc
  nccl_device/gin_scratch.cc
  nccl_device/ll_a2a.cc
  nccl_device/lsa_barrier.cc
  os/linux.cc
  include/os.h
  include/os/linux.h
  include/os/windows.h
  param/c_api.cc
  param/param.cc
  param/param_registry.cc
  include/param/common.h
  include/param/common_tmp.h
  include/param/param.h
  include/param/param_registry.h
  include/param/param_tmp.h
  include/param/parser_bitset.h
  include/param/parser_common.h
  include/param/parser_default.h
  include/param/parser_enum.h
  include/param/parser_list.h
  include/param/parsers.h
  include/param/utils.h
  include/param/utils_tmp.h
  rma/rma.cc
  rma/rma_ce.cc
  rma/rma_proxy.cc
  rma/rma_proxy_launch.cc
  rma/rma_proxy_progress.cc
  include/rma/rma.h
  include/rma/rma_ce.h
  include/rma/rma_proxy.h
  include/rma/rma_proxy_mem.h
  plugin/net.cc
  plugin/env.cc
  plugin/env/env_v1.cc
  plugin/plugin_open.cc
  plugin/profiler.cc
  plugin/tuner.cc
  plugin/net/net_v6.cc
  plugin/net/net_v7.cc
  plugin/net/net_v8.cc
  plugin/net/net_v9.cc
  plugin/net/net_v10.cc
  plugin/net/net_v11.cc
  plugin/net/net_v12.cc
  plugin/profiler/profiler_v1.cc
  plugin/profiler/profiler_v2.cc
  plugin/profiler/profiler_v3.cc
  plugin/profiler/profiler_v4.cc
  plugin/profiler/profiler_v5.cc
  plugin/profiler/profiler_v6.cc
  plugin/tuner/tuner_v2.cc
  plugin/tuner/tuner_v3.cc
  plugin/tuner/tuner_v4.cc
  plugin/tuner/tuner_v5.cc
  plugin/tuner/tuner_v6.cc
  plugin/tuner/csv_tuner.cc
  plugin/gin.cc
  plugin/gin/gin_v11.cc
  plugin/gin/gin_v12.cc
  plugin/gin/gin_v13.cc
  ras/client.cc
  ras/client_support.cc
  ras/collectives.cc
  ras/peers.cc
  ras/ras.cc
  ras/ras_internal.h
  ras/rasnet.cc
  register/coll_reg.cc
  register/register.cc
  register/sendrecv_reg.cc
  scheduler/symmetric_sched.cc
  scheduler/allgatherv_sched.cc
  transport/coll_net.cc
  transport/generic.cc
  transport/net.cc
  transport/net_ib_limits.h
  transport/net_ib/common.cc
  transport/net_ib/common.h
  transport/net_ib/connect.cc
  transport/net_ib/connect.h
  transport/net_ib/gdr.cc
  transport/net_ib/gin.cc
  transport/net_ib/gin.h
  transport/net_ib/init.cc
  transport/net_ib/p2p.cc
  transport/net_ib/p2p.h
  transport/net_ib/p2p_resiliency.cc
  transport/net_ib/p2p_resiliency.h
  transport/net_ib/p2p_resiliency_recovery.cc
  transport/net_ib/p2p_resiliency_recovery.h
  transport/net_ib/reg.cc
  transport/net_ib_cast/net_ib_cast_inspect.h
  transport/net_ib_cast/net_ib_fault_inject.h
  transport/net_ib/gdaki/gin_host_gdaki.h
  transport/net_socket.cc
  transport/nvls.cc
  transport/p2p.cc
  transport/profiler.cc
  transport/shm.cc
  gin/gin_host.cc
  gin/gin_host_proxy.cc
  include/latency_profiler/CollTrace.h
  include/latency_profiler/CollTraceEvent.h
  include/latency_profiler/CollTraceFunc.h
  include/latency_profiler/CollTraceUtils.h
  include/latency_profiler/EventQueue.h
  misc/latency_profiler/CollTrace.cc
  misc/latency_profiler/CollTraceEvent.cc
  misc/latency_profiler/CollTraceFunc.cc
  misc/latency_profiler/CollTraceUtils.cc
)
# Run this command to get META_FILES:
# find comms/rcclx/develop/meta -type f ! -path "*/scuba/*" ! -path "*/testinfra/*" ! -path "*/tests/*" ! -path "*/lib/*" ! -path "*/lpcoll/*" \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sed 's|^comms/rcclx/develop/\(build/release/hipify/\)\?||' | sort
set(RCCLX_FILES
  meta/algorithms/AlgoInit.cc
  meta/algorithms/AlgoInit.h
  meta/algorithms/AlgoUtils.h
  meta/ctran/BaselineBootstrap.cc
  meta/ctran/BaselineBootstrap.h
  meta/ctran/Hints.cc
  meta/ctran/MetaFactory.cc
  meta/ctran/MetaFactory.h
  meta/ctran/PersistentCollectives.cc
)

# Automatically collect all ctran source files. Glob paths must be absolute
# because the directory lives outside CMAKE_SOURCE_DIR.
file(GLOB_RECURSE CTRAN_SOURCES_ABS
    "${FBCODE_DIR}/comms/ctran/*.cc"
    "${FBCODE_DIR}/comms/ctran/*.cu"
    "${FBCODE_DIR}/comms/ctran/*.h"
    "${FBCODE_DIR}/comms/ctran/*.cuh"
)

# Convert absolute paths to fbcode-relative paths (e.g. "comms/ctran/Ctran.cc")
# so they work with the hipify pipeline which resolves via ${FBCODE_DIR}/${SRC_FILE}.
set(CTRAN_SOURCES "")
foreach(ABS_PATH ${CTRAN_SOURCES_ABS})
    file(RELATIVE_PATH REL_PATH "${FBCODE_DIR}" "${ABS_PATH}")
    list(APPEND CTRAN_SOURCES "${REL_PATH}")
endforeach()



list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/tests/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/benchmarks/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/examples/.*")
# Also exclude nested tests/benchmarks/examples dirs (e.g. comms/ctran/algos/tests/, comms/ctran/gpe/tests/)
list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/.*/tests/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/.*/benchmarks/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/.*/examples/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/issues/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/.*/issues/.*")
if (NOT DEFINED ENABLE_TCPDM OR NOT ENABLE_TCPDM)
    add_compile_definitions(CTRAN_DISABLE_TCPDM)
    list(FILTER CTRAN_SOURCES EXCLUDE REGEX "comms/ctran/backends/tcpdevmem/.*")
endif()
# Set path where CMake will look for header files

# Set path where CMake will look for library files
link_directories(${CONDA_LIB_DIR})


set(thrift_group_list
  ${CONDA_LIB_DIR}/libasync.a
  ${CONDA_LIB_DIR}/libconcurrency.a
  ${CONDA_LIB_DIR}/libthrift-core.a
  ${CONDA_LIB_DIR}/libthriftanyrep.a
  ${CONDA_LIB_DIR}/libthriftcpp2.a
  ${CONDA_LIB_DIR}/libthriftmetadata.a
  ${CONDA_LIB_DIR}/libthriftprotocol.a
  ${CONDA_LIB_DIR}/libthrifttype.a
  ${CONDA_LIB_DIR}/libthrifttyperep.a
  # New in thrift v2026.05.25.00: apache::thrift::dynamic (and its deps) are
  # referenced by the thrift libs above but live in separate archives.
  ${CONDA_LIB_DIR}/libthrift_dynamic_value.a
  ${CONDA_LIB_DIR}/libthrift_path.a
  ${CONDA_LIB_DIR}/libthriftannotation.a
  ${CONDA_LIB_DIR}/libthriftfrozen2.a
  ${CONDA_LIB_DIR}/librpcmetadata.a
  ${CONDA_LIB_DIR}/libruntime.a
  ${CONDA_LIB_DIR}/libserverdbginfo.a
  ${CONDA_LIB_DIR}/libtransport.a
  ${CONDA_LIB_DIR}/libcommon.a
)

# Other thrift service linker flags

# Folly linker flags
# To get this list, activate conda environment with folly installed, then run:
# export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}" && pkg-config --libs --static libfolly

# Other linker flags

# Set HIP/ROCm preprocessor flags

# Set non-FB infra preprocessor flags
# Prevent glog macro conflicts with HIP/AMD headers that define ERROR
# Undefine __HIPCC__ for host-only .cc files that need types guarded by
# #if !defined(__CUDACC__) (hipified to __HIPCC__) in AllReduce/Types.h.
set_source_files_properties(
  ${PROJECT_BINARY_DIR}/hipify/comms/ctran/gpe/CtranGpe.cc
  ${PROJECT_BINARY_DIR}/hipify/comms/ctran/algos/AllReduce/AllReduceRing.cc
  PROPERTIES COMPILE_OPTIONS "-U__HIPCC__"
)

# Files in fbcode/comms/rcclx/develop/meta/ are imported using just their file name
# We will include their parent directories so they are found by CMake
# target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/meta/colltrace)  # disabled: meta/colltrace excluded due to namespace incompatibility

# Files that are imported using their full path have been HIPified
# in fbcode/comms/rcclx/develop/build/release/hipify/<path/from/fbcode>
# To make these headers visible in a way that they can be imported with the full path,
# we need to use fbcode/comms/rcclx/develop/build/release/hipify/ as one of the target include directories.

# Abseil (absl::log / absl::check) is used by
# comms/utils/hrdw_ring_buffer/GpuClockCalibration.cc, which is compiled into
# librccl. Resolved from CONDA_PREFIX via CMAKE_PREFIX_PATH (env), same as fmt.
find_package(absl CONFIG REQUIRED)
find_package(spdlog 1.16.0 CONFIG QUIET)
if(NOT spdlog_FOUND)
  include(FetchContent)
  set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
  set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    spdlog
    SOURCE_DIR "${FBCODE_DIR}/../third-party/spdlog"
  )
  # spdlog treats BUILD_SHARED_LIBS as an override of SPDLOG_BUILD_SHARED.
  set(_rccl_saved_build_shared_libs "${BUILD_SHARED_LIBS}")
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(spdlog)
  set(BUILD_SHARED_LIBS "${_rccl_saved_build_shared_libs}")
  unset(_rccl_saved_build_shared_libs)
endif()



# Run this command to get RCCLX_LIB_FILES:
# find comms/rcclx/develop/meta -type f ! -path "*/tests/*" -path "*/lib/*" \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sort
# find comms/rcclx/develop/meta -type f ! -path "*/tests/*" -path "*/lpcoll/*" \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sort
set(RCCLX_LIB_FILES
  comms/rcclx/develop/meta/lib/CollTraceUtils.cc
  comms/rcclx/develop/meta/lib/CollTraceUtils.h
  comms/rcclx/develop/meta/lib/Common.h
  comms/rcclx/develop/meta/lib/EventQueue.h
  comms/rcclx/develop/meta/lib/ProxyTrace.cc
  comms/rcclx/develop/meta/lib/ProxyTrace.h
# comms/rcclx/develop/meta/lib/RcclxScubaEvent.cc
# comms/rcclx/develop/meta/lib/RcclxScubaEvent.h
# comms/rcclx/develop/meta/lib/ScubaLogger.cc
# comms/rcclx/develop/meta/lib/ScubaLogger.h
  comms/rcclx/develop/meta/lpcoll/low_precision_allgather.cc
  comms/rcclx/develop/meta/lpcoll/low_precision_allgather.h
  comms/rcclx/develop/meta/lpcoll/low_precision_allreduce.cc
  comms/rcclx/develop/meta/lpcoll/low_precision_allreduce.h
  comms/rcclx/develop/meta/lpcoll/low_precision_alltoall.cc
  comms/rcclx/develop/meta/lpcoll/low_precision_alltoall.h
  comms/rcclx/develop/meta/lpcoll/low_precision_buffer_pool.cc
  comms/rcclx/develop/meta/lpcoll/low_precision_buffer_pool.h
  comms/rcclx/develop/meta/lpcoll/low_precision_common.h
  comms/rcclx/develop/meta/lpcoll/low_precision_kernels.cu
  comms/rcclx/develop/meta/lpcoll/low_precision_kernels.h
  comms/rcclx/develop/meta/lpcoll/low_precision_reduce_scatter.cc
  comms/rcclx/develop/meta/lpcoll/low_precision_reduce_scatter.h
  comms/rcclx/develop/meta/lpcoll/low_precision_utility.h
  comms/rcclx/develop/meta/lpcoll/p2p_allgather.cc
  comms/rcclx/develop/meta/lpcoll/p2p_allgather.h
  # find comms/rcclx/develop/meta -type f ! -path "*/tests/*" -path "*/relay/*" \( -name "*.cc" -o -name "*.h" -o -name "*.cu" \) | sort
  comms/rcclx/develop/meta/relay/relay_control.cc
  comms/rcclx/develop/meta/relay/relay_control.h
  comms/rcclx/develop/meta/relay/sharded_relay_all_gather.cc
  comms/rcclx/develop/meta/relay/sharded_relay_all_gather.h
  comms/rcclx/develop/meta/relay/sharded_relay_all_to_all.cc
  comms/rcclx/develop/meta/relay/sharded_relay_all_to_all.h
  comms/rcclx/develop/meta/relay/sharded_relay_allreduce.cc
  comms/rcclx/develop/meta/relay/sharded_relay_allreduce.h
  comms/rcclx/develop/meta/relay/sharded_relay_allreduce_kernels.cu
  comms/rcclx/develop/meta/relay/sharded_relay_allreduce_kernels.h
  comms/rcclx/develop/meta/relay/sharded_relay_graph_scratch.cc
  comms/rcclx/develop/meta/relay/sharded_relay_graph_scratch.h
  comms/rcclx/develop/meta/relay/sharded_relay_lp.cc
  comms/rcclx/develop/meta/relay/sharded_relay_lp.h
  comms/rcclx/develop/meta/relay/sharded_relay_lp_arena.cc
  comms/rcclx/develop/meta/relay/sharded_relay_lp_arena.h
  comms/rcclx/develop/meta/relay/sharded_relay_lp_kernels.cu
  comms/rcclx/develop/meta/relay/sharded_relay_lp_kernels.h
  comms/rcclx/develop/meta/relay/sharded_relay_oneshot.cc
  comms/rcclx/develop/meta/relay/sharded_relay_oneshot.h
  comms/rcclx/develop/meta/relay/sharded_relay_reduce_scatter.cc
  comms/rcclx/develop/meta/relay/sharded_relay_reduce_scatter.h
  comms/rcclx/develop/meta/relay/sharded_relay_route.h
)

# find comms/ctran/ -type f ! -path "*/tests/*" ! -path "*/benchmarks/*" ! -path "*/examples/*" \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh" -o -name "*.cu" \) | sort
set(CTRAN_FILES
  ${CTRAN_SOURCES}
  comms/common/bootstrap/IBootstrap.h
  comms/ctran/bootstrap/ICtranBootstrap.h
  comms/tcp_devmem/transport.h # extra requirement
  comms/utils/MemUtils.h # extra requirement
)

# TODO: Add back comms/common/ files
# Run this command to get COMMON_FILES:
# find comms/utils/ comms/common -type f ! -path "*/tests/*" ! -path "*/benchmarks/*" \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh" -o -name "*.cu" \) | sort
set(COMMON_FILES
  comms/common/algorithms/AlgoFactory.cu
  comms/common/algorithms/AlgoFactory.cuh
  comms/common/algorithms/AlgoUtils.cc
  comms/common/algorithms/AlgoUtils.h
  comms/common/algorithms/all_gather/AlgoAllGather.cu
  comms/common/algorithms/all_gather/AlgoAllGather.cuh
  comms/common/algorithms/all_gather/AllGatherAlgoManager.cu
  comms/common/algorithms/all_gather/AllGatherAlgoManager.h
  comms/common/algorithms/all_gather/all_gather_dda.cuh
  comms/common/algorithms/all_reduce/AlgoAllReduce.cu
  comms/common/algorithms/all_reduce/AlgoAllReduce.cuh
  comms/common/algorithms/all_reduce/AllReduceAlgoManager.cu
  comms/common/algorithms/all_reduce/AllReduceAlgoManager.h
  comms/common/algorithms/all_reduce/all_reduce_dda.cuh
  comms/common/algorithms/all_to_all/AlgoAllToAll.cu
  comms/common/algorithms/all_to_all/AlgoAllToAll.cuh
  comms/common/algorithms/all_to_all/AllToAllAlgoManager.cu
  comms/common/algorithms/all_to_all/AllToAllAlgoManager.h
  comms/common/algorithms/all_to_all/all_to_all_dda.cuh
  comms/common/algorithms/CollCommon.cuh
  comms/common/algorithms/reduce_scatter/AlgoReduceScatter.cu
  comms/common/algorithms/reduce_scatter/AlgoReduceScatter.cuh
  comms/common/algorithms/reduce_scatter/ReduceScatterAlgoManager.cu
  comms/common/algorithms/reduce_scatter/ReduceScatterAlgoManager.h
  comms/common/algorithms/reduce_scatter/reduce_scatter_dda.cuh
  comms/common/AtomicUtils.cuh
  comms/common/BitOps.cuh
  comms/common/CudaWrap.h
  comms/common/DeviceConstants.cuh
  comms/common/IpcGpuBarrier.cu
  comms/common/IpcGpuBarrier.cuh
  comms/common/IpcMemHandler.cc
  comms/common/IpcMemHandler.h
  comms/common/bootstrap/IBootstrap.h
  comms/common/fault_tolerance/Abort.cc
  comms/common/fault_tolerance/Abort.h
  comms/utils/checks.h
  comms/utils/colltrace/AlgoStats.cc
  comms/utils/colltrace/AlgoStats.h
  comms/utils/colltrace/CollMetadata.h
  comms/utils/colltrace/CollMetadataImpl.cc
  comms/utils/colltrace/CollMetadataImpl.h
  comms/utils/colltrace/CollRecord.cc
  comms/utils/colltrace/CollRecord.h
  comms/utils/colltrace/CollTrace.cc
  comms/utils/colltrace/CollTraceEvent.h
  comms/utils/colltrace/CollTrace.h
  comms/utils/colltrace/CollTraceHandle.cc
  comms/utils/colltrace/CollTraceHandle.h
  comms/utils/colltrace/CollTraceInterface.h
  comms/utils/colltrace/CollTracePlugin.h
  comms/utils/colltrace/CollWaitEvent.h
  comms/utils/colltrace/CommLogDataSerialize.cc
  comms/utils/colltrace/CommLogDataSerialize.h
  comms/utils/colltrace/CPUWaitEvent.cc
  comms/utils/colltrace/CPUWaitEvent.h
  comms/utils/colltrace/CudaEventPool.cc
  comms/utils/colltrace/CudaEventPool.h
  comms/utils/colltrace/CudaWaitEvent.cc
  comms/utils/colltrace/CudaWaitEvent.h
  comms/utils/colltrace/DummyCollTraceHandle.h
  comms/utils/colltrace/GenericMetadata.cc
  comms/utils/colltrace/GenericMetadata.h
  comms/utils/colltrace/GraphCollTraceEvent.h
  comms/utils/colltrace/GraphCollTraceHandle.h
  comms/utils/colltrace/GraphCollTraceState.h
  comms/utils/colltrace/GraphCudaWaitEvent.cc
  comms/utils/colltrace/GraphCudaWaitEvent.h
  comms/utils/colltrace/HRDWRingBufferInstantiations.cu
  comms/utils/colltrace/HRDWRingBufferInstantiations.cuh
  comms/utils/colltrace/NetworkPerfMonitor.cc
  comms/utils/colltrace/NetworkPerfMonitor.h
  comms/utils/colltrace/plugins/CommDumpPlugin.cc
  comms/utils/colltrace/plugins/CommDumpPlugin.h
  comms/utils/colltrace/plugins/WatchdogPlugin.cc
  comms/utils/colltrace/plugins/WatchdogPlugin.h
  comms/utils/CommsMaybeChecks.h
  comms/utils/commSpecs.cc
  comms/utils/commSpecs.h
  comms/utils/Conversion.cc
  comms/utils/Conversion.h
  comms/utils/hrdw_ring_buffer/GpuClockCalibration.cc
  comms/utils/hrdw_ring_buffer/GpuClockCalibration.cu
  comms/utils/hrdw_ring_buffer/GpuClockCalibration.h
  comms/utils/CudaChecks.h
  comms/utils/CudaRAII.cc
  comms/utils/CudaRAII.h
  comms/utils/cvars/nccl_baseline_adapter.cc
  comms/utils/cvars/nccl_baseline_adapter.h
  comms/utils/cvars/nccl_cvars.cc
  comms/utils/cvars/nccl_cvars.h
  comms/utils/EnvUtils.cc
  comms/utils/EnvUtils.h
  comms/utils/GraphCaptureSideStream.cc
  comms/utils/GraphCaptureSideStream.h
  comms/utils/InitFolly.cc
  comms/utils/InitFolly.h
  comms/utils/kernels/rng/philox_rng.cuh
  comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh
  comms/utils/logger/alloc.cc
  comms/utils/logger/alloc.h
  comms/utils/logger/BackendTopologyUtil.cc
  comms/utils/logger/BackendTopologyUtil.h
  comms/utils/logger/CommsLogFormatter.cc
  comms/utils/logger/CommsLogFormatter.h
  comms/utils/logger/CudaLog.cc
  comms/utils/logger/CudaLog.h
  comms/utils/logger/DataSink.h
  comms/utils/logger/DataTable.cc
  comms/utils/logger/DataTable.h
  comms/utils/logger/DataTableWrapper.cc
  comms/utils/logger/DataTableWrapper.h
  comms/utils/logger/ErrorStackUtil.cc
  comms/utils/logger/ErrorStackUtil.h
  comms/utils/logger/EventMgr.cc
  comms/utils/logger/EventMgr.h
  comms/utils/logger/EventMgrHelperTypes.cc
  comms/utils/logger/EventMgrHelperTypes.h
  comms/utils/logger/EventsScubaUtil.cc
  comms/utils/logger/EventsScubaUtil.h
  comms/utils/logger/Logger.cc
  comms/utils/logger/Logger.h
  comms/utils/logger/LoggerRuntime.cc
  comms/utils/logger/LoggerRuntime.h
  comms/utils/logger/CommsLogging.cc
  comms/utils/logger/CommsLogging.h
  comms/utils/logger/LoggingFormat.cc
  comms/utils/logger/LoggingFormat.h
  comms/utils/logger/LogTypes.cc
  comms/utils/logger/LogTypes.h
  comms/utils/logger/LogUtils.cc
  comms/utils/logger/LogUtils.h
  comms/utils/logger/NcclScubaSample.cc
  comms/utils/logger/NcclScubaSample.h
  comms/utils/logger/NcclWriterWrapper.cc
  comms/utils/logger/NcclWriterWrapper.h
  comms/utils/logger/ScubaFileUtils.cc
  comms/utils/logger/ScubaFileUtils.h
  comms/utils/logger/ScubaLogger.cc
  comms/utils/logger/ScubaLogger.h
  comms/utils/logger/SpdlogLogger.cc
  comms/utils/logger/SpdlogLogger.h
  comms/utils/MemUtils.cc
  comms/utils/MemUtils.h
  comms/utils/memtrace/GpuMemoryTracker.cc
  comms/utils/memtrace/GpuMemoryTracker.h
  comms/utils/memtrace/McclCudaMemory.cc
  comms/utils/memtrace/McclCudaMemory.h
  comms/utils/memtrace/MemoryTrace.cc
  comms/utils/memtrace/MemoryTrace.h
  comms/utils/PrecisionClock.cc
  comms/utils/PrecisionClock.h
  comms/utils/RankUtils.cc
  comms/utils/RankUtils.h
  comms/utils/StrUtils.h
  comms/utils/trainer/TrainerContext.cc
  comms/utils/trainer/TrainerContext.h
  comms/utils/TypeUtils.h
)


# Prepend src/ to all source files for proper paths from root
list(TRANSFORM SRC_FILES PREPEND "src/")

# Conditional source files based on build options
#==================================================================================================
if(USE_AMDSMI)
  set(SMI_SOURCES
    src/include/amdsmi_wrap.h
    src/misc/amdsmi_wrap.cc
  )
else()
  set(SMI_SOURCES
    src/include/amdsmi_wrap.h
    src/misc/amdsmi_wrap.cc
    src/include/rocm_smi_wrap.h
    src/misc/rocm_smi_wrap.cc
  )
endif()
list(APPEND SRC_FILES ${SMI_SOURCES})
# NET_IB_CAST_SOURCES already carry the src/ prefix (set by transport/net_ib_cast/CMakeLists.txt)
list(APPEND SRC_FILES ${NET_IB_CAST_SOURCES})
# DEVCOMM_SOURCES already carry the src/ prefix (set by devcomm/CMakeLists.txt)
list(APPEND SRC_FILES ${DEVCOMM_SOURCES})

# Hipify source files (copy of source generated into hipify directory)
#==================================================================================================
find_program(hipify-perl_executable hipify-perl)
if(NOT hipify-perl_executable)
  message(FATAL_ERROR "hipify-perl not found")
endif()
# Use PROJECT_BINARY_DIR to keep hipify at project root level
set(HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
set(NCCL_DEVICE_HIP_FILES "")

# Meta hipify blocks, relocated here: they need hipify-perl_executable and
# HIPIFY_DIR (set just above) plus RCCLX_LIB_FILES/COMMON_FILES/CTRAN_FILES.
# All append to HIP_SOURCES, consumed later by add_library(rccl).
# Hipify RCCLX_FILES (relative to develop/)
set(RCCLX_DEV_DIR "${CMAKE_SOURCE_DIR}/../..")
foreach(SRC_FILE ${RCCLX_FILES})
  if (NOT EXISTS ${RCCLX_DEV_DIR}/${SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${RCCLX_DEV_DIR}/${SRC_FILE}")
  endif()
  set(HIP_FILE "${HIPIFY_DIR}/${SRC_FILE}")
  get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)
  add_file_unique(HIP_SOURCES ${HIP_FILE})
  string(REPLACE "\.cuh" "\.h" HIP_FILE ${HIP_FILE})
  string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${HIP_FILE})
  list(APPEND HIP_SOURCES ${HIP_FILE})
  add_custom_command(
    OUTPUT ${HIP_FILE}
    COMMAND mkdir -p ${HIP_FILE_DIR}
            && ${hipify-perl_executable} -quiet-warnings ${RCCLX_DEV_DIR}/${SRC_FILE} -o ${HIP_FILE}
            && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
    MAIN_DEPENDENCY ${RCCLX_DEV_DIR}/${SRC_FILE}
    COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
  )
endforeach()
# Different build rules for files that we want to import with their full path.
# They will be stored in fbcode/comms/rcclx/develop/build/release/hipify/<path/from/fbcode>
foreach(SRC_FILE ${RCCLX_LIB_FILES} ${COMMON_FILES} ${CTRAN_FILES})
  # Check that file exists
  if (NOT EXISTS ${FBCODE_DIR}/${SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${FBCODE_DIR}/${SRC_FILE}")
  endif()

  # Establish hipified copy of the source file
  set(HIP_FILE "${HIPIFY_DIR}/${SRC_FILE}")
  get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)

  # Make sure the file name is unique and there is no duplicate
  add_file_unique(HIP_SOURCES ${HIP_FILE})

  # Convert .cu files to .cpp so that they get processed properly
  if(HIP_FILE MATCHES "\\.cu$" AND NOT HIP_FILE MATCHES "\\.cuh$")
    string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${HIP_FILE})
  endif()
  list(APPEND HIP_SOURCES ${HIP_FILE})

  # Create a custom command to create hipified source code
  if (FAULT_INJECTION)
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -quiet-warnings ${FBCODE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_faults.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${FBCODE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  else()
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -quiet-warnings ${FBCODE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${FBCODE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  endif()
endforeach()

## Loop over each source file to hipify
foreach(SRC_FILE ${SRC_FILES})
  # Check that file exists
  if (NOT EXISTS ${RCCL_SOURCE_DIR}/${SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${RCCL_SOURCE_DIR}/${SRC_FILE}")
  endif()

  # Establish hipified copy of the source file
  set(HIP_FILE "${HIPIFY_DIR}/${SRC_FILE}")
  get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)

  # Make sure the file name is unique and there is no duplicate
  add_file_unique(HIP_SOURCES ${HIP_FILE})

  # Convert .cu files to .cpp so that they get processed properly
  string(REPLACE "\.cuh" "\.h" HIP_FILE ${HIP_FILE})
  string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${HIP_FILE})
  list(APPEND HIP_SOURCES ${HIP_FILE})

  if(SRC_FILE STREQUAL "src/include/nccl_device.h" OR SRC_FILE MATCHES "^src/include/nccl_device/")
    list(APPEND NCCL_DEVICE_HIP_FILES ${HIP_FILE})
  endif()

  # Create a custom command to create hipified source code
  if (FAULT_INJECTION)
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -experimental -quiet-warnings ${RCCL_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_faults.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${RCCL_SOURCE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  else()
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -experimental -quiet-warnings ${RCCL_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${RCCL_SOURCE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  endif()
endforeach()

set(NCCL_DEVICE_HEADER "${PROJECT_BINARY_DIR}/include/rccl/nccl_device.h")
# Copy hip_compat.h as-is (no hipification — contains both CUDA and HIP paths)
set(HIP_COMPAT_SRC "${RCCL_SOURCE_DIR}/src/include/nccl_device/hip_compat.h")
set(HIP_COMPAT_DST "${HIPIFY_DIR}/src/include/nccl_device/hip_compat.h")
add_custom_command(
  OUTPUT ${HIP_COMPAT_DST}
  COMMAND ${CMAKE_COMMAND} -E copy ${HIP_COMPAT_SRC} ${HIP_COMPAT_DST}
  MAIN_DEPENDENCY ${HIP_COMPAT_SRC}
  COMMENT "Copying hip_compat.h (no hipification)"
)
list(APPEND HIP_SOURCES ${HIP_COMPAT_DST})
list(APPEND NCCL_DEVICE_HIP_FILES ${HIP_COMPAT_DST})

# copy_directory_if_different was added in CMake 3.26.  On older toolchains
# fall back to the unconditional copy_directory (same behaviour, just causes
# unnecessary recompilation on no-op rebuilds).
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.26")
  add_custom_command(
    OUTPUT ${NCCL_DEVICE_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/include"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HIPIFY_DIR}/src/include/nccl_device.h" "${PROJECT_BINARY_DIR}/include/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${HIPIFY_DIR}/src/include/nccl_device" "${PROJECT_BINARY_DIR}/include/nccl_device"
    DEPENDS ${NCCL_DEVICE_HIP_FILES}
    VERBATIM
  )
else()
  add_custom_command(
    OUTPUT ${NCCL_DEVICE_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/include"
    COMMAND ${CMAKE_COMMAND} -E copy "${HIPIFY_DIR}/src/include/nccl_device.h" "${PROJECT_BINARY_DIR}/include/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HIPIFY_DIR}/src/include/nccl_device" "${PROJECT_BINARY_DIR}/include/nccl_device"
    DEPENDS ${NCCL_DEVICE_HIP_FILES}
    VERBATIM
  )
endif()
add_custom_target(copy_nccl_device_headers DEPENDS ${NCCL_DEVICE_HEADER})

# Adding custom target to hipify all the source files
# This is required to make sure that all the hipified source files are
# available before compiling the unit tests executable(s)
add_custom_target(hipify_all DEPENDS ${HIP_SOURCES})
# Hipify external comms/ headers (comms/common, comms/utils) that are not part of RCCL source
#==================================================================================================
set(HIPIFY_EXT_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipify_ext")
# Prepend HIPIFY_EXT_DIR to CXX flags so it takes priority over -I fbcode in env CXXFLAGS
set(CMAKE_CXX_FLAGS "-I${HIPIFY_EXT_DIR} ${CMAKE_CXX_FLAGS}")
add_custom_target(hipify_external_headers
  COMMAND ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/hipify_external_headers.sh
    ${hipify-perl_executable} ${FBCODE_DIR} ${HIPIFY_EXT_DIR}
  COMMENT "Hipifying external comms/ headers"
)
add_dependencies(hipify_all hipify_external_headers)


# Serialize copy after full hipify: prevents previous race condition where copying of nccl device headers
# could proceed before hipify_all stage had completed.
add_dependencies(copy_nccl_device_headers hipify_all)

# net_ib/ headers collide with other basenames (common.h, gin.h, p2p.h) and are hipified
# with _tmp suffixes. Create symlinks so that #include "common.h" etc. resolve correctly
# when compiling files in hipify/src/transport/net_ib/.
set(NET_IB_HIPIFY_DIR "${HIPIFY_DIR}/src/transport/net_ib")
set(NET_IB_HEADER_SYMLINKS "")
foreach(_hdr common gin p2p)
  add_custom_command(
    OUTPUT  "${NET_IB_HIPIFY_DIR}/${_hdr}.h"
    DEPENDS "${NET_IB_HIPIFY_DIR}/${_hdr}_tmp.h"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${_hdr}_tmp.h" "${NET_IB_HIPIFY_DIR}/${_hdr}.h"
    COMMENT "Symlinking net_ib/${_hdr}_tmp.h -> ${_hdr}.h"
  )
  list(APPEND NET_IB_HEADER_SYMLINKS "${NET_IB_HIPIFY_DIR}/${_hdr}.h")
endforeach()
add_custom_target(net_ib_header_symlinks DEPENDS ${NET_IB_HEADER_SYMLINKS})


# Generate device/host tables and all the collective functions that are going to be in librccl.so
#==================================================================================================
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if (NOT Python3_FOUND)
  message(FATAL_ERROR "RCCL requires Python3 for generating host/device tables")
endif()

set(GEN_DIR "${HIPIFY_DIR}/gensrc")
set(GEN_SYM_DIR "${GEN_DIR}/symmetric")

if(ONLY_FUNCS)
  message(WARNING "Using ONLY_FUNCS = ${ONLY_FUNCS}. Not meant for release builds.")
endif()

# Execute the python script to generate required collective functions
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${RCCL_SOURCE_DIR}/src/device/generate.py ${GEN_DIR} ${IFC_ENABLED} OFF ${BUILD_LOCAL_GPU_TARGET_ONLY} ${ENABLE_ROCSHMEM} ${ONLY_FUNCS}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_py_result
    ERROR_VARIABLE gen_py_error
)
if (gen_py_result)
  message(SEND_ERROR "Error: ${gen_py_error}")
  message(FATAL_ERROR "${RCCL_SOURCE_DIR}/src/device/generate.py failed")
endif()

if (GENERATE_SYM_KERNELS)
  # Execute the python script to generate required symmetric memory kernels
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${RCCL_SOURCE_DIR}/src/device/symmetric/generate.py ${GEN_SYM_DIR}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_sym_py_result
    ERROR_VARIABLE gen_sym_py_error
  )
  if (gen_sym_py_result)
    message(SEND_ERROR "Error: ${gen_sym_py_error}")
    message(FATAL_ERROR "${RCCL_SOURCE_DIR}/src/device/symmetric/generate.py failed")
  endif()
endif()

# Find the generated files in the output directory
file(GLOB_RECURSE GENERATED_FILES "${GEN_DIR}/*")
# Exclude the per-function "specialized" kernel files (gensrc/specialized/*.cpp
# and gensrc/specialized_files.txt). They are an alternative parallel-build code
# path that re-emits DEFINE_ncclDevFunc for the same ncclDevFunc_* symbols
# already defined by the grouped impl files above; compiling both causes
# "duplicate symbol" errors at device (amdgcn) link time. The grouped impl
# files plus device_table.cpp are the complete, referenced set.
list(FILTER GENERATED_FILES EXCLUDE REGEX "/specialized")


# Append all found generated files to the list
foreach(file ${GENERATED_FILES})
  list(APPEND HIP_SOURCES ${file})
endforeach()

# Specialized kernels are only used by the device linker pipeline;
# always exclude them from the normal HIP compilation to avoid
# duplicate symbols with the aggregate per-collective .cpp files.
list(FILTER HIP_SOURCES EXCLUDE REGEX "/gensrc/specialized/")

# Generate ctran kernel instantiation files
#==================================================================================================
# These files instantiate the kernel templates defined in .cuh files for each data type.
# Without these, the kernel symbols would be undefined at link time.
set(CTRAN_GEN_DIR "${HIPIFY_DIR}/ctran_gensrc")
message(STATUS "Generating ctran kernel instantiation files in ${CTRAN_GEN_DIR}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${FBCODE_DIR}/comms/ctran/algos/genctran.py ${CTRAN_GEN_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE gen_ctran_result
    ERROR_VARIABLE gen_ctran_error
)
if (gen_ctran_result)
  message(SEND_ERROR "Error: ${gen_ctran_error}")
  message(FATAL_ERROR "${FBCODE_DIR}/comms/ctran/algos/genctran.py failed")
endif()

# Find and hipify the generated ctran kernel files
file(GLOB CTRAN_GENERATED_CU_FILES "${CTRAN_GEN_DIR}/*.cu")
list(LENGTH CTRAN_GENERATED_CU_FILES CTRAN_GEN_COUNT)
message(STATUS "Device Linker: ${CTRAN_GEN_COUNT} ctran kernel instantiation files")

foreach(GEN_CU_FILE ${CTRAN_GENERATED_CU_FILES})
  get_filename_component(GEN_CU_FILENAME ${GEN_CU_FILE} NAME)
  # Convert .cu to .cu.cpp for HIP compilation
  string(REPLACE ".cu" ".cu.cpp" HIP_GEN_FILENAME ${GEN_CU_FILENAME})
  set(HIP_GEN_FILE "${CTRAN_GEN_DIR}/${HIP_GEN_FILENAME}")

  # Create a custom command to hipify the generated file
  add_custom_command(
    OUTPUT ${HIP_GEN_FILE}
    COMMAND ${hipify-perl_executable} -quiet-warnings ${GEN_CU_FILE} -o ${HIP_GEN_FILE}
    DEPENDS ${GEN_CU_FILE}
    COMMENT "Hipifying generated ctran kernel: ${GEN_CU_FILENAME} -> ${HIP_GEN_FILENAME}"
  )
  # These TUs exist only to emit __global__ template instantiations, so they
  # cannot go through the host-only rccl target -- the kernel bodies would be
  # stripped and every ncclKernel* symbol would be left undefined. Hand them to
  # the device pipeline instead, which compiles them monolithically. Unlike
  # META_DEVICE_SOURCES this needs no explicit list: genctran.py emits them into
  # a directory of its own, so the glob above is an exact predicate.
  if(ENABLE_DEVICE_LINKER)
    list(APPEND CTRAN_GEN_HIP_FILES ${HIP_GEN_FILE})
  else()
    list(APPEND HIP_SOURCES ${HIP_GEN_FILE})
  endif()
endforeach()

# Device Linker: filter out device sources that are built by the pipeline
#==================================================================================================
if(ENABLE_DEVICE_LINKER)
  # Remove per-collective .cpp files from gensrc/ and any subdirs (replaced by specialized/symmetric pipelines)
  # Leading slash matters: without it this also matches comms/ctran's
  # ctran_gensrc/, whose kernel instantiations are not part of this pipeline.
  list(FILTER HIP_SOURCES EXCLUDE REGEX "/gensrc/.*\\.cpp$")
  # Add back host-only generated files
  list(APPEND HIP_SOURCES "${GEN_DIR}/host_table.cpp")
  if(EXISTS "${GEN_DIR}/device_table.cpp")
    list(APPEND HIP_SOURCES "${GEN_DIR}/device_table.cpp")
  endif()
  # Remove files compiled separately by the device pipeline
  list(FILTER HIP_SOURCES EXCLUDE REGEX "common\\.cu\\.cpp$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "onerank\\.cu\\.cpp$")
  # Path-qualified so this does not also drop src/ras/collectives.cc, which is
  # host-only RAS code and defines rasCollFree.
  list(FILTER HIP_SOURCES EXCLUDE REGEX "/src/collectives\\.cc$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "dda_all_reduce_ipc\\.cu\\.cpp$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "dda_reduce_scatter_ipc\\.cu\\.cpp$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "dda_all_gather_ipc\\.cu\\.cpp$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "dda_alltoall_ipc\\.cu\\.cpp$")
  message(STATUS "Device Linker: filtered HIP_SOURCES (removed device .cpp under gensrc/ + subdirs, common.cu, onerank.cu, collectives.cc, dda_all_reduce_ipc.cu.cpp, dda_reduce_scatter_ipc.cu.cpp, dda_all_gather_ipc.cu.cpp, dda_alltoall_ipc.cu.cpp)")

  # Meta translation units that emit GPU code but are not part of AMD's device
  # pipeline. target_compile_options(rccl PRIVATE --offload-host-only) below
  # makes clang emit each TU's __hip_fatbin_* reference without the device
  # image, so they end up undefined in librccl.so and abort dlopen. Build them
  # monolithically alongside the dda_*/sym_* objects instead. Buck reaches the
  # same result by globbing meta/{relay,lpcoll}/*.cu into their own cpp_library
  # targets -- see get_rccl_relay_kernel_srcs_develop in
  # comms/rcclx/develop/def_build.bzl.
  #
  # This set cannot be derived: some entries contain no literal __global__
  # (they instantiate kernel templates from .cuh headers) and some arrive via
  # CTRAN_FILES rather than the explicit source list above. Nothing enforces
  # it -- a missing entry links cleanly and only surfaces as an undefined
  # __hip_fatbin_* when librccl.so is dlopen'd.
  set(META_DEVICE_SOURCES
    comms/common/algorithms/all_gather/AlgoAllGather.cu.cpp
    comms/common/algorithms/all_reduce/AlgoAllReduce.cu.cpp
    comms/common/algorithms/all_to_all/AlgoAllToAll.cu.cpp
    comms/common/algorithms/reduce_scatter/AlgoReduceScatter.cu.cpp
    comms/ctran/algos/AllGather/AllGatherBrucksFF.cu.cpp
    comms/ctran/algos/AllGather/AllGatherRing.cu.cpp
    comms/ctran/algos/AllGather/StreamedRd/Impl.cu.cpp
    comms/ctran/algos/AllGatherP/DirectImpl.cu.cpp
    comms/ctran/algos/AllGatherP/PipelineImpl.cu.cpp
    comms/ctran/algos/AllToAll/AllToAllPImpl.cu.cpp
    comms/ctran/algos/Barrier.cu.cpp
    comms/ctran/algos/Broadcast/Broadcast.cu.cpp
    comms/ctran/algos/Checksum.cu.cpp
    comms/ctran/algos/RMA/Get.cu.cpp
    comms/ctran/algos/RMA/PutSignal.cu.cpp
    comms/ctran/algos/SendRecv/SendRecv.cu.cpp
    comms/rcclx/develop/meta/lpcoll/low_precision_kernels.cu.cpp
    comms/rcclx/develop/meta/relay/sharded_relay_allreduce_kernels.cu.cpp
    comms/rcclx/develop/meta/relay/sharded_relay_lp_kernels.cu.cpp
    comms/utils/colltrace/HRDWRingBufferInstantiations.cu.cpp
    comms/utils/hrdw_ring_buffer/GpuClockCalibration.cu.cpp
  )
  foreach(_meta_rel IN LISTS META_DEVICE_SOURCES)
    string(REPLACE "." "\\." _meta_re "${_meta_rel}")
    list(FILTER HIP_SOURCES EXCLUDE REGEX "${_meta_re}$")
  endforeach()
  list(LENGTH META_DEVICE_SOURCES _meta_dev_count)
  message(STATUS "Device Linker: ${_meta_dev_count} Meta device sources routed to the device pipeline")
endif()

# Create an initial git_version.cpp file (that will be updated with latest git version)
#==================================================================================================
# Create initial empty file at configure time
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "")

# Add a custom target that always runs at build time to update git version
add_custom_target(update_git_version
  ALL
  COMMAND ${CMAKE_COMMAND} -DRCCL_SOURCE_DIR=${RCCL_SOURCE_DIR} -DRCCL_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${RCCL_SOURCE_DIR}/cmake/scripts/git_version.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp
  COMMENT "Updating git version information"
  VERBATIM
)

list(APPEND HIP_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)

# Set up RCCL library
#==================================================================================================
## Set RCCL source files
add_library(rccl ${HIP_SOURCES})

target_compile_definitions(rccl PRIVATE
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
  SPDLOG_FMT_EXTERNAL
)

# Meta target_* settings, relocated here: they must follow add_library(rccl).
target_include_directories(rccl PRIVATE
  ${CONDA_INCLUDE_DIR}
)
# SMI_LIBRARIES is resolved in the top-level CMakeLists.txt, which explains why
# librccl carries this dependency even though RCCL itself dlopens amd-smi.
target_link_libraries(rccl PRIVATE ${SMI_LIBRARIES})
target_link_libraries(rccl PRIVATE
  ${CONDA_LIB_DIR}/libcomms_tracing_service.a
)
target_link_libraries(rccl PRIVATE
  "$<LINK_GROUP:RESCAN,${thrift_group_list}>"
)
target_link_libraries(rccl PRIVATE
  ${CONDA_LIB_DIR}/libwangle.a
  ${CONDA_LIB_DIR}/libfizz.a
  ${CONDA_LIB_DIR}/libxxhash.a
)
target_link_libraries(rccl PRIVATE
  ${CONDA_LIB_DIR}/libfolly.a
  ${CONDA_LIB_DIR}/libdouble-conversion.a
  ${CONDA_LIB_DIR}/libglog.a
  ${CONDA_LIB_DIR}/libevent.a
  ${CONDA_LIB_DIR}/libz.a
  ${CONDA_LIB_DIR}/libssl.a
  ${CONDA_LIB_DIR}/libcrypto.a
  ${CONDA_LIB_DIR}/libzstd.so
  ${CONDA_LIB_DIR}/libsnappy.a
  ${CONDA_LIB_DIR}/libbz2.a
  ${CONDA_LIB_DIR}/libsodium.a
)
target_link_libraries(rccl PRIVATE
  ${CONDA_LIB_DIR}/libglog.a
  ${CONDA_LIB_DIR}/libgflags.a
  ${CONDA_LIB_DIR}/libboost_context.a
  ${CONDA_LIB_DIR}/libfmt.a
  ${CONDA_LIB_DIR}/libssl.a
  ${CONDA_LIB_DIR}/libcrypto.a
)
target_compile_definitions(rccl PRIVATE __HIP_PLATFORM_AMD__)
target_compile_definitions(rccl PRIVATE USE_ROCM)
target_compile_definitions(rccl PRIVATE __HIPCC__)
target_compile_definitions(rccl PRIVATE GLOG_NO_ABBREVIATED_SEVERITIES)
target_compile_definitions(rccl PRIVATE MOCK_SCUBA_DATA) # removes some Scuba logging in comms/utils/logger.
target_compile_definitions(rccl PRIVATE CTRAN_DISABLE_TCPDM) # TCPDM has strict topology requirements, which is why it isn't supported NCCLX/RCCLX Conda packages. See T239012482.
target_compile_definitions(rccl PRIVATE NVTX_DISABLE) # NVTX is for annotating code for profilers like NSight systems. It's disabled by default for the NCCLX Conda package.
# comms/utils/PrecisionClock.cc finds time/fbclock/fbclock.h via the `-I fbcode`
# in env CXXFLAGS, but this build does not link fbclock_c. Opt out so PrecisionClock
# falls back to std::chrono::system_clock instead of leaving fbclock_init /
# fbclock_gettime_utc undefined in librccl.so (which breaks PyTorch's dlopen).
target_compile_definitions(rccl PRIVATE COMMS_NO_FBCLOCK)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/meta/algorithms)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/meta/ctran)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR})
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/comms/rcclx/develop)

## Set RCCL dependencies
## Ensure git version is updated before building rccl
add_dependencies(rccl update_git_version copy_nccl_device_headers net_ib_header_symlinks)
add_dependencies(rccl hipify_external_headers)

## Set RCCL include directories
target_include_directories(rccl BEFORE PRIVATE ${HIPIFY_EXT_DIR})
target_include_directories(rccl PRIVATE ${PROJECT_BINARY_DIR}/include)        # for generated rccl.h header
target_include_directories(rccl PRIVATE ${PROJECT_BINARY_DIR}/include/rccl)   # for #include "rccl.h"
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src)                    # for hipified headers
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device/network/unpack)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/mlx5)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/nccl_device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/ionic)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/plugin)
target_include_directories(rccl PRIVATE ${NET_IB_CAST_INCLUDE_DIRS})
target_include_directories(rccl PRIVATE ${DEVCOMM_INCLUDE_DIRS})
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/transport)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/transport/net_ib)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/gensrc)
target_include_directories(rccl PRIVATE ${HSA_INCLUDE_PATH})
target_include_directories(rccl PRIVATE ${SMI_INCLUDE_DIR})
target_include_directories(rccl PRIVATE ${ROCMCORE_PATH}/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/comms/rcclx/develop/meta/ctran)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/comms/ctran/utils)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/comms/ctran)

if(DEMANGLE_DIR)
  target_include_directories(rccl PRIVATE ${DEMANGLE_DIR})
endif()
if(ROCTX_ENABLE)
  target_include_directories(rccl PRIVATE ${ROCTRACER_INCLUDE_DIR})
endif()

## Set RCCL compile definitions
if(NCCL_OS_LINUX)
  target_compile_definitions(rccl PRIVATE NCCL_OS_LINUX)
elseif(NCCL_OS_WINDOWS)
  target_compile_definitions(rccl PRIVATE NCCL_OS_WINDOWS)
endif()
if(USE_AMDSMI)
  target_compile_definitions(rccl PRIVATE USE_AMDSMI)
  if(AMDSMI_FABRIC_API)
    target_compile_definitions(rccl PRIVATE AMDSMI_FABRIC_DIRECT)
  endif()
else()
  if(HAVE_ROCM_SMI64CONFIG)
    target_compile_definitions(rccl PRIVATE USE_ROCM_SMI64CONFIG)
  endif()
  if(HAVE_ROCM_SMI_THREAD_ONLY_MUTEX)
    target_compile_definitions(rccl PRIVATE USE_ROCM_SMI_THREAD_ONLY_MUTEX)
  endif()
endif()
if(ENABLE_WARP_SPEED)
  target_compile_definitions(rccl PRIVATE ENABLE_WARP_SPEED)
endif()
if(ENABLE_ROCSHMEM)
  target_compile_definitions(rccl PRIVATE ENABLE_ROCSHMEM)
endif()
if("${hip_version_string}" VERSION_GREATER_EQUAL "7.12.60540")
  target_compile_definitions(rccl PRIVATE HIP_VMM_UNCACHED_MEMORY)
endif()

# ==== rocSHMEM integration (optional) ====
if (ENABLE_ROCSHMEM)
  add_rocshmem_targets()
  # Ensure rocSHMEM is fully built/installed before compiling rccl
  if (TARGET rocshmem_ext)
    add_dependencies(rccl rocshmem_ext)
  endif()

  if (ROCSHMEM_INCLUDE_DIR)
    target_include_directories(rccl PRIVATE ${ROCSHMEM_INCLUDE_DIR})
  endif()

  # Moved to where rocSHMEM target_links
  ## target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE ${IBVERBS})
endif()

if(ROCTX_ENABLE)
  target_compile_definitions(rccl PRIVATE ROCTX_ENABLE)
else()
  target_compile_definitions(rccl PRIVATE NVTX_NO_IMPL)
  target_compile_definitions(rccl PRIVATE NVTX_DISABLE)
endif()
if(TRACE)
  target_compile_definitions(rccl PRIVATE ENABLE_TRACE)
endif()
if(QP_TRACKING)
  target_compile_definitions(rccl PRIVATE ENABLE_QP_TRACKING)
endif()
if(${HIP_CONTIGUOUS_MEMORY})
  target_compile_definitions(rccl PRIVATE HIP_CONTIGUOUS_MEMORY)
  message(STATUS "HIP_CONTIGUOUS_MEMORY enabled")
else()
  message(STATUS "HIP_CONTIGUOUS_MEMORY disabled")
endif()
if("${hip_version_string}" VERSION_GREATER_EQUAL "5.7.31920")
  target_compile_definitions(rccl PRIVATE HIP_UNCACHED_MEMORY)
  message(STATUS "HIP_UNCACHED_MEMORY enabled")
else()
  message(STATUS "HIP_UNCACHED_MEMORY disabled - requires HIP version >= 5.7.31920")
  # keep --hipcc-func-supp on older HIP and compiler
  if(NOT IFC_ENABLED)
    target_compile_options(rccl PRIVATE --hipcc-func-supp)
    message(STATUS "--hipcc-func-supp enabled")
  else()
    message(STATUS "--hipcc-func-supp disabled")
  endif()
endif()
if (HIP_HOST_UNCACHED_MEMORY)
  target_compile_definitions(rccl PRIVATE HIP_HOST_UNCACHED_MEMORY)
  message(STATUS "HIP_HOST_UNCACHED_MEMORY enabled")
else()
  message(STATUS "HIP_HOST_UNCACHED_MEMORY disabled")
endif()
if(HIP_FABRIC_API_FUNC AND HIP_FABRIC_HANDLE_TYPE)
  # HIP_FABRIC_API is used as both a cmake variable (ON/OFF, for status messages below)
  # and a compile definition (bare #define, for #ifdef guards in C++). The names are
  # intentionally identical; the cmake variable is not queried by any downstream cmake logic.
  set(HIP_FABRIC_API ON)
  target_compile_definitions(rccl PRIVATE HIP_FABRIC_API)
  message(STATUS "HIP Fabric API enabled (hipMemImportFromShareableHandle and hipMemFabricHandle_t found)")
else()
  set(HIP_FABRIC_API OFF)
  if(NOT HIP_FABRIC_API_FUNC)
    message(STATUS "hipMemImportFromShareableHandle not found - disabling HIP_FABRIC_API")
  endif()
  if(NOT HIP_FABRIC_HANDLE_TYPE)
    message(STATUS "hipMemFabricHandle_t type not found - disabling HIP_FABRIC_API")
  endif()
endif()
if (BUILD_BFD)
  if (HAVE_BFD)
    target_compile_definitions(rccl PRIVATE HAVE_BFD)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_FLAGS)
    target_compile_definitions(rccl PRIVATE HAVE_DECL_BFD_GET_SECTION_FLAGS)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_VMA)
    target_compile_definitions(rccl PRIVATE HAVE_DECL_BFD_GET_SECTION_VMA)
  endif()
  if (HAVE_TWO_ARG_BFD_SECTION_SIZE)
    target_compile_definitions(rccl PRIVATE HAVE_TWO_ARG_BFD_SECTION_SIZE)
  endif()
endif()
if (IFC_ENABLED)
  target_compile_definitions(rccl PRIVATE USE_INDIRECT_FUNCTION_CALL)
endif()
if(DEMANGLE_DIR)
  target_compile_definitions(rccl PRIVATE "HAVE_CPLUS_DEMANGLE=1")
  target_compile_definitions(rccl PRIVATE "HAVE_DECL_BASENAME=1")
endif()
if(LL128_ENABLED)
  target_compile_definitions(rccl PRIVATE ENABLE_LL128)
endif()
if(CE_ENABLED)
  target_compile_definitions(rccl PRIVATE CE_BATCH_ASYNC_SUPPORTED)
  message(STATUS "CE_BATCH_ASYNC_SUPPORTED compile definition enabled")
endif()

## Set RCCL compile options
if (HAVE_PARALLEL_JOBS)
  target_compile_options(rccl PRIVATE -parallel-jobs=12)
endif()

if (ROCM_VERSION VERSION_GREATER_EQUAL "60200")
  target_compile_options(rccl PRIVATE --offload-compress)    # Compress GPU code at compile time.
  target_link_libraries(rccl PRIVATE --offload-compress)     # Compress GPU code at link time.
  message(STATUS "--offload-compress enabled - ROCm version >= 6.2.0")
else()
  message(STATUS "--offload-compress disabled - ROCm version < 6.2.0")
endif()

target_compile_options(rccl PRIVATE -Werror=uninitialized)
target_compile_options(rccl PRIVATE -Werror=sometimes-uninitialized)
target_compile_options(rccl PRIVATE -Wall)
target_compile_options(rccl PRIVATE -Werror=deprecated-copy-with-user-provided-copy)
target_compile_options(rccl PRIVATE -Wno-format-nonliteral)
target_compile_options(rccl PRIVATE -Wno-unused-function)
if(ENABLE_DEVICE_LINKER)
  target_compile_definitions(rccl PRIVATE RCCL_DEVICE_LINKER)
  target_compile_options(rccl PRIVATE --offload-host-only)
else()
  target_compile_options(rccl PRIVATE -fgpu-rdc)
endif()

# Enable SSE4.2 for folly F14 CRC intrinsics (host code only) to match libfolly.a
# build. Required in both device-linker modes: build_rcclx.sh compiles folly with
# -msse4.2, so libfolly.a only defines F14LinkCheck<SimdAndCrc>. Without the flag
# these TUs select F14IntrinsicsMode::Simd and reference a F14LinkCheck<Simd>
# that nothing defines -- a link check folly emits deliberately to catch exactly
# this mismatch, and which surfaces as an undefined symbol at dlopen.
target_compile_options(rccl PRIVATE -Xarch_host -msse4.2)

if(QUIET_WARNINGS)
  target_compile_options(rccl PRIVATE -Wno-invalid-offsetof)
  target_compile_options(rccl PRIVATE -Wno-unused-result)
  target_compile_options(rccl PRIVATE -Wno-macro-redefined)
  target_compile_options(rccl PRIVATE -Wno-unused-label)
  target_compile_options(rccl PRIVATE -Wno-unused-variable)
  target_compile_options(rccl PRIVATE -Wno-unused-private-field)
  target_compile_options(rccl PRIVATE -Wno-null-conversion)
  target_compile_options(rccl PRIVATE -Wno-missing-braces)
endif()

## Set RCCL compile and linker options for unit tests and code coverage
if(ENABLE_CODE_COVERAGE)
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    message(FATAL_ERROR "Code coverage is enabled, but the build type is '${CMAKE_BUILD_TYPE}'. "
                        "Code coverage requires 'Debug' build types to expose internal symbols. "
                        "Please set CMAKE_BUILD_TYPE to 'Debug' and reconfigure.")
  endif()

  message(STATUS "Code coverage is enabled with build type '${CMAKE_BUILD_TYPE}'.")

  target_compile_options(rccl PRIVATE
    -fvisibility=default
    "SHELL:-Xarch_host -fprofile-instr-generate"
    "SHELL:-Xarch_host -fcoverage-mapping")

  set(COVERAGE_SHARED_LINKER_FLAGS
    -fprofile-generate
    -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN
  )

  set(COVERAGE_EXE_LINKER_FLAGS
    -fprofile-generate
    -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN/../lib
  )

  target_link_options(rccl PRIVATE ${COVERAGE_SHARED_LINKER_FLAGS})
  target_link_options(rccl PRIVATE ${COVERAGE_EXE_LINKER_FLAGS})

  # Expose explicit profile-flush wrappers so test harnesses that bypass
  # atexit() handlers (e.g. ProcessIsolatedTestRunner using _exit()) can
  # still flush librccl's private LLVM profile runtime instance.
  target_sources(rccl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/misc/coverage_flush.cc)
elseif(BUILD_TESTS) # Enable default/hidden visibility based on build type and ROCM_VERSION
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(rccl PRIVATE -fvisibility=default)
  else()
    target_compile_options(rccl PRIVATE -fvisibility=hidden)
  endif()
else()  # Enable hidden visibility for library without tests/code coverage enabled
  target_compile_options(rccl PRIVATE -fvisibility=hidden)
endif()

if (HAVE_KERNARG_PRELOAD)
  target_compile_options(rccl PRIVATE -mllvm --amdgpu-kernarg-preload-count=16)
endif()

if (REPORT_KERNEL_RESOURCE_USE)
  target_link_options(rccl PRIVATE -Rpass-analysis=kernel-resource-usage)
endif()

if (DUMP_ASM) # Save temporary files from kernel compilation
  message(STATUS "Disassembling librccl.so to asm")
  # Maintain symbols but without changing code.  Keep additional data in dwarf section of binary.
  target_compile_options(rccl PRIVATE -gline-tables-only)
  set(OBJ_DUMP ${ROCM_PATH}/llvm/bin/llvm-objdump)

  add_custom_command(TARGET rccl POST_BUILD
    COMMENT "Disassembling RCCL library"
    COMMAND /bin/bash -c "${OBJ_DUMP} --offload-fatbin librccl.so"
    VERBATIM
  )
  foreach(GPUARCH ${GPU_TARGETS})
    add_custom_command(TARGET rccl POST_BUILD
        COMMENT "Disassembling RCCL library to dump assembly for ${GPUARCH}"
        COMMAND /bin/bash -c "${OBJ_DUMP} -d -l --source --symbolize-operands librccl.so.0.hipv4-amdgcn-amd-amdhsa--${GPUARCH} > librccl.${GPUARCH}.s"
        VERBATIM
    )
  endforeach()
endif()

## NOTE: This is currently being handled by rocm-cmake, however may need to be re-enabled in the future
#foreach(target ${GPU_TARGETS})
#  target_compile_options(rccl PRIVATE --offload-arch=${target})
#endforeach()

if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rccl PRIVATE -fsanitize=address)
  target_link_options(rccl PRIVATE -fsanitize=address -shared-libasan)
  if(DEFINED ASAN_RUNTIME_DIR)
    target_link_options(rccl PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
  endif()
endif()
if(TIMETRACE)
  target_compile_options(rccl PRIVATE -ftime-trace)
endif()
if (FAULT_INJECTION)
  target_compile_definitions(rccl PRIVATE ENABLE_FAULT_INJECTION)
  message(STATUS "Fault injection enabled")
endif()

if (NOT DWORDX4_INTRINSICS)
  message(STATUS "Force disable dwordx4 intrinsics")
  target_compile_definitions(rccl PRIVATE DWORDX4_INTRINSICS_FORCE_OFF)
endif()
## Set RCCL linked library directories
target_link_directories(rccl PRIVATE ${SMI_LIB_DIR})

if (ROCM_VERSION VERSION_GREATER_EQUAL "60100")
    option(RCCL_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
else()
    if(RCCL_ROCPROFILER_REGISTER)
        message(AUTHOR_WARNING "RCCL_ROCPROFILER_REGISTER is not valid option for ROCm < 6.2. Current ROCm version: ${ROCM_VERSION}")
    endif()
    set(RCCL_ROCPROFILER_REGISTER OFF CACHE BOOL "" FORCE)
endif()
if(RCCL_ROCPROFILER_REGISTER)
  find_package(rocprofiler-register REQUIRED)
  target_compile_definitions(rccl PRIVATE RCCL_ROCPROFILER_REGISTER=1)
  target_link_libraries(
      rccl PRIVATE rocprofiler-register::rocprofiler-register)
endif()

# Device Linker: include pipeline and wire up dependencies
# (placed after all target_compile_definitions so DeviceLinker.cmake
#  can read the rccl target's definitions directly)
#==================================================================================================
if(ENABLE_DEVICE_LINKER)
  include(${PROJECT_SOURCE_DIR}/cmake/DeviceLinker.cmake)
  add_dependencies(rccl device_linker_build)
  target_link_libraries(rccl PRIVATE ${DEVICE_LINKER_OBJECTS})
endif()

## Set RCCL linked libraries
if (HAVE_BFD)
  target_link_libraries(rccl PRIVATE bfd)
  if(HAVE_IBERTY)
    target_link_libraries(rccl PRIVATE iberty z)
  endif()
endif()
if (ROCTX_ENABLE)
  target_link_libraries(rccl PRIVATE ${ROCTX_LIB})
endif()
if(NOT ENABLE_DEVICE_LINKER)
  target_link_libraries(rccl PRIVATE   -fgpu-rdc)             # Required when linking relocatable device code
endif()
target_link_libraries(rccl PRIVATE   Threads::Threads)
target_link_libraries(rccl INTERFACE hip::host)
if(ENABLE_DEVICE_LINKER)
  # Device code is compiled separately by the DeviceLinker pipeline.
  # Link hip::host for the runtime library but add -x hip manually so HIP
  # builtins (warpSize, etc.) are available. Crucially, do NOT propagate
  # --offload-arch from hip::device — that would make the compiler try to
  # compile device code in host-only .cc files.
  target_link_libraries(rccl PRIVATE hip::host)
  target_compile_options(rccl PRIVATE -x hip)
  # api_trace.c exists only to include api_trace.h under strict C rules; global
  # -x hip makes Clang treat it as C++ and hits #error there. Per-file -x c is
  # appended after target flags so this unit is compiled as C (same driver).
  set_source_files_properties(
    "${HIPIFY_DIR}/src/misc/api_trace.c"
    PROPERTIES
      COMPILE_OPTIONS "-x;c"
  )
else()
  target_link_libraries(rccl PRIVATE hip::device)
endif()
target_link_libraries(rccl PRIVATE   dl)
# 128-bit atomics (__atomic_load_16/__atomic_store_16, emitted as libcalls on
# x86_64) are not lock-free, so the compiler lowers them to libatomic. Without
# this librccl.so has undefined __atomic_*_16 and PyTorch's dlopen fails.
target_link_libraries(rccl PRIVATE   atomic)
target_link_libraries(rccl PRIVATE   ${SMI_LIBRARIES})
# librocm-core provides getROCmVersion() for reporting the runtime ROCm version.
if(ROCM_MAJOR_VERSION GREATER_EQUAL 6)
  target_link_libraries(rccl PRIVATE -L${ROCM_PATH}/lib -lrocm-core)
endif()
# Wrap fmt in $<BUILD_INTERFACE:...> so it's only used while building rccl
# and is omitted from the exported rccl-targets export set. Without this,
# a FetchContent-built fmt-header-only target leaks into rccl's link
# interface and breaks install(EXPORT) for static builds.
target_link_libraries(rccl PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)
target_link_libraries(rccl PRIVATE spdlog::spdlog)
target_link_libraries(rccl PRIVATE absl::log absl::check)  # GpuClockCalibration.cc
if(ENABLE_ROCSHMEM)
  target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE ${IBVERBS})
  # librocshmem.a is compiled with -fgpu-rdc (relocatable device code).
  # The __hip_gpubin_handle_* symbols referenced by __hip_module_ctor in the
  # librocshmem.a objects are only defined by the RDC device-link step.
  # --hip-link tells amdclang++ to perform that step as part of the final
  # librccl.so link; -fgpu-rdc marks the link as RDC-mode.  --offload-arch
  # flags use RCCL's GPU_TARGETS (stripped of feature suffixes like :xnack-)
  # so the device-link covers every arch RCCL is built for.
  # RCCL's own device code (already embedded in device_build/common.o by the
  # custom device-linker pipeline) is NOT compiled with -fgpu-rdc and is
  # therefore unaffected by this device-link step.
  set(_rocshmem_offload_arch_flags "")
  foreach(_gpu_raw ${GPU_TARGETS})
    string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
    list(APPEND _rocshmem_offload_arch_flags "--offload-arch=${_gpu}")
  endforeach()
  list(REMOVE_DUPLICATES _rocshmem_offload_arch_flags)
  target_link_options(rccl PRIVATE -fgpu-rdc --hip-link ${_rocshmem_offload_arch_flags})
endif()

## Set RCCL link options
## Find out available memory
execute_process(
  COMMAND bash "-c" "test -r /sys/fs/cgroup/memory.max && cat /sys/fs/cgroup/memory.max || echo"
  OUTPUT_VARIABLE memory_max_string)
if (${memory_max_string} MATCHES "^[0-9]+")
  math(EXPR detected_memory_gb "${memory_max_string} / (1024 * 1024 * 1024)")
else()
  execute_process(
    COMMAND bash "-c" "free | grep -o '[[:digit:]]*' | head -1"
    OUTPUT_VARIABLE memory_max_string)
  ## memory_max_string holds the free memory in KB
  if (${memory_max_string} MATCHES "^[0-9]+")
    math(EXPR detected_memory_gb "${memory_max_string} / (1024 * 1024)") ## KB to GB conversion
  else()
    cmake_host_system_information(RESULT memory_max_string QUERY AVAILABLE_PHYSICAL_MEMORY )
    math(EXPR detected_memory_gb "${memory_max_string} / 1024")
  endif()
endif()
set(RCCL_MAX_MEMORY ${detected_memory_gb} CACHE STRING "Maximum memory (in GB) during linking")
set(RCCL_MEMORY_PER_LINK_JOB 32 CACHE STRING "Memory (in GB) reserved for each linker job")
# Use the minimum of detected memory and user-specified memory
if (RCCL_MAX_MEMORY LESS detected_memory_gb)
  set(memory_in_gb ${RCCL_MAX_MEMORY})
else()
  set(memory_in_gb ${detected_memory_gb})
endif()

## Reserve 16GB for each linker job. Limit max number of linker jobs to 16
if (HAVE_PARALLEL_JOBS)
  math(EXPR num_linker_jobs "(${memory_in_gb} + ${RCCL_MEMORY_PER_LINK_JOB} - 1) / ${RCCL_MEMORY_PER_LINK_JOB}")
  if (${num_linker_jobs} GREATER_EQUAL "16")
    set(num_linker_jobs "16")
  elseif (${num_linker_jobs} LESS_EQUAL "0")
    set(num_linker_jobs "1")
  endif()
  message(STATUS "Use ${num_linker_jobs} jobs for linking")
  target_link_options(rccl PRIVATE -parallel-jobs=${num_linker_jobs})       # Use multiple threads to link
endif()
if(BUILD_ADDRESS_SANITIZER)
  target_link_options(rccl PRIVATE -fuse-ld=lld)
  if(NOT ENABLE_DEVICE_LINKER)
    ## ---- ASAN device-link acceleration ----------------------------------------
    ## With -fgpu-rdc, the offload linker merges device bitcode from all object
    ## files and performs full LTO (optimization + codegen) for each GPU target.
    ## --lto-O0 + --lto-CGO0: skip LTO optimization and backend optimization.
    ## --------------------------------------------------------------------------
    target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-O0")
    target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-CGO0")
  endif()
endif()
if(CMAKE_BUILD_SUBTYPE MATCHES "DebugFast" AND NOT ENABLE_DEVICE_LINKER)
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-O0")
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-CGO0")
endif()
if(TIMETRACE)
  target_link_options(rccl PRIVATE -ftime-trace)
endif()

if(NOT BUILD_SHARED_LIBS)
  message(STATUS "Building static RCCL library")
else()
  message(STATUS "Building shared RCCL library")
endif()
if (HAVE_KERNARG_PRELOAD AND NOT ENABLE_DEVICE_LINKER)
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker -mllvm=-amdgpu-kernarg-preload-count=16")
endif()

## Track linking time
set_property(TARGET rccl PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

## Setup librccl.so version
rocm_set_soversion(rccl "1.0")

if(NOT BUILD_SHARED_LIBS)
  # To create a static lib with `-fgpu-rdc`, you need `--emit-static-lib` and `--hip-link`.
  # You also need to invoke amdclang++ again to trigger GPU code generation.
  set(static_link_flags
    ${CXXFLAGS}
    --hip-link
    -fgpu-rdc
    --emit-static-lib
  )

  # Find all the libraries we need to link at link time to include them in the clang link
  # command line.
  get_target_property(rccl_libs rccl LINK_LIBRARIES)
  foreach(target ${rccl_libs})
    if(TARGET ${target})
      get_target_property(location ${target} LOCATION)
      if(location)
        LIST(APPEND static_link_flags -l${location})
      endif()
    endif()
  endforeach()

  foreach(target ${GPU_TARGETS})
    list(APPEND static_link_flags --offload-arch=${target})
  endforeach()
  list(JOIN static_link_flags " " flags_str)

  # Invoking amdclang++ this way will produce a static archive, so just override ARCHIVE_CREATE.
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_CXX_COMPILER> ${flags_str} -o <TARGET> <OBJECTS>")
endif()

# Build RCCL RAS client executable
#==================================================================================================
message("Building rccl RAS client executable")

add_executable(rcclras "${HIPIFY_DIR}/src/ras/client.cc")

target_include_directories(rcclras PRIVATE ${PROJECT_BINARY_DIR}/include)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src/include)

target_link_libraries(rcclras PRIVATE hip::host)
target_link_libraries(rcclras PRIVATE dl)

if(BUILD_SHARED_LIBS)
  target_link_libraries(rcclras PRIVATE rccl hip::device)
else()
  add_dependencies(rcclras rccl)
  target_link_libraries(rcclras PRIVATE dl rt -lrccl -L${CMAKE_CURRENT_BINARY_DIR} -lamdhip64 -L${ROCM_PATH}/lib)
endif()

# Export HIPIFY_DIR to parent scope for use in install rules
set(HIPIFY_DIR ${HIPIFY_DIR} PARENT_SCOPE)
