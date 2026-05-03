#!/bin/bash

export PAPI_CUDA_TEST_QUIET=1    # Comment this line to see standard output from tests

evt_names=("cuda:::dram__bytes_read.sum:device=0" \
           "cuda:::sm__cycles_active.sum:device=0" \
           "cuda:::smsp__warps_launched.sum:device=0")

multi_gpu_evt_names=("cuda:::dram__bytes_read.sum" \
                     "cuda:::sm__cycles_active.sum" \
                     "cuda:::smsp__warps_launched.sum")

multi_pass_evt_name="cuda:::gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed:device=0"

concurrent_evt_names=("cuda:::sm__cycles_active.sum:device=" \
                      "cuda:::sm__cycles_elapsed.max:device=")

make test_multipass_event_fail
echo -e "Running: \e[36m./test_multipass_event_fail\e[0m" "${evt_names[@]}" $multi_pass_evt_name
./test_multipass_event_fail "${evt_names[@]}" $multi_pass_evt_name
echo -e "-------------------------------------\n"

make test_multi_read_and_reset
echo -e "Running: \e[36m./test_multi_read_and_reset\e[0m" "${evt_names[@]}"
./test_multi_read_and_reset "${evt_names[@]}"
echo -e "-------------------------------------\n"

make test_2thr_1gpu_not_allowed
echo -e "Running: \e[36m./test_2thr_1gpu_not_allowed\e[0m" "${evt_names[@]}"
./test_2thr_1gpu_not_allowed "${evt_names[@]}"
echo -e "-------------------------------------\n"

make HelloWorld
echo -e "Running: \e[36m./HelloWorld\e[0m" "${evt_names[@]}"
./HelloWorld "${evt_names[@]}"
echo -e "-------------------------------------\n"

make HelloWorld_noCuCtx
echo -e "Running: \e[36m./HelloWorld_noCuCtx\e[0m" "${evt_names[@]}"
./HelloWorld_noCuCtx "${evt_names[@]}"
echo -e "-------------------------------------\n"

make simpleMultiGPU
echo -e "Running: \e[36m./simpleMultiGPU\e[0m" "${multi_gpu_evt_names[@]}"
./simpleMultiGPU "${multi_gpu_evt_names[@]}"
echo -e "-------------------------------------\n"

make simpleMultiGPU_noCuCtx
echo -e "Running: \e[36m./simpleMultiGPU_noCuCtx\e[0m" "${multi_gpu_evt_names[@]}"
./simpleMultiGPU_noCuCtx "${multi_gpu_evt_names[@]}"
echo -e "-------------------------------------\n"

make pthreads_noCuCtx
echo -e "Running: \e[36m./pthreads_noCuCtx\e[0m" "${multi_gpu_evt_names[@]}"
./pthreads_noCuCtx "${multi_gpu_evt_names[@]}"
echo -e "-------------------------------------\n"

make pthreads
echo -e "Running: \e[36m./pthreads\e[0m" "${multi_gpu_evt_names[@]}"
./pthreads "${multi_gpu_evt_names[@]}"
echo -e "-------------------------------------\n"

make cudaOpenMP
echo -e "Running: \e[36m./cudaOpenMP\e[0m" "${multi_gpu_evt_names[@]}"
./cudaOpenMP "${multi_gpu_evt_names[@]}"
echo -e "-------------------------------------\n"

make cudaOpenMP_noCuCtx
echo -e "Running: \e[36m./cudaOpenMP_noCuCtx\e[0m" "${multi_gpu_evt_names[@]}"
./cudaOpenMP_noCuCtx "${multi_gpu_evt_names[@]}"
echo -e "-------------------------------------\n"

make concurrent_profiling_noCuCtx
echo -e "Running: \e[36m./concurrent_profiling_noCuCtx\e[0m" "${concurrent_evt_names[@]}"
./concurrent_profiling_noCuCtx "${concurrent_evt_names[@]}"
echo -e "-------------------------------------\n"

make concurrent_profiling
echo -e "Running: \e[36m./concurrent_profiling\e[0m" "${concurrent_evt_names[@]}"
./concurrent_profiling "${concurrent_evt_names[@]}"
echo -e "-------------------------------------\n"

# Finalize tests
unset PAPI_CUDA_TEST_QUIET
