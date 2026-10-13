#!/usr/bin/env bash
# NVCC architecture selection shared by the comms build scripts, so every
# library in a build gets the same fatbins. Source this file; it defines
# functions only.

# Appends b200 to a comma-separated NVCC_ARCH list when the CUDA toolkit at
# $2 is 12.8 or newer (the first release that can target sm_100a).
nvcc_arch_with_b200() {
  local arch_list="$1"
  local cuda_home="$2"
  local cuda_version cuda_major cuda_minor
  cuda_version=$("${cuda_home}/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
  cuda_major=$(echo "$cuda_version" | cut -d. -f1)
  cuda_minor=$(echo "$cuda_version" | cut -d. -f2)
  if [[ "$cuda_major" -gt 12 ]] || [[ "$cuda_major" -eq 12 && "$cuda_minor" -ge 8 ]]; then
    arch_list="${arch_list},b200"
  fi
  printf '%s\n' "$arch_list"
}

# Prints the -gencode flags for a comma-separated NVCC_ARCH list.
nvcc_gencode_from_arch() {
  local arch_list="$1"
  local arch_gencode=""
  local arch_array arch
  IFS=',' read -ra arch_array <<< "$arch_list"
  for arch in "${arch_array[@]}"; do
    case "$arch" in
      "p100") arch_gencode="$arch_gencode -gencode=arch=compute_60,code=sm_60" ;;
      "v100") arch_gencode="$arch_gencode -gencode=arch=compute_70,code=sm_70" ;;
      "a100") arch_gencode="$arch_gencode -gencode=arch=compute_80,code=sm_80" ;;
      "h100") arch_gencode="$arch_gencode -gencode=arch=compute_90,code=sm_90" ;;
      "b200") arch_gencode="$arch_gencode -gencode=arch=compute_100a,code=sm_100a" ;;
      "b300") arch_gencode="$arch_gencode -gencode=arch=compute_103a,code=sm_103a" ;;
    esac
  done
  printf '%s\n' "$arch_gencode"
}
