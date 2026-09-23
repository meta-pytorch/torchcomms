# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Exercise the real RCCL shared-buffer lifecycle with CPU allocation stubs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

STUBS = r"""
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
using ncclResult_t = int;
constexpr int ncclSuccess = 0, ncclInternalError = 1, ncclInvalidArgument = 2;
constexpr int hipDeviceMallocFinegrained = 0, hipDeviceMallocDefault = 0;
constexpr int ncclMemPersist = 0;
constexpr int NCCL_SHARED_STEPS = 16;
static int lazy = 0, allocations = 0, releases = 0, cumem = 0;
int ncclParamLazyNetSharedBuffers() { return lazy; }
int ncclCuMemEnable() { return cumem; }
#define WARN(...) ((void)0)
#define NCCLCHECK(call) do { int code = (call); if (code) return code; } while (0)
struct ncclIpcDesc { int marker; };
struct ncclProxySharedP2p {
  int refcount = 0;
  int64_t size = 0;
  char* cudaBuff = nullptr;
  char* hostBuff = nullptr;
  ncclIpcDesc ipcDesc = {};
};
struct ncclProxyPeer { ncclProxySharedP2p send, recv; };
struct ncclProxyProgressState { ncclProxyPeer** localPeers = nullptr; };
struct ncclProxyState {
  ncclProxyProgressState progressState;
  int tpLocalnRanks = 2, p2pChunkSize = 8;
  void* memManager = nullptr;
};
struct ncclProxyConnection { int sameProcess = 1; };
template<class T> int ncclCalloc(T** out, size_t n) {
  *out = static_cast<T*>(calloc(n, sizeof(T))); return *out ? 0 : 1;
}
int ncclCudaCalloc(char** out, size_t n, void*, int, int) {
  ++allocations; *out = static_cast<char*>(calloc(n, 1)); return *out ? 0 : 1;
}
int ncclCudaHostCalloc(char** out, size_t n) {
  return ncclCudaCalloc(out, n, nullptr, ncclMemPersist, 0);
}
int ncclP2pAllocateShareableBuffer(size_t n, int, ncclIpcDesc* desc, void** out,
                                 int, void* memManager) {
  desc->marker = 47;
  return ncclCudaCalloc(reinterpret_cast<char**>(out), n, memManager, ncclMemPersist, 0);
}
int ncclP2pFreeShareableBuffer(ncclIpcDesc*) { return 0; }
int ncclCudaFree(char* ptr, void*) { assert(ptr); ++releases; free(ptr); return 0; }
int ncclCudaHostFree(char* ptr) { return ncclCudaFree(ptr, nullptr); }
int ncclAtomicRefCountDecrement(int* count) { return --*count; }
"""

TESTS = r"""
int main() {
  for (int mode : {0, 1}) for (int device : {0, 1})
  for (int useCuMem : {0, 1}) for (int direction : {0, 1}) {
    cumem = useCuMem;
    lazy = mode; allocations = releases = 0;
    ncclProxyState p; ncclProxyConnection c;
    assert(sharedNetBuffersInit(&p, device, 0, direction, 1, 4, nullptr, nullptr, nullptr, nullptr) == 0);
    assert(allocations == (mode ? 0 : 1));
    auto* state = direction == 0 ? &p.progressState.localPeers[0]->send : &p.progressState.localPeers[0]->recv;
    assert(state->refcount == 1 && state->size == 512);
    // The metadata-only reservation must be destructible without a connection.
    assert(sharedNetBuffersDestroy(&p, 0, direction, &c) == 0);
    assert(p.progressState.localPeers == nullptr && releases == allocations);

    allocations = releases = 0;
    assert(sharedNetBuffersInit(&p, device, 0, direction, 1, 4, nullptr, nullptr, nullptr, nullptr) == 0);
    char* gpu = nullptr; char* cpu = nullptr; int size = 0;
    assert(sharedNetBuffersInit(&p, device, 0, direction, 1, 4, &gpu, &cpu, &size, nullptr) == 0);
    assert(allocations == 1 && cpu && gpu == cpu && size == 512);
    char* first = cpu;
    assert(sharedNetBuffersInit(&p, device, 0, direction, 1, 4, &gpu, &cpu, &size, nullptr) == 0);
    assert(allocations == 1 && cpu == first);
    assert(sharedNetBuffersDestroy(&p, 0, direction, &c) == 0 && releases == 0);
    assert(sharedNetBuffersDestroy(&p, 0, direction, &c) == 0 && releases == 0);
    assert(sharedNetBuffersDestroy(&p, 0, direction, &c) == 0 && releases == 1);
    assert(p.progressState.localPeers == nullptr);
  }
  lazy = 1; allocations = releases = 0;
  ncclProxyState p; ncclProxyConnection remote; remote.sameProcess = 0;
  assert(sharedNetBuffersInit(&p, 0, 0, 0, 0, 4, nullptr, nullptr, nullptr, nullptr) == ncclInternalError);
  assert(p.progressState.localPeers == nullptr && allocations == 0);
  assert(sharedNetBuffersInit(&p, 1, 0, 0, 0, 4, nullptr, nullptr, nullptr, nullptr) == 0);
  ncclIpcDesc desc;
  assert(sharedNetBuffersInit(&p, 1, 0, 0, 0, 4, nullptr, nullptr, nullptr, &desc) == 0);
  assert(allocations == 1 && desc.marker == 47);
  assert(sharedNetBuffersDestroy(&p, 0, 0, &remote) == 0 && releases == 0);
  assert(sharedNetBuffersDestroy(&p, 0, 0, &remote) == 0 && releases == 1);
  std::cout << "PASS CPU-stub lifecycle: default/lazy, host/device, send/recv, cuMem branches, reuse, teardown, IPC descriptor\n";
}
"""


def function(source: str, name: str) -> str:
    start = source.index("static ncclResult_t " + name + "(")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text()
    cpp = args.output / "shared_buffers_test.cc"
    cpp.write_text(
        STUBS
        + function(source, "sharedNetBuffersInit")
        + function(source, "sharedNetBuffersDestroy")
        + TESTS
    )
    binary = args.output / "shared_buffers_test"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
