// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// Internal bootstrap state for the NCCLX fork.
//
// Upstream NCCL keeps `bootstrapState` and its helper structs private to
// bootstrap.cc, and the forked include/bootstrap.h is kept byte-identical to
// upstream. NCCLX-only code still needs the full definition (the ctran
// bootstrap-cleanup helper and its tests reach into the ring/listen endpoints),
// so the definition lives here -- an NCCLX-owned header outside the forked
// tree -- rather than being added back to the upstream header.
//
// This is NOT part of any public NCCL API. Only bootstrap.cc and NCCLX code
// that manipulates bootstrap endpoints should include it.
//
// Older forks (< 2.30) still carry these structs in their own
// include/bootstrap.h, so this header is empty for them and they keep using
// that definition unchanged.

#include "nccl.h"
#include "nccl_net.h"
#include "socket.h"

// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

struct bootstrapRing_t {
  union {
    struct {
      void *sendComm, *recvComm;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
    } net;
    struct {
      struct ncclSocket recv;
      struct ncclSocket send;
    } socket;
  };
};

struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
    } net;
    struct ncclSocket socket; // socket to be used for the ring
  };
};

struct bootstrapState {
  struct bootstrapRing_t ring;
  struct bootstrapListen_t listen;
  ncclNet_t* net;
  uint64_t* peerProxyAddressesUDS;
  union ncclSocketAddress* peerProxyAddresses;
  union ncclSocketAddress* peerP2pAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;

  // Reference to CommLogData to object to facilicate logging
  struct CommLogData* logMetaDataPtr{nullptr};
  bool ringUsesOobNet{false};
};

#endif
