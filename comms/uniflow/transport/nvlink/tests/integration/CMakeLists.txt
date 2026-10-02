# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_test(
  uniflow_nvlink_integration_test NVLinkTransportIntegrationTest.cpp
  "cuda;nvlink;multi_gpu")
uniflow_add_test(
  uniflow_peer_to_peer_transfer_test PeerToPeerTransferTest.cpp
  "cuda;nvlink;multi_gpu")
