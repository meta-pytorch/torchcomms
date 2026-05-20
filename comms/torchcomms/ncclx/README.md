# torchcomms-ncclx

NCCLX backend for [torchcomms](https://pypi.org/project/torchcomms/).

NCCLX is Meta's extended NCCL fork. This package builds the
`torchcomms_ncclx._comms_ncclx` extension and registers it under the
`torchcomms.backends` entry-point group so it can be loaded by torchcomms
via `backend="ncclx"`.

## Install

```bash
pip install torchcomms torchcomms-ncclx
```

## Build from source

```bash
# (optional) pre-build the vendored NCCLX C++ libraries
./build_ncclx.sh

# Build & install the wheel
pip install --no-build-isolation -v ./comms/torchcomms/ncclx
```

The setup script invokes the same `build_ncclx.sh` and CMake configuration
the bundled build used; only the NCCLX-specific bits are compiled and
packaged.
