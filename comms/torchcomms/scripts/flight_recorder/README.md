# Flight Recorder Verification

This directory contains scripts to verify and demonstrate the FlightRecorderHook functionality in torchcomms.

## Overview

The FlightRecorder is a debugging tool that records collective operations performed through torchcomms. It maintains a circular buffer of operations with metadata that can be dumped to JSON format for post-mortem analysis. This is particularly useful for debugging hangs, timeouts, and desync issues in distributed training.

## Files

- `verify_flight_recorder.py`: Example script demonstrating FlightRecorderHook usage

## Prerequisites

- PyTorch 2.8 or higher
- torchcomms with NCCL or NCCLX backend
- CUDA-capable GPUs

## Running the Verification Script

```bash
# Set the dump directory prefix (traces will be written as <prefix><rank>)
export TORCHCOMM_FR_DUMP_TEMP_FILE=/tmp/flight_recorder_traces/rank_

# Using torchrun with 2 GPUs
torchrun --nproc_per_node=2 verify_flight_recorder.py

# With a specific backend
TEST_BACKEND=nccl torchrun --nproc_per_node=2 verify_flight_recorder.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TEST_BACKEND` | Backend to use (nccl, ncclx, gloo) | `gloo` |
| `TEST_DEVICE` | Device type | `cuda` |
| `TORCHCOMM_FR_DUMP_TEMP_FILE` | File prefix for trace dumps (rank is appended) | `~/.cache/torchcomm_fr_trace_` |
| `TORCHCOMM_FR_DUMP_DYNAMIC_FILE_NAME` | Enable dynamic file naming | `false` |

## Analyzing Flight Recorder Traces

After running the script, traces are saved as JSON files (one per rank). Use PyTorch's built-in trace analyzer to analyze them:

### Basic Analysis

```bash
# Analyze all traces in a directory
python -m torch.distributed.flight_recorder.fr_trace -j /path/to/traces

# Example with default location
python -m torch.distributed.flight_recorder.fr_trace -j /tmp/flight_recorder_traces
```

## Debugging Common Issues

### Hangs / Timeouts

When a collective operation hangs:

1. The flight recorder automatically dumps traces when an abort hook is triggered
2. Look for operations in `scheduled` or `started` state (not `completed`)
3. Compare sequence IDs across ranks to find where they diverge

### Desync Detection

The trace analyzer automatically detects desync issues by:

1. Matching operations across ranks by sequence ID
2. Verifying all ranks issue the same collective type
3. Checking tensor sizes and dtypes match

### Incomplete Traces

If some ranks crash before dumping:
```bash
python -m torch.distributed.flight_recorder.fr_trace /tmp/traces --allow-incomplete-ranks
```

## Programmatic Usage

You can also analyze traces programmatically:

```python
import json
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.loader import read_dir
from torch.distributed.flight_recorder.components.config_manager import JobConfig

# Load traces
config = JobConfig()
args = config.parse_args(["/path/to/traces"])
details, version = read_dir(args)

# Build database
db = build_db(details, args, version)

# Access parsed data
for collective in db.collectives:
    print(f"Collective: {collective.profiling_name}, seq_id={collective.collective_seq_id}")

for group in db.groups:
    print(f"Process Group: {group.name}, {group.desc}")
```

## See Also

- [PyTorch Flight Recorder Documentation](https://pytorch.org/docs/stable/distributed.html#flight-recorder)
- [FlightRecorderTest.py](../../torchcomms/tests/integration/py/FlightRecorderTest.py) - Unit tests with more examples
