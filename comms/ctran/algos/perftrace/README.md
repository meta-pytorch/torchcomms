# Ctran PerfTrace

A lightweight tracing utility for profiling CTRAN (Communication Transport) algorithm via visualized trace. It generates Chrome Trace Format JSON files that can be visualized in Chrome's `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).

## Table of Contents
- [Overview](#overview)
- [Class Descriptions](#class-descriptions)
- [Configuration](#configuration)
- [Algorithm Code Instrumentation](#algorithm-code-instrumentation)
- [Output Format](#output-format)
- [Viewing Traces](#viewing-traces)
- [Run Command to Collect Traces from a Ctran Algorithm Test](#run-command-to-collect-traces-from-a-ctran-algorithm-test)

---

## Overview

Ctran PerfTrace provides a low-overhead tracing mechanism for CTRAN algorithms. When enabled, it captures:

- **Timestamp Points**: Instantaneous events at specific moments
- **Time Intervals**: Duration-based events with start and end times
- **Metadata**: Key-value pairs for additional context

The logger is **disabled by default** and incurs minimal overhead when disabled (all recording methods are no-ops).

---

## Class Descriptions

The tracing system consists of three main classes:

```
┌─────────────────────────────────────────────────────────────────┐
│                           Tracer                                 │
│  - Thread-safe container for Records                            │
│  - Dumps JSON on destruction                                     │
├─────────────────────────────────────────────────────────────────┤
│                         Record                                   │
│  - Represents one algorithm execution (e.g., one collective)    │
|  - Thread Safety: **Not thread-safe** - use from single thread only |
│  - Contains points, intervals, and metadata                      │
├─────────────────────────────────────────────────────────────────┤
│     TimestampPoint              │         TimeInterval           │
│  - Instant "X" event            │  - Duration "X" event          │
│  - Records single timestamp     │  - Records start + end         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

**Environment Variables:**

- **`NCCL_CTRAN_ENABLE_PERFTRACE`** (`bool`, default: `false`): Enable/disable trace logging
- **`NCCL_CTRAN_PERFTRACE_DIR`** (`string`, default: `/tmp/`): Directory for trace output files

## Algorithm Code Instrumentation

See Hello World example in `comms/ctran/algos/perftrace/tests/PerfTraceHelloWorldUT.cc`

## How to run

Depending on how developers want to visualize the trace, best practice to run single GPU profiling is to follow example run command `comms/ctran/algos/perftrace/examples/run.sh`

If the developer wish to visualize the algorithm on multiple ranks, best practice is to follow the run command for distributed mode `comms/ctran/algos/perftrace/examples/run.sh dist`

---

## Viewing Traces

### Perfetto

1. Open [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. Drag and drop your JSON file
3. Use the timeline to explore events
