# timemory-nvml

Command-line tool which provides similar data as `nvidia-smi`:

- GPU and Memory Utilization
- Total/Free/Used GPU Memory
- GPU Processes and Used GPU Memory
- GPU Temperature

## Installation

`timemory-nvml` is automatically built whenever `TIMEMORY_USE_NVML=ON` and `TIMEMORY_BUILD_NVML=ON`.
These options are automatically enabled when `TIMEMORY_USE_CUDA=ON`.

## Execution Modes

`timemory-nvml` can be run in daemon mode or on an executable.

### Daemon Mode

```console
timemory-nvml <options> --
```

### Per-Executable

```console
timemory-nvml <options> -- <command>
```

## Options

| Option            | Environment                     | Type                | Description                                                   |
| ----------------- | ------------------------------- | ------------------- | ------------------------------------------------------------- |
| `buffer-count`    | `TIMEMORY_NVML_BUFFER_COUNT`    | integer             | maximum number of measurements                                |
| `max-samples`     | `TIMEMORY_NVML_MAX_SAMPLES`     | integer             | exit after this many measurements have been collected         |
| `dump-interval`   | `TIMEMORY_NVML_DUMP_INTERVAL`   | integer             | write output file after this many measurements                |
| `sample-interval` | `TIMEMORY_NVML_SAMPLE_INTERVAL` | seconds  (floating) | time to wait between measurements                             |
| `output`          | `TIMEMORY_NVML_OUTPUT`          | string              | name of the output file to write                              |
| `time-format`     | `TIMEMORY_NVML_TIME_FORMAT`     | string              | the `strftime` format string this identifies each measurement |

```console
$ timemory-nvml --help
[./timemory-nvml] Usage: ./timemory-nvml [ --help (count: 0, dtype: bool)
                                           --debug (count: 0, dtype: bool)
                                           --verbose (max: 1, dtype: bool)
                                           --quiet (count: 0, dtype: bool)
                                           --buffer-count (count: 1, dtype: integer)
                                           --max-samples (count: 1, dtype: integer)
                                           --dump-interval (count: 1, dtype: integer)
                                           --sample-interval (count: 1, dtype: double)
                                           --time-format (count: 1, dtype: string)
                                           --output (max: 1, dtype: string)
                                         ] -- <CMD> <ARGS>


Options:
    -h, -?, --help                 Shows this page
    --debug                        Debug output (env: TIMEMORY_NVML_DEBUG)
    -v, --verbose                  Verbose output (env: TIMEMORY_NVML_VERBOSE)
    -q, --quiet                    Suppress as much reporting as possible
    -c, --buffer-count             Buffer count (env: TIMEMORY_NVML_BUFFER_COUNT)
    -m, --max-samples              Maximum samples (env: TIMEMORY_NVML_MAX_SAMPLES)
    -i, --dump-interval            Dump interval (env: TIMEMORY_NVML_DUMP_INTERVAL)
    -s, --sample-interval          Sample interval (env: TIMEMORY_NVML_SAMPLE_INTERVAL)
    -f, --time-format              strftime format for labels (env: TIMEMORY_NVML_TIME_FORMAT)
    -o, --output                   Write results to JSON output file (env: TIMEMORY_NVML_OUTPUT).
                                   Use:
                                   - '%p' to encode the process ID
                                   - '%j' to encode the SLURM job ID
                                   E.g. '-o timemory-nvml-output-%p'.
```
