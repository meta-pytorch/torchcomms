# timemory-pid

This application is used by timemory MPI applications to provide a PID for an MPI process which was launched via MPI_Comm_spawn_multiple by writing a temporary file `${TMPDIR}/.timemory-pid-${PID}`
and then uses `TIMEM_PID_SIGNAL` (defaults to `SIGCONT`) to signal to the process that it has been written.

## Usage

```console
timemory-pid <PID> <CMD> [<ARG> [<ARGS...>]]
timemory-pid 23533 sleep 20
```

This tool is not intended to be invoked directly, instead use the `--mpi` argument for tools
which require MPI parallelism.

## Future Work

- May also apply to UPC++ parallelism.
- May be extended to support command line options

## Known Issues

No known issues.
