There are two types of loggers provided here. Text loggers, and sample/event
loggers.

# Sample Loggers

## Overview

The rough overview of sample based loggers:

- `DataSink` - A class that writes out data
- `DataTable` - Collects samples, and writes them to a `DataSink`
- `DataTableWrapper` - Not used directly, handles some conveniences
- `NcclScubaSample` - Samples added to a `DataTable` and written to a `DataSink`
- `NcclScubaEvent` - Structures a `NcclScubaSample` to be event oriented
- `LoggerEvent` - Well structured events :)

## Usage

If you want to sample something new:

- Extend `LoggerEvent`, e.g. `CoolEvent`
    - Ensure `CoolEvent` has appropriate fields, validation, etc
    - Implement `NcclScubaSample CoolEvent::toSample()`
- Create table description
    - In `DataTableWrapper.h`
        - `DECLARE_scuba_table(nccl_cool_logging)`
        - `#define SCUBA_nccl_cool_logging (*SCUBA_nccl_cool_logging_ptr)`
    - In `DataTableWrapper.cc`
        - Add table name string at top
        - Add lookup for table name to `getAllTableNames()`
        - `DEFINE_scuba_table(nccl_cool_logging);`
        - In `init()` add `INIT_scuba_table(nccl_cool_logging);`
        - In `shutdown()` add `SHUTDOWN_scuba_table(nccl_cool_logging);`
    - In `ScubaLogger.cc`
        - Add appropriate logger event type to `getTablePtrFromEvent`
- Use your event. Most typically that would be something like:

```c++
NcclScubaEvent event(std::make_unique<CoolEvent>(arg1, arg2, arg3));
// code code code
event.record();
```

`record()` will look up the table associated with the sample, call `addSample`
on the table, which will call `addRawData` on the sink.

Easy peasy!

## Future

The above is pretty damn clunky. There's a lot of clean to be done here.

- Can just extend `DataTable` or a related registry instead of needing to spread table definition over several files
- Naming cleanup; `NcclScubaSample` and `NcclScubaEvent` aren't actually that related to `Scuba` anymore
- Remove `NcclScubaSample` usage besides the `toSample` methods. It's too low level for most users.
