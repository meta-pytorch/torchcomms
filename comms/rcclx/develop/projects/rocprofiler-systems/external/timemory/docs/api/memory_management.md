# Memory Management

[Detailed Doxygen Documentation](../doxygen.md).

## Manager Class

The `tim::manager` class is a thread-local singleton which handles the memory management for each thread.
It is stored as a `std::shared_ptr` which automatically deletes itself when the thread exits.
When the thread-local singleton for each component storage class is created, it increments the reference
count for this class to ensure that it exists while any storage classes are allocated. When the
storage singletons are created, it registers functors with the manager for destorying itself.

```eval_rst
.. doxygenclass:: tim::manager
    :members:
```

## Memory Buffers

```eval_rst
.. doxygenstruct:: tim::base::ring_buffer
    :members:
    :undoc-members:
.. doxygenstruct:: tim::data_storage::ring_buffer
    :members:
    :undoc-members:
```

## Allocators

```eval_rst
.. doxygenclass:: tim::data::ring_buffer_allocator
    :members:
    :undoc-members:
.. doxygenclass:: tim::ert::aligned_allocator
    :members:
    :undoc-members:
```

## Graph Classes

The graph classes are responsible for maintaining the hierarchy of the calling context tree.
`tim::graph`, `tim::graph_data`, and `tim::node::graph<T>` are rarely interacted with directly.
Storage results are reported within nested `std::vector`s of `tim::node::result<T>` and
`tim::node::tree<T>`. The former provides the data in a flat heirarchy where the calling-context
is represented through indentation and a depth value, the latter represents the
calling-context through a recursive structure.

```eval_rst
.. doxygenclass:: tim::graph
.. doxygenclass:: tim::graph_data
.. doxygenstruct:: tim::node::graph
    :members:
.. doxygenstruct:: tim::node::result
    :members:
.. doxygenstruct:: tim::basic_tree
    :members:
.. doxygenstruct:: tim::node::entry
    :members:
.. doxygenstruct:: tim::node::tree
    :members:
```

### Graph Result and Tree Sample

```cpp
using namespace tim;
using wall_clock = component::wall_clock;
using node_type  = node::result<wall_clock>;
using tree_type  = basic_tree<node::tree<wall_clock>>;

// the flat data for the process
std::vector<node_type> foo = storage<wall_clock>::instance()->get();

// aggregated flat data from distributed memory process parallelism
// depending on settings, maybe contain all data on rank 0, partial data, or no data
// on non-zero ranks
std::vector<std::vector<node_type>> bar = storage<wall_clock>::instance()->dmp_get();

// the tree data for the process
std::vector<tree_type> baz{};
baz = storage<wall_clock>::instance()->get(baz);

// aggregated tree data from distributed memory process parallelism
// depending on settings, maybe contain all data on rank 0, partial data, or no data
// on non-zero ranks
std::vector<std::vector<tree_type>> spam{};
spam = storage<wall_clock>::instance()->dmp_get(spam);
```

### Graph Result and Tree Comparison

```console

#----------------------------------------#
# Storage Result
#----------------------------------------#
  Thread id            : 0
  Process id           : 4385
  Depth                : 0
  Hash                 : 9631199822919835227
  Rolling hash         : 9631199822919835227
  Prefix               : >>> foo
  Hierarchy            : [9631199822919835227]
  Data object          :    6.534 sec wall
  Statistics           : [sum: 6.53361] [min: 6.53361] [max: 6.53361] [sqr: 42.6881] [count: 1]
#----------------------------------------#
  Thread id            : 0
  Process id           : 4385
  Depth                : 1
  Hash                 : 11474628671133349553
  Rolling hash         : 2659084420343633164
  Prefix               : >>> |_bar
  Hierarchy            : [9631199822919835227, 11474628671133349553]
  Data object          :    5.531 sec wall
  Statistics           : [sum: 5.53115] [min: 0.307581] [max: 0.307581] [sqr: 7.71154] [count: 5]

#----------------------------------------#
# Storage Tree
#----------------------------------------#
Thread id            : {0}
Process id           : {4385}
Depth                : -1
Hash                 : 0
Prefix               : unknown-hash=0
Inclusive data       :    0.000 sec wall
Inclusive stat       : [sum: 0] [min: 0] [max: 0] [sqr: 0] [count: 0]
Exclusive data       :   -6.534 sec wall
Exclusive stat       : [sum: 0] [min: 0] [max: 0] [sqr: 0] [count: 0]
  #----------------------------------------#
  Thread id            : {0}
  Process id           : {4385}
  Depth                : 0
  Hash                 : 9631199822919835227
  Prefix               : foo
  Inclusive data       :    6.534 sec wall
  Inclusive stat       : [sum: 6.53361] [min: 6.53361] [max: 6.53361] [sqr: 42.6881] [count: 1]
  Exclusive data       :    1.002 sec wall
  Exclusive stat       : [sum: 1.00246] [min: 6.53361] [max: 6.53361] [sqr: 34.9765] [count: 1]
    #----------------------------------------#
    Thread id            : {0}
    Process id           : {4385}
    Depth                : 1
    Hash                 : 11474628671133349553
    Prefix               : bar
    Inclusive data       :    5.531 sec wall
    Inclusive stat       : [sum: 5.53115] [min: 0.307581] [max: 0.307581] [sqr: 7.71154] [count: 5]
    Exclusive data       :    5.531 sec wall
    Exclusive stat       : [sum: 5.53115] [min: 0.307581] [max: 0.307581] [sqr: 7.71154] [count: 5]
```

Note the first entry of storage tree has a negative depth and hash of zero. Nodes such of these
are "dummy" nodes which timemory keeps internally as bookmarks for root nodes and thread-forks
(parent call-graph location when a child thread was initialized or returned to "sea-level").
These may be removed in future versions of timemory.

## Storage Class

The `tim::storage` class is a thread-local singleton which handles the call-graph and persistent
data accumulation for each component. It is stored as a `std::unique_ptr` which automatically deletes
itself when the thread exits. On the non-primary thread, destruction of the singleton merges it's
call-graph data into the storage singleton on the primary thread. Initialization and finalization
of the storage class is the ONLY time that thread synchronization and inter-process communication
occurs. This characteristic enables timemory storage to arbitrarily scale to any number of threads and/or
processes without performance degradation. If you want to information of the state of the call-graph,
the `tim::storage<T>` is the structure to do so, e.g. the current size of the call-graph, a serialization
of the current process- and thread-specific, etc. Invoking the `get()` member function will return
the data for the current thread on worker threads and invoking the `get()` member function on the primary
thread will return the data for _all_ the threads. Invoking `mpi_get()` will aggregate the results
across all MPI processes, `upc_get()` will aggregate the results across all the UPC++ results, and
`dmp_get()` (dmp == distributed memory parallelism) will aggregate all the results across MPI and UPC++
processes.

```eval_rst
.. doxygenclass:: tim::base::storage
    :members:
    :undoc-members:
.. doxygenclass:: tim::storage
    :members:
```
