# C++ Operations

> Namespace: `tim::operation`

These structs are invoked by the variadic template bundlers when performing
operations on components. The constructor of the class is invoked with a
reference to the component and any additional arguments either provided by
the bundler or passed in by the user.

```cpp
component_tuple<Foo, Bar> handle("example");

handle.start();
for(int i = 0; i < N; ++i)
{
    handle.store(i);
    // ...
}
handle.stop();
```

will roughly translate to:

```cpp
component_tuple<Foo, Bar> handle("example");

Foo& f = std::get<0>(handle.m_data);
Bar& b = std::get<1>(handle.m_data);

// constructor
auto ... = operation::set_prefix<Foo>(f, "example");
auto ... = operation::set_prefix<Bar>(b, "example");

// handle.start()
auto ... = operation::start<Foo>(f);
auto ... = operation::start<Bar>(b);

for(int i = 0; i < N; ++i)
{
    // handle.store(i)
    auto ... = operation::store<Foo>(f, i);
    auto ... = operation::store<Bar>(b, i);
    // ...
}
// handle.stop()
auto ... = operation::stop<Foo>(f);
auto ... = operation::stop<Bar>(b);
```

and, after the optimization by the compiler, those temporary objects
will be eliminated and only the contents of happen inside the constructor (if any)
will remain. This can be verified by executing the `nm` utility and searching for
any symbols in the binary with the name `operation`:

```console
nm --demangle <EXE or LIB> | grep operation
```

```eval_rst
.. doxygenstruct:: tim::operation::init
    :members:
.. doxygenstruct:: tim::operation::init_storage
    :members:
.. doxygenstruct:: tim::operation::fini_storage
    :members:
.. doxygenstruct:: tim::operation::construct
    :members:
.. doxygenstruct:: tim::operation::set_prefix
    :members:
.. doxygenstruct:: tim::operation::set_scope
    :members:
.. doxygenstruct:: tim::operation::push_node
    :members:
.. doxygenstruct:: tim::operation::pop_node
    :members:
.. doxygenstruct:: tim::operation::record
    :members:
.. doxygenstruct:: tim::operation::reset
    :members:
.. doxygenstruct:: tim::operation::measure
    :members:
.. doxygenstruct:: tim::operation::sample
    :members:
.. doxygenstruct:: tim::operation::compose
    :members:
.. doxygenstruct:: tim::operation::is_running
    :members:
.. doxygenstruct:: tim::operation::set_started
    :members:
.. doxygenstruct:: tim::operation::start
    :members:
.. doxygenstruct:: tim::operation::priority_start
    :members:
.. doxygenstruct:: tim::operation::standard_start
    :members:
.. doxygenstruct:: tim::operation::delayed_start
    :members:
.. doxygenstruct:: tim::operation::set_stopped
    :members:
.. doxygenstruct:: tim::operation::stop
    :members:
.. doxygenstruct:: tim::operation::priority_stop
    :members:
.. doxygenstruct:: tim::operation::standard_stop
    :members:
.. doxygenstruct:: tim::operation::delayed_stop
    :members:
.. doxygenstruct:: tim::operation::mark
    :members:
.. doxygenstruct:: tim::operation::mark_begin
    :members:
.. doxygenstruct:: tim::operation::mark_end
    :members:
.. doxygenstruct:: tim::operation::store
    :members:
.. doxygenstruct:: tim::operation::audit
    :members:
.. doxygenstruct:: tim::operation::plus
    :members:
.. doxygenstruct:: tim::operation::minus
    :members:
.. doxygenstruct:: tim::operation::multiply
    :members:
.. doxygenstruct:: tim::operation::divide
    :members:
.. doxygenstruct:: tim::operation::get
    :members:
.. doxygenstruct:: tim::operation::get_data
    :members:
.. doxygenstruct:: tim::operation::get_labeled_data
    :members:
.. doxygenstruct:: tim::operation::base_printer
    :members:
.. doxygenstruct:: tim::operation::print
    :members:
.. doxygenstruct:: tim::operation::print_header
    :members:
.. doxygenstruct:: tim::operation::print_statistics
    :members:
.. doxygenstruct:: tim::operation::print_storage
    :members:
.. doxygenstruct:: tim::operation::add_secondary
    :members:
.. doxygenstruct:: tim::operation::add_statistics
    :members:
.. doxygenstruct:: tim::operation::serialization
    :members:
.. doxygenstruct:: tim::operation::echo_measurement
    :members:
.. doxygenstruct:: tim::operation::copy
    :members:
.. doxygenstruct:: tim::operation::assemble
    :members:
.. doxygenstruct:: tim::operation::derive
    :members:
.. doxygenstruct:: tim::operation::cache
    :members:
.. doxygenstruct:: tim::operation::fini
    :members:
.. doxygenstruct:: tim::operation::finalize::get
    :members:
.. doxygenstruct:: tim::operation::finalize::mpi_get
    :members:
.. doxygenstruct:: tim::operation::finalize::upc_get
    :members:
.. doxygenstruct:: tim::operation::finalize::dmp_get
    :members:
.. doxygenstruct:: tim::operation::finalize::print
    :members:
.. doxygenstruct:: tim::operation::finalize::merge
    :members:
.. doxygenstruct:: tim::operation::finalize::flamegraph
    :members:
.. doxygenstruct:: tim::operation::generic_deleter
    :members:
.. doxygenstruct:: tim::operation::generic_counter
    :members:
.. doxygenstruct:: tim::operation::generic_operator
    :members:
```
