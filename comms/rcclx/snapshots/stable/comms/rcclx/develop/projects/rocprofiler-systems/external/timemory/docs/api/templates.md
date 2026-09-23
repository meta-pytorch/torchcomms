# C++ Variadic Template Bundlers

[Detailed Doxygen Documentation](https://timemory.readthedocs.io/en/develop/doxygen-docs/index.html).

Several flavors of variadic template classes for creating a single
handle to multiple components are provided. In general, these
components all have identical interfaces and the vast majority
of their member functions accept any and all arguments.
These member functions will take the given arguments and attempt
to invoke the member function of the component with a similar name,
e.g. `component_tuple<wall_clock, cpu_clock>::start()` will attempt to
invoke `wall_clock::start()` and `cpu_clock::start()`. When arguments are
provided, these bundlers will go through a series of checks at compile-time:

1. Attempt to find the component member function that accepts those exact set of arguments
2. Attempt to find the component member function which accepts a subset of those arguments (requires type-trait configuration)
3. Attempt to find the component member function which accepts no arguments
4. Accept the component does not have the member function and invoke no member function

In the event, this workflow require specialization for a particular component, a custom specialization
of the struct in the `tim::operation` namespace can be written.

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Table of Contents

   templates/auto_bundle
   templates/auto_list
   templates/auto_tuple
   templates/component_bundle
   templates/component_list
   templates/component_tuple
   templates/lightweight_tuple
```

## Static Polymorphic Base Classes

```eval_rst
.. doxygenclass:: tim::bundle
   :members:
.. doxygenclass:: tim::auto_base_bundle
   :members:
.. doxygenclass:: tim::bundle< Tag, BundleT, TupleT >
   :members:
.. doxygenclass:: tim::auto_base_bundle< Tag, CompT, BundleT >
   :members:
```
