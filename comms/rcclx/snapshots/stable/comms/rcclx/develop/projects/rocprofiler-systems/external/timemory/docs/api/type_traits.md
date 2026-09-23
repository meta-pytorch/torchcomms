# C++ Type-Traits

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2
```

Type-traits are used to specialize behavior at compile-time. In general, timemory
tries to avoid specialization in the core library (when possible) so that users are
not restricted for specializing downstream. Please note, specializations may be ignored
or cause compilation errors if extern templates are used. Ignoring specializations commonly happen in
output routines like `tim::operation::print<T>` where the value of the specialization
is used at runtime and the body of the output routine is not actually instantiated in the
user code.

## Component Implementation

```eval_rst
.. doxygenstruct:: tim::trait::is_available
   :members:
.. doxygenstruct:: tim::trait::python_args
   :members:
.. doxygenstruct:: tim::trait::default_runtime_enabled
   :members:
.. doxygenstruct:: tim::trait::runtime_enabled
   :members:
.. doxygenstruct:: tim::trait::api_components
   :members:
```

## Base Class Modifications

```eval_rst
.. doxygenstruct:: tim::trait::base_has_accum
   :members:
.. doxygenstruct:: tim::trait::base_has_last
   :members:
.. doxygenstruct:: tim::trait::dynamic_base
   :members:
```

## Priority Ordering

```eval_rst
.. doxygenstruct:: tim::trait::start_priority
   :members:
.. doxygenstruct:: tim::trait::stop_priority
   :members:
.. doxygenstruct:: tim::trait::fini_priority
   :members:
```

## Data Sharing

```eval_rst
.. doxygenstruct:: tim::trait::cache
   :members:
.. doxygenstruct:: tim::trait::derivation_types
   :members:
```

## Data Collection

```eval_rst
.. doxygenstruct:: tim::trait::sampler
   :members:
.. doxygenstruct:: tim::trait::file_sampler
   :members:
```

## Feature Support

```eval_rst
.. doxygenstruct:: tim::trait::supports_custom_record
   :members:
.. doxygenstruct:: tim::trait::supports_flamegraph
   :members:
```

## Archive Serialization

```eval_rst
.. doxygenstruct:: tim::trait::api_input_archive
   :members:
.. doxygenstruct:: tim::trait::api_output_archive
   :members:
.. doxygenstruct:: tim::trait::input_archive
   :members:
.. doxygenstruct:: tim::trait::output_archive
   :members:
.. doxygenstruct:: tim::trait::pretty_archive
   :members:
.. doxygenstruct:: tim::trait::requires_json
   :members:
```

## Units and Formatting

```eval_rst
.. doxygenstruct:: tim::trait::is_memory_category
   :members:
.. doxygenstruct:: tim::trait::is_timing_category
   :members:
.. doxygenstruct:: tim::trait::uses_memory_units
   :members:
.. doxygenstruct:: tim::trait::uses_timing_units
   :members:
.. doxygenstruct:: tim::trait::uses_percent_units
   :members:
.. doxygenstruct:: tim::trait::units
   :members:
```

## Output Reporting

```eval_rst
.. doxygenstruct:: tim::trait::report
   :members:
.. doxygenstruct:: tim::trait::report_count
   :members:
.. doxygenstruct:: tim::trait::report_depth
   :members:
.. doxygenstruct:: tim::trait::report_metric_name
   :members:
.. doxygenstruct:: tim::trait::report_units
   :members:
.. doxygenstruct:: tim::trait::report_sum
   :members:
.. doxygenstruct:: tim::trait::report_mean
   :members:
.. doxygenstruct:: tim::trait::report_statistics
   :members:
.. doxygenstruct:: tim::trait::report_self
   :members:
.. doxygenstruct:: tim::trait::custom_label_printing
   :members:
.. doxygenstruct:: tim::trait::custom_serialization
   :members:
.. doxygenstruct:: tim::trait::custom_unit_printing
   :members:
.. doxygenstruct:: tim::trait::echo_enabled
   :members:
.. doxygenstruct:: tim::trait::iterable_measurement
   :members:
```

## Statistics

```eval_rst
.. doxygenstruct:: tim::trait::statistics
   :members:
.. doxygenstruct:: tim::trait::record_statistics
   :members:
.. doxygenstruct:: tim::trait::permissive_statistics
   :members:
```

## Storage

```eval_rst
.. doxygenstruct:: tim::trait::uses_storage
   :members:
.. doxygenstruct:: tim::trait::uses_value_storage
   :members:
.. doxygenstruct:: tim::trait::tree_storage
   :members:
.. doxygenstruct:: tim::trait::flat_storage
   :members:
.. doxygenstruct:: tim::trait::timeline_storage
   :members:
.. doxygenstruct:: tim::trait::thread_scope_only
   :members:
.. doxygenstruct:: tim::trait::data
   :members:
.. doxygenstruct:: tim::trait::secondary_data
   :members:
.. doxygenstruct:: tim::trait::collects_data
   :members:
.. doxygenstruct:: tim::trait::generates_output
   :members:
```

## Deprecated

These type-traits are either:

- Removed from the source code entirely
- Automatically detected
- Migrated to concepts

```eval_rst
.. doxygenstruct:: tim::trait::is_component
   :members:
.. doxygenstruct:: tim::trait::is_gotcha
   :members:
.. doxygenstruct:: tim::trait::is_user_bundle
   :members:
.. doxygenstruct:: tim::trait::record_max
   :members:
.. doxygenstruct:: tim::trait::array_serialization
   :members:
.. doxygenstruct:: tim::trait::requires_prefix
   :members:
.. doxygenstruct:: tim::trait::supports_args
   :members:
```
