# Custom Components

Timemory supports user-specifed components. A custom component can utilize the static polymorphic base class
`tim::component::base` to inherit many features but ultimately, the goal is to not require a specific
base class. The `tim::component::base` class provides the integration into the API and requires two template
parameters:

1. component type (i.e. itself)
2. the data type being stored

```cpp
//
// generic static polymorphic base class
//
template <typename _Tp, typename value_type>
struct base;

//
// create cpu_clock component and use int64_t as data type
//
struct cpu_clock
: public base<cpu_clock, int64_t>
{
    // ...
};

//
// create cpu_util component and specify data type as pair of 64-bit integers
//
struct cpu_util
: public base<cpu_util, std::pair<int64_t, int64_t>>
{
    // ...
};

//
// create papi_tuple component and specify data type as an array of long long types
//
template <int... Events>
struct papi_tuple
: public base<papi_tuple<Events...>,                    // self type
              std::array<long long, sizeof...(Events)>> // data type
{
    // ...
};
```

## Type Traits

Type traits are provided to customize how operations on the data type are handled. For example, adding two `wall_clock`
components together is a simple `a += b` operation but adding two `peak_rss` components is better defined as a
max operation: `a = max(a, b)` since `+=` a "peak" memory value could return a result larger than the total amount of
memory on the system. Additionally, certain components are not available on certain systems, e.g. components that
provide data via the POSIX rusage struct on Windows. The most update-to-date list of type-traits can be found in
`timemory/mpl/types.hpp` in the `trait` namespace. A subset of the available type-traits is provided below.

> Namespace: `tim::trait`

| Type Trait                  | Description                                                                            | Default Setting   |
| --------------------------- | -------------------------------------------------------------------------------------- | ----------------- |
| **`is_available`**          | Specify the availablity of the component's implementation                              | `std::true_type`  |
| **`record_max`**            | Specify that addition operations should use comparisions                               | `std::false_type` |
| **`custom_label_printing`** | Specify the component provides its own labeling for output (typically multiple labels) | `std::false_type` |
| **`custom_unit_printing`**  | Specify the component provides its own units for output (typically multiple units)     | `std::false_type` |
| **`custom_laps_printing`**  | Specify the component provides its own "lap" printing for output                       | `std::false_type` |
| **`is_timing_category`**    | Designates the width and precision should apply environment-specified timing settings  | `std::false_type` |
| **`is_memory_category`**    | Designates the width and precision should apply environment-specified memory settings  | `std::false_type` |
| **`uses_timing_units`**     | Designates the width and precision should apply environment-specified timing units     | `std::false_type` |
| **`uses_memory_units`**     | Designates the width and precision should apply environment-specified memory units     | `std::false_type` |
| **`requires_json`**         | Specify the component should always output the JSON file format                        | `std::false_type` |
| **`start_priority`**        | Specify the component should be started before non-prioritized components              | `std::false_type` |
| **`stop_priority`**         | Specify the component should be stopped before non-prioritized components              | `std::false_type` |

### Type Traits Example

```cpp
template <typename _Tp> struct is_available : std::true_type { };

// on Windows, stack_rss is not available because stack_rss requires POSIX rusage struct
#if defined(_WIN32) || defined(_WIN64)
//
// using macro to specialize type-trait
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::stack_rss, false_type)
//
// explicit specialization of type-trait without macro
//
namespace tim
{
namespace trait
{

template <>
struct is_available<component::stack_rss> : std::false_type
{};

}  // namespace trait
}  // namespace tim
#endif
```

### The `is_available` Trait

`tim::trait::is_available` is a very important type-trait. Marking a component as not available results in this type
being filtered out of the variadic wrapper classes `auto_tuple`, `auto_list`, `component_tuple`, and `component_list`.
This means that with the above example, on Windows, the `example_real_stack_type` type below will implement the
same behavor as the `example_real_type` type:

```cpp
using example_real_type       = tim::auto_tuple< wall_clock >;
using example_real_stack_type = tim::auto_tuple< wall_clock, stack_rss >;
// On Windows, example_real_stack_type type will filter out stack_rss at compile time
// and will produce the exact same result and performance as example_real_type
```

In addition to a performance benefit (no operations on a component that does not provide anything) and a much
cleaner code base (significant reduction of `#ifdef` blocks), this can be used to enable "add-on" features for
performance analysis that won't be included in production code, e.g. including `cpu_roofline_dp_flops` in a specification
but not enabling the PAPI backend in a production code.

## Custom Component Example

Timemory uses SFINAE (substitution-failure-is-not-an-error) to query for the presence of member functions at compile-time.
Adding new capabilities to a component is commonly as simple as just providing the member function for the capability.
For example, if a component needs a string to forward on to another tool, the component can just add a `void set_prefix(const char* str)`
and this member function will be detected by `operation::set_prefix<T>` (invoked by the variadic bundler). If this member
function does not exist, then `operation::set_prefix<T>` will be a no-op for all instances of component `T`.

| Field                 | Required | Type                      | Description                                                                                 |
| --------------------- | :------: | ------------------------- | ------------------------------------------------------------------------------------------- |
| `value_type`          |   Yes    | Alias/typedef             | Data type used for storage                                                                  |
| `base_type`           |    No    | Alias/typedef             | Base class directly inheriting from                                                         |
| `precision`           |    No    | Constant integer          | Default output precision                                                                    |
| `width`               |    No    | Constant integer          | Default output width                                                                        |
| `format_flags`        |    No    | `std::ios_base::fmtflags` | Bitset of formatting flags                                                                  |
| `unit()`              |    No    | numerical value           | Units of data type                                                                          |
| `label()`             |    No    | string                    | Short-hand description without spaces                                                       |
| `description()`       |    No    | string                    | Full description of component                                                               |
| `display_unit()`      |    No    | string                    | String representation of `unit()`                                                           |
| `get() const`         |    No    | const member function     | Returns the current measurement                                                             |
| `get_display() const` |    No    | const member function     | Returns the current measurement in format desired for reporting                             |
| `start()`             |    No    | member function           | Defines operation when component recording is started                                       |
| `stop()`              |    No    | member function           | Defines operation when component recording is stopped                                       |
| `set_prefix(T)`       |    No    | member function           | Sets label before `start()` is called. `T` can be `std::string`, `const char*`, or `size_t` |

```cpp
struct wall_clock : public base<wall_clock, int64_t>
{
    // alias to the type of data being recorded is stored
    using value_type = int64_t;

    // alias to base type that implements a lot of the functionality
    using base_type  = base<wall_clock, value_type>;

    // this is specific to this particular component
    using ratio_t    = std::nano;

    // these are formatting guidelines
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    // these handle units conversions and descriptions
    static int64_t     unit() { return units::sec; }
    static std::string label() { return "real"; }
    static std::string description() { return "wall time"; }
    static std::string display_unit() { return "sec"; }

    // this defines how to record a value for the component
    static value_type  record()
    {
        return tim::get_clock_real_now<int64_t, ratio_t>();
    }

    // this defines how to get the relevant data from the component and can
    // return any type
    float get() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }

    // this defines how the value is represented in '<<' and can return any type
    float get_display()() const
    {
        return this->get();
    }

    // this defines what happens when the components is started
    void start()
    {
        base_type::set_started();
        value = record();
    }

    // this defines what happens when the components is stopped
    void stop()
    {
        value = (record() - value);
        accum += value;
        base_type::set_stopped();
    }
};
```
