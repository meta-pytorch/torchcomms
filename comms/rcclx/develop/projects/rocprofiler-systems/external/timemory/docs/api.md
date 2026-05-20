# API

This section provides the high-level details for general and toolkit usage.
The library and Python APIs are recommended for general use and the C++ toolkit
is recommended for building performance monitoring frameworks into their APIs.
Detailed documentation on the toolkit API can be found in the [Doxygen](doxygen.md)
section.

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 3

   api/library
   api/python
   api/templates
   api/type_traits
   api/concepts
   api/policies
   api/operations
   api/utilities
   api/memory_management
```

## Basic Usage

See the [Getting Started Basics Section](getting_started/basics.md).

## Creating Custom Tools/Components

- Written in C++
- Direct access to performance analysis data in Python and C++
- Create your own components: any one-time measurement or start/stop paradigm can be wrapped with timemory
    - Flexible and easily extensible interface: no data type restrictions in custom components

### Custom Logging Component Example

Thanks to the ability to pass pretty much _anything_ to a component, timemory components are not limited to
performance measurements: they can serve as very efficient logging and debugging components since they can
easily be compiled out of the code by marking them with the `is_available` type-trait set to false.

```cpp
// This "component" is for conceptual demonstration only
// It is not intended to be copy+pasted
struct log_message : base<log_message, void>
{
public:
    // use return type SFINAE to check whether "val" supports operator<<
    // if Tp does not support <<, then this function will not be called
    template <typename Tp>
    auto start(const char* label, const Tp& val) -> decltype(std::cerr << val, void())
    {
        std::cerr << "[LOG:START][" << label << "]> " << msg << std::endl;
    }

    // use return type SFINAE to check whether "val" supports operator<<
    // if Tp does not support <<, then this function will not be called
    template <typename Tp>
    auto stop(const char* label, const Tp& val) -> decltype(std::cerr << val, void())
    {
        std::cerr << "[LOG:STOP][" << label << "]> " << val << std::endl;
    }
};
```

Where usage could look below, where the usage within the loop, e.g. `logger_t{}.start("foo", val)`,
will never even create an instance of `log_message` unless `ENABLE_LOG_MESSAGE` is defined at compile-time.
Since the construction of `logger_t{}` and `start("foo", val)` do not have any side-effects,
the entire line will likely be optimized entirely away (depending on the optimization settings).

```cpp
// log_message will NEVER be called when ENABLE_LOG_MESSAGE is not defined
#if !defined(ENABLE_LOG_MESSAGE)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::log_message, false_type)
#endif

// just a general bundle that uses TIMEMORY_GLOBAL_COMPONENTS environ variable
using bundle_t = tim::component_tuple<global_user_bundle>;

// a dedicated bundle for logging
using logger_t = tim::lightweight_bundle<log_message>;

void foo(double val)
{
    // basic label == function name
    TIMEMORY_BASIC_MARKER(bundle_t, "")
    for(int i = 0; i < 10; ++i)
    {
        logger_t{}.start("foo", val); // write log message or nothing if not available
        val += i;
        logger_t{}.stop("foo", val); // write log message or nothing if not available
    }
}
```

### Composable Components Example

Building a brand-new component is simple and straight-forward.
In fact, new components can simply be composites of existing components.
For example, if a component for measuring the FLOP-rate (floating point operations per second)
is desired, it is arbitrarily easy to create and this new component will have all the
features of `wall_clock` and `papi_tuple` component:

```cpp
// This "component" is for conceptual demonstration only
// It is not intended to be copy+pasted
struct flop_rate : base<flop_rate, double>
{
private:
    wall_clock              wc;
    papi_tuple<PAPI_DP_OPS> hw;

public:
    void start()
    {
        wc.start();
        hw.start();
    }

    void stop()
    {
        wc.stop();
        hw.stop();
    }

    auto get() const
    {
        return hw.get() / wc.get();
    }
};
```

### Extended Example

The simplicity of creating a custom component that inherits category-based formatting properties
(`is_timing_category`) and timing unit conversion (`uses_timing_units`)
can be easily demonstrated with the `wall_clock` component and the simplicity and adaptability
of forwarding timemory markers to external instrumentation is easily demonstrated with the
`tau_marker` component:

```cpp
TIMEMORY_DECLARE_COMPONENT(wall_clock)
TIMEMORY_DECLARE_COMPONENT(tau_marker)

// type-traits for wall-clock
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::wall_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::wall_clock, true_type)
TIMEMORY_STATISTICS_TYPE(component::wall_clock, double)

namespace tim
{
namespace component
{
//
// the system's real time (i.e. wall time) clock, expressed as the
// amount of time since the epoch.
//
// NOTE: 'value', 'accum', 'get_units()', etc. are provided by base class
//
struct wall_clock : public base<wall_clock, int64_t>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<wall_clock, value_type>;

    static std::string label() { return "wall_clock"; }
    static std::string description() { return "wall-clock timer"; }

    static value_type  record()
    {
        // use STL steady_clock to get time-stamp in nanoseconds
        using clock_type    = std::chrono::steady_clock;
        using duration_type = std::chrono::duration<clock_type::rep, ratio_t>;
        return std::chrono::duration_cast<duration_type>(
            clock_type::now().time_since_epoch()).count();
    }

    double get_display() const { return get(); }

    double get() const
    {
        // get_unit() provided by base_clock via uses_timing_units type-trait
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val) / ratio_t::den * get_unit();
    }

    void start()
    {
        value = record();
    }

    void stop()
    {
        value = (record() - value);
        accum += value;
    }
};

//
// forwards timemory instrumentation to TAU instrumentation.
//
struct tau_marker : public base<tau_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = tau_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "tau"; }
    static std::string description() { return "TAU_start and TAU_stop instrumentation"; }

    static void global_init(storage_type*) { Tau_set_node(dmp::rank()); }
    static void thread_init(storage_type*) { TAU_REGISTER_THREAD();     }

    tau_marker() = default;
    tau_marker(const std::string& _prefix) : m_prefix(_prefix) {}

    void start() { Tau_start(m_prefix.c_str()); }
    void stop()  { Tau_stop(m_prefix.c_str());  }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }
    // This 'set_prefix(...)' member function is a great example of the template
    // meta-programming provided by timemory: at compile-time, timemory checks
    // whether components have this member function and, if and only if it exists,
    // timemory will call this member function for the component and provide the
    // marker label.

private:
    std::string m_prefix = "";
};

}  // namespace component
}  // namespace tim
```

Using the two tools together in C++ is as easy as the following:

```cpp
#include <timemory/timemory.hpp>

using namespace tim::component;
using comp_bundle_t = tim::component_tuple_t <wall_clock, tau_marker>;
using auto_bundle_t = tim::auto_tuple_t      <wall_clock, tau_marker>;
// "auto" types automatically start/stop based on scope

void foo()
{
    comp_bundle_t t("foo");
    t.start();
    // do something
    t.stop();
}

void bar()
{
    auto_bundle_t t("foo");
    // do something
}

int main(int argc, char** argv)
{
    tim::init(argc, argv);
    foo();
    bar();
    tim::finalize();
}
```

Using the pure template interface will cause longer compile-times and is only available in C++
so a library interface for C, C++, and Fortran is also available:

```cpp
#include <timemory/library.h>

void foo()
{
    uint64_t idx = timemory_get_begin_record("foo");
    // do something
    timemory_end_record(idx);
}

void bar()
{
    timemory_push_region("bar");
    // do something
    timemory_pop_region("bar");
}

int main(int argc, char** argv)
{
    timemory_init_library(argc, argv);
    timemory_push_components("wall_clock,tau_marker");
    foo();
    bar();
    timemory_pop_components();
    timemory_finalize_library();
}
```

In Python:

```python
import timemory
from timemory.profiler import profile
from timemory.util import auto_tuple

def get_config(items=["wall_clock", "tau_marker"]):
    """
    Converts strings to enumerations
    """
    return [getattr(timemory.component, x) for x in items]

@profile(["wall_clock", "tau_marker"])
def foo():
    """
    @profile (also available as context-manager) enables full python instrumentation
    of every subsequent python call
    """
    # ...

@auto_tuple(get_config())
def bar():
    """
    @auto_tuple (also available as context-manager) enables instrumentation
    of only this function
    """
    # ...

if __name__ == "__main__":
    foo()
    bar()
    timemory.finalize()
```
