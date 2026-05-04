# About

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2
```

Timemory is a modular API for performance measurements and analysis with a very lightweight overhead.
If timemory does not support a particular measurement type or analysis method, user applications
can easily create their own component that accomplishes the desired task.

Timemory is implemented as a generic C++14 template library but supports implementation
in C, C++, Fortran, CUDA, and Python codes.
The design goal of timemory is to create an easy-to-use framework for generating
performance measurements and analysis methods which are extremely flexible
with respect to both how the data is stored/accumulated and which methods the measurement
or analysis supports.

## Design Goals

- __*Toolkit*__ for creating new performance analysis tools
- __*Common instrumentation framework*__
    - Eliminate need for projects to explicitly support multiple instrumentation frameworks
- __*High performance*__ during data collection
- __*Low overhead*__ when dormant (disabled at runtime)
- Zero overhead when disabled at compile time
- Support arbitrarily intermixing components:
    - Instrument measurements of A, B, and C around arbitrary region 1
    - Instrument measurements of A and C around arbitrary region 1.1 (nested with Section 1)
    - Instrument measurements of C around arbitrary region 2
    - Instrument measurements of D around arbitrary region 3
    - No instrumentation around arbitrary region 4
- Intuitive and simple API to use and extend

## Support for timemory in external tools

Currently, timemory provides compatibility with multiple tools internally but the end
goal is for the majority of this to be maintained by the authors of the tool. This will
benefits users by provided a single method for using all of their favorite tools and
make it extremely easy for them to try out new tools.
This will benefit the authors of the tools because there will be a significantly
lower the introduction barrier required for users to try out the new tool -- if the
user is familiar with timemory, the tool can be trivially integrated into either their
code or into the profiler.

An external tool can easily provide compatibility with timemory and leverage
all of its work creating a low-overhead measurement system in parallel environments,
Python extensions, and dynamic instrumentation, by simply providing a header in their
source code which defines the interface the tool wants to provide and the tools can
add/remove support at will without having to maintain any source code in
timemory or worry about version compatability with timemory. Versioning issues do
not inherently exist because for several reasons which are detailed
the [CONTRIBUTING.md](CONTRIBUTING) documentation.

## Credits

Timemory is actively developed by NERSC at Lawrence Berkeley National Laboratory

| Name               |                                        Affiliation                                        |                    GitHub                     |
| ------------------ | :---------------------------------------------------------------------------------------: | :-------------------------------------------: |
| Jonathan R. Madsen | [NERSC](https://www.nersc.gov/about/nersc-staff/application-performance/jonathan-madsen/) |    [jrmadsen](https://github.com/jrmadsen)    |
| Yunsong Wang       |       [NERSC](https://www.nersc.gov/about/nersc-staff/nesap-postdocs/yunsong-wang/)       | [PointKernel](https://github.com/PointKernel) |
| Muhammad Haseeb    |                   [NERSC](https://sites.google.com/a/fiu.edu/mhaseeb/)                    |  [mhaseeb123](https://github.com/mhaseeb123)  |

## Contributions

Timemory encourages contributions via GitHub pull-requests.
For more information about contributing new components, please read the
[CONTRIBUTING](https://github.com/NERSC/timemory/blob/develop/CONTRIBUTING.md)
document on GitHub.
