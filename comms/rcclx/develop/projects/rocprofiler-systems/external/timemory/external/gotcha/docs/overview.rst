========
Overview
========

Gotcha is an API that provides function wrapping, interposing a wrapper 
function between a function and its callsites. Many tools rely on function wrapping 
as a fundamental building block. For example, a performance analysis 
tool which wants to measure time an application spends in IO might put
wrappers around "read()" and "write()" which trigger stopwatches.

Tools traditionally implemented function wrapping with the LD_PRELOAD
environment variable on glibc-based systems. This environment variable
allowed the tool to inject a tool library into the target application.
Any functions that the tool library exports, such as a "read()" or
"write()" function, will intercept calls to the matching function names
from the original application. While powerful, the LD_PRELOAD approach
had several limitations:

-  Tool libraries can have challenges matching ABI-compatibility with
   the application.

-  Multiple tools cannot wrap the same function.

-  The set of wrapped functions are determined at tool build-time and
   cannot be changed in response to application behavior.

Gotcha addresses these limitations by providing an API for function
wrapping. Tool libraries make wrapping requests to Gotcha that say, for
example, “wrap all calls to the read() function with my tool_read()
function, and give me a function pointer to the original read().”
Gotcha’s API allows tool wrapping decisions to be made at runtime, and
it handles cases of multiple tools wrapping the same function. It does
not, however, provide any new mechanisms for injecting the tool library
into an application. Gotcha-based tools should be added to the
application at link-time or injected with LD_PRELOAD.

Gotcha works by rewriting the Global Offset Table (GOT) that links
inter-library callsites and variable references to their targets.
Because of this Gotcha cannot wrap intra-library calls (such as a call
to a static function in C) or calls in statically-linked binaries.
Binary rewriting technology such as DyninstAPI_ is more appropriate for these use
cases.

.. _DyninstAPI: https://github.com/dyninst/dyninst