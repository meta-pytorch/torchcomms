# Template Component

The Template component showcases a possible approach to component design that
varies from the more traditional monolitic approach, used in some existing
components.

The Template component is split into a front-end (template.c) and a back-end
(vendor\_{dispatch,common,profiler}.c). The goal of this layered design is to
decouple changes in the vendor libraries from the existing implementation.

If, for example, the vendor library introduces a new set of interfaces and
deprecates the old ones, a new vendor\_dispatch\_v2.c can be written to add
support for the new vendor library interface without affecting the old vendor
related code (i.e., vendor\_profiler.c). This can still be kept for backward
compatibility with older vendor library versions. The dispatch layer (i.e.,
vendor\_dispatch.c) takes care of forwarding profiling calls to the right
vendor library version.

Any code that is shared between the different vendor library versions is placed
in vendor\_common.c. This can contain common init routines (e.g., for the vendor
runtime system initialization, like cuda or hsa), utility routines and shared
data structures (e.g., device information tables).

The template component emulates support for a generic vendor library to profile
associated device hardware counters. The implementation is fairly detailed in
stating the design decisions. For example, there is a clear separation between
the front-end and the back-end with no sharing of data between the two. The
front-end only takes care of doing some book keeping work (perhaps rearrange the
order of events as it sees fit) and calling into the back-end functions to do
actual work.
