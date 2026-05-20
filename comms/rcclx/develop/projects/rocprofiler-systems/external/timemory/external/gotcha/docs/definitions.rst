================
Definitions
================

This section defines some terms used throughout the document.

GOT
****

The Global Offset Table, or GOT, is a section of a computer program's (executables and shared libraries) memory used to enable computer program code compiled as an ELF file to run correctly, independent of the memory address where the program's code or data is loaded at runtime.
More details can be read at `GOT Documentation`_.


ELF
****

In computing, the Executable and Linkable Format[2] (ELF, formerly named Extensible Linking Format), is a common standard file format for executable files, object code, shared libraries, and core dumps.


LD_PRELOAD
**********

LD_PRELOAD is a powerful and advanced feature in the Linux dynamic linker that allows users to preload shared object files into the address space of a process.
Read more at `LD_PRELOAD Documentation`_.

ABI-compatibility
*****************

An application binary interface (ABI) is an interface between two binary program modules. An ABI defines how data structures or computational routines are accessed in machine code, which is a low-level, hardware-dependent format.

.. explicit external hyperlink targets

.. _`GOT Documentation`: https://refspecs.linuxfoundation.org/ELF/zSeries/lzsabi0_zSeries/x2251.html
.. _`LD_PRELOAD Documentation`: https://man7.org/linux/man-pages/man8/ld.so.8.html


