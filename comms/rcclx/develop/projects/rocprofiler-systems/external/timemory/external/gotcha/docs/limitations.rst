===========================
Limitations
===========================

-------------------
General Limitations
-------------------

Operating system support
*************************

As the ELF is the file format used for .o object files, binaries, shared libraries and core dumps in Linux.
We currently only support Linux OS.

Intra and Intra-library calls
*****************************

Gotcha works by rewriting the Global Offset Table (GOT) that links inter-library callsites and variable references to their targets. 
Because of this Gotcha cannot wrap intra-library calls (such as a call to a static function in C) or calls in statically-linked binaries. 
Binary rewriting technology such as DyninstAPI_ is more appropriate for these use cases.
Additionally, the function pointer wrapping feature with GOTCHA only applies to function pointers created after wrapping functions.
The function pointers created before wrapping would not be wrapped by gotcha.

.. _DyninstAPI: https://github.com/dyninst/dyninst
