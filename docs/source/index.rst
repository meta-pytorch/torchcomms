:github_url: https://github.com/meta-pytorch/torchcomms

torchcomms
==========

**torchcomms** is an experimental, lightweight communication API for PyTorch Distributed (PTD).
It provides a simplified, object-oriented interface for distributed collective operations with multiple
out-of-the-box backends, including Meta's production-tested **NCCLX** backend that powers all generative AI services.

.. raw:: html

   <div style="margin-bottom: 2em;"></div>

----

Browse the documentation and Examples
-------------------------------------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🚀 Quick Start
        :link: getting_started
        :link-type: doc

        New to torchcomms? Start here to learn how to install and use
        torchcomms for distributed communication.

    .. grid-item-card:: 📚 API Reference
        :link: api
        :link-type: doc

        Complete API documentation for all torchcomms classes,
        functions, and backends.

    .. grid-item-card:: 💻 Code Examples
        :link: https://github.com/meta-pytorch/torchcomms/tree/main/comms/torchcomms/examples

        Explore practical examples showing how to use torchcomms
        in real-world distributed applications.

    .. grid-item-card:: 🐛 Report Issues
        :link: https://github.com/meta-pytorch/torchcomms/issues

        Found a bug or have a feature request? Let us know on GitHub.

----

.. raw:: html

   <div style="margin-bottom: 2em;"></div>

Why torchcomms?
---------------

torchcomms addresses several key challenges in distributed PyTorch training:

* **Simplified API**: Clean, object-oriented interface that abstracts away low-level communication details
* **Backend Flexibility**: Easily switch between different communication backends (NCCLX, NCCL, Gloo, RCCL) without changing your code
* **Production-Ready**: NCCLX backend is battle-tested in Meta's production environments powering large-scale AI workloads
* **Type Safety**: Full type hints and validation for better development experience
* **Performance**: Optimized implementations with support for GPU-accelerated communication

Key Features
------------

Multiple Backends
^^^^^^^^^^^^^^^^^

torchcomms supports several communication backends out of the box:

* **NCCLX**: Meta's enhanced NCCL implementation with additional optimizations
* **NCCL**: NVIDIA's Collective Communications Library for multi-GPU communication
* **Gloo**: Facebook's collective communications library for both CPU and GPU
* **RCCL**: AMD ROCm Collective Communications Library for AMD GPUs

Comprehensive Collective Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All standard distributed operations are supported:

* AllReduce, ReduceScatter, AllGather
* Broadcast, Reduce, Scatter, Gather
* Send, Recv for point-to-point communication
* Support for both synchronous and asynchronous operations

Flexible Group Management
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and manage process groups with ease:

* Initialize groups with different backends
* Support for sub-groups and hierarchical communication patterns
* Automatic resource management and cleanup

.. raw:: html

   <div style="margin-bottom: 2em;"></div>

----

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting_started
   api
