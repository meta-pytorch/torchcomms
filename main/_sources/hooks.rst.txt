Hooks
=====

torchcomms provides a hooks mechanism that allows you to intercept and monitor
collective operations. Hooks are useful for debugging, profiling, and
implementing custom monitoring solutions.

Hook Types
----------

torchcomms supports three types of hooks:

**Pre-hooks**
    Called before each collective operation starts. Receives operation metadata
    including operation name, tensors, and a unique operation ID.

**Post-hooks**
    Called after each collective operation completes. Receives the operation
    name, work object, and the same operation ID for correlation.

**Abort-hooks**
    Called before the process aborts due to a collective timeout or failure.
    Useful for capturing debug information before termination.

Custom Hooks
------------

You can register custom hook callbacks directly on a communicator using the
``register_pre_hook``, ``register_post_hook``, and ``register_abort_hook``
methods. This allows you to implement custom monitoring, logging, or debugging
logic.

Registering Hooks
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torchcomms
    from torchcomms._comms import OpName, PreHookArgs, PostHookArgs

    # Create a communicator
    device = torch.device("cuda:0")
    comm = torchcomms.new_comm("ncclx", device)

    # Define hook callbacks
    def my_pre_hook(args: PreHookArgs) -> None:
        print(f"Starting operation: {args.name} (op_id={args.op_id})")

    def my_post_hook(args: PostHookArgs) -> None:
        print(f"Completed operation: {args.name} (op_id={args.op_id})")

    def my_abort_hook() -> None:
        print("Process is about to abort, saving debug info...")

    # Register hooks - each returns a RemovableHandle
    pre_handle = comm.register_pre_hook(my_pre_hook)
    post_handle = comm.register_post_hook(my_post_hook)
    abort_handle = comm.register_abort_hook(my_abort_hook)

    # Run collective operations - hooks will be called
    tensor = torch.ones(10, device=device)
    comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

    # Unregister hooks when done
    pre_handle.remove()
    post_handle.remove()
    abort_handle.remove()

    comm.finalize()

Thread Safety Note
^^^^^^^^^^^^^^^^^^

Hooks are not thread-safe and must not be modified (registered or removed)
while a collective operation is in progress. Register hooks before starting
collective operations and remove them after all operations have completed.

FlightRecorderHook
------------------

The ``FlightRecorderHook`` is a built-in hook implementation that tracks all
collective operations for debugging purposes. It records operation metadata,
timing information, and completion status in a ring buffer.

The output format matches the OSS FlightRecorder format from PyTorch's
distributed module, so traces can be analyzed using the same ``fr_trace``
analysis tools.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    import torch
    import torchcomms
    from torchcomms.hooks import FlightRecorderHook

    # Create a communicator
    device = torch.device("cuda:0")
    comm = torchcomms.new_comm("ncclx", device)

    # Create and register a flight recorder hook
    recorder = FlightRecorderHook(max_entries=1024)
    recorder.register_with_comm(comm)

    # Run some collective operations
    tensor = torch.ones(10, device=device)
    comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
    comm.barrier(async_op=False)

    # Dump the recorded trace as JSON
    json_trace = recorder.dump_json()
    print(json_trace)

    # Optionally dump to a file
    recorder.dump_file(rank=comm.get_rank())

    # Unregister when done
    recorder.unregister()

    # Finalize the communicator
    comm.finalize()

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

The FlightRecorderHook uses the following environment variable:

``TORCHCOMM_FR_DUMP_TEMP_FILE``
    Controls the output location for ``dump_file()``. Files are written as
    ``<prefix><rank>`` where the prefix is the value of this variable.

API Reference
-------------

Hooks Module
^^^^^^^^^^^^

.. automodule:: torchcomms.hooks
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
    :member-order: bysource
