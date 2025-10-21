#!/usr/bin/env python3
"""
Reproduction script for CUDA initialization error with multiprocessing.

The issue occurs when:
1. The static variable might initialize CUDA (e.g. static const bool cuMemSysSupported = ctran::utils::commCudaLibraryInit();)
2. Python multiprocessing with various start methods spawns a child process
3. The child process tries to call torch.cuda.set_device()

This causes a CUDA initialization error because CUDA was already initialized
in the parent process before the fork.

Usage:
    python repro_cuda_multiprocess.py

    start_method: fork, forkserver, or spawn (default: forkserver)
"""

import multiprocessing as mp
import sys
import unittest


def child_process_func(device_id: int):
    """
    Function executed in the child process.
    Attempts to set CUDA device, which should fail if CUDA was already initialized.
    """
    print(f"Child process started, attempting to set CUDA device {device_id}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"Attempting to set device to {device_id}...")

            torch.cuda.set_device(device_id)
            device = torch.cuda.current_device()

            print(f"Successfully got current device: {device}")
        else:
            print("CUDA not available")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


class TestCudaInit(unittest.TestCase):
    """Test class for CUDA initialization error with multiprocessing."""

    def test_cuda_init(self):
        for start_method in ["fork", "forkserver", "spawn"]:
            print(f"\nTesting start method: {start_method}")
            ctx = mp.get_context(start_method)
            device_id = 0
            process = ctx.Process(target=child_process_func, args=(device_id,))

            process.start()
            process.join()

            print(f"\nChild process exited with code: {process.exitcode}")
            if process.exitcode != 0:
                print("❌ CUDA initialization error occurred!")
                sys.exit(1)
            else:
                print("✓ Success: No CUDA initialization error")


if __name__ == "__main__":
    unittest.main()
