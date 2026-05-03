# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import cython

def subtract_exclusive_metric_vals(long nid, long parent_nid, double[:] metrics, long num_stmt_nodes, long stride):
  cdef long ref_nid = nid
  cdef long ref_pnid = parent_nid

  # compiler directive
  with cython.boundscheck(False):
  # we are modifying metrics in place here
  # since they are passed by refrence via their
  # memory
    for i in range(num_stmt_nodes):
      metrics[ref_pnid-1] -= metrics[ref_nid-1]

      ref_nid += stride
      ref_pnid += stride
