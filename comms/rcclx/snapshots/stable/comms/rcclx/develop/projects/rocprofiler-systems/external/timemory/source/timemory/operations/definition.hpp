// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * \file timemory/operations/definition.hpp
 * \brief The definitions for the types in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/types/add_secondary.hpp"
#include "timemory/operations/types/add_statistics.hpp"
#include "timemory/operations/types/assemble.hpp"
#include "timemory/operations/types/audit.hpp"
#include "timemory/operations/types/base_printer.hpp"
#include "timemory/operations/types/cache.hpp"
#include "timemory/operations/types/cleanup.hpp"
#include "timemory/operations/types/compose.hpp"
#include "timemory/operations/types/construct.hpp"
#include "timemory/operations/types/copy.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/operations/types/derive.hpp"
#include "timemory/operations/types/echo_measurement.hpp"
#include "timemory/operations/types/finalize/dmp_get.hpp"
#include "timemory/operations/types/finalize/flamegraph.hpp"
#include "timemory/operations/types/finalize/get.hpp"
#include "timemory/operations/types/finalize/merge.hpp"
#include "timemory/operations/types/finalize/mpi_get.hpp"
#include "timemory/operations/types/finalize/print.hpp"
#include "timemory/operations/types/finalize/upc_get.hpp"
#include "timemory/operations/types/fini.hpp"
#include "timemory/operations/types/fini_storage.hpp"
#include "timemory/operations/types/generic.hpp"
#include "timemory/operations/types/get.hpp"
#include "timemory/operations/types/init.hpp"
#include "timemory/operations/types/init_storage.hpp"
#include "timemory/operations/types/mark.hpp"
#include "timemory/operations/types/math.hpp"
#include "timemory/operations/types/measure.hpp"
#include "timemory/operations/types/node.hpp"
#include "timemory/operations/types/print.hpp"
#include "timemory/operations/types/print_header.hpp"
#include "timemory/operations/types/print_statistics.hpp"
#include "timemory/operations/types/print_storage.hpp"
#include "timemory/operations/types/record.hpp"
#include "timemory/operations/types/reset.hpp"
#include "timemory/operations/types/sample.hpp"
#include "timemory/operations/types/serialization.hpp"
#include "timemory/operations/types/set.hpp"
#include "timemory/operations/types/start.hpp"
#include "timemory/operations/types/stop.hpp"
#include "timemory/operations/types/storage_initializer.hpp"
#include "timemory/operations/types/store.hpp"
//
#include "timemory/components/types.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/tpls/cereal/archives.hpp"
//
//--------------------------------------------------------------------------------------//
//
