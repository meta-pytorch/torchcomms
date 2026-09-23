/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Stands in for the staged name add_file_unique() would produce for
// param/utils.h, whose basename clashes with include/utils.h. RCCLX disables
// that renaming, so the _tmp name the sources include must resolve on its own.

#ifndef PARAM_UTILS_TMP_H_INCLUDED
#define PARAM_UTILS_TMP_H_INCLUDED

#include "utils.h"

#endif // PARAM_UTILS_TMP_H_INCLUDED
