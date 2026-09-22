/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Stands in for the staged name add_file_unique() would produce for
// param/common.h, whose basename clashes with device/common.h. RCCLX disables
// that renaming, so the _tmp name the sources include must resolve on its own.

#ifndef PARAM_COMMON_TMP_H_INCLUDED
#define PARAM_COMMON_TMP_H_INCLUDED

#include "common.h"

#endif // PARAM_COMMON_TMP_H_INCLUDED
