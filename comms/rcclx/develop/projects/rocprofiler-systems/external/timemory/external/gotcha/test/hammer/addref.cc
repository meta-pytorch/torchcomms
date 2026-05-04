/*
This file is part of GOTCHA.  For copyright information see the COPYRIGHT
file in the top level directory, or at
https://github.com/LLNL/gotcha/blob/master/COPYRIGHT
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (as published by the Free
Software Foundation) version 2.1 dated February 1999.  This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the terms and conditions of the GNU Lesser General Public License
for more details.  You should have received a copy of the GNU Lesser General
Public License along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "math.hpp"
#include "externals.h"

template <int A, int B>
class AddChecker {
 public:
  static bool doCheck(bool is_neg) {
    (void)AddChecker<A / 2, B>::doCheck;
    (void)AddChecker<A, B / 2>::doCheck;

    if (!AddChecker<A, B - 1>::doCheck(is_neg)) return false;

    result_t add = Add<A, B>::math();
    result_t target = (A + B) * (is_neg ? -1 : 1);
    return (add == target);
  }
};

template <int A>
class AddChecker<A, 0> {
 public:
  static bool doCheck(bool is_neg) {
    if (!AddChecker<A - 1, X>::doCheck(is_neg)) return false;

    result_t add = Add<A, 0>::math();
    result_t target = A * (is_neg ? -1 : 1);
    return (add == target);
  }
};

template <>
class AddChecker<0, 0> {
 public:
  static bool doCheck(bool) {
    result_t add = Add<0, 0>::math();
    return (add == 0);
  }
};

static bool check(bool is_neg) {
  return AddChecker<YSIZE, XSIZE>::doCheck(is_neg);
}

static void onLoad() __attribute__((constructor));
static void onLoad() { addinfo.checkFunctions.insert(check); }
