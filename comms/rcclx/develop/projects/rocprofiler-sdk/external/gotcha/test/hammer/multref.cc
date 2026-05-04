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

#include <set>

#if !defined(STARTX) || !defined(STARTY)
#error Build ref.cc with STARTX and STARTY defined
#endif

using namespace std;

template <int A, int B>
class MultChecker {
 public:
  static bool doCheck(bool is_neg) {
    if (!MultChecker<A, B - 1>::doCheck(is_neg)) return false;

    result_t mult = Mult<A, B>::math();
    result_t target = A * B * (is_neg ? -1 : 1);
    return (mult == target);
  }
};

template <int A>
class MultChecker<A, 0> {
 public:
  static int doCheck(bool is_neg __attribute__((unused))) {
    return Mult<A, 0>::math() == 0;
  }
};

static bool check(bool is_neg) {
  return MultChecker<STARTY, STARTX>::doCheck(is_neg);
}

static void onLoad() __attribute__((constructor));
static void onLoad() { multinfo.checkFunctions.insert(check); }
