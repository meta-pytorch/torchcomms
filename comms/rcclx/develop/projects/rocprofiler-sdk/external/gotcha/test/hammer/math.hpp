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

#if !defined(MULT_HPP_)
#define MULT_HPP_

#include <gotcha/gotcha.h>

#include <map>
#include <set>
#include <string>
#include <utility>

#include "wrap.h"

#define X XSIZE
#define Y YSIZE

typedef signed long result_t;

template <int A, int B>
class Mult {
 public:
  static void init() { multinfo.addName(A, B, (void *)Mult<A, B>::math); }

  static result_t math() { return A * B; }
};

template <int A, int B>
class Add {
 public:
  static void init() { addinfo.addName(A, B, (void *)Add<A, B>::math); }

  static result_t math() { return A + B; }
};

template <int A, int B>
class Neg {
 public:
  static gotcha_wrappee_handle_t *mathfn_add_handle;
  static gotcha_wrappee_handle_t *mathfn_mult_handle;
  static void init() {
    multinfo.addNegPtr(A, B, (void **)Neg<A, B>::mult_math_wrapper,
                       (void **)&mathfn_mult_handle);
    addinfo.addNegPtr(A, B, (void **)Neg<A, B>::add_math_wrapper,
                      (void **)&mathfn_add_handle);
  }

  static result_t mult_math_wrapper() {
    auto mathfn_mult = reinterpret_cast<decltype(&mult_math_wrapper)>(
        gotcha_get_wrappee(mathfn_mult_handle));
    return mathfn_mult() * -1;
  }

  static result_t add_math_wrapper() {
    auto mathfn_add = reinterpret_cast<decltype(&add_math_wrapper)>(
        gotcha_get_wrappee(mathfn_add_handle));
    return mathfn_add() * -1;
  }
};

#endif
