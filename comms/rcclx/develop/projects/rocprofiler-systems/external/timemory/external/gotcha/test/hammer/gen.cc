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

using namespace std;

template <int A, int B>
class Generate {
 public:
  static void init() {
    Mult<A, B>::init();
    Add<A, B>::init();
    Neg<A, B>::init();

    // Extra gens here prevents recursive template overflow
    (void)Generate<A, B / 2>::init;
    (void)Generate<A / 2, B>::init;

    Generate<A, B - 1>::init();
  }
};

template <int A>
class Generate<A, 0> {
 public:
  static void init() {
    Mult<A, 0>::init();
    Add<A, 0>::init();
    Neg<A, 0>::init();

    Generate<A - 1, X>::init();
  }
};

template <>
class Generate<0, 0> {
 public:
  static void init() {
    Mult<0, 0>::init();
    Add<0, 0>::init();
    Neg<0, 0>::init();
  }
};

template <int A, int B>
gotcha_wrappee_handle_t *Neg<A, B>::mathfn_mult_handle = NULL;

template <int A, int B>
gotcha_wrappee_handle_t *Neg<A, B>::mathfn_add_handle = NULL;

void initMath() { Generate<Y, X>::init(); }
