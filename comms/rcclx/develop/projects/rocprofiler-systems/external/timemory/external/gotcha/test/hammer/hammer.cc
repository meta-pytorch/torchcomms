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

/**
 * This test is meant to hammer gotcha_wrap with an excessively large workload.
 * It should be linked with:
 *  -- one libmath.so that contains:
 *      - (X+1)*(Y+1) template instantiations for each Add, Mult
 *        class, which does the appropriately named math() operation on its
 *        template arguments. (e.g, Mult<5, 4>::math() == 20)
 *      - (X+1)*(Y+1) template instantiations of Neg, which we will wrap
 *        around the Add and Mult calls.  Each Neg<Y, X> instance contains a
 *        pair of function pointers that will be pointed at Add<Y, X>::math()
 *        and Mult<Y, X>::math, and functions that call those Add/Mult function
 *        pointer and negate the result.
 *  -- (X+1)*(Y+1) librefY_X.so libraries.  Each libref$y_$x.so contains:
 *      -  (X+1)*(Y+1) calls to Add<Y, X>::math(), and functions that validate
 *         whether that call returned the positive or negative result of its
 *         operation (which tells whether it was wrapped or not).
 *      - $x calls to Mult<Y, X>::math(), along with similar validation calls
 *        that are used in the Add case.
 *  -- one libwrap.so, which contains bookkeeping datastructures to collect
 *     function pointers and references to all the various Add/Mult data
 *structures, and gotcha calls to trigger wrapping.
 *
 * In summary, there are a bunch of calls across many libraries to Add<Y,
 *X>::math() and Mult<Y, X>::math() that do a simple math operation.  After
 *gotcha wrapping, they will start to return -1 * their result.
 **/

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "gotcha/gotcha.h"
#include "math.hpp"
#include "wrap.h"

using namespace std;

int main() {
  initMath();
  bool had_error = false;

  if (!multinfo.validate(false)) {
    cerr << "ERROR: mult sanity check failed" << endl;
    had_error = true;
  }
  if (!addinfo.validate(false)) {
    cerr << "ERROR: add sanity check failed" << endl;
    had_error = true;
  }

  multinfo.dowrap("Multiplies");

  if (!multinfo.validate(true)) {
    cerr << "ERROR: mult was not wrapped" << endl;
    had_error = true;
  }
  if (!addinfo.validate(false)) {
    cerr << "ERROR: add was modified by mult" << endl;
    had_error = true;
  }

  addinfo.dowrap("Additions");

  if (!multinfo.validate(true)) {
    cerr << "ERROR: mult was modified by add" << endl;
    had_error = true;
  }
  if (!addinfo.validate(true)) {
    cerr << "ERROR: add was not wrapped" << endl;
    had_error = true;
  }

  if (had_error)
    cout << "Failed hammering:" << endl;
  else
    cout << "Successfully hammered:" << endl;
  cout << "\t" << multinfo.checkFunctions.size() << "/"
       << addinfo.checkFunctions.size() << " mult/add libraries" << endl
       << "\twrapped " << multinfo.mathFunctionNames.size()
       << " mult functions and " << addinfo.mathFunctionNames.size()
       << " add functions" << endl
       << "\twith " << (X + 1) * (X + 2) * (Y + 1) / 2
       << " mult functions references" << endl
       << "\twith " << (X + 1) * (Y + 1) * addinfo.checkFunctions.size()
       << " add functions references" << endl;
  return had_error ? -1 : 0;
}
