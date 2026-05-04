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

#include "wrap.h"

#include <dlfcn.h>

#include <chrono>
#include <iostream>
#include <map>
#include <set>
#include <utility>

#include "gotcha/gotcha.h"

using namespace std;

WrapperInfo multinfo;
WrapperInfo addinfo;

void WrapperInfo::addName(int a, int b, void *func) {
  Dl_info info;
  if (mathFunctionNames.find(make_pair(a, b)) != mathFunctionNames.end())
    return;

  int result = dladdr(func, &info);
  if (result && info.dli_sname) {
    mathFunctionNames[make_pair(a, b)] = string(info.dli_sname);
  }
}

void WrapperInfo::addNegPtr(int a, int b, void **negfunc, void **multfptr) {
  negFunctionPtrs[make_pair(a, b)] = make_pair(negfunc, multfptr);
}

bool WrapperInfo::validate(bool is_neg) {
  for (auto i = checkFunctions.begin(); i != checkFunctions.end(); i++)
    if (!(*i)(is_neg)) return false;
  return true;
}

bool WrapperInfo::dowrap(std::string toolname) {
  int j = 0;

  size_t size = mathFunctionNames.size();
  gotcha_binding_t *bindings = new gotcha_binding_t[size];

  for (auto i = mathFunctionNames.begin(); i != mathFunctionNames.end();
       i++, j++) {
    auto neg = negFunctionPtrs.find(make_pair(i->first.first, i->first.second));

    bindings[j].name = i->second.c_str();
    bindings[j].wrapper_pointer = neg->second.first;
    bindings[j].function_handle = neg->second.second;
  }

  auto start = chrono::steady_clock::now();
  gotcha_error_t result =
      gotcha_wrap(bindings, size, const_cast<char *>(toolname.c_str()));
  if (result != GOTCHA_SUCCESS) {
    cerr << "gotcha_wrap did not return success" << endl;
    return false;
  }
  auto end = chrono::steady_clock::now();
  cout << "gotcha_wrap took "
       << chrono::duration<double, milli>(end - start).count() / 1000.0
       << " seconds for " << toolname << endl;

  return true;
}
