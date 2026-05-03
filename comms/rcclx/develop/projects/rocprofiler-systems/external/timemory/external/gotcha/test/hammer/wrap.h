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

#if !defined(_WRAP_H_)
#define _WRAP_H_

#include <gotcha/gotcha_types.h>

#include <map>
#include <set>
#include <string>
#include <utility>

extern void initMath();
class WrapperInfo {
 public:
  std::map<std::pair<int, int>, std::string> mathFunctionNames;
  std::map<std::pair<int, int>, std::pair<void *, gotcha_wrappee_handle_t *>>
      negFunctionPtrs;
  std::set<bool (*)(bool)> checkFunctions;
  void addName(int a, int b, void *func);
  void addNegPtr(int a, int b, void **negfunc, void **mathfptr);
  bool validate(bool is_neg);
  bool dowrap(std::string toolname);
};

extern WrapperInfo multinfo;
extern WrapperInfo addinfo;

void initMath();

#endif
