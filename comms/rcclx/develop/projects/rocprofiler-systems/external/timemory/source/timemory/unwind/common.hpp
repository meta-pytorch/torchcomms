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

#pragma once

#include "timemory/defines.h"
#include "timemory/unwind/types.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tim
{
namespace unwind
{
// finds the minimum size b/t LhsN and RhsN and then searches
// for the overlap.
template <size_t LhsN, size_t RhsN>
inline auto
get_common_stack(stack<LhsN> _lhs_v, stack<RhsN> _rhs_v)
{
    if constexpr(LhsN == RhsN)
    {
        return std::make_tuple(_lhs_v, _rhs_v);
    }
    else
    {
        constexpr size_t MinN = (LhsN < RhsN) ? LhsN : RhsN;
        constexpr size_t MaxN = (LhsN < RhsN) ? RhsN : LhsN;
        using return_type     = std::tuple<stack<MinN>, stack<MinN>>;
        stack<MinN>* _lhs     = nullptr;
        stack<MaxN>* _rhs     = nullptr;

        if constexpr(LhsN < RhsN)
        {
            _lhs = &_lhs_v;
            _rhs = &_rhs_v;
        }
        else
        {
            _lhs = &_rhs_v;
            _rhs = &_lhs_v;
        }

        auto _copy = [](const auto* _stack, size_t _beg) {
            constexpr size_t N    = (LhsN < RhsN) ? LhsN : RhsN;
            auto             _v   = stack<N>{};
            size_t           _idx = _beg;
            for(size_t i = 0; i < N; ++i, ++_idx)
            {
                if(_idx < _stack->call_stack.size())
                    _v.call_stack.at(i) = _stack->call_stack.at(_idx);
            }
            return _v;
        };

        //
        size_t _didx = _lhs->size() - (MaxN - MinN);
        size_t _lidx = 0;
        size_t _ridx = 0;
        for(size_t j = 0; j < _rhs->size(); ++j)
        {
            // only search, at most, in range of i to i + (MaxN - MinN)
            // we don't want to return them the overlap of just main/start_thread
            for(size_t i = 0; i < _didx; ++i)
            {
                if(_lhs->at(i)->address() == _rhs->at(j)->address())
                {
                    // first match
                    if(_lidx == 0 && _ridx == 0)
                    {
                        _lidx = i;
                        _ridx = j;
                    }
                    break;
                    if(_copy(_lhs, i) == _copy(_rhs, j))
                        return return_type{ _copy(_lhs, i), _copy(_rhs, j) };
                }
            }
        }

        return return_type{ _copy(_lhs, _lidx), _copy(_rhs, _ridx) };
    }
}
}  // namespace unwind
}  // namespace tim
