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

#include "timemory/data/macros.hpp"

#ifndef TIMEMORY_DATA_STREAM_CPP_
#    define TIMEMORY_DATA_STREAM_CPP_
#endif

#if !defined(TIMEMORY_DATA_STREAM_HPP_)
#    include "timemory/data/stream.hpp"
#endif

#include "timemory/mpl/stl.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/types.hpp"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tim
{
namespace data
{
TIMEMORY_DATA_INLINE
entry::entry(const std::string& _val, header& _hdr, bool _center, bool _left)
: base::stream_entry(_hdr)
, m_hdr(&_hdr)
, m_permit_empty(true)
{
    m_center = _center;
    m_left   = _left;
    base::stream_entry::construct(_val);
}

TIMEMORY_DATA_INLINE
entry::entry(const entry& _rhs)
: base::stream_entry(_rhs)
, m_hdr(_rhs.m_hdr)
{}

TIMEMORY_DATA_INLINE
stream::stream(char _delim, char _fill, format_flags _fmt, int _width, int _prec,
               bool _center)
: m_center(_center)
, m_fill(_fill)
, m_delim(_delim)
, m_width(_width)
, m_precision(_prec)
, m_format(_fmt)
{}

TIMEMORY_DATA_INLINE
int64_t
stream::index(const string_t& _val, const std::vector<string_t>& _obj)
{
    for(size_t i = 0; i < _obj.size(); ++i)
    {
        if(_obj.at(i) == _val)
            return static_cast<int64_t>(i);
    }
    return -1;
}

TIMEMORY_DATA_INLINE
int64_t
stream::insert(const string_t& _val, std::vector<string_t>& _obj)
{
    auto idx = index(_val, _obj);
    if(idx < 0)
    {
        idx = _obj.size();
        _obj.push_back(_val);
    }
    return idx;
}

TIMEMORY_DATA_INLINE
void
stream::set_prefix_begin(int val)
{
    m_prefix_begin = (val < 0) ? ((int) m_order.size()) : val;
}

TIMEMORY_DATA_INLINE
void
stream::set_prefix_end(int val)
{
    m_prefix_end = (val < 0) ? ((int) m_order.size()) : val;
}

TIMEMORY_DATA_INLINE
void
stream::insert_break(int val)
{
    m_break.insert((val < 0) ? ((int) m_order.size()) : val);
}

TIMEMORY_DATA_INLINE
void
stream::operator()(header _hdr)
{
    if(_hdr.get().empty())
        throw std::runtime_error("Header has no value");

    auto _w = std::max<int>(m_width, _hdr.get().length() + 2);
    _hdr.width(_w);

    m_order.push_back(m_name);
    auto _h = insert(m_name, m_headers);
    auto _n = m_headers[_h].second.size();
    _hdr.center(true);
    _hdr.row(0);
    _hdr.column(_n);
    m_headers[_h].second.push_back(_hdr);
}

TIMEMORY_DATA_INLINE
void
stream::operator()(entry _obj)
{
    if(_obj.get().empty() && !_obj.permit_empty())
        throw std::runtime_error("Entry has no value");

    auto _w = std::max<int>(m_width, _obj.get().length() + 2);
    _w      = std::max<int>(_w, _obj.get_header().width());

    _obj.width(_w);
    _obj.get_header().width(_w);

    auto _o = index(m_name, m_order);
    if(_o < 0)
        throw std::runtime_error(string_t("Missing entry for ") + m_name);

    auto _r = insert(m_name, m_entries);
    _obj.center(false);
    _obj.row(m_rows + 1);
    _obj.column(m_cols);
    m_entries[_r].second.push_back(_obj);
    ++m_cols;
}

TIMEMORY_DATA_INLINE
std::ostream&
stream::write(std::ostream& os) const
{
    // return if completely empty
    if(m_headers.empty())
        return os;

    // return if not entries
    if(m_entries.empty())
        return os;

    stringstream_t       ss;
    map_t<string_t, int> offset;

    write_banner(ss);

    write_separator(ss, '-');

    int64_t norder_col = 0;
    for(const auto& itr : m_order)
    {
        int64_t col = ++norder_col;

        stringstream_t _ss;
        const auto&    _key    = itr;
        auto           _offset = offset[_key]++;
        auto           _idx    = index(_key, m_headers);
        if(_idx < 0 || (_idx >= 0 && !(_offset < (int) m_headers[_idx].second.size())))
        {
            throw std::runtime_error("Error! indexing issue!");
        }

        const auto& hitr = m_headers[_idx].second.at(_offset);
        base::write_entry(_ss, hitr);

        ss << delim() << ' ' << _ss.str() << ' ';

        if(m_break.count(col) > 0)
            break;
    }

    // end the line
    ss << delim() << '\n';

    write_separator(ss, m_delim);

    auto write_empty = [&](stringstream_t& _ss, int64_t _hidx, int64_t _offset) {
        const auto& _hitr  = m_headers[_hidx].second;
        auto        _hsize = _hitr.size();
        const auto& _hdr   = _hitr.at(_offset % _hsize);
        _ss << delim() << ' ' << std::setw(_hdr.width() - 2) << "" << ' ';
    };

    offset.clear();

    for(int i = 0; i < m_rows; ++i)
    {
        bool just_broke = false;
        norder_col      = 0;
        for(const auto& itr : m_order)
        {
            just_broke  = false;
            int64_t col = ++norder_col;

            stringstream_t _ss;
            const auto&    _key    = itr;
            auto           _offset = offset[_key]++;

            auto _hidx = index(_key, m_headers);
            auto _eidx = index(_key, m_entries);

            if(_eidx < 0 && _hidx >= 0)
            {
                write_empty(ss, _hidx, _offset);
            }
            else
            {
                assert(_hidx >= 0);
                assert(_eidx >= 0);

                const auto& _eitr  = m_entries[_eidx].second;
                auto        _esize = _eitr.size();
                const auto& _itr   = _eitr.at(_offset % _esize);

                base::write_entry(_ss, _itr);
                ss << delim() << ' ' << _ss.str() << ' ';
            }

            if(col < (int64_t) m_order.size() && m_break.count(col) > 0)
            {
                ss << m_delim << '\n';
                just_broke = true;
                for(auto j = m_prefix_begin; j < m_prefix_end; ++j)
                    write_empty(ss, j, 0);
            }
        }
        if(!just_broke)
            ss << m_delim << '\n';

        if(m_separator_freq > 0)
        {
            if((i + 1) < m_rows && (i % m_separator_freq) == (m_separator_freq - 1))
                write_separator(ss, m_delim);
        }
    }

    write_separator(ss, '-');

    os << ss.str();
    return os;
}

TIMEMORY_DATA_INLINE
void
stream::clear()
{
    m_name = "";
    m_headers.clear();
    m_entries.clear();
}

TIMEMORY_DATA_INLINE
header&
stream::get_header(const string_t& _key, int64_t _n)
{
    auto idx = index(_key, m_headers);
    if(idx < 0)
    {
        stringstream_t ss;
        ss << "Missing header '" << _key << "'";
        throw std::runtime_error(ss.str());
    }

    if(!(_n < (int64_t) m_headers[idx].second.size()))
    {
        auto _size = m_headers[idx].second.size();
        return m_headers[idx].second[_n % _size];
    }

    return m_headers[idx].second[_n];
}

TIMEMORY_DATA_INLINE
int
stream::add_row()
{
    m_cols = 0;
    return ++m_rows;
}

TIMEMORY_DATA_INLINE
void
stream::sort(const std::function<bool(const std::string&, const std::string&)>& sorter,
             std::vector<std::string> keys, const std::set<std::string>& exclude)
{
    // if no keys were provided, add all of them
    if(keys.empty())
    {
        for(const auto& itr : m_order)
            keys.push_back(itr);
    }

    // the new headers
    order_map_t _order{};

    // sort the keys
    std::sort(keys.begin(), keys.end(), sorter);

    // generate the new layout in the order specified
    for(const auto& itr : keys)
    {
        bool found = false;
        for(auto hitr = m_order.begin(); hitr != m_order.end(); ++hitr)
        {
            if(*hitr == itr)
            {
                _order.push_back(*hitr);
                // remove entry
                m_order.erase(hitr);
                found = true;
                break;
            }
        }
        if(!found)
        {
            TIMEMORY_PRINT_HERE(
                "Warning! Expected header tag '%s' not found when sorting", itr.c_str());
        }
    }

    // insert any remaining not excluded
    for(const auto& itr : m_order)
    {
        if(exclude.count(itr) == 0)
            _order.push_back(itr);
    }

    // set the new headers
    m_order = _order;
}
}  // namespace data
}  // namespace tim
