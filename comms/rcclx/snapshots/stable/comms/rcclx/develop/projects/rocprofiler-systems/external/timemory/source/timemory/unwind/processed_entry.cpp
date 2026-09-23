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

#include "timemory/unwind/processed_entry.hpp"

#include "timemory/utility/filepath.hpp"
#include "timemory/utility/procfs/maps.hpp"

#include <cstdlib>

namespace tim
{
namespace unwind
{
void
processed_entry::construct(processed_entry& _v, file_map_t* _files, bool _prefer_dlinfo)
{
    _v.info = dlinfo::construct(_v.address - _v.offset);

    auto _dlinfo_update = [&](bool _realpath) {
        if(_realpath)
        {
            _v.location =
                filepath::realpath(std::string{ _v.info.location.name }, nullptr, false);
            _v.line_address =
                (_v.info.symbol.address() - _v.info.location.address()) + _v.offset;
        }
        else
        {
            _v.location = std::string{ _v.info.location.name };
            _v.line_address =
                (_v.info.symbol.address() - _v.info.location.address()) + _v.offset;
        }
    };

    auto _procfs_update = [&]() {
        auto _map = procfs::find_map(_v.address);
        if(!_map.is_empty() && !_map.pathname.empty())
        {
            _v.location     = _map.pathname;
            _v.line_address = (_v.address - _map.load_address) + _map.offset;
        }
        return _map.is_empty();
    };

    if(_prefer_dlinfo)
    {
        if(_v.info && _v.location.empty())
            _dlinfo_update(true);

        if(_v.location.empty())
            _procfs_update();
    }
    else
    {
        auto _empty_map = _procfs_update();
        if(_v.info && (_empty_map || _v.location.empty()))
            _dlinfo_update(false);
    }

    if(_files != nullptr && !_v.location.empty() && filepath::exists(_v.location))
    {
        auto _get_file = [&_files](const auto& _val) {
            auto _val_real = filepath::realpath(_val, nullptr, false);
            if(_files->find(_val) == _files->end())
                _files->emplace(_val, std::make_shared<bfd_file>(_val));
            return _files->at(_val);
        };

        auto _bfd = _get_file(_v.location);
        if(_bfd && *_bfd)
        {
            _v.lineinfo = addr2line(_bfd, { _v.line_address, _v.address });
            if(_v.lineinfo && !_v.lineinfo.lines.empty())
                _v.lineno = _v.lineinfo.lines.front().line;
        }
    }
}
}  // namespace unwind
}  // namespace tim
