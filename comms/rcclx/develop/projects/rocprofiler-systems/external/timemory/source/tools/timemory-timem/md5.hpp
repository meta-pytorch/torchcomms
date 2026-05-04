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

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

// helper function
std::string
compute_md5(const std::string& inp);

// a small class for calculating MD5 hashes of strings or byte arrays
//
// usage: 1) feed it blocks of uchars with update()
//      2) finalize()
//      3) get hexdigest() string
//      or
//      MD5(std::string).hexdigest()
//
// assumes that char is 8 bit and int is 32 bit
class md5sum
{
public:
    using size_type                = uint32_t;  // must be 32bit
    static constexpr int blocksize = 64;

    md5sum(const std::string& text);
    md5sum()              = default;
    ~md5sum()             = default;
    md5sum(const md5sum&) = default;
    md5sum(md5sum&&)      = default;

    md5sum& operator=(const md5sum&) = default;
    md5sum& operator=(md5sum&&) = default;

    md5sum&              update(const unsigned char* buf, size_type length);
    md5sum&              update(const char* buf, size_type length);
    md5sum&              finalize();
    std::string          hexdigest() const;
    friend std::ostream& operator<<(std::ostream&, md5sum md5);

private:
    void transform(const uint8_t block[blocksize]);

    bool finalized = false;
    // 64bit counter for number of bits (lo, hi)
    std::array<uint32_t, 2>        count = { 0, 0 };
    std::array<uint8_t, blocksize> buffer{};  // overflow bytes from last 64 byte chunk
    // digest so far, initialized to magic initialization constants.
    std::array<uint32_t, 4> state = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };
    std::array<uint8_t, 16> digest{};  // result
};
