// MIT License
//
// Copyright (c) 2019 Jonathan R. Madsen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#pragma once

#include "macros.hh"

//======================================================================================//

namespace constants
{
// pi numbers
constexpr float pi     = static_cast<float>(M_PI);
constexpr float halfpi = 0.5f * pi;
constexpr float twopi  = 2.0f * pi;
// 2 * the hardware epsilon for floating point numbers
constexpr float epsilonf = 2.0f * std::numeric_limits<float>::epsilon();
// convert radians to degrees: theta_in_radians * degrees
// convert degrees to radians: theta_in_degrees / degrees
constexpr float degrees = 180.0f / pi;
}  // namespace constants

//======================================================================================//
