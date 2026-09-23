//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/allinea/components.hpp
 * \brief Implementation of the allinea component(s)
 */

#pragma once

#include "timemory/components/allinea/backends.hpp"
#include "timemory/components/allinea/types.hpp"
#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
// Control AllineaMap sampler
//
/// \struct tim::component::allinea_map
/// \brief Controls the AllineaMap sampling profiler
///
struct allinea_map
: public base<allinea_map, void>
, private policy::instance_tracker<allinea_map, false>
{
    using value_type   = void;
    using this_type    = allinea_map;
    using base_type    = base<this_type, value_type>;
    using tracker_type = policy::instance_tracker<allinea_map, false>;

    static std::string label() { return "allinea_map"; }
    static std::string description() { return "Controls the AllineaMAP sampler"; }

    static void global_init() { backend::allinea::stop_sampling(); }
    static void global_finalize() { backend::allinea::stop_sampling(); }

    void start()
    {
        tracker_type::start();
        if(tracker_type::m_tot == 0)
            backend::allinea::start_sampling();
    }

    void stop()
    {
        tracker_type::stop();
        if(tracker_type::m_tot == 0)
            backend::allinea::stop_sampling();
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
