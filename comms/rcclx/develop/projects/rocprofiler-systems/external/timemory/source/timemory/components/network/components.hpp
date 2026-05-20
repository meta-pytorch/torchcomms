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

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/network/backends.hpp"
#include "timemory/components/network/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
/// \struct tim::component::network
/// \brief Records the network activity
//
struct network_stats : public base<network_stats, cache::network_stats>
{
    static constexpr size_t data_size = cache::network_stats::data_size;

    using value_type        = cache::network_stats;
    using this_type         = network_stats;
    using base_type         = base<this_type, value_type>;
    using data_array_type   = typename cache::network_stats::data_type;
    using string_array_type = typename cache::network_stats::string_array_type;
    using iface_map_type    = std::unordered_map<std::string, string_array_type>;

    static std::string               label();
    static std::string               description();
    static data_array_type           data_array(value_type);
    static data_array_type           unit_array();
    static string_array_type         label_array();
    static string_array_type         display_unit_array();
    static string_array_type         description_array();
    static value_type                record(const std::string& _interface);
    static value_type                record(const std::vector<std::string>& _interfaces);
    static std::vector<std::string>& get_interfaces();

    network_stats(std::string);
    network_stats(std::vector<std::string>);

    network_stats()                     = default;
    ~network_stats()                    = default;
    network_stats(const network_stats&) = default;
    network_stats(network_stats&&)      = default;

    network_stats& operator=(const network_stats&) = default;
    network_stats& operator=(network_stats&&) = default;

    data_array_type get() const;
    std::string     get_display() const;

    void sample();
    void start();
    void stop();

private:
    bool                     m_first      = true;
    std::vector<std::string> m_interfaces = get_interfaces();

private:
    static const string_array_type& get_iface_paths(const std::string& _name);
};
//
inline std::string
network_stats::label()
{
    return "network_stats";
}
//
inline std::string
network_stats::description()
{
    return "Reports network bytes, packets, errors, dropped";
}
//
inline network_stats::value_type
network_stats::record(const std::string& _interface)
{
    if(_interface.empty())
        return cache::network_stats{};
    // const auto& _iface_paths = get_iface_paths(_name);
    return cache::network_stats{ _interface };
}
//
inline network_stats::value_type
network_stats::record(const std::vector<std::string>& _interfaces)
{
    if(_interfaces.empty())
        return cache::network_stats{};
    cache::network_stats _data{};
    for(const auto& itr : _interfaces)
    {
        if(!itr.empty())
            _data += cache::network_stats{ itr };
    }
    return _data;
}
//
inline network_stats::network_stats(std::string _interface)
: m_interfaces{ std::move(_interface) }
{}
//
inline network_stats::network_stats(std::vector<std::string> _interfaces)
: m_interfaces{ std::move(_interfaces) }
{}
//
inline std::vector<std::string>&
network_stats::get_interfaces()
{
    static std::string              _last{};
    static std::string              _value{};
    static std::vector<std::string> _instance{};

    auto* _settings = settings::instance();
    if(!_settings)
        return _instance;
    _settings->get(TIMEMORY_SETTINGS_KEY("NETWORK_INTERFACE"), _value, true);

    auto _update_instance = []() {
        _instance.clear();
        for(auto&& itr : delimit(_value, " ,\t;"))
        {
            if(itr != "." && itr != "..")
                _instance.emplace_back(std::move(itr));
        }
    };

    if(_instance.empty())
    {
        if(!_value.empty())
        {
            _update_instance();
            _last = _value;
        }
        else
        {
            for(auto&& itr : utility::filesystem::list_directory("/sys/class/net"))
            {
                if(itr != "." && itr != "..")
                    _instance.emplace_back(std::move(itr));
            }
            if(_instance.empty())
                return _instance;
            std::stringstream _ss;
            for(const auto& itr : _instance)
                _ss << ", " << itr;
            _value = _last = (_ss.str().length() > 2) ? _ss.str().substr(2) : _ss.str();
            _settings->set(TIMEMORY_SETTINGS_KEY("NETWORK_INTERFACE"), _value, true);
        }
    }
    else if(_last != _value)
    {
        _update_instance();
        _last = _value;
    }

    return _instance;
}
//
inline network_stats::data_array_type
network_stats::get() const
{
    auto _data = load();
    if(!m_first && !get_is_transient())
    {
        _data.get_data().fill(0);
        return _data.get_data();
    }

    static auto _data_units = cache::network_stats::data_units();
    for(size_t i = 0; i < data_size; ++i)
        _data.get_data()[i] /= _data_units[i];
    return _data.get_data();
}
//
inline std::string
network_stats::get_display() const
{
    auto&& _get_display = [](const int64_t& _val, const std::string& _lbl,
                             const std::string& _units) {
        std::stringstream ss;
        ss.setf(base_type::get_format_flags());
        ss << std::setw(base_type::get_width())
           << std::setprecision(base_type::get_precision()) << _val;
        if(!_units.empty())
            ss << " " << _units;
        if(!_lbl.empty())
            ss << " " << _lbl;
        return ss.str();
    };

    auto&&            _val    = value_type{ get() };
    auto              _disp   = cache::network_stats::data_display_units();
    auto              _labels = cache::network_stats::data_labels();
    std::stringstream ss;
    for(size_t i = 0; i < data_size; ++i)
        ss << ", " << _get_display(_val.get_data()[i], _labels[i], _disp[i]);
    return (ss.str().length() > 2) ? ss.str().substr(2) : ss.str();
}
//
inline void
network_stats::sample()
{
    if(m_first)
    {
        m_first = false;
        value   = record(m_interfaces);
    }
    else
    {
        auto _current = record(m_interfaces);
        accum += _current - value;
        value = _current;
        set_is_transient(true);
    }
}
//
inline void
network_stats::start()
{
    if(m_interfaces.empty())
        m_interfaces = get_interfaces();
    value = record(m_interfaces);
}
//
inline void
network_stats::stop()
{
    value = record(m_interfaces) - value;
    accum += value;
}
//
inline network_stats::data_array_type
network_stats::data_array(value_type _data)
{
    return _data.get_data();
}
//
inline network_stats::data_array_type
network_stats::unit_array()
{
    return cache::network_stats::data_units();
}
//
inline network_stats::string_array_type
network_stats::label_array()
{
    return cache::network_stats::data_labels();
}
//
inline network_stats::string_array_type
network_stats::display_unit_array()
{
    return cache::network_stats::data_display_units();
}
//
inline network_stats::string_array_type
network_stats::description_array()
{
    return cache::network_stats::data_descriptions();
}
//
inline const network_stats::string_array_type&
network_stats::get_iface_paths(const std::string& _name)
{
    static iface_map_type _iface_paths{};
    auto                  itr = _iface_paths.find(_name);
    if(itr == _iface_paths.end())
    {
        string_array_type        _data{};
        static const std::string _base = "/sys/class/net/";
        for(size_t i = 0; i < data_size; ++i)
        {
            _data.at(i) = _base + _name + "/statistics/" +
                          cache::network_stats::data_labels().at(i);
        }
        auto_lock_t _lk(type_mutex<network_stats>());
        _iface_paths.insert({ _name, _data });
    }
    itr = _iface_paths.find(_name);
    return itr->second;
}
}  // namespace component
}  // namespace tim
//
//======================================================================================//
