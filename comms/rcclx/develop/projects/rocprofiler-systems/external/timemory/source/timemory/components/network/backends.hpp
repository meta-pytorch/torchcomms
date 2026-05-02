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

#pragma once

#include "timemory/backends/process.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/tpls/cereal/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/macros.hpp"

#include <iostream>
#include <string>
#include <vector>

#if defined(CXX17) && defined(__cpp_lib_filesystem)
#    include <filesystem>
#elif defined(_UNIX)
#    include <dirent.h>
#endif

namespace tim
{
//
namespace utility
{
namespace filesystem
{
#if defined(CXX17) && defined(__cpp_lib_filesystem)
inline auto
list_directory(string_view_cref_t _path)
{
    namespace fs = std::filesystem;
    std::vector<std::string> _entries{};
    for(const auto& itr : fs::directory_iterator(_path))
    {
        _entries.emplace_back(itr.path().filename());
    }
    return _entries;
}
#elif defined(_UNIX)
inline auto
list_directory(string_view_cref_t _path)
{
    DIR*                     _dir;
    struct dirent*           _diread;
    std::vector<std::string> _entries{};

    if((_dir = opendir(_path.data())) != nullptr)
    {
        while((_diread = readdir(_dir)) != nullptr)
        {
            _entries.emplace_back(_diread->d_name);
        }
        closedir(_dir);
    }
    else
    {
        perror("opendir");
    }
    return _entries;
}
#else
inline auto list_directory(string_view_cref_t) { return std::vector<std::string>{}; }
#endif
}  // namespace filesystem
}  // namespace utility
//
namespace cache
{
struct network_stats
{
    static constexpr size_t data_size = 8;
    using value_type                  = int64_t;
    using string_array_type           = std::array<std::string, data_size>;
    using strvec_type                 = std::vector<std::string>;
    using data_type                   = std::array<value_type, data_size>;

    static inline auto get_filename()
    {
        return TIMEMORY_JOIN('/', "/proc", process::get_target_id(), "net/dev");
    }

    template <typename Tp, size_t N>
    static inline auto& read(const std::array<std::string, N>& _paths,
                             std::array<Tp, data_size>&        _data)
    {
        _data.fill(0);
        read(_paths, _data, std::make_index_sequence<N>{});
        return _data;
    }

    template <size_t N>
    static inline auto read(const std::array<std::string, N>& _paths)
    {
        data_type _data{};
        _data = read(_paths, _data);
        return _data;
    }

public:
#if defined(_LINUX)
    template <size_t N>
    explicit network_stats(const std::array<std::string, N>& _paths)
    : m_data{ read(_paths) }
    {}

    explicit network_stats(const std::string& _iface)
    : m_data{ read(get_filename(), _iface) }
    {}
#else
    template <size_t N>
    explicit network_stats(const std::array<std::string, N>&)
    {}

    explicit network_stats(const std::string&) {}
#endif

    explicit network_stats(data_type _data)
    : m_data{ _data }
    {}

    network_stats() { m_data.fill(0); }
    ~network_stats()                        = default;
    network_stats(const network_stats&)     = default;
    network_stats(network_stats&&) noexcept = default;

    network_stats& operator=(const network_stats&) = default;
    network_stats& operator=(network_stats&&) noexcept = default;

    data_type&       get_data() { return m_data; }
    const data_type& get_data() const { return m_data; }

public:
    bool operator==(const network_stats& rhs) const
    {
        for(size_t i = 0; i < data_size; ++i)
            if(m_data[i] != rhs.m_data[i])
                return false;
        return true;
    }

    bool operator<(const network_stats& rhs) const
    {
        for(size_t i = 0; i < data_size; ++i)
            if(m_data[i] != rhs.m_data[i])
                return m_data[i] < rhs.m_data[i];
        return false;
    }

    bool operator>(const network_stats& rhs) const
    {
        return !(*this < rhs && *this == rhs);
    }

    bool operator!=(const network_stats& rhs) const { return !(*this == rhs); }
    bool operator<=(const network_stats& rhs) const { return !(*this > rhs); }
    bool operator>=(const network_stats& rhs) const { return !(*this < rhs); }

    network_stats& operator+=(const network_stats& rhs)
    {
        for(size_t i = 0; i < data_size; ++i)
            m_data[i] += rhs.m_data[i];
        return *this;
    }

    network_stats& operator-=(const network_stats& rhs)
    {
        for(size_t i = 0; i < data_size; ++i)
        {
            if(m_data[i] > rhs.m_data[i])
                m_data[i] -= rhs.m_data[i];
            else
                m_data[i] = 0;
        }
        return *this;
    }

    network_stats& operator/=(int64_t rhs)
    {
        for(size_t i = 0; i < data_size; ++i)
            m_data[i] /= rhs;
        return *this;
    }

    friend network_stats operator+(const network_stats& lhs, const network_stats& rhs)
    {
        return network_stats{ lhs } += rhs;
    }

    friend network_stats operator-(const network_stats& lhs, const network_stats& rhs)
    {
        return network_stats{ lhs } -= rhs;
    }

    friend network_stats operator/(const network_stats& lhs, int64_t rhs)
    {
        return network_stats{ lhs } /= rhs;
    }

public:
    static const string_array_type& data_labels()
    {
        static string_array_type _data{ "rx_bytes",   "rx_packets", "rx_errors",
                                        "rx_dropped", "tx_bytes",   "tx_packets",
                                        "tx_errors",  "tx_dropped" };
        return _data;
    }

    static const string_array_type& data_descriptions()
    {
        static string_array_type _data{ "bytes received over network",
                                        "packets received over network",
                                        "errors during network reception",
                                        "dropped packets during network reception",
                                        "bytes transmitted over network",
                                        "packets transmitted over network",
                                        "errors during network transmission",
                                        "dropped packets during network transmission" };
        return _data;
    }

    static const auto& data_units()
    {
        static std::array<value_type, data_size> _data{ units::kilobyte, 1, 1, 1,
                                                        units::kilobyte, 1, 1, 1 };
        if(!settings::memory_units().empty())
        {
            for(auto&& idx : { 0, 4 })
                _data.at(idx) =
                    std::get<1>(units::get_memory_unit(settings::memory_units()));
        }
        return _data;
    }

    static const auto& data_display_units()
    {
        static string_array_type _data{};
        for(auto&& idx : { 0, 4 })
            _data.at(idx) = units::mem_repr(data_units().at(idx));
        return _data;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned int)
    {
        for(size_t i = 0; i < data_size; ++i)
        {
            try
            {
                ar(cereal::make_nvp(data_labels().at(i).c_str(), m_data.at(i)));
            } catch(cereal::Exception& e)
            {
                TIMEMORY_PRINT_HERE("Warning! '%s': %s\n", data_labels().at(i).c_str(),
                                    e.what());
            }
        }
    }

public:
    inline int64_t get_rx_bytes() const { return std::get<0>(m_data); }
    inline int64_t get_rx_packets() const { return std::get<1>(m_data); }
    inline int64_t get_rx_errors() const { return std::get<2>(m_data); }
    inline int64_t get_rx_dropped() const { return std::get<3>(m_data); }
    inline int64_t get_tx_bytes() const { return std::get<4>(m_data); }
    inline int64_t get_tx_packets() const { return std::get<5>(m_data); }
    inline int64_t get_tx_errors() const { return std::get<6>(m_data); }
    inline int64_t get_tx_dropped() const { return std::get<7>(m_data); }

    std::string str() const
    {
        std::stringstream ss{};
        ss << "bytes: " << get_rx_bytes() << " / " << get_tx_bytes();
        ss << ", packets: " << get_rx_packets() << " / " << get_tx_packets();
        ss << ", errors: " << get_rx_errors() << " / " << get_tx_errors();
        ss << ", dropped: " << get_rx_dropped() << " / " << get_tx_dropped();
        return ss.str();
    }

private:
    data_type m_data{};

private:
    template <typename Tp, size_t N, size_t Idx>
    static inline auto read(const std::array<std::string, N>& _paths,
                            std::array<Tp, data_size>&        _data,
                            std::integral_constant<size_t, Idx>)
    {
        static_assert(Idx < N, "Error! index exceeds array size");
        std::ifstream ifs{ _paths.at(Idx) };
        if(ifs)
            ifs >> _data[Idx];
    }

    template <typename Tp, size_t N, size_t... Idx>
    static inline auto read(const std::array<std::string, N>& _paths,
                            std::array<Tp, data_size>& _data, std::index_sequence<Idx...>)
    {
        TIMEMORY_FOLD_EXPRESSION(
            read(_paths, _data, std::integral_constant<size_t, Idx>{}));
    }

    static inline std::string get_entry(std::ifstream& _ifs, const std::string& _iface)
    {
        std::string _line{};
        getline(_ifs, _line, '\n');
        if(!_ifs)
            return std::string{};
        auto _idx = _line.find(':');
        if(_idx != std::string::npos &&
           _line.substr(0, _idx).find(_iface) != std::string::npos)
        {
            return _line.substr(_idx + 1);
        }
        return get_entry(_ifs, _iface);
    }

    static inline data_type read(const std::string& _line)
    {
        data_type         _data{};
        value_type        _dummy{};
        std::stringstream _sdata{ _line };

        _sdata >> _data[0] >> _data[1] >> _data[2] >> _data[3];
        _sdata >> _dummy >> _dummy >> _dummy >> _dummy;
        _sdata >> _data[4] >> _data[5] >> _data[6] >> _data[7];
        return _data;
    }

    static inline data_type read(const std::string& _fname, const std::string& _iface)
    {
        std::ifstream _ifs{ _fname.c_str() };
        if(_ifs)
        {
            auto&& _entry = get_entry(_ifs, _iface);
            // std::cout << _iface << " :: " << _entry << std::endl;
            if(!_entry.empty())
                return read(_entry);
        }
        return data_type{};
    }
};
}  // namespace cache
}  // namespace tim
