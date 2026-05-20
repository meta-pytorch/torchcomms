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

#include "timemory/plotting/definition.hpp"
#include "timemory/plotting/extern.hpp"

#ifndef TIMEMORY_PLOTTING_DECLARATION_HPP_
#    include "timemory/plotting/declaration.hpp"
#endif

#include "timemory/plotting/declaration.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/utility/launch_process.hpp"
#include "timemory/utility/locking.hpp"

#include <fstream>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace plotting
{
TIMEMORY_PLOTTING_INLINE void
plot(const std::string& _label, const std::string& _prefix, const std::string& _dir,
     bool _echo_dart, const std::string& _json_file)
{
    auto_lock_t lk(type_mutex<std::ostream>());

    if(settings::debug() || settings::verbose() > 2)
        TIMEMORY_PRINT_HERE("%s", "");

    if(settings::python_exe().empty())
    {
        fprintf(stderr, "[%s]> Empty '%s' (env: '%s'). Plot generation is disabled...\n",
                _label.c_str(), "tim::settings::python_exe()", "TIMEMORY_PYTHON_EXE");
        return;
    }

    if(settings::debug() || settings::verbose() > 2)
        TIMEMORY_PRINT_HERE("%s", "");

    const auto& _file = _json_file;
    {
        std::ifstream ifs{ _file };
        bool          exists = ifs.good();
        ifs.close();
        if(!exists)
        {
            fprintf(stderr,
                    "[%s]> file '%s' does not exist. Plot generation is disabled...\n",
                    _label.c_str(), _file.c_str());
            return;
        }
    }

    // if currently in plotting mode, we dont want to plot again
    if(get_env<bool>("TIMEMORY_CXX_PLOT_MODE", false, false))
        return;

    auto _settings = settings::shared_instance();
    auto _tag      = _settings ? _settings->get_tag() : settings::get_fallback_tag();
    auto _fdir     = settings::format(_dir, _tag);

    using strpair_t             = std::pair<std::string, std::string>;
    std::vector<strpair_t> _env = { strpair_t{ "TIMEMORY_LIBRARY_CTOR", "OFF" },
                                    strpair_t{ "TIMEMORY_BANNER", "OFF" },
                                    strpair_t{ "TIMEMORY_CXX_PLOT_MODE", "1" } };
    auto                   cmd =
        operation::join(" ", settings::python_exe(), "-m", "timemory.plotting", "-f",
                        _file, "-t", TIMEMORY_JOIN("\"", "", _prefix, ""), "-o", _fdir);

    if(_echo_dart)
        cmd += " -e";

    if(settings::verbose() > 2 || settings::debug())
        TIMEMORY_PRINT_HERE("PLOT COMMAND: '%s'", cmd.c_str());

    std::stringstream _log{};
    auto              _success = launch_process(
        cmd.c_str(), TIMEMORY_LABEL("") + " plot generation failed", &_log, _env);

    std::ostream& _os = (_success) ? std::cout : std::cerr;
    _os << log::color::source() << "[" << TIMEMORY_PROJECT_NAME << "]["
        << process::get_id() << "]" << _log.str() << '\n'
        << log::color::end();
}

TIMEMORY_PLOTTING_INLINE
void
echo_dart_file(std::string filepath, attributes_t attributes)
{
    auto _settings = settings::shared_instance();
    auto _tag      = _settings ? _settings->get_tag() : settings::get_fallback_tag();
    filepath       = settings::format(filepath, _tag);

    auto attribute_string = [](const std::string& key, const std::string& item) {
        return operation::join("", key, '=', "\"", item, "\"");
    };

    auto lowercase = [](std::string _str) {
        for(auto& itr : _str)
            itr = tolower(itr);
        return _str;
    };

    auto contains = [&lowercase](const std::string&           str,
                                 const std::set<std::string>& items) {
        auto lstr = lowercase(str);
        return std::any_of(items.begin(), items.end(), [&lstr](const auto& itr) {
            return lstr.find(itr) != std::string::npos;
        });
    };

    auto is_numeric = [](const std::string& str) -> bool {
        return (str.find_first_not_of("0123456789.e+-*/") == std::string::npos);
    };

    if(attributes.find("name") == attributes.end())
    {
        auto name = filepath;
        if(name.find('/') != std::string::npos)
            name = name.substr(name.find_last_of('/') + 1);
        if(name.find('\\') != std::string::npos)
            name = name.substr(name.find_last_of('\\') + 1);
        if(name.find('.') != std::string::npos)
            name.erase(name.find_last_of('.'));
        attributes["name"] = name;
    }

    if(attributes.find("type") == attributes.end())
    {
        if(contains(filepath, { ".jpeg", ".jpg" }))
        {
            attributes["type"] = "image/jpeg";
        }
        else if(contains(filepath, { ".png" }))
        {
            attributes["type"] = "image/png";
        }
        else if(contains(filepath, { ".tiff", ".tif" }))
        {
            attributes["type"] = "image/tiff";
        }
        else if(contains(filepath, { ".txt" }))
        {
            bool          numeric_file = true;
            std::ifstream ifs;
            ifs.open(filepath);
            if(ifs)
            {
                while(!ifs.eof())
                {
                    std::string entry;
                    ifs >> entry;
                    if(ifs.eof())
                        break;
                    if(!is_numeric(entry))
                    {
                        numeric_file = false;
                        break;
                    }
                }
            }
            ifs.close();
            if(numeric_file)
            {
                attributes["type"] = "numeric/double";
            }
            else
            {
                attributes["type"] = "text/string";
            }
        }
    }

    std::stringstream ss;
    ss << "<DartMeasurementFile";
    for(const auto& itr : attributes)
        ss << " " << attribute_string(itr.first, itr.second);
    ss << ">" << filepath << "</DartMeasurementFile>";
    std::cout << ss.str() << std::endl;
}
}  // namespace plotting
}  // namespace tim
