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
//

// "all-inclusive" header ensuring extern templates are included properly
#include "timemory/timemory.hpp"

// select headers for reference if you want to explore the code some
#include "timemory/components/base/declaration.hpp"         /// base class from components
#include "timemory/components/data_tracker/components.hpp"  /// data tracker component
#include "timemory/components/timing/components.hpp"        /// other timing components
#include "timemory/components/timing/wall_clock.hpp"        /// wall-clock component
#include "timemory/mpl/type_traits.hpp"                     /// type-traits
#include "timemory/settings.hpp"                            /// available settings
#include "timemory/variadic/component_bundle.hpp"           /// component bundler
#include "timemory/variadic/lightweight_tuple.hpp"          /// component bundler

//
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

// Example demonstrating:
// 1. Differences between bundles
// 2. How to control the output
// 3. Creating custom data trackers

using tim::component::cpu_clock;
using tim::component::data_tracker;
using tim::component::data_tracker_unsigned;
using tim::component::wall_clock;

// alias to a time-point type
using time_point_t = std::chrono::time_point<std::chrono::system_clock>;

// alias the templated data-tracker component for storing time-stamps
using time_point_tracker = data_tracker<time_point_t>;

namespace tim
{
namespace trait
{
// disable value-based call-stack tracking for time_pointer_tracker since time_point_t
// does not have overloads for math operations and because MSVC is instantiating
// templates which use them (even though they are unused in this application and therefore
// should not be instantiated)
template <>
struct uses_value_storage<time_point_tracker, time_point_t> : std::false_type
{};
}  // namespace trait
}  // namespace tim

// TODO: remove the need for this
// Below is due to a bug in data-tracker
namespace std
{
time_point_t
operator/(const time_point_t& _tp, const int64_t&);
//
ostream&
operator<<(ostream& _os, const time_point_t&);
}  // namespace std

int
main(int argc, char** argv)
{
    // configure these default settings before timemory_init to allow for
    // environment overrides
    tim::settings::cout_output()       = true;   // print to stdout
    tim::settings::text_output()       = true;   // write text file
    tim::settings::json_output()       = false;  // don't write flat json files
    tim::settings::tree_output()       = false;  // don't write hierarchical json files
    tim::settings::time_output()       = false;  // time-stamp the output folder
    tim::settings::flamegraph_output() = false;  // don't write flamegraph outputs

    //--------------------------------------------------------------------------------//
    // THINGS TO TRY:
    // - Use timemory-avail command-line tool with -S option to view available settings
    // - Add -d option to above to see descriptions of the settings
    //--------------------------------------------------------------------------------//

    // parse command line options
    tim::timemory_argparse(&argc, &argv);

    // initialize timemory
    tim::timemory_init(argc, argv);

    //---------------------------------------------------------------------//
    // Demonstrate the difference b/t these two bundles
    //---------------------------------------------------------------------//
    // this bundle does not implicitly store persistent data. All data will
    // be local to the components in this bundle unless "push()" and "pop()"
    // are explicitly called. Thus, this bundle is useful if you want to
    // handle your own data-storage or do one initial push, and then take
    // several short measurements before doing one final pop which updates
    // the global call-stack stack with the final value.
    using lbundle_t = tim::lightweight_tuple<wall_clock, cpu_clock, data_tracker_unsigned,
                                             time_point_tracker>;
    // NOTE: time_point_tracker can be used in lightweight_tuple as long as
    // it does not interact with storage. Interacting with storage requires
    // some overloads and specializations due to the fact that time_point_t
    // doesn't support operator such as +=, -=, <<, and so on. It is
    // safe to remove from above if you want to experiment with pushing and
    // popping (nothing else below needs to be modified beyond that)
    //
    // EDIT: by setting trait::uses_value_storage to false, the above no longer applies
    // but it will not be tracked in the call-graph

    // this bundle will implicitly "push" to persistent storage when start is
    // called. A "push" operation makes a bookmark in the global call-stack.
    // When stop is called, this bundle will implicitly "pop" itself off the
    // global call-stack. As part of the "pop" operation, this component will
    // add its internal measurement data to the component instance in the global
    // call-stack and then reset its values to zero.
    using cbundle_t = tim::component_tuple<wall_clock, cpu_clock, data_tracker_unsigned>;

    //--------------------------------------------------------------------------------//
    // THINGS TO TRY:
    // - add "cpu_util" to one or both of the aliases above
    // - add "user_global_bundle" to one or both of the alias above
    //   - set environment variable TIMEMORY_GLOBAL_COMPONENTS to "system_clock,
    //   user_clock"
    //     and run the application
    //   - user_global_bundle::configure<system_clock>()
    //--------------------------------------------------------------------------------//

    // standard container for storing each iteration instance
    std::vector<lbundle_t> _ldata{};
    std::vector<cbundle_t> _cdata{};

    // number of iterations
    int nitr = 5;
    if(argc > 1)
        nitr = std::stoi(argv[1]);
    int ndump = 1;
    if(argc > 2)
        ndump = std::max<int>(1, std::stoi(argv[2]));

    for(int i = 0; i < nitr; ++i)
    {
        std::cout << "Performing iteration " << i << "..." << std::endl;
        // the base label for the measurements
        std::string _label = TIMEMORY_JOIN("#", "iteration", i);

        // create an instance for this loop and add a custom suffix for the
        // different types. When the application terminates, you will see
        // the "component_tuple" entries in the final report only.
        _ldata.emplace_back(lbundle_t{ _label + "/lightweight_bundle" });
        _cdata.emplace_back(cbundle_t{ _label + "/component_tuple" });

        //----------------------------------------------------------------------------//
        // THINGS TO TRY:
        //  _ldata.back.push();
        //----------------------------------------------------------------------------//

        // start the measurements
        _ldata.back().start();
        _cdata.back().start();

        // get a time-stamp
        time_point_t _now = std::chrono::system_clock::now();

        // store the time-stamp in time_point_tracker
        _ldata.back().store(_now);

        // convert the time-stamp to the number of seconds since the epoch
        size_t _epoch =
            std::chrono::duration_cast<std::chrono::seconds>(_now.time_since_epoch())
                .count();

        // store the epoch value in data_tracker_unsigned
        _ldata.back().store(_epoch);
        _cdata.back().store(_epoch);

        // sleep for one second for half the iteration
        if(i < nitr / 2)
            std::this_thread::sleep_for(std::chrono::seconds(1));
        else  // consume cpu-cycles for the other half
            while(std::chrono::system_clock::now() <
                  (_now + std::chrono::milliseconds(1005)))
                ;

        // stop the measurements
        _ldata.back().stop();
        _cdata.back().stop();

        //----------------------------------------------------------------------------//
        // THINGS TO TRY:
        //  _ldata.back().pop();
        //----------------------------------------------------------------------------//

        // skip based on modulus
        if(i % ndump != 0)
            continue;

        // write a json serialization of the global call-stack
        std::string outfname = tim::settings::compose_input_filename(
            TIMEMORY_JOIN('-', "iteration", i, "results"), ".json");
        std::ofstream ofs(outfname);
        if(ofs)
        {
            std::cout << "--> Writing " << outfname << "..." << std::endl;
            ofs << tim::manager::get_storage<tim::available_types_t>::serialize();
        }
    }

    std::cout << "Finished " << nitr << " iterations." << std::endl;

    // write a json serialization of the global call-stack
    if(nitr % ndump == 0)
    {
        // basename for the file that will be written
        std::string _bname = TIMEMORY_JOIN('-', "iteration", nitr, "results");
        // compose_output_filename appends the output_path and output_prefix
        // in settings to the first parameter, appends a time-stamped directory
        // if that is enabled, and so on. Then, it just tacks on the extension provided.
        std::string   outfname = tim::settings::compose_output_filename(_bname, ".json");
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, outfname))
        {
            std::cout << "--> Writing " << outfname << "..." << std::endl;
            // tim::available_types_t is just a tim::type_list<...> of all the
            // components that were not marked as unavailable (and thus
            // definitions are provided for them). This manager function opens a
            // cereal::JSONOutputArchive on a stringstream and does a
            // compile-time loop over all of those types. For each type, it
            // checks if the storage has any data and if it does, it serializes
            // the call-graph for that type into the archive. Once the loop
            // is completed, it converts the sstream into a string and returns
            // the string.
            ofs << tim::manager::get_storage<tim::available_types_t>::serialize();
            //------------------------------------------------------------------------//
            // THINGS TO TRY:
            // - replace available_types_t with specific components, e.g. wall_clock
            //------------------------------------------------------------------------//
        }
    }

    // use the stream operator of the bundle to print out data
    size_t _cnt = 0;
    std::cout << "\nLIGHTWEIGHT BUNDLE DATA:" << std::endl;
    for(auto& itr : _ldata)
        std::cout << _cnt++ << " : " << itr << std::endl;

    _cnt = 0;
    std::cout << "\nCOMPONENT TUPLE DATA:" << std::endl;
    for(auto& itr : _cdata)
        std::cout << _cnt++ << " : " << itr << std::endl;

    //--------------------------------------------------------------------------------//
    // THINGS TO TRY:
    // - add this to top of file after the include statement:
    //
    //  TIMEMORY_DECLARE_COMPONENT(fake_component)
    //  TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::fake_component,
    //                                 false_type)
    //
    // - add fake_component to the bundles and then compile and run.
    //   - you will not see anything related to fake_component
    //   - a component marked as unavailable is never instantiated in the template -> no
    //     definition of the type is ever needed
    //
    // - change "false_type" to "true_type" and then try to compile (hint: will fail)
    //    - when a component is available, a definition will be needed
    //--------------------------------------------------------------------------------//

    // demonstrate data access
    _cnt = 0;
    std::cout << "\nLIGHTWEIGHT BUNDLE DATA ACCESS:" << std::endl;
    for(auto& itr : _ldata)
    {
        // get the instance of a wall-clock component in the bundle
        wall_clock* _wc = itr.get<wall_clock>();
        // since wall-clock was part of a "tuple" type, this will always
        // be a valid pointer as long as tim::is_available<wall_clock>::value
        // is true at compile-time. If _ldata was a "list" bundle, this
        // would be a null pointer unless you previously called
        // _ldata.initialize<wall_clock>()
        if(!_wc)
            continue;

        // the "get()" member function returns the accumulated measurement value
        double _elapsed = _wc->get();

        // units are static properties of type so convert this measurement to milliseconds
        _elapsed *= tim::units::msec / wall_clock::unit();

        // get the instance of the unsigned data-tracker component
        time_point_tracker* _tp = itr.get<time_point_tracker>();
        if(_tp)
        {
            // get the time-point value
            time_point_t _point = _tp->get();
            // convert to time_t
            std::time_t _today = std::chrono::system_clock::to_time_t(_point);
            std::string _ctime = std::ctime(&_today);

            // report when the measurement was taken and how long it took.
            std::cout << _cnt++ << " : " << itr.key() << " started on "
                      << _ctime.substr(0, _ctime.find('\n')) << " and took " << _elapsed
                      << " milliseconds" << std::endl;
        }
        else
        {
            // report how long the the measurement took.
            std::cout << _cnt++ << " : " << itr.key() << " took " << _elapsed
                      << " milliseconds" << std::endl;
        }
    }

    puts("\nFinalizing...\n");

    // generate final reports, cleanup memory
    tim::timemory_finalize();

    //----------------------------------------------------------------------------//
    // THINGS TO TRY:
    // - view the text files
    // - enable json output and import the data into python
    //----------------------------------------------------------------------------//

    puts("Done!");
}

// TODO: remove the need for this
// Below is due to a bug in data-tracker
namespace std
{
time_point_t
operator/(const time_point_t& _tp, const int64_t&)
{
    return _tp;
}
//
std::ostream&
operator<<(std::ostream& _os, const time_point_t&)
{
    return _os;
}
}  // namespace std
