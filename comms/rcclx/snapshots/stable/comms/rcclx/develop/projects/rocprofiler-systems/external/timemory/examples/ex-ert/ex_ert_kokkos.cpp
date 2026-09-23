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

#include "timemory/components/types.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::cpu_util, std::false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::wall_clock, std::false_type)

#include "timemory/ert/configuration.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/timemory.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <cstdint>
#include <set>

//--------------------------------------------------------------------------------------//
// make the namespace usage a little clearer
//
using namespace tim;
namespace tim {
namespace device {
struct kokkos {
    using stream_t = cuda::stream_t;
    using fp16_t   = float;

    template <typename Tp>
    static Tp* alloc(size_t nsize)
    {
        return static_cast<Tp>(Kokkos::kokkos_malloc(nsize * sizeof(Tp)));
    }

    template <typename Tp>
    static void free(Tp* ptr)
    {
        Kokkos::kokkos_free(ptr);
    }

    static std::string name() { return "kokkos"; }
};
}  // namespace device
namespace ert {
template <typename Tp, typename CounterT>
class counter<device::kokkos, Tp, CounterT> {
public:
    using DeviceT       = device::kokkos;
    using mutex_t       = std::recursive_mutex;
    using lock_t        = std::unique_lock<mutex_t>;
    using counter_type  = CounterT;
    using ert_data_t    = exec_data<_Counter>;
    using this_type     = counter<DeviceT, Tp, CounterT>;
    using callback_type = std::function<void(uint64_t, this_type&)>;
    using data_type     = typename ert_data_t::value_type;
    using data_ptr_t    = std::shared_ptr<ert_data_t>;
    using ull           = unsigned long long;
    using skip_ops_t    = std::unordered_set<size_t>;

public:
    //----------------------------------------------------------------------------------//
    //  default construction
    //
    counter()               = delete;
    ~counter()              = default;
    counter(const counter&) = default;
    counter(counter&&)      = default;
    counter& operator=(const counter&) = delete;
    counter& operator=(counter&&) = delete;

    //----------------------------------------------------------------------------------//
    // standard creation
    //
    explicit counter(const exec_params& _params, data_ptr_t _exec_data,
                     uint64_t _align = 8 * sizeof(Tp))
    : params(_params)
    , align(_align)
    , data(_exec_data)
    {
        compute_internal();
    }

    //----------------------------------------------------------------------------------//
    // overload how to create the counter with a callback function
    //
    counter(const exec_params& _params, const callback_type& _func, data_ptr_t _exec_data,
            uint64_t _align = 8 * sizeof(Tp))
    : params(_params)
    , align(_align)
    , data(_exec_data)
    , configure_callback(_func)
    {
        compute_internal();
    }

public:
    //----------------------------------------------------------------------------------//
    ///  allocate a buffer for the ERT calculation
    ///     uses this function if device is CPU or device is GPU and type is not half2
    ///
    Kokkos::View<Tp*> get_buffer() const
    {
        align = std::max<uint64_t>(align, 8 * sizeof(Tp));
        compute_internal();
        Kokkos::View<Tp*> buffer("counter_buffer", nsize);
        for(uint64_t i = 0; i < nsize; ++i) buffer(i) = Tp(1.0);
        return buffer;
    }

    //----------------------------------------------------------------------------------//
    //  destroy associated buffer
    //
    void destroy_buffer(Kokkos::View<Tp*>) const {}

    //----------------------------------------------------------------------------------//
    // execute the callback that may customize the thread before returning the object
    // that provides the measurement
    //
    void configure(uint64_t tid) { configure_callback(tid, *this); }

    //----------------------------------------------------------------------------------//
    // execute the callback that may customize the thread before returning the object
    // that provides the measurement
    //
    counter_type get_counter() const { return counter_type(); }

    //----------------------------------------------------------------------------------//
    // record the data from a thread/process. Extra exec_params (_itrp) should contain
    // the computed grid size for serialization
    //
    inline void record(counter_type& _counter, int n, int trials, uint64_t nops,
                       const exec_params& _itrp) const
    {
        uint64_t working_set_size = n * params.nthreads * params.nproc;
        uint64_t working_set      = working_set_size * bytes_per_element;
        uint64_t total_bytes      = trials * working_set * memory_accesses_per_element;
        uint64_t total_ops        = trials * working_set_size * nops;

        std::stringstream ss;
        ss << label;
        if(label.length() == 0)
        {
            if(nops > 1)
                ss << "vector_op";
            else
                ss << "scalar_op";
        }

        auto      _label = tim::demangle<Tp>();
        data_type _data(ss.str(), working_set, trials, total_bytes, total_ops, nops,
                        _counter, DeviceT::name(), _label, _itrp);

#if !defined(TIMEMORY_WINDOWS)
        using namespace tim::stl::ostream;
        if(settings::verbose() > 1 || settings::debug())
            std::cout << "[RECORD]> " << _data << std::endl;
#endif

        static std::mutex _mutex;
        // std::unique_lock<std::mutex> _lock(_mutex);
        _mutex.lock();
        *data += _data;
        _mutex.unlock();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename FuncT>
    void set_callback(FuncT&& _f)
    {
        configure_callback = std::forward<FuncT>(_f);
    }

    //----------------------------------------------------------------------------------//
    //      provide ability to write to JSON/XML
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        if(!data.get())  // for input
            data = data_ptr_t(new ert_data_t());
        ar(tim::cereal::make_nvp("params", params), tim::cereal::make_nvp("data", *data));
    }

    //----------------------------------------------------------------------------------//
    //      write to stream
    //
    friend std::ostream& operator<<(std::ostream& os, const counter& obj)
    {
        std::stringstream ss;
        ss << obj.params << ", "
           << "bytes_per_element = " << obj.bytes_per_element << ", "
           << "memory_accesses_per_element = " << obj.memory_accesses_per_element << ", "
           << "alignment = " << obj.align << ", "
           << "nsize = " << obj.nsize << ", "
           << "label = " << obj.label << ", "
           << "data entries = " << ((obj.data.get()) ? obj.data->size() : 0);
        os << ss.str();
        return os;
    }

    //----------------------------------------------------------------------------------//
    //  Get the data pointer
    //
    data_ptr_t&       get_data() { return data; }
    const data_ptr_t& get_data() const { return data; }

    //----------------------------------------------------------------------------------//
    //  Skip the flop counts
    //
    void add_skip_ops(size_t _nops) { skip_ops.insert(_nops); }

    void add_skip_ops(std::initializer_list<size_t> _args)
    {
        for(const auto& itr : _args) skip_ops.insert(itr);
    }

    bool skip(size_t _nops) { return (skip_ops.count(_nops) > 0); }

public:
    //----------------------------------------------------------------------------------//
    //  public data members, modify as needed
    //
    exec_params        params                      = exec_params();
    int                bytes_per_element           = 0;
    int                memory_accesses_per_element = 0;
    mutable uint64_t   align                       = sizeof(Tp);
    mutable uint64_t   nsize                       = 0;
    mutable data_ptr_t data                        = std::make_shared<ert_data_t>();
    std::string        label                       = "";
    skip_ops_t         skip_ops                    = skip_ops_t();

protected:
    callback_type configure_callback = [](uint64_t, this_type&) {};

private:
    //----------------------------------------------------------------------------------//
    //  compute the data size
    //
    void compute_internal() const
    {
        nsize = params.memory_max / params.nproc / params.nthreads;
        nsize = nsize & (~(align - 1));
        nsize = nsize / sizeof(Tp);
        nsize = std::max<uint64_t>(nsize, 1);
    }
};

//--------------------------------------------------------------------------------------//
///
///     This is the "main" function for ERT
///
template <size_t NopsT, size_t... NextraT, typename Tp, typename CounterT,
          typename FuncOpsT, typename FuncStoreT,
          enable_if_t<(sizeof...(NextraT) == 0), int> = 0>
void
ops_kokkos(counter<device::kokkos, Tp, CounterT>& _counter, FuncOpsT&& ops_func,
           FuncStoreT&& store_func)
{
    using ull = long long unsigned;
    if(_counter.skip(NopsT)) return;

    if(settings::verbose() > 0 || settings::debug())
        printf("[%s] Executing %li ops...\n", __FUNCTION__, (long int) NopsT);

    if(_counter.bytes_per_element == 0)
        fprintf(stderr, "[%s:%i]> bytes-per-element is not set!\n", __FUNCTION__,
                __LINE__);

    if(_counter.memory_accesses_per_element == 0)
        fprintf(stderr, "[%s:%i]> memory-accesses-per-element is not set!\n",
                __FUNCTION__, __LINE__);

    using ThreadPolicyType =
        Kokkos::TeamPolicy<Kokkos::Threads, Kokkos::Schedule<Kokkos::Static>>;
    using member_type = typename ThreadPolicyType::member_type;

    // using KernelPolicyType =
    //    Kokkos::TeamPolicy<Kokkos::Threads, Kokkos::Schedule<Kokkos::Dynamic>>;
    // using kernel_type = typename KernelPolicyType::member_type;

    // Create an instance of the policy
    ThreadPolicyType policy(1, _counter.params.nthreads);
    // ThreadPolicyType _policy(1, 1);

    dmp::barrier();  // synchronize distributed memory processes
    Kokkos::fence();

    // printf("Number of thread: %lu\n", (unsigned long) _counter.params.nthreads);

    Kokkos::parallel_for(policy, [&](member_type team_member) {
        int tid =
            team_member.league_rank() * team_member.team_size() + team_member.team_rank();
        // execute the callback
        _counter.configure(tid);

        // allocate buffer
        auto     buf = _counter.get_buffer();
        uint64_t n   = _counter.params.working_set_min;
        //
        if(n > _counter.nsize)
        {
            fprintf(stderr,
                    "[%s@'%s':%i]> Warning! ERT not running any trials because working "
                    "set min > nsize: %llu > %llu\n",
                    TIMEMORY_ERROR_FUNCTION_MACRO, __FILE__, __LINE__, (ull) n,
                    (ull) _counter.nsize);
        }

        while(n <= _counter.nsize)
        {
            // working set - nsize
            uint64_t ntrials = _counter.nsize / n;
            if(ntrials < 1) ntrials = 1;

            auto itr_params = _counter.params;

            // wait master thread notifies to proceed
            team_member.team_barrier();

            // get instance of object measuring something during the calculation
            CounterT ct = _counter.get_counter();
            // start the timer or anything else being recorded
            ct.start();

            Tp alpha = static_cast<Tp>(0.5);
            for(decltype(ntrials) j = 0; j < ntrials; ++j)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n), [=](int i) {
                    // Kokkos::parallel_for(_policy, KOKKOS_LAMBDA(kernel_type
                    // kernel_member) { int i = kernel_member.league_rank() *
                    // kernel_member.team_size() +
                    //         kernel_member.team_rank();
                    //
                    // divide by two here because macros halve,
                    // e.g. ERT_FLOP == 4 means 2 calls
                    constexpr size_t NUM_REP = NopsT / 2;
                    constexpr size_t MOD_REP = NopsT % 2;
                    Tp               beta    = static_cast<Tp>(0.8);
                    mpl::apply<void>::unroll<NUM_REP + MOD_REP, device::gpu>(
                        ops_func, beta, buf[i], alpha);
                    store_func(buf[i], beta);
                });
                alpha *= static_cast<Tp>(1.0 - 1.0e-8);
            }

            // wait master thread notifies to proceed
            team_member.team_barrier();

            // stop the timer or anything else being recorded
            ct.stop();

            // store the result
            if(tid == 0) _counter.record(ct, n, ntrials, NopsT, itr_params);

            n = ((1.1 * n) == n) ? (n + 1) : (1.1 * n);
        }

        team_member.team_barrier();
        _counter.destroy_buffer(buf);
    });

    Kokkos::fence();

    dmp::barrier();  // synchronize distributed memory processes
}

//--------------------------------------------------------------------------------------//
///
///     This is invokes the "main" function for ERT for all the desired "FLOPs" that
///     are unrolled in the kernel
///
template <size_t NopsT, size_t... NextraT, typename DeviceT, typename Tp,
          typename CounterT, typename FuncOpsT, typename FuncStoreT,
          enable_if_t<(sizeof...(NextraT) > 0), int> = 0>
void
ops_kokkos(counter<DeviceT, Tp, CounterT>& _counter, FuncOpsT&& ops_func,
           FuncStoreT&& store_func)
{
    // execute a single parameter
    ops_kokkos<NopsT>(std::ref(_counter).get(), ops_func, store_func);
    // continue the recursive loop
    ops_kokkos<_Nextra...>(std::ref(_counter).get(), ops_func, store_func);
}

//--------------------------------------------------------------------------------------//
///
///     This is invoked when TIMEMORY_USER_ERT_FLOPS is empty
///
template <size_t... NopsT, typename DeviceT, typename Tp, typename CounterT,
          typename FuncOpsT, typename FuncStoreT,
          enable_if_t<(sizeof...(NopsT) == 0), int> = 0>
void
ops_kokkos(counter<DeviceT, Tp, CounterT>&, FuncOpsT&&, FuncStoreT&&)
{}

//======================================================================================//

template <typename Tp, typename CounterT>
struct executor<device::kokkos, Tp, CounterT> {
    //----------------------------------------------------------------------------------//
    // useful aliases
    //
    using device_type        = device::kokkos;
    using value_type         = Tp;
    using configuration_type = configuration<device::kokkos, value_type, CounterT>;
    using counter_type       = counter<device::kokkos, value_type, CounterT>;
    using this_type          = executor<device::kokkos, value_type, CounterT>;
    using callback_type      = std::function<void(counter_type&)>;
    using ert_data_t         = exec_data<_Counter>;

public:
    //----------------------------------------------------------------------------------//
    //  standard invocation with no callback specialization
    //
    executor(configuration_type& config, std::shared_ptr<ert_data_t> _data)
    {
        try
        {
            auto _counter = config.executor(_data);
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
    }

    //----------------------------------------------------------------------------------//
    //  specialize the counter callback
    //
    template <typename FuncT>
    executor(configuration_type& config, std::shared_ptr<ert_data_t> _data,
             FuncT&& _counter_callback)
    {
        try
        {
            auto _counter = config.executor(_data);
            _counter.set_callback(std::forward<FuncT>(_counter_callback));
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
    }

public:
    //----------------------------------------------------------------------------------//
    //
    callback_type callback = get_callback();

public:
    //----------------------------------------------------------------------------------//
    //
    static callback_type& get_callback()
    {
        static callback_type _instance = [](counter_type& _counter) {
            this_type::execute(_counter);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    static void execute(counter_type& _counter)
    {
        // vectorization number of ops
        static constexpr const int SIZE_BITS = sizeof(Tp) * 8;
        static_assert(SIZE_BITS > 0, "Calculated bits size is not greater than zero");
        static constexpr const int VEC = TIMEMORY_VEC / SIZE_BITS;
        static_assert(VEC > 0, "Calculated vector size is zero");

        // functions
        auto store_func = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b) { a = b; };
        auto add_func   = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
            a = b + c;
        };
        auto fma_func = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
            a = a * b + c;
        };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = "scalar_add";
        // run the kernels
        ops_kokkos<1>(_counter, add_func, store_func);

        // set the label
        _counter.label = "vector_fma";
        // run the kernels
        ops_kokkos<VEC / 2, VEC, 2 * VEC, 4 * VEC>(_counter, fma_func, store_func);
        ops_kokkos<TIMEMORY_USER_ERT_FLOPS>(_counter, fma_func, store_func);
    }
};

}  // namespace ert
}  // namespace tim

//--------------------------------------------------------------------------------------//
// some short-hand aliases
//
using counter_type   = component::wall_clock;
using ert_data_t     = ert::exec_data<counter_type>;
using ert_data_ptr_t = std::shared_ptr<ert_data_t>;
using init_list_t    = std::set<uint64_t>;

//--------------------------------------------------------------------------------------//
//  this will invoke ERT with the specified settings
//
template <typename Tp, typename DeviceT = device::kokkos>
void
run_ert(ert_data_ptr_t, int64_t num_threads, int64_t min_size, int64_t max_data,
        int64_t num_streams = 0, int64_t block_size = 0, int64_t num_gpus = 0);

//--------------------------------------------------------------------------------------//

int
finalize(int, int, void*, void*)
{
    Kokkos::finalize();
    return MPI_SUCCESS;
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    settings::verbose() = 0;
    tim::timemory_init(&argc, &argv);
    tim::signals::enable_signal_detection();

    Kokkos::InitArguments args;
    args.num_threads =
        std::thread::hardware_concurrency();  // CPU threads per NUMA region
    args.num_numa  = 1;                       // CPUNUMA regions per process
    args.device_id = 0;  // If Kokkos was built with CUDA enabled, use the GPU #1.
    args.ndevices  = 0;
    Kokkos::initialize(args);

    int comm_key = 0;
    MPI_Comm_create_keyval(nullptr, &finalize, &comm_key, nullptr);
    MPI_Comm_set_attr(MPI_COMM_SELF, comm_key, nullptr);

    auto data  = ert_data_ptr_t(new ert_data_t());
    auto nproc = dmp::size();

    auto cpu_min_size = 64;
    auto cpu_max_data = 10 * ert::cache_size::get_max();

    init_list_t cpu_num_threads;

    if(argc > 1) cpu_min_size = atol(argv[1]);
    if(argc > 2) cpu_max_data = tim::from_string<long>(argv[2]);

    auto default_thread_init_list = init_list_t({ 1, 2 });

    if(argc > 3)
    {
        default_thread_init_list.clear();
        for(int i = 3; i < argc; ++i) default_thread_init_list.insert(atoi(argv[i]));
    }

    for(auto itr : default_thread_init_list)
    {
        auto entry = itr / nproc;
        if(entry > 0) cpu_num_threads.insert(entry);
    }

    TIMEMORY_BLANK_AUTO_TIMER("run_ert");

    for(int i = 0; i < tim::get_env<int>("NUM_ITER", 1); ++i)
    {
        // execute the single-precision ERT calculations
        for(auto nthread : cpu_num_threads)
            run_ert<float, device::kokkos>(data, nthread, cpu_min_size, cpu_max_data);

        // execute the double-precision ERT calculations
        for(auto nthread : cpu_num_threads)
            run_ert<double, device::kokkos>(data, nthread, cpu_min_size, cpu_max_data);
    }

    Kokkos::finalize();

    printf("\n");
    std::string fname = "ert_results";
    ert::serialize(fname, *data);

    tim::timemory_finalize();

    return 0;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
void
run_ert(ert_data_ptr_t data, int64_t num_threads, int64_t min_size, int64_t max_data,
        int64_t num_streams, int64_t block_size, int64_t num_gpus)
{
    // create a label for this test
    auto dtype = tim::demangle(typeid(Tp).name());
    auto htype = DeviceT::name();
    auto label = TIMEMORY_JOIN("_", __FUNCTION__, dtype, htype, num_threads, "threads",
                               min_size, "min-ws", max_data, "max-size");

    printf("\n[ert-example]> Executing %s...\n", label.c_str());

    using ert_executor_type = ert::executor<DeviceT, Tp, counter_type>;
    using ert_config_type   = typename ert_executor_type::configuration_type;
    using ert_counter_type  = ert::counter<DeviceT, Tp, counter_type>;

    //
    // simple modifications to override method number of threads, number of streams,
    // block size, minimum working set size, and max data size
    //
    ert_config_type::get_num_threads()      = [=]() { return num_threads; };
    ert_config_type::get_num_streams()      = [=]() { return num_streams; };
    ert_config_type::get_block_size()       = [=]() { return block_size; };
    ert_config_type::get_min_working_size() = [=]() { return min_size; };
    ert_config_type::get_max_data_size()    = [=]() { return max_data; };

    //
    // create a callback function that sets the device based on the thread-id
    //
    auto set_counter_device = [=](uint64_t tid, ert_counter_type&) {
        if(num_gpus > 0) cuda::set_device(tid % num_gpus);
    };

    //
    // create a configuration object -- this handles a lot of the setup
    //
    ert_config_type config;
    //
    // start generic timemory timer
    //
    TIMEMORY_BLANK_CALIPER(config, tim::auto_timer, label);
    TIMEMORY_CALIPER_APPLY(config, report_at_exit, true);
    //
    // "construct" an ert::executor that executes the configuration and inserts
    // into the data object.
    //
    // NOTE: the ert::executor has callbacks that allows one to customize the
    //       the ERT execution
    //
    ert_executor_type(config, data, set_counter_device);
    if(data && (settings::verbose() > 0 || settings::debug()))
        std::cout << "\n" << *(data) << std::endl;
    printf("\n");
}

//======================================================================================//
