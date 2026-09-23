
#include "gpu_device_timer.hpp"

#include "timemory/timemory.hpp"

namespace tim
{
namespace component
{
gpu_device_timer::gpu_device_timer(const gpu_device_timer& rhs)
: base_type{ rhs }
, m_copy{ true }
, m_device_num{ rhs.m_device_num }
, m_count{ rhs.m_count }
, m_threads{ rhs.m_threads }
, m_incr{ rhs.m_incr }
, m_data{ rhs.m_data }
{}

gpu_device_timer&
gpu_device_timer::operator=(const gpu_device_timer& rhs)
{
    if(this != &rhs)
    {
        base_type::operator=(rhs);
        m_copy             = true;
        m_device_num       = rhs.m_device_num;
        m_count            = rhs.m_count;
        m_threads          = rhs.m_threads;
        m_incr             = rhs.m_incr;
        m_data             = rhs.m_data;
    }
    return *this;
}

void
gpu_device_timer::preinit()
{
    gpu_device_timer_data::label()       = label();
    gpu_device_timer_data::description() = description();
}

std::string
gpu_device_timer::label()
{
    return "gpu_device_timer";
}

std::string
gpu_device_timer::description()
{
    return "instruments clock64() within a GPU kernel";
}

void
gpu_device_timer::stop()
{
    if(m_incr && m_data)
    {
        auto _clock_rate = gpu::get_device_clock_rate(m_device_num) * 1.0e-6;  // GHz

        gpu::stream_sync(gpu::default_stream_v);
        std::vector<CLOCK_DTYPE>  _host(m_threads);
        std::vector<unsigned int> _incr(m_threads);

        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(_host.data(), m_data, m_threads, gpu::device_to_host_v));
        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(_incr.data(), m_incr, m_threads, gpu::device_to_host_v));

        std::vector<double> _values(m_threads, 0.0);

        for(size_t i = 0; i < m_threads; ++i)
            _values.at(i) = _host.at(i) / static_cast<double>(_clock_rate);

        for(size_t i = 0; i < m_threads; ++i)
        {
            if(_incr.at(i) == 0)
                continue;

            double _value = _values.at(i);

            m_tracker.store(
                [](double lhs, double rhs) { return std::max<double>(lhs, rhs); },
                _value);

            if(add_secondary())
            {
                m_tracker.get<gpu_device_timer_data>([&](auto* _obj) {
                    auto* _child = _obj->add_secondary(TIMEMORY_JOIN("_", "thread", i),
                                                       std::plus<double>{}, _value);
                    _child->set_laps(_child->get_laps() + _incr.at(i) - 1);
                });
            }
        }

        auto* _data = m_tracker.get<gpu_device_timer_data>();
        if(_data)
            _data->set_laps(_data->get_laps() + m_count - 1);

        m_tracker.stop();
    }
}

void
gpu_device_timer::mark()
{
    ++m_count;
}

size_t&
gpu_device_timer::max_threads()
{
    static size_t _value = 0;
    return _value;
}
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(gpu_device_timer, false, void)
