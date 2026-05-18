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

#include "timemory/macros/attributes.hpp"
#include "timemory/variadic/types.hpp"

#include <type_traits>
#include <utility>

namespace tim
{
namespace mpl
{
//
template <typename BundleT, typename FuncT, typename... Args>
auto
execute(BundleT&& _bundle, FuncT&& _func, Args&&... _args,
        enable_if_t<is_invocable<FuncT, Args...>::value &&
                        !std::is_void<std::result_of_t<FuncT(Args...)>>::value,
                    int>)
{
    using result_type  = std::result_of_t<FuncT(Args...)>;
    using handler_type = execution_handler<decay_t<BundleT>, result_type>;
    return handler_type{ std::forward<BundleT>(_bundle),
                         std::forward<FuncT>(_func)(std::forward<Args>(_args)...) };
}
//
template <typename BundleT, typename FuncT, typename... Args>
auto
execute(BundleT&& _bundle, FuncT&& _func, Args&&... _args,
        enable_if_t<is_invocable<FuncT, Args...>::value &&
                        std::is_void<std::result_of_t<FuncT(Args...)>>::value,
                    int>)
{
    _func(std::forward<Args>(_args)...);
    return std::forward<BundleT>(_bundle);
}
//
template <typename BundleT, typename ValueT>
auto
execute(BundleT&& _bundle, ValueT&& _value,
        enable_if_t<!is_invocable<ValueT>::value, long>)
{
    using handler_type = execution_handler<decay_t<BundleT>, ValueT>;
    return handler_type{ std::forward<BundleT>(_bundle), std::forward<ValueT>(_value) };
}
//
template <typename BundleT, typename DataT>
class execution_handler
{
public:
    static_assert(!std::is_function<DataT>::value,
                  "Error! should be result, not function!");

    using this_type = execution_handler<BundleT, DataT>;

    execution_handler()                         = delete;
    execution_handler(const execution_handler&) = delete;
    execution_handler& operator=(const execution_handler&) = delete;

    execution_handler(execution_handler&&) noexcept = default;
    execution_handler& operator=(execution_handler&&) noexcept = default;

    execution_handler(BundleT& _bundle, DataT&& _data) noexcept
    : m_bundle(_bundle)
    , m_data(std::move(_data))
    {}

    operator BundleT() const { return m_bundle; }
    operator DataT() const { return m_data; }
    operator std::pair<BundleT, DataT>() const
    {
        return std::pair<BundleT, DataT>{ m_bundle, m_data };
    }
    operator std::tuple<BundleT, DataT>() const
    {
        return std::tuple<BundleT, DataT>{ m_bundle, m_data };
    }

    auto get_bundle_and_result() { return std::pair<BundleT, DataT>{ m_bundle, m_data }; }

    TIMEMORY_NODISCARD auto get_bundle() noexcept { return BundleT{ m_bundle }; }
    TIMEMORY_NODISCARD auto get_result() noexcept { return DataT{ m_data }; }

    TIMEMORY_NODISCARD auto& return_bundle() noexcept { return m_bundle; }
    TIMEMORY_NODISCARD auto  return_result() noexcept { return std::move(m_data); }

    // NOTE: since inheriting from BundleT would result in subsequent functions returning
    // the this pointer of BundleT, we would discard DataT
    template <typename... Args>
    this_type& push(Args&&... args)
    {
        m_bundle.push(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& pop(Args&&... args)
    {
        m_bundle.pop(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& measure(Args&&... args)
    {
        m_bundle.measure(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& sample(Args&&... args)
    {
        m_bundle.sample(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& start(Args&&... args)
    {
        m_bundle.start(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& stop(Args&&... args)
    {
        m_bundle.stop(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& assemble(Args&&... args)
    {
        m_bundle.assemble(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& derive(Args&&... args)
    {
        m_bundle.derive(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& mark(Args&&... args)
    {
        m_bundle.mark(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& mark_begin(Args&&... args)
    {
        m_bundle.mark_begin(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& mark_end(Args&&... args)
    {
        m_bundle.mark_end(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& store(Args&&... args)
    {
        m_bundle.store(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& audit(Args&&... args)
    {
        m_bundle.audit(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& add_secondary(Args&&... args)
    {
        m_bundle.add_secondary(std::forward<Args>(args)...);
        return *this;
    }
    template <template <typename> class OpT, typename... Args>
    this_type& invoke(Args&&... _args)
    {
        m_bundle.template invoke<OpT>(std::forward<Args>(_args)...);
        return *this;
    }
    template <typename... Args>
    decltype(auto) get(Args&&... args)
    {
        return execute(*this, m_bundle.get(std::forward<Args>(args)...));
    }
    template <typename... Args>
    decltype(auto) get_labeled(Args&&... args)
    {
        return execute(*this, m_bundle.get_labeled(std::forward<Args>(args)...));
        return m_bundle.get_labeled(std::forward<Args>(args)...);
    }

private:
    BundleT& m_bundle;
    DataT    m_data;
};
//
//
template <typename BundleT>
class execution_handler<BundleT, void>
{
public:
    using DataT     = void;
    using this_type = execution_handler<BundleT, DataT>;

    execution_handler()                         = delete;
    execution_handler(const execution_handler&) = delete;
    execution_handler& operator=(const execution_handler&) = delete;

    execution_handler(execution_handler&&) noexcept = default;
    execution_handler& operator=(execution_handler&&) noexcept = default;

    execution_handler(BundleT& _bundle) noexcept
    : m_bundle(_bundle)
    {}

    operator BundleT() const { return m_bundle; }

    operator std::pair<BundleT, null_type>() const
    {
        return std::pair<BundleT, DataT>{ m_bundle, null_type{} };
    }

    operator std::tuple<BundleT, null_type>() const
    {
        return std::tuple<BundleT, DataT>{ m_bundle, null_type{} };
    }

    auto get_bundle_and_result()
    {
        return std::pair<BundleT, DataT>{ m_bundle, null_type{} };
    }

    TIMEMORY_NODISCARD auto& return_bundle() noexcept { return m_bundle; }
    void                     return_result() noexcept {}

    // NOTE: since inheriting from BundleT would result in subsequent functions returning
    // the this pointer of BundleT, we would discard DataT
    template <typename... Args>
    this_type& push(Args&&... args)
    {
        m_bundle.push(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& pop(Args&&... args)
    {
        m_bundle.pop(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& measure(Args&&... args)
    {
        m_bundle.measure(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& sample(Args&&... args)
    {
        m_bundle.sample(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& start(Args&&... args)
    {
        m_bundle.start(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& stop(Args&&... args)
    {
        m_bundle.stop(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& assemble(Args&&... args)
    {
        m_bundle.assemble(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& derive(Args&&... args)
    {
        m_bundle.derive(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& mark(Args&&... args)
    {
        m_bundle.mark(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& mark_begin(Args&&... args)
    {
        m_bundle.mark_begin(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& mark_end(Args&&... args)
    {
        m_bundle.mark_end(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& store(Args&&... args)
    {
        m_bundle.store(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& audit(Args&&... args)
    {
        m_bundle.audit(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Args>
    this_type& add_secondary(Args&&... args)
    {
        m_bundle.add_secondary(std::forward<Args>(args)...);
        return *this;
    }
    template <template <typename> class OpT, typename... Args>
    this_type& invoke(Args&&... _args)
    {
        m_bundle.template invoke<OpT>(std::forward<Args>(_args)...);
        return *this;
    }
    template <typename... Args>
    decltype(auto) get(Args&&... args)
    {
        return execute(*this, m_bundle.get(std::forward<Args>(args)...));
    }
    template <typename... Args>
    decltype(auto) get_labeled(Args&&... args)
    {
        return execute(*this, m_bundle.get_labeled(std::forward<Args>(args)...));
        return m_bundle.get_labeled(std::forward<Args>(args)...);
    }

private:
    BundleT& m_bundle;
};
//
}  // namespace mpl
}  // namespace tim
