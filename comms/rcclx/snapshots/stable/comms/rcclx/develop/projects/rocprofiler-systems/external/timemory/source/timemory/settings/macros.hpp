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

#include "timemory/compat/macros.h"
#include "timemory/defines.h"

#include <cassert>

#if defined(TIMEMORY_CORE_SOURCE)
#    define TIMEMORY_SETTINGS_SOURCE
#elif defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_SETTINGS_EXTERN
#endif
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_SETTINGS_EXTERN)
#    define TIMEMORY_USE_SETTINGS_EXTERN
#endif
//
#if defined(TIMEMORY_SETTINGS_SOURCE)
#    define TIMEMORY_SETTINGS_COMPILE_MODE
#    define TIMEMORY_SETTINGS_INLINE
#    define TIMEMORY_SETTINGS_LINKAGE(...) __VA_ARGS__
#elif defined(TIMEMORY_USE_SETTINGS_EXTERN)
#    define TIMEMORY_SETTINGS_EXTERN_MODE
#    define TIMEMORY_SETTINGS_INLINE
#    define TIMEMORY_SETTINGS_LINKAGE(...) __VA_ARGS__
#else
#    define TIMEMORY_SETTINGS_HEADER_MODE
#    define TIMEMORY_SETTINGS_INLINE inline
#    define TIMEMORY_SETTINGS_LINKAGE(...) inline __VA_ARGS__
#endif
//
// this can be configured via cmake w/ -DTIMEMORY_SETTINGS_PREFIX=MY_PREFIX_
#if !defined(TIMEMORY_SETTINGS_PREFIX)
#    define TIMEMORY_SETTINGS_PREFIX "TIMEMORY_"
#endif
//
#if !defined(TIMEMORY_PROJECT_NAME)
#    define TIMEMORY_PROJECT_NAME "timemory"
#endif
//
#if !defined(TIMEMORY_SETTINGS_KEY)
#    define TIMEMORY_SETTINGS_KEY(...) TIMEMORY_SETTINGS_PREFIX __VA_ARGS__
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_MEMBER_DECL)
// memory leak w/ _key is intentional due to potential calls during _cxa_finalize
// which may have already deleted a non-heap allocation
#    define TIMEMORY_SETTINGS_MEMBER_DECL(TYPE, FUNC)                                    \
    public:                                                                              \
        TYPE& get_##FUNC() TIMEMORY_NEVER_INSTRUMENT TIMEMORY_VISIBILITY("default");     \
        TYPE                                         get_##FUNC()                        \
            const TIMEMORY_NEVER_INSTRUMENT          TIMEMORY_VISIBILITY("default");     \
        static TYPE&                                 FUNC()                              \
            TIMEMORY_NEVER_INSTRUMENT TIMEMORY_HOT   TIMEMORY_VISIBILITY("default");
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_REFERENCE_DECL)
// memory leak w/ _key is intentional due to potential calls during _cxa_finalize
// which may have already deleted a non-heap allocation
#    define TIMEMORY_SETTINGS_REFERENCE_DECL(TYPE, FUNC)                                 \
    public:                                                                              \
        TYPE& get_##FUNC() TIMEMORY_NEVER_INSTRUMENT TIMEMORY_VISIBILITY("default");     \
        TYPE                                         get_##FUNC()                        \
            const TIMEMORY_NEVER_INSTRUMENT          TIMEMORY_VISIBILITY("default");     \
        static TYPE&                                 FUNC()                              \
            TIMEMORY_NEVER_INSTRUMENT TIMEMORY_HOT   TIMEMORY_VISIBILITY("default");
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_MEMBER_DEF)
// memory leak w/ _key is intentional due to potential calls during _cxa_finalize
// which may have already deleted a non-heap allocation
#    define TIMEMORY_SETTINGS_MEMBER_DEF(TYPE, FUNC, ENV_VAR)                            \
        TIMEMORY_SETTINGS_INLINE TYPE& settings::get_##FUNC()                            \
        {                                                                                \
            return static_cast<tsettings<TYPE>*>(m_data.at(ENV_VAR).get())->get();       \
        }                                                                                \
                                                                                         \
        TIMEMORY_SETTINGS_INLINE TYPE settings::get_##FUNC() const                       \
        {                                                                                \
            auto ret = m_data.find(ENV_VAR);                                             \
            if(ret == m_data.end())                                                      \
                return TYPE{};                                                           \
            if(!ret->second)                                                             \
                return TYPE{};                                                           \
            return static_cast<tsettings<TYPE>*>(ret->second.get())->get();              \
        }                                                                                \
                                                                                         \
        TIMEMORY_SETTINGS_INLINE TYPE& settings::FUNC()                                  \
        {                                                                                \
            try                                                                          \
            {                                                                            \
                return shared_instance()->get_##FUNC();                                  \
            } catch(std::exception & _e)                                                 \
            {                                                                            \
                handle_exception(ENV_VAR, #FUNC, #TYPE, _e.what());                      \
                std::rethrow_exception(std::current_exception());                        \
            }                                                                            \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_REFERENCE_DEF)
// memory leak w/ _key is intentional due to potential calls during _cxa_finalize
// which may have already deleted a non-heap allocation
#    define TIMEMORY_SETTINGS_REFERENCE_DEF(TYPE, FUNC, ENV_VAR)                         \
        TIMEMORY_SETTINGS_INLINE TYPE& settings::get_##FUNC()                            \
        {                                                                                \
            return static_cast<tsettings<TYPE, TYPE&>*>(m_data.at(ENV_VAR).get())        \
                ->get();                                                                 \
        }                                                                                \
                                                                                         \
        TIMEMORY_SETTINGS_INLINE TYPE settings::get_##FUNC() const                       \
        {                                                                                \
            auto ret = m_data.find(ENV_VAR);                                             \
            if(ret == m_data.end())                                                      \
                return TYPE{};                                                           \
            if(!ret->second)                                                             \
                return TYPE{};                                                           \
            return static_cast<tsettings<TYPE, TYPE&>*>(ret->second.get())->get();       \
        }                                                                                \
                                                                                         \
        TIMEMORY_SETTINGS_INLINE TYPE& settings::FUNC()                                  \
        {                                                                                \
            try                                                                          \
            {                                                                            \
                return shared_instance()->get_##FUNC();                                  \
            } catch(std::exception & _e)                                                 \
            {                                                                            \
                handle_exception(ENV_VAR, #FUNC, #TYPE, _e.what());                      \
                std::rethrow_exception(std::current_exception());                        \
            }                                                                            \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_MEMBER_IMPL)
#    define TIMEMORY_SETTINGS_MEMBER_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT, CATEGORIES)   \
                                                                                         \
        if(m_data                                                                        \
               .emplace(ENV_VAR, std::make_shared<tsettings<TYPE>>(                      \
                                     INIT, std::string{ #FUNC }, std::string{ ENV_VAR }, \
                                     std::string{ DESC }, CATEGORIES))                   \
               .second)                                                                  \
            m_order.emplace_back(ENV_VAR);
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_HIDDEN_MEMBER_IMPL)
#    define TIMEMORY_SETTINGS_HIDDEN_MEMBER_IMPL(TYPE, ENV_VAR, DESC, INIT, CATEGORIES)  \
                                                                                         \
        if(m_data                                                                        \
               .emplace(ENV_VAR, std::make_shared<tsettings<TYPE>>(                      \
                                     INIT, std::string{}, std::string{ ENV_VAR },        \
                                     std::string{ DESC }, CATEGORIES))                   \
               .second)                                                                  \
            m_order.emplace_back(ENV_VAR);
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_MEMBER_ARG_IMPL)
#    define TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT, ...)      \
                                                                                         \
        if(m_data                                                                        \
               .emplace(ENV_VAR, std::make_shared<tsettings<TYPE>>(                      \
                                     INIT, std::string{ #FUNC }, std::string{ ENV_VAR }, \
                                     std::string{ DESC }, __VA_ARGS__))                  \
               .second)                                                                  \
            m_order.emplace_back(ENV_VAR);
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_HIDDEN_MEMBER_ARG_IMPL)
#    define TIMEMORY_SETTINGS_HIDDEN_MEMBER_ARG_IMPL(TYPE, ENV_VAR, DESC, INIT, ...)     \
                                                                                         \
        if(m_data                                                                        \
               .emplace(ENV_VAR, std::make_shared<tsettings<TYPE>>(                      \
                                     INIT, std::string{}, std::string{ ENV_VAR },        \
                                     std::string{ DESC }, __VA_ARGS__))                  \
               .second)                                                                  \
            m_order.emplace_back(ENV_VAR);
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_REFERENCE_IMPL)
#    define TIMEMORY_SETTINGS_REFERENCE_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT,            \
                                             CATEGORIES)                                 \
                                                                                         \
        if(m_data                                                                        \
               .emplace(ENV_VAR, std::make_shared<tsettings<TYPE, TYPE&>>(               \
                                     INIT, std::string{ #FUNC }, std::string{ ENV_VAR }, \
                                     std::string{ DESC }, CATEGORIES))                   \
               .second)                                                                  \
            m_order.emplace_back(ENV_VAR);
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL)
#    define TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT, ...)   \
        if(m_data                                                                        \
               .emplace(ENV_VAR, std::make_shared<tsettings<TYPE, TYPE&>>(               \
                                     INIT, std::string{ #FUNC }, std::string{ ENV_VAR }, \
                                     std::string{ DESC }, __VA_ARGS__))                  \
               .second)                                                                  \
            m_order.emplace_back(ENV_VAR);
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_ERROR_FUNCTION_MACRO)
#    if defined(__PRETTY_FUNCTION__)
#        define TIMEMORY_ERROR_FUNCTION_MACRO __PRETTY_FUNCTION__
#    else
#        define TIMEMORY_ERROR_FUNCTION_MACRO __FUNCTION__
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_TRY_CATCH_NVP)
#    define TIMEMORY_SETTINGS_TRY_CATCH_NVP(ENV_VAR, FUNC)                               \
        try                                                                              \
        {                                                                                \
            ar(cereal::make_nvp(ENV_VAR, FUNC()));                                       \
        } catch(...)                                                                     \
        {}
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_EXTERN_TEMPLATE)
//
#    if defined(TIMEMORY_SETTINGS_SOURCE)
//
#        define TIMEMORY_SETTINGS_EXTERN_TEMPLATE(API)                                   \
            namespace tim                                                                \
            {                                                                            \
            template const std::shared_ptr<settings>& settings::shared_instance<API>();  \
            template settings*                        settings::instance<API>();         \
            template void settings::serialize_settings(cereal::JSONInputArchive&);       \
            template void settings::serialize_settings(                                  \
                cereal::PrettyJSONOutputArchive&);                                       \
            template void settings::serialize_settings(                                  \
                cereal::MinimalJSONOutputArchive&);                                      \
            template void settings::serialize_settings(cereal::JSONInputArchive&,        \
                                                       settings&);                       \
            template void settings::serialize_settings(cereal::PrettyJSONOutputArchive&, \
                                                       settings&);                       \
            template void settings::serialize_settings(                                  \
                cereal::MinimalJSONOutputArchive&, settings&);                           \
            template void settings::save(cereal::PrettyJSONOutputArchive&,               \
                                         const unsigned int) const;                      \
            template void settings::save(cereal::MinimalJSONOutputArchive&,              \
                                         const unsigned int) const;                      \
            template void settings::load(cereal::JSONInputArchive&, const unsigned int); \
            }
//
#    elif defined(TIMEMORY_USE_SETTINGS_EXTERN)
//
#        define TIMEMORY_SETTINGS_EXTERN_TEMPLATE(API)                                   \
            namespace tim                                                                \
            {                                                                            \
            extern template const std::shared_ptr<settings>&                             \
                                      settings::shared_instance<API>();                  \
            extern template settings* settings::instance<API>();                         \
            extern template void      settings::serialize_settings(                      \
                cereal::JSONInputArchive&);                                         \
            extern template void settings::serialize_settings(                           \
                cereal::PrettyJSONOutputArchive&);                                       \
            extern template void settings::serialize_settings(                           \
                cereal::MinimalJSONOutputArchive&);                                      \
            extern template void settings::serialize_settings(cereal::JSONInputArchive&, \
                                                              settings&);                \
            extern template void settings::serialize_settings(                           \
                cereal::PrettyJSONOutputArchive&, settings&);                            \
            extern template void settings::serialize_settings(                           \
                cereal::MinimalJSONOutputArchive&, settings&);                           \
            extern template void settings::save(cereal::PrettyJSONOutputArchive&,        \
                                                const unsigned int) const;               \
            extern template void settings::save(cereal::MinimalJSONOutputArchive&,       \
                                                const unsigned int) const;               \
            extern template void settings::load(cereal::JSONInputArchive&,               \
                                                const unsigned int);                     \
            }
//
#    else
//
#        define TIMEMORY_SETTINGS_EXTERN_TEMPLATE(...)
//
#    endif
#endif
//
//======================================================================================//
//
