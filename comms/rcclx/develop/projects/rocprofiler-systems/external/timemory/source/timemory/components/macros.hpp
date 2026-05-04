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

#include "timemory/components/metadata.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/macros.hpp"
#include "timemory/macros/attributes.hpp"

#include <string>
#include <unordered_set>

//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_COMPONENT_EXTERN)
#    define TIMEMORY_USE_COMPONENT_EXTERN 1
#endif

#if defined(TIMEMORY_COMPONENT_SOURCE)
#    if !defined(TIMEMORY_COMPONENT_INLINE)
#        define TIMEMORY_COMPONENT_INLINE
#    endif
#elif defined(TIMEMORY_USE_COMPONENT_EXTERN)
#    if !defined(TIMEMORY_COMPONENT_INLINE)
#        define TIMEMORY_COMPONENT_INLINE
#    endif
#else
#    if !defined(TIMEMORY_COMPONENT_INLINE)
#        define TIMEMORY_COMPONENT_INLINE inline
#    endif
#    if !defined(TIMEMORY_COMPONENT_HEADER_MODE)
#        define TIMEMORY_COMPONENT_HEADER_MODE 1
#    endif
#endif

//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_DECLARE_COMPONENT
 * \brief Declare a non-templated component type in the tim::component namespace
 */

#if !defined(TIMEMORY_DECLARE_COMPONENT)
#    define TIMEMORY_DECLARE_COMPONENT(NAME)                                             \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        struct TIMEMORY_VISIBLE NAME;                                                    \
        }                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct is_component<component::NAME> : true_type                                 \
        {};                                                                              \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_DECLARE_API_COMPONENTS
 * \brief Declare a non-templated component type in the tim::component namespace
 */

#if !defined(TIMEMORY_DECLARE_API_COMPONENTS)
#    define TIMEMORY_DECLARE_API_COMPONENTS(API, ...)                                    \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct api_components<API, void>                                                 \
        {                                                                                \
            using type = type_list<__VA_ARGS__>;                                         \
        };                                                                               \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_SET_COMPONENT_APIS
 * \brief Declare a component is part of one or more APIs
 */

#if !defined(TIMEMORY_SET_COMPONENT_API)
#    define TIMEMORY_SET_COMPONENT_API(COMP, ...)                                        \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct component_apis<COMP>                                                      \
        {                                                                                \
            using type = type_list<__VA_ARGS__>;                                         \
        };                                                                               \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_SET_TEMPLATE_COMPONENT_API
 * \brief Declare a component is part of one or more APIs
 */

#if !defined(TIMEMORY_SET_TEMPLATE_COMPONENT_API)
#    define TIMEMORY_SET_TEMPLATE_COMPONENT_API(TARGS, TSPECIAL, ...)                    \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TARGS>                                                                 \
        struct component_apis<TSPECIAL>                                                  \
        {                                                                                \
            using type = type_list<__VA_ARGS__>;                                         \
        };                                                                               \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_BUNDLE_INDEX
 * \brief Declare a bundle index
 */

#if !defined(TIMEMORY_BUNDLE_INDEX)
#    define TIMEMORY_BUNDLE_INDEX(NAME, IDX)                                             \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        static constexpr size_t NAME = IDX;                                              \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_DECLARE_TEMPLATE_COMPONENT
 * \brief Declare a templated component type in the tim::component namespace
 */

#if !defined(TIMEMORY_DECLARE_TEMPLATE_COMPONENT)
#    define TIMEMORY_DECLARE_TEMPLATE_COMPONENT(NAME, ...)                               \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        template <__VA_ARGS__>                                                           \
        struct TIMEMORY_VISIBLE NAME;                                                    \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_TEMPLATE_COMPONENT
 * \brief Declare a templated component type in the tim::component namespace
 */
#if !defined(TIMEMORY_TEMPLATE_COMPONENT)
#    define TIMEMORY_TEMPLATE_COMPONENT(NAME, TEMPLATE_PARAM, ...)                       \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        template <TEMPLATE_PARAM>                                                        \
        struct TIMEMORY_VISIBLE NAME;                                                    \
        }                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TEMPLATE_PARAM>                                                        \
        struct is_component<component::NAME<__VA_ARGS__>> : true_type                    \
        {};                                                                              \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_COMPONENT_ALIAS
 * \brief Declare a non-templated alias to a component in the tim::component namespace
 */

#if !defined(TIMEMORY_COMPONENT_ALIAS)
#    define TIMEMORY_COMPONENT_ALIAS(NAME, ...)                                          \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        using NAME = __VA_ARGS__;                                                        \
        }                                                                                \
        }
#endif

//======================================================================================//

/**
 * \macro TIMEMORY_PROPERTY_SPECIALIZATION
 * \brief Specialization of the property specialization
 */

#if !defined(TIMEMORY_PROPERTY_SPECIALIZATION) && !defined(TIMEMORY_DISABLE_PROPERTIES)
#    define TIMEMORY_PROPERTY_SPECIALIZATION(TYPE, ENUM, ID, ...)                        \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        template <>                                                                      \
        struct TIMEMORY_VISIBLE properties<TYPE> : static_properties<TYPE>               \
        {                                                                                \
            using type                        = TYPE;                                    \
            using value_type                  = TIMEMORY_COMPONENT;                      \
            static constexpr value_type value = ENUM;                                    \
            static constexpr bool       specialized() { return true; }                   \
            static const char*          enum_string()                                    \
            {                                                                            \
                static const char*  _enum   = #ENUM;                                     \
                static const size_t _offset = std::string{ _enum }.find('_') + 1;        \
                return static_cast<const char*>(&_enum[_offset]);                        \
            }                                                                            \
            static const char*    id() { return ID; }                                    \
            static const idset_t& ids()                                                  \
            {                                                                            \
                static auto _instance = []() {                                           \
                    auto _val = idset_t{ ID, __VA_ARGS__ };                              \
                    if(_val.find("") != _val.end())                                      \
                        _val.erase("");                                                  \
                    return _val;                                                         \
                }();                                                                     \
                return _instance;                                                        \
            }                                                                            \
            template <typename Archive>                                                  \
            void save(Archive& ar, const unsigned int) const                             \
            {                                                                            \
                ar(cereal::make_nvp("value", ENUM),                                      \
                   cereal::make_nvp("enum", std::string{ enum_string() }),               \
                   cereal::make_nvp("id", std::string{ id() }),                          \
                   cereal::make_nvp("ids", ids()));                                      \
            }                                                                            \
            template <typename Archive>                                                  \
            void load(Archive&, const unsigned int)                                      \
            {}                                                                           \
            TIMEMORY_COMPONENT operator()() { return ENUM; }                             \
                                                                                         \
            constexpr operator TIMEMORY_COMPONENT() const { return ENUM; }               \
        };                                                                               \
        template <>                                                                      \
        struct enumerator<ENUM> : properties<TYPE>                                       \
        {                                                                                \
            using type                  = TYPE;                                          \
            static constexpr bool value = ::tim::trait::is_available<TYPE>::value;       \
                                                                                         \
            static constexpr bool specialized() { return true; }                         \
        };                                                                               \
        }                                                                                \
        }
#elif !defined(TIMEMORY_PROPERTY_SPECIALIZATION) && defined(TIMEMORY_DISABLE_PROPERTIES)
#    define TIMEMORY_PROPERTY_SPECIALIZATION(...)
#endif

//======================================================================================//

/**
 * \macro TIMEMORY_METADATA_SPECIALIZATION
 * \brief Specialization of the property specialization
 */

#if !defined(TIMEMORY_METADATA_SPECIALIZATION) && !defined(TIMEMORY_DISABLE_METADATA)
#    define TIMEMORY_METADATA_SPECIALIZATION(TYPE, LABEL, BASIC_DESC, ...)               \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        template <>                                                                      \
        struct TIMEMORY_VISIBLE metadata<TYPE>                                           \
        {                                                                                \
            using type       = TYPE;                                                     \
            using value_type = TIMEMORY_COMPONENT;                                       \
                                                                                         \
            static constexpr bool       specialized() { return true; }                   \
            static constexpr value_type value = properties<TYPE>::value;                 \
            static std::string          name() { return LABEL; }                         \
            static std::string          label() { return LABEL; }                        \
            static std::string          description() { return BASIC_DESC; }             \
            static std::string          extra_description()                              \
            {                                                                            \
                return TIMEMORY_JOIN(" ", __VA_ARGS__);                                  \
            }                                                                            \
        };                                                                               \
        }                                                                                \
        }
#elif !defined(TIMEMORY_METADATA_SPECIALIZATION) && defined(TIMEMORY_DISABLE_METADATA)
#    define TIMEMORY_METADATA_SPECIALIZATION(...)
#endif

//======================================================================================//

/**
 * \macro TIMEMORY_TOOLSET_ALIAS
 * \brief Creates an alias for a complex type when declaring the statistics type
 */

#if !defined(TIMEMORY_TOOLSET_ALIAS)
#    define TIMEMORY_TOOLSET_ALIAS(NAME, WRAPPER, ...)                                   \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        namespace aliases                                                                \
        {                                                                                \
        using NAME = WRAPPER<__VA_ARGS__>;                                               \
        }                                                                                \
        }                                                                                \
        }                                                                                \
        using tim::component::aliases::NAME;
#endif

//======================================================================================//
//
//      EXTERN TEMPLATE DECLARE AND INSTANTIATE
//
//======================================================================================//

#if !defined(_EXTERN_NAME_COMBINE)
#    define _EXTERN_NAME_COMBINE(X, Y) X##Y
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(_EXTERN_TUPLE_ALIAS)
#    define _EXTERN_TUPLE_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_tuple_, Y)
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(_EXTERN_LIST_ALIAS)
#    define _EXTERN_LIST_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_list_, Y)
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_INITIALIZE_STORAGE)
#    define TIMEMORY_INITIALIZE_STORAGE(...)                                             \
        namespace tim                                                                    \
        {                                                                                \
        namespace internal                                                               \
        {                                                                                \
        namespace initialization                                                         \
        {                                                                                \
        namespace                                                                        \
        {                                                                                \
        using namespace ::tim::component;                                                \
        namespace component = ::tim::component;                                          \
        auto TIMEMORY_STORAGE_INIT_VARIABLE(__COUNTER__) =                               \
            ::tim::storage_initializer::get<__VA_ARGS__>();                              \
        }                                                                                \
        }                                                                                \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_INVOKE_PREINIT)
#    define TIMEMORY_INVOKE_PREINIT(...)                                                 \
        namespace tim                                                                    \
        {                                                                                \
        namespace internal                                                               \
        {                                                                                \
        namespace initialization                                                         \
        {                                                                                \
        namespace                                                                        \
        {                                                                                \
        using namespace ::tim::component;                                                \
        namespace component = ::tim::component;                                          \
        auto                                 TIMEMORY_PREINIT_VARIABLE(__COUNTER__) =    \
            ::tim::preinitializer{}.template operator()<__VA_ARGS__>();                  \
        }                                                                                \
        }                                                                                \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//
// backwards compatibility
#if !defined(TIMEMORY_STORAGE_INITIALIZER)
#    define TIMEMORY_STORAGE_INITIALIZER(...) TIMEMORY_INITIALIZE_STORAGE(__VA_ARGS__)
#endif

//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_DECLARE_EXTERN_STORAGE)
#    define TIMEMORY_DECLARE_EXTERN_STORAGE(TYPE)                                        \
        namespace tim                                                                    \
        {                                                                                \
        extern template class impl::storage<TYPE,                                        \
                                            trait::uses_value_storage<TYPE>::value>;     \
        extern template class storage<TYPE, typename TYPE::value_type>;                  \
        extern template storage_initializer storage_initializer::get<TYPE>();            \
        namespace node                                                                   \
        {                                                                                \
        extern template struct data<TYPE>;                                               \
        extern template struct graph<TYPE>;                                              \
        extern template struct result<TYPE>;                                             \
        extern template struct tree<TYPE>;                                               \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_INSTANTIATE_EXTERN_STORAGE)
#    define TIMEMORY_INSTANTIATE_EXTERN_STORAGE(TYPE)                                    \
        namespace tim                                                                    \
        {                                                                                \
        template class impl::storage<TYPE, trait::uses_value_storage<TYPE>::value>;      \
        template class storage<TYPE, typename TYPE::value_type>;                         \
        template storage_initializer storage_initializer::get<TYPE>();                   \
        namespace node                                                                   \
        {                                                                                \
        template struct data<TYPE>;                                                      \
        template struct graph<TYPE>;                                                     \
        template struct result<TYPE>;                                                    \
        template struct tree<TYPE>;                                                      \
        }                                                                                \
        }                                                                                \
        namespace                                                                        \
        {                                                                                \
        using namespace ::tim::component;                                                \
        namespace component = ::tim::component;                                          \
        auto TIMEMORY_STORAGE_INIT_VARIABLE(__COUNTER__) =                               \
            ::tim::storage_initializer::get<TYPE>();                                     \
        }
#endif

//======================================================================================//
//
//      EXTERN OPERATION DECLARE AND INSTANTIATE
//
//======================================================================================//
//
#if !defined(TIMEMORY_DECLARE_EXTERN_OPERATIONS)
#    define TIMEMORY_DECLARE_EXTERN_OPERATIONS(TYPE, HAS_DATA)                           \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        namespace factory                                                                \
        {                                                                                \
        extern template opaque           get_opaque<TYPE>();                             \
        extern template opaque           get_opaque<TYPE>(scope::config);                \
        extern template std::set<size_t> get_typeids<TYPE>();                            \
        }                                                                                \
        }                                                                                \
        namespace operation                                                              \
        {                                                                                \
        namespace internal                                                               \
        {                                                                                \
        extern template struct add_secondary<TYPE, trait::secondary_data<TYPE>::value>;  \
        extern template struct serialization_base<TYPE>;                                 \
        extern template struct serialization<TYPE, is_enabled<TYPE>::value>;             \
        }                                                                                \
        extern template struct add_secondary<TYPE>;                                      \
        extern template struct add_statistics<TYPE>;                                     \
        extern template struct assemble<TYPE>;                                           \
        extern template struct audit<TYPE>;                                              \
        extern template struct cache<TYPE>;                                              \
        extern template struct cleanup<TYPE>;                                            \
        extern template struct construct<TYPE>;                                          \
        extern template struct copy<TYPE>;                                               \
        extern template struct derive<TYPE>;                                             \
        extern template struct fini<TYPE>;                                               \
        extern template struct fini_storage<TYPE>;                                       \
        extern template struct echo_measurement<TYPE, trait::echo_enabled<TYPE>::value>; \
        extern template struct extra_serialization<TYPE>;                                \
        extern template struct get<TYPE>;                                                \
        extern template struct get_is_flat<TYPE, true>;                                  \
        extern template struct get_is_flat<TYPE, false>;                                 \
        extern template struct get_is_invalid<TYPE, true>;                               \
        extern template struct get_is_invalid<TYPE, false>;                              \
        extern template struct get_is_on_stack<TYPE, true>;                              \
        extern template struct get_is_on_stack<TYPE, false>;                             \
        extern template struct get_depth<TYPE, 0>;                                       \
        extern template struct init<TYPE>;                                               \
        extern template struct init_storage<TYPE>;                                       \
        extern template struct is_running<TYPE, true>;                                   \
        extern template struct is_running<TYPE, false>;                                  \
        extern template struct minus<TYPE>;                                              \
        extern template struct plus<TYPE>;                                               \
        extern template struct pop_node<TYPE>;                                           \
        extern template struct print<TYPE>;                                              \
        extern template struct print_header<TYPE>;                                       \
        extern template struct print_statistics<TYPE>;                                   \
        extern template struct print_storage<TYPE>;                                      \
        extern template struct push_node<TYPE>;                                          \
        extern template struct record<TYPE>;                                             \
        extern template struct reset<TYPE>;                                              \
        extern template struct serialization<TYPE>;                                      \
        extern template struct set_depth_change<TYPE>;                                   \
        extern template struct set_is_flat<TYPE>;                                        \
        extern template struct set_is_invalid<TYPE>;                                     \
        extern template struct set_is_on_stack<TYPE>;                                    \
        extern template struct set_iterator<TYPE>;                                       \
        extern template struct set_prefix<TYPE>;                                         \
        extern template struct set_scope<TYPE>;                                          \
        extern template struct set_state<TYPE>;                                          \
        extern template struct set_started<TYPE>;                                        \
        extern template struct set_stopped<TYPE>;                                        \
        extern template struct store<TYPE>;                                              \
        namespace finalize                                                               \
        {                                                                                \
        extern template struct dmp_get<TYPE, enabled_value_storage<TYPE>::value>;        \
        extern template struct get<TYPE, enabled_value_storage<TYPE>::value>;            \
        extern template struct merge<TYPE, enabled_value_storage<TYPE>::value>;          \
        extern template struct mpi_get<TYPE, enabled_value_storage<TYPE>::value>;        \
        extern template struct print<TYPE, enabled_value_storage<TYPE>::value>;          \
        extern template struct upc_get<TYPE, enabled_value_storage<TYPE>::value>;        \
        }                                                                                \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS)
#    define TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(TYPE, HAS_DATA)                       \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        namespace factory                                                                \
        {                                                                                \
        template opaque           get_opaque<TYPE>();                                    \
        template opaque           get_opaque<TYPE>(scope::config);                       \
        template std::set<size_t> get_typeids<TYPE>();                                   \
        }                                                                                \
        }                                                                                \
        namespace operation                                                              \
        {                                                                                \
        namespace internal                                                               \
        {                                                                                \
        template struct add_secondary<TYPE, trait::secondary_data<TYPE>::value>;         \
        template struct serialization_base<TYPE>;                                        \
        template struct serialization<TYPE, is_enabled<TYPE>::value>;                    \
        }                                                                                \
        template struct add_secondary<TYPE>;                                             \
        template struct add_statistics<TYPE>;                                            \
        template struct assemble<TYPE>;                                                  \
        template struct audit<TYPE>;                                                     \
        template struct cache<TYPE>;                                                     \
        template struct cleanup<TYPE>;                                                   \
        template struct construct<TYPE>;                                                 \
        template struct copy<TYPE>;                                                      \
        template struct derive<TYPE>;                                                    \
        template struct echo_measurement<TYPE, trait::echo_enabled<TYPE>::value>;        \
        template struct extra_serialization<TYPE>;                                       \
        template struct fini<TYPE>;                                                      \
        template struct fini_storage<TYPE>;                                              \
        template struct get<TYPE>;                                                       \
        template struct get_is_flat<TYPE, true>;                                         \
        template struct get_is_flat<TYPE, false>;                                        \
        template struct get_is_invalid<TYPE, true>;                                      \
        template struct get_is_invalid<TYPE, false>;                                     \
        template struct get_is_on_stack<TYPE, true>;                                     \
        template struct get_is_on_stack<TYPE, false>;                                    \
        template struct get_depth<TYPE, 0>;                                              \
        template struct init<TYPE>;                                                      \
        template struct init_storage<TYPE>;                                              \
        template struct is_running<TYPE, true>;                                          \
        template struct is_running<TYPE, false>;                                         \
        template struct minus<TYPE>;                                                     \
        template struct plus<TYPE>;                                                      \
        template struct pop_node<TYPE>;                                                  \
        template struct print<TYPE>;                                                     \
        template struct print_header<TYPE>;                                              \
        template struct print_statistics<TYPE>;                                          \
        template struct print_storage<TYPE>;                                             \
        template struct push_node<TYPE>;                                                 \
        template struct record<TYPE>;                                                    \
        template struct reset<TYPE>;                                                     \
        template struct serialization<TYPE>;                                             \
        template struct set_depth_change<TYPE>;                                          \
        template struct set_is_flat<TYPE>;                                               \
        template struct set_is_invalid<TYPE>;                                            \
        template struct set_is_on_stack<TYPE>;                                           \
        template struct set_iterator<TYPE>;                                              \
        template struct set_prefix<TYPE>;                                                \
        template struct set_scope<TYPE>;                                                 \
        template struct set_state<TYPE>;                                                 \
        template struct set_started<TYPE>;                                               \
        template struct set_stopped<TYPE>;                                               \
        template struct store<TYPE>;                                                     \
        namespace finalize                                                               \
        {                                                                                \
        template struct dmp_get<TYPE, enabled_value_storage<TYPE>::value>;               \
        template struct get<TYPE, enabled_value_storage<TYPE>::value>;                   \
        template struct merge<TYPE, enabled_value_storage<TYPE>::value>;                 \
        template struct mpi_get<TYPE, enabled_value_storage<TYPE>::value>;               \
        template struct print<TYPE, enabled_value_storage<TYPE>::value>;                 \
        template struct upc_get<TYPE, enabled_value_storage<TYPE>::value>;               \
        }                                                                                \
        }                                                                                \
        }
#endif

//======================================================================================//

#if defined(TIMEMORY_COMPONENT_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_OPERATIONS)
#        define TIMEMORY_EXTERN_OPERATIONS(COMPONENT_NAME, HAS_DATA)                     \
            TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(TIMEMORY_ESC(COMPONENT_NAME), HAS_DATA)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_STORAGE)
#        define TIMEMORY_EXTERN_STORAGE(...)                                             \
            TIMEMORY_INSTANTIATE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_TEMPLATE)
#        define TIMEMORY_EXTERN_TEMPLATE(...) template __VA_ARGS__;
#    endif
//
//--------------------------------------------------------------------------------------//
//
#elif defined(TIMEMORY_USE_COMPONENT_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_OPERATIONS)
#        define TIMEMORY_EXTERN_OPERATIONS(COMPONENT_NAME, HAS_DATA)                     \
            TIMEMORY_DECLARE_EXTERN_OPERATIONS(TIMEMORY_ESC(COMPONENT_NAME), HAS_DATA)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_STORAGE)
#        define TIMEMORY_EXTERN_STORAGE(...) TIMEMORY_DECLARE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_TEMPLATE)
#        define TIMEMORY_EXTERN_TEMPLATE(...) extern template __VA_ARGS__;
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_OPERATIONS)
#        define TIMEMORY_EXTERN_OPERATIONS(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_STORAGE)
#        define TIMEMORY_EXTERN_STORAGE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_TEMPLATE)
#        define TIMEMORY_EXTERN_TEMPLATE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif

//======================================================================================//

#if !defined(TIMEMORY_EXTERN_COMPONENT)
#    define TIMEMORY_EXTERN_COMPONENT(NAME, HAS_DATA, ...)                               \
        TIMEMORY_EXTERN_TEMPLATE(                                                        \
            struct tim::component::base<TIMEMORY_ESC(tim::component::NAME),              \
                                        __VA_ARGS__>)                                    \
        TIMEMORY_EXTERN_OPERATIONS(TIMEMORY_ESC(component::NAME), HAS_DATA)              \
        TIMEMORY_EXTERN_STORAGE(TIMEMORY_ESC(component::NAME))
#endif

//======================================================================================//

#if !defined(TIMEMORY_DECLARE_EXTERN_COMPONENT)
#    define TIMEMORY_DECLARE_EXTERN_COMPONENT(NAME, HAS_DATA, ...)                       \
        TIMEMORY_DECLARE_EXTERN_TEMPLATE(                                                \
            struct tim::component::base<TIMEMORY_ESC(tim::component::NAME),              \
                                        __VA_ARGS__>)                                    \
        TIMEMORY_DECLARE_EXTERN_OPERATIONS(TIMEMORY_ESC(component::NAME), HAS_DATA)      \
        TIMEMORY_DECLARE_EXTERN_STORAGE(TIMEMORY_ESC(component::NAME))
#endif

//======================================================================================//

#if !defined(TIMEMORY_INSTANTIATE_EXTERN_COMPONENT)
#    define TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(NAME, HAS_DATA, ...)                   \
        TIMEMORY_INSTANTIATE_EXTERN_TEMPLATE(                                            \
            struct tim::component::base<TIMEMORY_ESC(tim::component::NAME),              \
                                        __VA_ARGS__>)                                    \
        TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(TIMEMORY_ESC(component::NAME), HAS_DATA)  \
        TIMEMORY_INSTANTIATE_EXTERN_STORAGE(TIMEMORY_ESC(component::NAME))
#endif

//======================================================================================//
