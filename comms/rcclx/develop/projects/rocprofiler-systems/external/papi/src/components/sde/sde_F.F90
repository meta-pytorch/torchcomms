module papi_sde_fortran_wrappers
  use, intrinsic :: ISO_C_BINDING

  implicit none

#include "f90papi.h"

  integer, parameter :: i_kind = 0
  integer, parameter :: PAPI_SDE_RO      = int( Z'00', kind=kind(i_kind))
  integer, parameter :: PAPI_SDE_RW      = int( Z'01', kind=kind(i_kind))
  integer, parameter :: PAPI_SDE_DELTA   = int( Z'00', kind=kind(i_kind))
  integer, parameter :: PAPI_SDE_INSTANT = int( Z'10', kind=kind(i_kind))

  integer, parameter :: PAPI_SDE_long_long = int( Z'00', kind=kind(i_kind))
  integer, parameter :: PAPI_SDE_int       = int( Z'01', kind=kind(i_kind))
  integer, parameter :: PAPI_SDE_double    = int( Z'02', kind=kind(i_kind))
  integer, parameter :: PAPI_SDE_float     = int( Z'03', kind=kind(i_kind))

! -------------------------------------------------------------------
! ------------ Interfaces for F08 bridge-to-C functions -------------
! -------------------------------------------------------------------

  type, bind(C) :: fptr_struct_t
      type(C_funptr) init
      type(C_funptr) register_counter
      type(C_funptr) register_counter_cb
      type(C_funptr) unregister_counter
      type(C_funptr) describe_counter
      type(C_funptr) add_counter_to_group
      type(C_funptr) create_counter
      type(C_funptr) inc_counter
      type(C_funptr) create_recorder
      type(C_funptr) record
      type(c_funptr) reset_recorder
      type(c_funptr) reset_counter
      type(c_funptr) get_counter_handle
  end type fptr_struct_t

  interface papif_sde_init_F08
    type(C_ptr) function papif_sde_init_F08(lib_name_C_str) result(handle) bind(C, name="papi_sde_init")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr
      type(C_ptr), value, intent(in) :: lib_name_C_str
    end function papif_sde_init_F08
  end interface papif_sde_init_F08

  interface papif_sde_register_counter_F08
    integer(kind=C_int) function papif_sde_register_counter_F08(handle, event_name_C_str, cntr_mode, cntr_type, counter) result(error) bind(C, name="papi_sde_register_counter")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      integer(kind=C_int), value, intent(in) :: cntr_type
      integer(kind=C_int), value, intent(in) :: cntr_mode
      type(C_ptr), value, intent(in) :: counter
    end function papif_sde_register_counter_F08
  end interface papif_sde_register_counter_F08

  interface papif_sde_unregister_counter_F08
    integer(kind=C_int) function papif_sde_unregister_counter_F08(handle, event_name_C_str) result(error) bind(C, name="papi_sde_unregister_counter")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
    end function papif_sde_unregister_counter_F08
  end interface papif_sde_unregister_counter_F08

  interface papif_sde_register_counter_cb_F08
    integer(kind=C_int) function papif_sde_register_counter_cb_F08(handle, event_name_C_str, cntr_mode, cntr_type, func_ptr, param) result(error) bind(C, name="papi_sde_register_counter_cb")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_funptr, C_int
      type(C_ptr), value, intent(in)    :: handle
      type(C_ptr), value, intent(in)    :: event_name_C_str 
      integer(kind=C_int), value, intent(in) :: cntr_type
      integer(kind=C_int), value, intent(in) :: cntr_mode
      type(C_funptr), value, intent(in) :: func_ptr
      type(C_ptr), value, intent(in)    :: param
    end function papif_sde_register_counter_cb_F08
  end interface papif_sde_register_counter_cb_F08

  interface papif_sde_describe_counter_F08
    integer(kind=C_int) function papif_sde_describe_counter_F08(handle, event_name_C_str, event_desc_C_str) result(error) bind(C, name="papi_sde_describe_counter")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      type(C_ptr), value, intent(in) :: event_desc_C_str
    end function papif_sde_describe_counter_F08
  end interface papif_sde_describe_counter_F08

  interface papif_sde_add_counter_to_group_F08
    integer(kind=C_int) function papif_sde_add_counter_to_group_F08(handle, event_name_C_str, group_name_C_str, flags) result(error) bind(C, name="papi_sde_add_counter_to_group")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int, C_int32_t
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      type(C_ptr), value, intent(in) :: group_name_C_str
      integer(kind=C_INT32_T), value, intent(in) :: flags
    end function papif_sde_add_counter_to_group_F08
  end interface papif_sde_add_counter_to_group_F08

  interface papif_sde_create_counter_F08
    integer(kind=C_int) function papif_sde_create_counter_F08(handle, event_name_C_str, cntr_type, counter_handle) result(error) bind(C, name="papi_sde_create_counter")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      integer(kind=C_int), value, intent(in) :: cntr_type
      type(C_ptr), value, intent(in) :: counter_handle   ! this argument is "intent(in)" because we will modify the address in which it points to, not the argument itself.
    end function papif_sde_create_counter_F08
  end interface papif_sde_create_counter_F08

  interface papif_sde_inc_counter_F08
    integer(kind=C_int) function papif_sde_inc_counter_F08(counter_handle, increment) result(error) bind(C, name="papi_sde_inc_counter")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_long_long, C_int
      type(C_ptr), value, intent(in) :: counter_handle
      integer(kind=C_long_long), value, intent(in) :: increment
    end function papif_sde_inc_counter_F08
  end interface papif_sde_inc_counter_F08

  interface papif_sde_create_recorder_F08
    integer(kind=C_int) function papif_sde_create_recorder_F08(handle, event_name_C_str, typesize, cmpr_func_ptr, recorder_handle) result(error) bind(C, name="papi_sde_create_recorder")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_funptr, C_size_t, C_int
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      integer(kind=C_size_t), value, intent(in) :: typesize
      type(C_funptr), value, intent(in) :: cmpr_func_ptr
      type(C_ptr), value, intent(in) :: recorder_handle   ! this argument is "intent(in)" because we will modify the address in which it points to, not the argument itself.
    end function papif_sde_create_recorder_F08
  end interface papif_sde_create_recorder_F08

  interface papif_sde_record_F08
    integer(kind=C_int) function papif_sde_record_F08(recorder_handle, typesize, value_to_rec) result(error) bind(C, name="papi_sde_record")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_size_t, C_int
      type(C_ptr), value, intent(in) :: recorder_handle
      integer(kind=C_size_t), value, intent(in) :: typesize
      type(C_ptr), value, intent(in) :: value_to_rec
    end function papif_sde_record_F08
  end interface papif_sde_record_F08

  interface papif_sde_reset_recorder_F08
    integer(kind=C_int) function papif_sde_reset_recorder_F08(recorder_handle) result(error) bind(C, name="papi_sde_reset_recorder")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: recorder_handle
    end function papif_sde_reset_recorder_F08
  end interface papif_sde_reset_recorder_F08

  interface papif_sde_reset_counter_F08
    integer(kind=C_int) function papif_sde_reset_counter_F08(counter_handle) result(error) bind(C, name="papi_sde_reset_counter")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: counter_handle
    end function papif_sde_reset_counter_F08
  end interface papif_sde_reset_counter_F08

  interface papif_sde_get_counter_handle_F08
    type(C_ptr) function papif_sde_get_counter_handle_F08(handle, event_name_C_str) result(counter_handle) bind(C, name="papi_sde_get_counter_handle")
      use, intrinsic :: ISO_C_BINDING, only : C_ptr
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
    end function papif_sde_get_counter_handle_F08
  end interface papif_sde_get_counter_handle_F08


! -------------------------------------------------------------------
! ----------------- Interfaces for helper functions -----------------
! -------------------------------------------------------------------

  interface C_malloc
  type(C_ptr) function C_malloc(size) bind(C,name="malloc")
    use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_size_t
    integer(C_size_t), value, intent(in) :: size
  end function C_malloc
  end interface C_malloc

  interface C_free
  subroutine C_free(ptr) bind(C,name="free")
    use, intrinsic :: ISO_C_BINDING, only : C_ptr
    type(C_ptr), value, intent(in) :: ptr
  end subroutine C_free
  end interface C_free

! -------------------------------------------------------------------
! ----------------- Interfaces for function pointers ----------------
! -------------------------------------------------------------------

  interface
    function init_t(lib_name) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only: C_ptr
      type(C_ptr), value :: lib_name
      type(C_ptr) :: ret_val
    end function init_t
  end interface

  interface
    function register_counter_t(lib_handle, event_name, cntr_mode, cntr_type, cntr) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only: C_ptr, C_int
      type(C_ptr), value, intent(in) :: lib_handle
      type(C_ptr), value, intent(in) :: event_name
      integer(kind=C_int), value, intent(in) :: cntr_mode
      integer(kind=C_int), value, intent(in) :: cntr_type
      type(C_ptr), intent(in) :: cntr
      integer(kind=C_int) :: ret_val
    end function register_counter_t
  end interface

  interface
    function register_counter_cb_t(lib_handle, event_name, cntr_mode, cntr_type, c_func_ptr, param ) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_funptr, C_int
      type(C_ptr), value, intent(in)    :: lib_handle
      type(C_ptr), value, intent(in)    :: event_name
      integer(kind=C_int), value, intent(in) :: cntr_type
      integer(kind=C_int), value, intent(in) :: cntr_mode
      type(C_funptr), value, intent(in) :: c_func_ptr
      type(C_ptr), value, intent(in)    :: param
      integer(kind=C_int) :: ret_val
    end function register_counter_cb_t
  end interface

  interface
    function unregister_counter_t(lib_handle, event_name) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: lib_handle
      type(C_ptr), value, intent(in) :: event_name
      integer(kind=C_int) :: ret_val
    end function unregister_counter_t
  end interface

  interface
    function describe_counter_t(lib_handle, event_name, event_desc) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: lib_handle
      type(C_ptr), value, intent(in) :: event_name
      type(C_ptr), value, intent(in) :: event_desc
      integer(kind=C_int) :: ret_val
    end function describe_counter_t
  end interface

  interface
    function add_counter_to_group_t(handle, event_name_C_str, group_name_C_str, flags) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int, C_int32_t
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      type(C_ptr), value, intent(in) :: group_name_C_str
      integer(kind=C_INT32_T), value, intent(in) :: flags
      integer(kind=C_int) :: ret_val
    end function add_counter_to_group_t
  end interface

  interface
    function create_counter_t(handle, event_name_C_str, cntr_type, counter_handle) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      integer(kind=C_int), value, intent(in) :: cntr_type
      type(C_ptr), value, intent(in) :: counter_handle   ! this argument is "intent(in)" because we will modify the address in which it points to, not the argument itself.
      integer(kind=C_int) :: ret_val
    end function create_counter_t
  end interface

  interface
    function inc_counter_t(counter_handle, increment) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_long_long, C_int
      type(C_ptr), value, intent(in) :: counter_handle
      integer(kind=C_long_long), value, intent(in) :: increment
      integer(kind=C_int) :: ret_val
    end function inc_counter_t
  end interface

  interface
    function create_recorder_t(handle, event_name_C_str, typesize, recorder_handle) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_size_t, C_int
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      integer(kind=C_size_t), value, intent(in) :: typesize
      type(C_ptr), value, intent(in) :: recorder_handle   ! this argument is "intent(in)" because we will modify the address in which it points to, not the argument itself.
      integer(kind=C_int) :: ret_val
    end function create_recorder_t
  end interface

  interface
    function record_t(recorder_handle, typesize, value_to_rec) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_size_t, C_int
      type(C_ptr), value, intent(in) :: recorder_handle
      integer(kind=C_size_t), value, intent(in) :: typesize
      type(C_ptr), value, intent(in) :: value_to_rec
      integer(kind=C_int) :: ret_val
    end function record_t
  end interface


  interface
    function reset_recorder_t(recorder_handle) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: recorder_handle
      integer(kind=C_int) :: ret_val
    end function reset_recorder_t
  end interface

  interface
    function reset_counter_t(counter_handle) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr, C_int
      type(C_ptr), value, intent(in) :: counter_handle
      integer(kind=C_int) :: ret_val
    end function reset_counter_t
  end interface

  interface
    function get_counter_handle_t(handle, event_name_C_str) result(ret_val)
      use, intrinsic :: ISO_C_BINDING, only : C_ptr
      type(C_ptr), value, intent(in) :: handle
      type(C_ptr), value, intent(in) :: event_name_C_str
      type(C_ptr) :: ret_val
    end function get_counter_handle_t
  end interface


! -------------------------------------------------------------------
! ------------------------ END OF INTERFACES ------------------------
! -------------------------------------------------------------------


  contains



! -------------------------------------------------------------------
! ---------------------- F08 API subroutines ------------------------
! -------------------------------------------------------------------

  subroutine papif_sde_init(lib_name, handle, error)
    character(len=*), intent(in) :: lib_name
    type(C_ptr), intent(out) :: handle
    integer, intent(out), optional :: error

    type(C_ptr) :: lib_name_C_str

    lib_name_C_str = F_str_to_C(lib_name)
    handle = papif_sde_init_F08(lib_name_C_str)
    call C_free(lib_name_C_str)
    if( present(error) ) then
        error = PAPI_OK
    end if
  end subroutine papif_sde_init

! ---------------------------------------------------------

  subroutine papif_sde_register_counter( handle, event_name, cntr_mode, cntr_type, counter, error )
    type(C_ptr), intent(in)      :: handle
    character(len=*), intent(in) :: event_name
    integer, intent(in) :: cntr_type
    integer, intent(in) :: cntr_mode
    type(C_ptr), value, intent(in) :: counter
    integer, intent(out), optional :: error
    integer :: tmp

    type(C_ptr) :: event_name_C_str

    event_name_C_str = F_str_to_C(event_name)
    tmp = papif_sde_register_counter_F08(handle, event_name_C_str, cntr_mode, cntr_type, counter)
    if( present(error) ) then
        error = tmp
    end if
    call C_free(event_name_C_str)
  end subroutine papif_sde_register_counter

! ---------------------------------------------------------

  subroutine papif_sde_register_counter_cb( handle, event_name, cntr_mode, cntr_type, c_func_ptr, param, error )
    type(C_ptr), intent(in)         :: handle
    character(len=*), intent(in)    :: event_name
    integer, intent(in) :: cntr_type
    integer, intent(in) :: cntr_mode
    type(C_ptr), value, intent(in)  :: param
    integer, intent(out), optional :: error
    integer :: tmp

    type(C_funptr) :: c_func_ptr
    type(C_ptr)    :: event_name_C_str

    event_name_C_str = F_str_to_C(event_name)
    tmp = papif_sde_register_counter_cb_F08(handle, event_name_C_str, cntr_mode, cntr_type, c_func_ptr, param)
    if( present(error) ) then
        error = tmp
    end if
    call C_free(event_name_C_str)
  end subroutine papif_sde_register_counter_cb

! ---------------------------------------------------------

  subroutine papif_sde_unregister_counter( handle, event_name, error )
    type(C_ptr), intent(in)      :: handle
    character(len=*), intent(in) :: event_name
    integer, intent(out), optional :: error
    integer :: tmp

    type(C_ptr) :: event_name_C_str

    event_name_C_str = F_str_to_C(event_name)
    tmp = papif_sde_unregister_counter_F08(handle, event_name_C_str)
    if( present(error) ) then
        error = tmp
    end if
    call C_free(event_name_C_str)
  end subroutine papif_sde_unregister_counter

! ---------------------------------------------------------

  subroutine papif_sde_describe_counter( handle, event_name, event_desc, error )
    type(C_ptr), intent(in)      :: handle
    character(len=*), intent(in) :: event_name
    character(len=*), intent(in) :: event_desc
    integer, intent(out), optional :: error
    integer :: tmp

    type(C_ptr) :: event_name_C_str
    type(C_ptr) :: event_desc_C_str

    event_name_C_str = F_str_to_C(event_name)
    event_desc_C_str = F_str_to_C(event_desc)
    tmp = papif_sde_describe_counter_F08(handle, event_name_C_str, event_desc_C_str)
    if( present(error) ) then
        error = tmp
    end if
    call C_free(event_name_C_str)
    call C_free(event_desc_C_str)
  end subroutine papif_sde_describe_counter

! ---------------------------------------------------------

  subroutine papif_sde_create_counter(handle, event_name, cntr_type, counter_handle, error)
    type(C_ptr), value, intent(in) :: handle
    character(len=*), intent(in) :: event_name
    integer(kind=C_int), value, intent(in) :: cntr_type
    type(C_ptr), value, intent(in) :: counter_handle
    integer, intent(out), optional :: error
    integer :: tmp

    type(C_ptr) :: event_name_C_str

    event_name_C_str = F_str_to_C(event_name)

    tmp = papif_sde_create_counter_F08(handle, event_name_C_str, cntr_type, counter_handle)
    if( present(error) ) then
        error = tmp
    end if
    call C_free(event_name_C_str)
  end subroutine papif_sde_create_counter

! ---------------------------------------------------------

  subroutine papif_sde_inc_counter(counter_handle, increment, error)
    type(C_ptr), value, intent(in) :: counter_handle
    integer(kind=C_long_long), value, intent(in) :: increment
    integer, intent(out), optional :: error
    integer :: tmp

    tmp = papif_sde_inc_counter_F08(counter_handle, increment)
    if( present(error) ) then
        error = tmp
    end if
  end subroutine papif_sde_inc_counter

! ---------------------------------------------------------

  subroutine papif_sde_create_recorder(handle, event_name, typesize, cmpr_c_func_ptr, recorder_handle, error)
    type(C_ptr), value, intent(in) :: handle
    character(len=*), intent(in) :: event_name
    integer(kind=C_size_t), value, intent(in) :: typesize
    type(C_funptr) :: cmpr_c_func_ptr
    type(C_ptr), value, intent(in) :: recorder_handle
    integer, intent(out), optional :: error
    integer :: tmp

    type(C_ptr) :: event_name_C_str

    event_name_C_str = F_str_to_C(event_name)

    tmp = papif_sde_create_recorder_F08(handle, event_name_C_str, typesize, cmpr_c_func_ptr, recorder_handle)
    if( present(error) ) then
        error = tmp
    end if
    call C_free(event_name_C_str)
  end subroutine papif_sde_create_recorder

! ---------------------------------------------------------

  subroutine papif_sde_record(recorder_handle, typesize, value_to_rec, error)
    type(C_ptr), value, intent(in) :: recorder_handle
    integer(kind=C_size_t), value, intent(in) :: typesize
    type(C_ptr), value, intent(in) :: value_to_rec
    integer, intent(out), optional :: error
    integer :: tmp

    tmp = papif_sde_record_F08(recorder_handle, typesize, value_to_rec)
    if( present(error) ) then
        error = tmp
    end if
  end subroutine papif_sde_record

! ---------------------------------------------------------

  subroutine papif_sde_reset_recorder(recorder_handle, error)
    type(C_ptr), value, intent(in) :: recorder_handle
    integer, intent(out), optional :: error
    integer :: tmp

    tmp = papif_sde_reset_recorder_F08(recorder_handle)
    if( present(error) ) then
        error = tmp
    end if
  end subroutine papif_sde_reset_recorder

! ---------------------------------------------------------

  subroutine papif_sde_reset_counter(counter_handle, error)
    type(C_ptr), value, intent(in) :: counter_handle
    integer, intent(out), optional :: error
    integer :: tmp

    tmp = papif_sde_reset_counter_F08(counter_handle)
    if( present(error) ) then
        error = tmp
    end if
  end subroutine papif_sde_reset_counter

! ---------------------------------------------------------

  subroutine papif_sde_get_counter_handle(handle, event_name, counter_handle, error)
    type(C_ptr), value, intent(in) :: handle
    character(len=*), intent(in) :: event_name
    integer, intent(out), optional :: error
    type(C_ptr), intent(out) :: counter_handle

    type(C_ptr) :: event_name_C_str

    event_name_C_str = F_str_to_C(event_name)
    counter_handle = papif_sde_get_counter_handle_F08(handle, event_name_C_str)
    call C_free(event_name_C_str)
    if( present(error) ) then
        error = PAPI_OK
    end if
  end subroutine papif_sde_get_counter_handle


! -------------------------------------------------------------------
! ------------------------ Helper functions -------------------------
! -------------------------------------------------------------------

  type(C_ptr) function F_str_to_C(F_str) result(C_str)
    implicit none
    character(len=*), intent(in) :: F_str

    character(len=1,kind=C_char), pointer :: tmp_str_ptr(:)
    integer(C_size_t) :: i, strlen

    strlen = len(F_str)

    C_str = C_malloc(strlen+1)
    if (C_associated(C_str)) then
      call C_F_pointer(C_str,tmp_str_ptr,[strlen+1])
      forall (i=1:strlen)
        tmp_str_ptr(i) = F_str(i:i)
      end forall
      tmp_str_ptr(strlen+1) = C_NULL_char
    end if
  end function F_str_to_C


end module papi_sde_fortran_wrappers







