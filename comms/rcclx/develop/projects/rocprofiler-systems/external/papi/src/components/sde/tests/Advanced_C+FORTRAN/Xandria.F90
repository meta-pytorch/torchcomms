        module xandria_mod
          use ISO_C_BINDING, only: C_LONG_LONG, C_DOUBLE, C_ptr, C_funptr
          use papi_sde_fortran_wrappers
          implicit none
          integer(kind=C_LONG_LONG), target :: cntr_i1, cntr_i2, cntr_rw_i1, cntr_iL
          integer(kind=C_LONG_LONG), target :: cntr_i10, cntr_i20, cntr_i30
          real(kind=C_DOUBLE), target       :: cntr_r1, cntr_r2, cntr_r3
          TYPE(C_ptr) :: xandria_sde_handle
        end module

        function papi_sde_hook_list_events(fptr_struct) result(tmp_handle) bind(C)
          use xandria_mod
          use, intrinsic :: ISO_C_BINDING, only: C_ptr, C_null_ptr, C_int, C_F_procpointer
          implicit none
          type(fptr_struct_t) :: fptr_struct
          type(C_ptr) :: tmp_handle
          integer(kind=C_int) :: error_code
          integer(kind=C_int) :: cntr_mode, rw_mode, cntr_type

          procedure(init_t), pointer :: init_fptr
          procedure(register_counter_t), pointer :: reg_cntr_fptr
          procedure(create_counter_t), pointer :: create_cntr_fptr

          cntr_mode = PAPI_SDE_RO+PAPI_SDE_DELTA
          rw_mode = PAPI_SDE_RW+PAPI_SDE_INSTANT
          cntr_type = PAPI_SDE_long_long

          call C_F_procpointer(fptr_struct%init, init_fptr)
          tmp_handle = init_fptr(F_str_to_C('Xandria'))

          call C_F_procpointer(fptr_struct%register_counter, reg_cntr_fptr)

          call C_F_procpointer(fptr_struct%create_counter, create_cntr_fptr)

          error_code = reg_cntr_fptr(tmp_handle, F_str_to_C('EV_I1'), cntr_mode, cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

          error_code = reg_cntr_fptr(tmp_handle, F_str_to_C('EV_I2'), cntr_mode, cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

          error_code = reg_cntr_fptr(tmp_handle, F_str_to_C('RW_I1'), rw_mode, cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

          error_code = reg_cntr_fptr(tmp_handle, F_str_to_C('EV_R1'), cntr_mode, cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

          error_code = reg_cntr_fptr(tmp_handle, F_str_to_C('EV_R2'), cntr_mode, cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

          error_code = reg_cntr_fptr(tmp_handle, F_str_to_C('EV_R3'), cntr_mode, cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

          error_code = reg_cntr_fptr(tmp_handle, F_str_to_C('LATE'), cntr_mode, cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

          error_code = create_cntr_fptr(tmp_handle, F_str_to_C('XND_CREATED'), cntr_type, C_null_ptr)
          if( error_code .ne. PAPI_OK ) then
              print *,'Error in Xandria:papi_sde_hook_list_events() '
              return
          endif

        end function papi_sde_hook_list_events

        subroutine xandria_init
          use, intrinsic :: ISO_C_BINDING
          use  :: xandria_mod
          implicit none

          integer :: cntr_mode, rw_mode, cntr_type, error

          cntr_i1 = 0
          cntr_i2 = 0
          cntr_rw_i1 = 0
          cntr_i10 = 0
          cntr_i20 = 0
          cntr_i30 = 0
          cntr_iL = 0
       

          cntr_mode = PAPI_SDE_RO+PAPI_SDE_DELTA
          rw_mode = PAPI_SDE_RW+PAPI_SDE_INSTANT
          cntr_type = PAPI_SDE_long_long
          call papif_sde_init('Xandria', xandria_sde_handle, error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

          call papif_sde_register_counter(xandria_sde_handle, 'EV_I1', cntr_mode, cntr_type, C_loc(cntr_i1), error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

          call papif_sde_register_counter(xandria_sde_handle, 'EV_I2', cntr_mode, cntr_type, C_loc(cntr_i2), error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

          call papif_sde_register_counter(xandria_sde_handle, 'RW_I1', rw_mode, cntr_type, C_loc(cntr_rw_i1), error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

          call papif_sde_register_counter(xandria_sde_handle, 'EV_R1', cntr_mode, cntr_type, C_loc(cntr_i10), error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

          call papif_sde_register_counter(xandria_sde_handle, 'EV_R2', cntr_mode, cntr_type, C_loc(cntr_i20), error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

          call papif_sde_register_counter(xandria_sde_handle, 'EV_R3', cntr_mode, cntr_type, C_loc(cntr_i30), error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

          call papif_sde_create_counter(xandria_sde_handle, 'XND_CREATED', cntr_type, C_null_ptr, error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_create_counter() '
              return
          endif
        end subroutine

        subroutine xandria_add_more
          use, intrinsic :: ISO_C_BINDING
          use  :: xandria_mod
          implicit none

          integer :: cntr_mode, cntr_type, error
          cntr_mode = PAPI_SDE_RO+PAPI_SDE_DELTA
          cntr_type = PAPI_SDE_long_long

          call papif_sde_register_counter(xandria_sde_handle, 'LATE', cntr_mode, cntr_type, C_loc(cntr_iL), error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_register_counter() '
              stop
          endif

        end subroutine

        subroutine xandria_do_work
          use, intrinsic :: ISO_C_BINDING
          use  :: xandria_mod

          implicit none

          TYPE(C_ptr) :: cntr_handle
          integer :: error

          cntr_i1 = cntr_i1+1
          cntr_i2 = cntr_i2+3

          cntr_rw_i1 = cntr_rw_i1 + 7

          cntr_i10 = cntr_i10+10
          cntr_i20 = cntr_i20+20
          cntr_i30 = cntr_i30+30

          cntr_iL = cntr_iL+7

          call papif_sde_get_counter_handle(xandria_sde_handle, 'XND_CREATED', cntr_handle, error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_get_counter_handle() '
              stop
          endif

          call papif_sde_inc_counter( cntr_handle, 9_8, error)
          if( error .ne. PAPI_OK ) then
              print *,'Error in papif_sde_inc_counter() '
              stop
          endif

        end subroutine

