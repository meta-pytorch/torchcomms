==========
GOTCHA API
==========

This section describes how to use the GOTCHA API in an application.

-----

--------------------------
Include the GOTCHA Header
--------------------------

In C or C++ applications, include ``gotcha.h``.

.. code-block:: c

    #include <gotcha.h>

--------------------------
Define your Gotcha wrappee
--------------------------

Gotcha wrappee enables the application to call the function it wrapped using GOTCHA.

.. code-block:: c

    static gotcha_wrappee_handle_t wrappee_fputs_handle;

----------------------------
Define your function wrapper
----------------------------

The function wrapper for wrapping functions from shared libraries.

.. code-block:: c

    static int fputs_wrapper(const char *str, FILE *f) {
      // insert clever tool logic here
      typeof(&fputs_wrapper) wrappee_fputs = gotcha_get_wrappee(wrappee_fputs_handle); // get my wrappee from Gotcha
      return wrappee_fputs(str, f); //wrappee_fputs was directed to the original fputs by GOTCHA
    }

----------------------
Define GOTCHA bindings
----------------------

GOTCHA works on binding a `function name`, `wrapper function`, and `wrappee handle`. 
Gotcha works on triplets containing this information.

.. code-block:: c

    struct gotcha_binding_t wrap_actions [] = {
      { "fputs", fputs_wrapper, &wrappee_fputs_handle },
    };

----------------------
Wrap the binding calls
----------------------

To initiate gotcha with the bindings defined in last step, tools can call the `gotcha_wrap` function.
This function should be called before any interception is expected by the tool.
Some popular places for calling this are `gnu constructor`_ or the start of `main` function.
The function will always be successful and would never throw error.

.. code-block:: c

    gotcha_error_t gotcha_wrap(wrap_actions, 
                sizeof(wrap_actions)/sizeof(struct gotcha_binding_t), // number of bindings
                "my_tool_name");

.. rubric:: Multiple gotcha_wrap Caveat

We allow tools to bind different set of functions to different tool names through multiple `gotcha_wrap` calls.
However, a tool within GOTCHA is designed to layer or prioritize the order of functions binding same symbol_ name.
For instance, if multiple tools bind the fputs functions, then GOTCHA layers them to call one after the other with the lowest level being the system call.
In this case, tools can prioritize which tools go first or second at runtime to determine the wrapper order for GOTCHA.
If an tool uses multiple bindings then they have to set priority to different bindings identified using `tool_name` defined within the same tool.


.. attention::

    The `gotcha_wrap` function modifies the `gotcha_binding_t wrap_actions[]` provided by the user.
    GOTCHA does not create a copy of the binding functions and is the responsibility of the user to maintain this binding.


----------------------------
Set priority of tool binding
----------------------------

To set priority of tool within GOTCHA, tools can utilize `gotcha_set_priority` function.
The priority is an integer value with lower values are for inner most call.
The lowest layer is the system call followed by GOTCHA layer and finally other tools based on priority.
The API would never fail. If it return GOTCHA_INTERNAL as error then there was issue with memory allocation of tool.
If multiple tools have same priority then they are wrapper in FIFO order with the first tool being the inner-most wrapper.
Without calling this API the default priority given to each tool is -1.

.. code-block:: c

    gotcha_error_t gotcha_set_priority(const char* tool_name, 
                                       int priority);



----------------------------
Get priority of tool binding
----------------------------

This API gets the priority of the tool. This could be default or as assigned by the tool.

.. code-block:: c

    gotcha_error_t gotcha_get_priority(const char* tool_name, 
                                       int *priority);


------------------------------------------
Get the wrapped function from GOTCHA stack
------------------------------------------

This API return the wrapped function to call based on the tool's handle.
The tools handle is used to locate the next element of the wrapper stack and return the function.
Returns the ptr of the wrapped function.

.. code-block:: c

    void* gotcha_get_wrappee(gotcha_wrappee_handle_t handle);


----------------
Filter libraries
----------------

Within GOTCHA, even bound symbol is updated in the GOT table for each shared library loaded within the tool.
In some cases, tools might not want to update these symbols on some libraries.
For these cases, GOTCHA has a series of filter functions that can assist tools to define which libraries should be updated.
CAUTION: this could lead to behaviors where calls from these libraries would not be intercepted by GOTCHA wrappers and need to handled by the tool.

Filter by Name
**************

This API allows GOTCHA to include only libraries given specified by the user.
This could be a partial match of string contains as defined by `strstr` function in C.

.. code-block:: c

    void gotcha_filter_libraries_by_name(const char* nameFilter);

Filter if Last
**************

This API allows GOTCHA to include only the last library defined in the linker of the tool.

.. code-block:: c

    void gotcha_only_filter_last();


Filter by user defined function
*******************************

This API allows users to define a function that selected the libraries that user wants to intercept.
The function should take `struct link_map*` as input and return true if it should be wrapped by GOTCHA.
TIP: the library name can be accessed by `map->l_name`.

.. code-block:: c

    void gotcha_set_library_filter_func(int(*new_func)(struct link_map*));


Restore default filter of GOTCHA
********************************

The default filter of gotcha selects all libraries loaded. This function set the default filter back for GOTCHA.

.. code-block:: c

    void gotcha_restore_library_filter_func();

.. explicit external hyperlink targets

---------------------------
Using Gotcha Version Macros
---------------------------

The source version of GOTCHA is defined by the `GOTCHA_VERSION` macro which uses the XYYYZZ format.
**Available since version 1.0.7.**  
Here, X signifies the major version, Y is the minor version, and Z is the patch.
Additionally, we define `GOTCHA_VERSION_MAJOR`, `GOTCHA_VERSION_MINOR`, and `GOTCHA_VERSION_PATCH` macros for convienience.
The codes can use the macros like


.. code-block:: c

    #if GOTCHA_VERSION > 100006 // this will check of version greater than 1.0.6
    #endif

    #if GOTCHA_VERSION_MAJOR > 1 // this will check of version greater than 2.0.0
    #endif

    #if GOTCHA_VERSION > GOTCHA_GET_VERSION(1,0,6)  // this will check of version greater than 1.0.6
    #endif




.. _`gnu constructor`: https://gcc.gnu.org/onlinedocs/gcc-4.7.2/gcc/Function-Attributes.html
.. _symbol: https://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-PDA/LSB-PDA.junk/symversion.html
