****************
Example Programs
****************

This example shows how to use gotcha to wrap the open and fopen libc
calls. This example is self-contained, though in typical gotcha
workflows the gotcha calls would be in a separate library from the
application.

The example logs the parameters and return result of every open and
fopen call to stderr.

.. code-block:: c
   :linenos:

   #include <stdio.h>
   #include <sys/types.h>
   #include <sys/stat.h>
   #include <fcntl.h>

   #include "gotcha/gotcha.h"

   typedef int (*open_fptr)(const char *pathname, int flags, mode_t mode);
   typedef FILE* (*fopen_fptr)(const char *pathname, const char *mode);

   static gotcha_wrappee_handle_t open_handle;
   static gotcha_wrappee_handle_t fopen_handle;

   static int open_wrapper(const char *pathname, int flags, mode_t mode) {
     open_fptr open_wrappee = (open_fptr) gotcha_get_wrappee(open_handle);
     int result = open_wrappee(pathname, flags, mode);
     fprintf(stderr, "open(%s, %d, %u) = %d\n",
             pathname, flags, (unsigned int) mode, result);
     return result;
   }

   static FILE *fopen_wrapper(const char *path, const char *mode) {
     fopen_fptr fopen_wrappee = (fopen_fptr) gotcha_get_wrappee(fopen_handle);
     FILE *result = fopen_wrappee(path, mode);
     fprintf(stderr, "fopen(%s, %s) = %p\n",
             path, mode, result);
     return result;
   }

   static gotcha_binding_t bindings[] = {
     { "open", open_wrapper, &open_handle },
     { "fopen", fopen_wrapper, &fopen_handle }
   };

   int main(int argc, char *argv[]) {
     gotcha_wrap(bindings, 2, "demotool");

     open("/dev/null", O_RDONLY);
     open("/dev/null", O_WRONLY | O_CREAT | O_EXCL);
     fopen("/dev/random", "r");
     fopen("/does/not/exist", "w");

     return 0;
   }

The fundamental data structure in the Gotcha API is the gotcha_binding_t
table, which is shown in lines 29-32. This table states that calls
to open should be rerouted to call open_wrapper, and similarly
for fopen and fopen_wrapper. The original open and fopen functions will
still be accessible via the handles open_handle and fopen_handle.

The binding table is passed to Gotcha on line 36, which specifies there
are two entries in the table and that these are part of the “demotool”
tool. The open_handle and fopen_handle variables are updated by this
call to gotcha_wrap and can now be used to look up function pointers to
the original open and fopen calls.

The subsequent callsites to open and fopen on lines 37-40 are redirected
to respectively call open_wrapper and fopen_wrapper on lines 14-20 and
22-27. Each of these functions looks up the original open and fopen
functions using the gotcha_get_wrappee API call and the open_handle and
fopen_handle on lines 15 and 23.

The wrappers call then call the underlying functions open and fopen
functions on lines 16 and 24. The print the parameters and results of
these calls on lines 17 and 25 and return.

Note that this example skips proper error handling for brevity. The call
to gotcha_wrap could have failed to find instances of fopen and open in
the process, which would have led to an error return. The calls to
fprintf on lines 17 and 25 are stomping on the value of errno, which
could be set in the open and fopen calls on lines 16 and 24.
