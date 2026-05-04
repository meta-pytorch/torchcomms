AC_INIT(host_micpower, version-0.1)

AC_PROG_CC

AC_ARG_WITH([sysmgmt-include-path], 
			[AS_HELP_STRING([--with-sysmgmt-include-path], 
				[location of the MPSS sysmgmt api headers, defaults to /opt/intel/mic/sysmgmt/sdk/include])],
			[SYSMGMT_CFLAGS="-I$withval"],
			[SYSMGMT_CFLAGS="-I/opt/intel/mic/sysmgmt/sdk/include"] )
AC_SUBST([SYSMGMT_CFLAGS])

AC_ARG_WITH([sysmgmt-lib-path],
			[AS_HELP_STRING([--with-sysmgmt-lib-path],
				[location of the MPSS sysmgmt libraries, feed to the runtime linker; \
defaults to /opt/intel/mic/sysmgmt/sdk/lib/Linux])],
			[SYSMGMT_LIBS="-Wl,-rpath,$withval"],
			[SYSMGMT_LIBS="-Wl,-rpath,/opt/intel/mic/sysmgmt/sdk/lib/Linux"])
AC_SUBST([SYSMGMT_LIBS])

#AC_ARG_WITH([scif-lib-path],
#			[AS_HELP_STRING([--with-scif-lib-path],[location of the SCIF library, needed by libMicAccessApi.so]),
#			[],
#			[])

OLD_CPPFLAGS=$CPPFLAGS
CPPFLAGS=["-DMICACCESSAPI -DLINUX $SYSMGMT_CFLAGS"]
AC_CHECK_HEADERS([MicAccessApi.h], [], AC_MSG_ERROR([Couldn't find MicAccessApi.h...try installing MPSS from \
http://software.intel.com/en-us/articles/intel-manycore-platform-software-stack-mpss]) )
CPPFLAGS=$OLD_CPPFLAGS

AC_CONFIG_FILES([Makefile.host_micpower])
AC_OUTPUT
