To run the SDE tests the LD_LIBRARY_PATH variable needs to contain the paths to the following:
libpapi.so
libpfm.so
and the libraries under $papi_root/src/components/sde/tests/lib

This can be done by adding to your LD_LIBRARY_PATH environment variable the following (from within the tests directory):
$PWD/lib:$PWD/../../..:$PWD/../../../libpfm4/lib
