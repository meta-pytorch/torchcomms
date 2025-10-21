This directory contains forwarding public headers for the NCCLX library.

A forwarding header is a header that pulls in the appropriate header dependening on build settings.
This allows support for header namespacing to guarantee that NCCLX headers will be over NCCL headers when building with both libraries
