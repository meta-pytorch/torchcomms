FROM fedora:latest
RUN dnf -y update
RUN dnf -y install check-devel cmake git gcc-c++ python3-sphinx
RUN dnf clean all
COPY GOTCHA GOTCHA
WORKDIR /GOTCHA
RUN cmake . -B build -DGOTCHA_ENABLE_TESTS=ON -DDEPENDENCIES_PREINSTALLED=TRUE -DCMAKE_BUILD_TYPE=PROFILE -DGOTCHA_ENABLE_WARNING_ERROR=ON
RUN cmake --build build
RUN ctest --test-dir build -VV
