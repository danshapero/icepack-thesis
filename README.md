
# icepack [![Build Status](https://travis-ci.org/icepack/icepack.svg?branch=master)](https://travis-ci.org/icepack/icepack)

This library is for simulating the flow of glaciers and ice sheets using the finite element method.


## Compilation & installation

You will first need a working deal.II installation.
If deal.II is not installed in some standard directory, e.g. `/usr` or `/usr/local`, the environment variable `DEAL_II_DIR` can be set to the directory of your deal.II installation.

We also rely on several C++14 features -- lambda, move semantics, etc. -- so you will need to compile deal.II with C++14 support.
When configuring deal.II with CMake, add the flag `-DDEAL_II_WITH_CXX14=True`.
If your compiler supports C++14, this should be detected automatically, but adding this flag will cause the build to fail immediately if not.

To build the icepack sources, run the following:

    mkdir <build>
    cd <build>
    cmake <path/to/icepack>
    make

To run the unit tests, run

    make test

from the build directory.
To generate test coverage information, configure icepack with

    cmake -DCMAKE_BUILD_TYPE=Debug -DICEPACK_TEST_COVERAGE=True <path/to/icepack>

and run

    make coverage

from the build directory.
Checking test coverage requires the program `lcov`.
To build documentation, configure icepack with

    cmake -DICEPACK_DOCUMENTATION=True <path/to/icepack>

and run

    make doc

from the build directory.
Building the documentation requires `doxygen` and `graphviz`.


## Dependencies

You will need the following packages installed in order to use icepack:

* a C++14 compiler, e.g. clang 3.8+, GCC 5.4+
* [CMake](http://www.cmake.org/) 3+. Both deal.II and icepacke use the CMake build system.
* [deal.II](http://dealii.org/) development branch. General-purpose finite element library on which icepack is built.

