
# icepack
Finite element modeling of glaciers and ice sheets

This library is for simulating the flow of glaciers and ice sheets using the finite element method.


## Compilation & installation

You will first need a working deal.II installation.
If deal.II is not installed in some standard directory, e.g. `/usr` or `/usr/local`, the environment variable `DEAL_II_DIR` can be set to the directory of your deal.II installation.

We also rely on several C++11 features -- lambda, move semantics, etc. -- so you will need to compile deal.II with C++11 support.
When configuring deal.II with CMake, add the flag `-DDEAL_II_WITH_CXX11:BOOL=True`.
If your compiler supports C++11, this should be detected automatically, but adding this flag will make sure.

To build the icepack sources, run the following:

    mkdir <build>
    cd <build>
    cmake <path/to/icepack>
    make

Unit tests can be run by invoking `make test`.


## Dependencies

You will need the following packages installed in order to use icepack:

* a C++11 compiler, e.g. clang 3.2+, GCC 4.7+, icc 12+
* [CMake](http://www.cmake.org/) 2.8.11+. Both deal.II and icepacke use the CMake build system.
* [deal.II](http://dealii.org/) development branch. General-purpose finite element library on which icepack is built.

