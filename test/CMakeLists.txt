
add_library(testing testing.cpp)
DEAL_II_SETUP_TARGET(testing)

# Build executables for each name in a list of source files
function(icepack_exec)
  foreach(arg ${ARGN})
    add_executable(${arg} ${arg}.cpp)
    DEAL_II_SETUP_TARGET(${arg})
    target_link_libraries(${arg} icepack testing)
  endforeach(arg)
endfunction(icepack_exec)

# Make a test from the name of a program and a list of command-line arguments
function(icepack_test test_name prog_name)
  add_test(${test_name} ${prog_name} ${ARGN})
endfunction(icepack_test)

# Build a program and make it a test with no arguments for each name in a list
# of source file names
function(icepack_tests)
  foreach(arg ${ARGN})
    icepack_exec(${arg})
    icepack_test(${arg} ${arg})
  endforeach(arg)
endfunction(icepack_tests)

icepack_tests(
  grid_data
  discretization
  optimization
  assembly
  viscosity
  friction
)

# Build a program and test it with several command line arguments to indicate
# refinement or quadratic elements
function(icepack_grid_tests)
  foreach(arg ${ARGN})
    icepack_exec(${arg})
    icepack_test(${arg}_basic ${arg})
    icepack_test(${arg}_refined ${arg} "--refined")
    icepack_test(${arg}_quadratic ${arg} "--quadratic")
  endforeach(arg)
endfunction(icepack_grid_tests)

icepack_grid_tests(
  field
  ice_shelf
  ice_stream
  mass_transport
)


# Create a make target to check how much of the library is covered by tests
if(ICEPACK_TEST_COVERAGE)
  setup_target_for_coverage(coverage ctest coverage)
endif()

