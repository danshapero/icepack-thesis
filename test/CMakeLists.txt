
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
)

icepack_exec(field)
icepack_test(field_basic field)
icepack_test(field_refined field "--refined")
icepack_test(field_quadratic field "--quadratic")

icepack_exec(ssa)
icepack_test(ssa_basic ssa)
icepack_test(ssa_refined ssa "--refined")
icepack_test(ssa_quadratic ssa "--quadratic")

icepack_exec(mass_transport)
icepack_test(mass_transport_basic mass_transport)
icepack_test(mass_transport_refined mass_transport "--refined")
icepack_test(mass_transport_quadratic mass_transport "--quadratic")

if(ICEPACK_TEST_COVERAGE)
  setup_target_for_coverage(coverage ctest coverage)
endif()
