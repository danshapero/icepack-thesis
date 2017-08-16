
# -------------
# Do not modify
CMAKE_MINIMUM_REQUIRED(VERSION 3.0.2)

FIND_PACKAGE(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
DEAL_II_INITIALIZE_CACHED_VARIABLES()
# ---------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")

project(icepack)

option(ICEPACK_TEST_COVERAGE
  "Check how much of the library is covered by the unit tests (requires lcov)")
if(ICEPACK_TEST_COVERAGE)
  include(CodeCoverage)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")

set(LIBRARY_OUTPUT_PATH ${icepack_BINARY_DIR}/lib)

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_SOURCE_DIR}/src")
add_subdirectory("${CMAKE_SOURCE_DIR}/test")
add_subdirectory("${CMAKE_SOURCE_DIR}/doc")

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

include (CMakePackageConfigHelpers)
export(EXPORT icepackTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/icepackTargets.cmake")
configure_file(cmake/icepack-config.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/icepack/icepack-config.cmake" COPYONLY)

set(ConfigPackageLocation share/cmake/icepack)
install(EXPORT icepackTargets
  FILE icepackTargets.cmake
  DESTINATION ${ConfigPackageLocation})
install(
  FILES cmake/icepack-config.cmake
  DESTINATION ${ConfigPackageLocation})

