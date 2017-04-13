
set(SOURCES
  utilities.cpp
  discretization.cpp
  field.cpp
  physics/ssa.cpp
)

add_library(icepack ${SOURCES})
DEAL_II_SETUP_TARGET(icepack)

install(TARGETS icepack DESTINATION lib)
