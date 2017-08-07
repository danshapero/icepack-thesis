
set(SOURCES
  utilities.cpp
  grid_data.cpp
  discretization.cpp
  field.cpp
  physics/viscosity.cpp
  physics/ice_shelf.cpp
  physics/mass_transport.cpp
)

add_library(icepack ${SOURCES})
DEAL_II_SETUP_TARGET(icepack)

install(TARGETS icepack DESTINATION lib)
