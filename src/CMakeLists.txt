
set(SOURCES
  utilities.cpp
  mesh.cpp
  grid_data.cpp
  discretization.cpp
  field.cpp
  physics/viscosity.cpp
  physics/friction.cpp
  physics/ice_shelf.cpp
  physics/ice_stream.cpp
  physics/mass_transport.cpp
)

add_library(icepack ${SOURCES})
DEAL_II_SETUP_TARGET(icepack)

install(TARGETS icepack DESTINATION lib)
