
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
  inverse/error_functionals.cpp
)

include (GenerateExportHeader)
add_library(icepack SHARED ${SOURCES})
DEAL_II_SETUP_TARGET(icepack)
generate_export_header(icepack)

install(TARGETS icepack
  DESTINATION lib
  EXPORT icepackTargets)

