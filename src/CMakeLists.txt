
set(SOURCES
  discretization.cpp
  )

add_library(icepack ${SOURCES})
DEAL_II_SETUP_TARGET(icepack)

install(TARGETS icepack DESTINATION lib)
