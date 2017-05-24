
# icepack

This is the documentation for the glacier and ice sheet modelling library icepack.
The most important classes in icepack are:
* The \ref icepack::Field and \ref icepack::VectorField classes represent scalar and vector fields, respectively.
Most of the things you'll want to do involve taking a bunch of fields, solving some PDE that represents some aspect of the physics of glaciers, and returning another field as a result.
* \ref icepack::IceShelf contains functions for computing the velocity of an ice shelf given its thickness and temperature, and updating the ice thickness given its current value and the ice velocity.
More glacier models are on the way.
