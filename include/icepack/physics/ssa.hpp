
#ifndef ICEPACK_SSA_HPP
#define ICEPACK_SSA_HPP

#include <icepack/physics/viscosity.hpp>
#include <icepack/field.hpp>
#include <icepack/numerics/optimization.hpp>

namespace icepack
{

  /**
   * @brief Computes the gravitational power dissipation and stress for shallow
   * shelf flows
   *
   * @ingroup physics
   */
  struct Gravity
  {
    Gravity();

    virtual double
    action(
      const Field<2>& thickness,
      const VectorField<2>& velocity
    ) const;

    virtual DualVectorField<2>
    derivative(
      const Field<2>& thickness,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    virtual double
    derivative(
      const Field<2>& thickness,
      const VectorField<2>& direction
    ) const;
  };


  /**
   * @brief Class for solving for the ice velocity and thickness, under the
   * shallow shelf approximation
   *
   * @ingroup physics
   */
  struct IceShelf
  {
    /**
     * Construct an ice shelf model object, given the IDs of the parts of the
     * ice shelf boundary that correspond to Dirichlet boundary conditions.
     */
    IceShelf(
      const std::set<dealii::types::boundary_id>& dirichlet_boundary_ids,
      const Viscosity& viscosity = Viscosity(MembraneStress()),
      const double convergence_tolerance = 1.0e-6
    );

    using SolveOptions = numerics::NewtonSearchOptions<VectorField<2>>;

    /**
     * Compute the ice velocity for a given thickness and temperature starting
     * from some initial guess, using the Dirichlet boundaries that were set
     * when the present `IceShelf` object was initialized. The boundary
     * conditions for the velocity are obtained from the initial guess.
     */
    VectorField<2> solve(
      const Field<2>& thickness,
      const Field<2>& theta,
      const VectorField<2>& velocity,
      const SolveOptions solve_options = SolveOptions()
    ) const;

    const std::set<dealii::types::boundary_id> dirichlet_boundary_ids;

    const Gravity gravity;
    const Viscosity viscosity;
    const double tolerance;
  };

} // namespace icepack

#endif
