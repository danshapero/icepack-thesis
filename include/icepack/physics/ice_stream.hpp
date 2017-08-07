
#ifndef ICEPACK_ICE_STREAM_HPP
#define ICEPACK_ICE_STREAM_HPP

#include <icepack/physics/viscosity.hpp>
#include <icepack/physics/friction.hpp>
#include <icepack/numerics/optimization.hpp>

namespace icepack
{
  /**
   * @brief Computes the gravitational power dissipation and stress for
   * the flow of a grounded ice stream
   *
   * @ingroup physics
   */
  struct Gravity
  {
    Gravity(const std::set<dealii::types::boundary_id>& dirichlet_ids);

    double action(
      const Field<2>& thickness,
      const Field<2>& surface,
      const VectorField<2>& velocity
    ) const;

    DualVectorField<2> derivative(
      const Field<2>& thickness,
      const Field<2>& surface,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    double derivative(
      const Field<2>& thickness,
      const Field<2>& surface,
      const VectorField<2>& direction
    ) const;

    const std::set<dealii::types::boundary_id>& dirichlet_ids;
  };


  /**
   * @brief Class for solving for the ice velocity and thickness, under the
   * shelfy stream approximation
   *
   * @ingroup physics
   */
  struct IceStream
  {
    IceStream(
      const std::set<dealii::types::boundary_id>& dirichlet_ids,
      const Viscosity& viscosity = Viscosity(MembraneStress()),
      const Friction& friction = Friction(BasalStress()),
      const double convergence_tolerance = 1.0e-6
    );

    using SolveOptions = numerics::NewtonSearchOptions<VectorField<2>>;

    VectorField<2> solve(
      const Field<2>& thickness,
      const Field<2>& surface,
      const Field<2>& theta,
      const Field<2>& beta,
      const VectorField<2>& velocity,
      const SolveOptions& solve_options = SolveOptions()
    ) const;

    const std::set<dealii::types::boundary_id> dirichlet_ids;

    const Gravity gravity;
    const Viscosity viscosity;
    const Friction friction;
    const double tolerance;
  };

}

#endif

