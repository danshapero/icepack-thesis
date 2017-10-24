
#ifndef ICEPACK_ICE_STREAM_HPP
#define ICEPACK_ICE_STREAM_HPP

#include <icepack/physics/viscosity.hpp>
#include <icepack/physics/friction.hpp>
#include <icepack/physics/mass_transport.hpp>
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
   * @brief Compute the surface elevation from the bed elevation and thickness
   *
   * When the bed elevation is below sea level, the surface elevation is
   * determined by the equation
   * \f[
   *   s = \max\left\{b + h, (1 - \rho_I/\rho_W)\cdot h\right\},
   * \f]
   * where \f$b, h\f$ are respectively the bed elevation and thickness,
   * depending on whether the ice is grounded or floating.
   */
  Field<2> compute_surface(const Field<2>& thickness, const Field<2>& bed);


  /**
   * @brief Class for solving for the ice velocity and thickness, under the
   * shelfy stream approximation
   *
   * Some definitions:
   *   * \f$\dot\varepsilon\f$, \f$M\f$: the ice strain rate tensor and
   *   membrane stress tensor (see the documentation for `MembraneStress`)
   *   * \f$u\f$, \f$\tau\f$: the ice velocity and basal shear stress (see
   *   the documentation for `BasalStress`)
   *   * \f$h\f$, \f$s\f$: the ice thickness and surface elevation (see the
   *   documentation for `Gravity`)
   *
   * The velocity of a grounded ice stream is the minimizer of the functional
   * \f[
   *   P = \int_\Omega\left(\frac{n}{n + 1}hM:\dot\varepsilon + \frac{m}{m + 1}\tau\cdot u + \rho gh\nabla s\cdot u\right)\hspace{1pt}dx.
   * \f]
   * These terms are respectively the power dissipation due to viscosity, to
   * sliding friction, and the power input from ice flowing under its own
   * weight. This classes uses the `Viscosity`, `Friction`, and `Gravity`
   * classes to approximate the ice velocity by numericalling minimizing
   * \f$P\f$.
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

    Field<2> solve(
      const double dt,
      const Field<2>& thickness,
      const Field<2>& accumulation,
      const VectorField<2>& velocity,
      const Field<2>& inflow_thickness
    ) const;

    const std::set<dealii::types::boundary_id> dirichlet_ids;

    const MassTransport mass_transport;
    const Gravity gravity;
    const Viscosity viscosity;
    const Friction friction;
    const double tolerance;
  };

}

#endif

