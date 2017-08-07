
#ifndef ICEPACK_MASS_TRANSPORT_HPP
#define ICEPACK_MASS_TRANSPORT_HPP

#include <icepack/field.hpp>

namespace icepack
{
  /**
   * @brief Return the timestep `dt` that gives a Courant number of a given
   * value for some velocity field.
   *
   * The Courant number for the advection equation with a velocity \f$u\f$,
   * mesh spacing \f$\delta x\f$, and timestep \f$\delta t\f$ is the quantity
   * \f[
   *   C = \max|u|\cdot \delta t/\delta x.
   * \f]
   * Many numerical methods for solving the advection equation are only stable
   * when the Courant number is below some upper limit.
   */
  double compute_timestep(
    const double courant_number,
    const VectorField<2>& u
  );


  /**
   * @brief Interface for updating the ice thickness from the continuity
   * equation
   *
   * The ice thickness \f$h\f$ evolves according to the advection equation
   * \f[
   *   \frac{\partial h}{\partial t} + \nabla\cdot hu = \dot a - \dot m
   * \f]
   * where \f$u\f$ is the ice velocity, \f$\dot a\f$ is the accumulation rate
   * and \f$m\f$ is the melt rate.
   *
   * There are several methods for solving these kinds of PDEs. For example,
   * fully implicit time-stepping schemes are unconditionally stable, but
   * require a more expensive linear solve. The Lax-Wendroff or streamlined
   * upwind Petrov-Galerkin methods, on the other hand, are explicit in time,
   * but they require that the Courant number be quite small for stability.
   *
   * This class is an interface for implementations of mass transport solvers;
   * it doesn't actually solve anything itself.
   *
   * @ingroup physics
   */
  struct MassTransport
  {
    /**
     * Solve the continuity equation over a single timestep.
     */
    virtual Field<2> solve(
      const double dt,
      const Field<2>& thickness,
      const Field<2>& accumulation,
      const VectorField<2>& velocity,
      const std::set<dealii::types::boundary_id>& inflow_boundary_ids,
      const Field<2>& inflow_thickness
    ) const = 0;
  };


  /**
   * @brief Updates the ice thickness from the continuity equation using an
   * implicit timestepping scheme
   *
   * @ingroup physics
   */
  struct MassTransportImplicit : public MassTransport
  {
    MassTransportImplicit();

    /**
     * Compute the matrix discretizing the action of the advective flux on a
     * given thickness field
     */
    dealii::SparseMatrix<double> flux_matrix(
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    /**
     * Implementation of the mass transport solve using an implicit time
     * discretization
     */
    Field<2> solve(
      const double dt,
      const Field<2>& thickness,
      const Field<2>& accumulation,
      const VectorField<2>& velocity,
      const std::set<dealii::types::boundary_id>& inflow_boundary_ids,
      const Field<2>& inflow_thickness
    ) const override;
  };

}

#endif

