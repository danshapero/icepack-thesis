
#ifndef ICEPACK_FIELD_HPP
#define ICEPACK_FIELD_HPP

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <icepack/utilities.hpp>
#include <icepack/discretization.hpp>

namespace icepack
{
  using dealii::Tensor;
  using dealii::Vector;


  /**
   * This type is for keeping track of whether a field is in the function space
   * \f$H^1\f$, i.e. the primal space, or in the dual space \f$H^{-1}\f$.
   */
  enum Duality {primal, dual};


  /**
   * This is a curiously recurring template pattern base class representing any
   * type that can be converted to a field, including algebraic expressions on
   * fields.
   */
  template <int rank, int dim, Duality duality, class Expr>
  class FieldExpr
  {
  public:
    /**
     * Implicit conversion to a reference to the underlying expression type.
     */
    operator const Expr&() const;

    /**
     * Evaluate one Galerkin expansion coefficient of a field expression.
     */
    double coefficient(const size_t i) const;

    /**
     * Return a reference to the discretization of the field expression.
     */
    const Discretization<dim>& discretization() const;
  };


  /**
   * This is a base class for any physical field discretized using a finite
   * element expansion. It is used as the return and argument types of all
   * glacier model objects (see `include/icepack/glacier_models`).
   *
   * Fields are distinguished by:
   *   - their tensor rank, i.e. 0 for a scalar field, 1 for a vector field
   *   - the dimension of the ambient space
   *   - whether the field is primal or dual
   *
   * Primal fields are the kind you are most familiar with, i.e. you can
   * evaluate them at a point. Dual fields are something that can be the right-
   * hand side of a PDE, e.g. the product of the mass matrix and a primal field
   * or the product of a differential operator and a primal field. The math
   * behind this follows, but if you aren't interested in that, you can get by
   * knowing that you will probably never need a dual field. The glacier model
   * classes' key functionalities are to solve the diagnostic and prognostic
   * equations for each model, and these routines all return primal fields.
   *
   * In finite element analysis, one uses a weak formulation of a PDE so that
   * its solution can be expressed in terms of functions have fewer derivatives
   * than the strong form mandates. For example, the Laplace operator is viewed
   * as a map from the space \f$H^1\f$ of functions with square-integrable
   * derivatives to its dual space \f$H^{-1}\f$. We cannot compute the value of
   * \f$-\nabla^2\phi\f$ for some field \f$\phi\f$, but we can compute the
   * integral
   \f[
   L[\phi, \psi] = -\int\nabla^2\phi\cdot\psi dx
   = \int\nabla\phi\cdot\nabla\psi dx
   \f]
   * for any field \f$\psi\f$ by pushing one of the derivatives onto \f$\psi\f$
   * using the divergence theorem. In other words, the Laplacian of a field is
   * thought of instead as a linear functional on the space of physical fields.
   *
   * Some procedures naturally give elements of \f$H^1\f$ as a result, while
   * others return elements of \f$H^{-1}\f$. For example, computing the driving
   * stress resulting from a given thickness and surface elevation gives a
   * dual field; computing the derivative of a nonlinear functional gives a
   * dual field; solving the diagnostic equations for a given driving stress
   * returns a primal field. In general, taking derivatives maps a primal to a
   * dual field, while solving a PDE takes a dual field to a primal field.
   *
   * The distinction between primal and dual fields is useful because, without
   * it, it might be unclear whether the result of some computation has a
   * factor of the finite element mass matrix or not, a common source of
   * mistakes. By introducing a type-level distinction between primal and dual
   * fields, we guarantee that the dual fields are those which have a factor of
   * the mass matrix, and primal fields do not. Moreover, dual fields have
   * different units: an extra factor of area in 2D and volume in 3D.
   */
  template <int rank, int dim, Duality duality = primal>
  class FieldType :
    public FieldExpr<rank, dim, duality, FieldType<rank, dim, duality>>
  {
  public:

    /**
     * Construct a field which is 0 everywhere given the data about its finite
     * element discretization.
     */
    FieldType(const Discretization<dim>& discretization);

    /**
     * Move constructor. This allows fields to be returned from functions,
     * so that one can write things like
     *
     *     Field<dim> u = solve_pde(kappa, f);
     *
     * without an expensive and unnecessary copy operation by using C++11
     * move semantics.
     */
    FieldType(FieldType<rank, dim, duality>&& phi);

    /**
     * Copy the values of another field. Note that the discretization member is
     * just a `dealii::SmartPointer`, so this copies the address of the object
     * and not its contents.
     *
     * This method allocates memory and should be used sparingly, so the method
     * is explicit -- copies will only be made if you really want them.
     */
    explicit FieldType(const FieldType<rank, dim, duality>& phi) = default;


    /**
     * The destructor for a field deallocates all the memory used for e.g. the
     * coefficients, and releases the hold on the discretization.
     */
    virtual ~FieldType() = default;


    /**
     * Move assignment operator. Like the move constructor, this allows fields
     * to be returned from functions, even after their declaration:
     *
     *     Field<dim> kappa = initial_guess();
     *     Field<dim> u = solve_pde(kappa, f);
     *     kappa = update_guess(u);
     *
     * This functionality is useful when solving nonlinear PDE iteratively.
     */
    FieldType<rank, dim, duality>&
    operator=(FieldType<rank, dim, duality>&& phi);

    /**
     * Copy assignment operator
     */
    FieldType<rank, dim, duality>&
    operator=(const FieldType<rank, dim, duality>& phi);


    /**
     * Return a const reference to the underlying discretization for this field,
     * from which various other useful data can be accessed -- the degree-of-
     * freedom handler, the hanging node constraints, etc.
     */
    const Discretization<dim>& discretization() const;

    /**
     * Return a const reference to the Galerkin expansion coefficients for this
     * field.
     */
    const Vector<double>& coefficients() const;

    /**
     * Return a reference to the Galerkin expansion coefficients for this field.
     */
    Vector<double>& coefficients();

    /**
     * Return a single Galerkin expansion coefficient for this field.
     */
    double coefficient(const size_t i) const;


    /**
     * The `value_type` for a scalar field is a real number, for a vector field
     * a rank-1 tensor, and so on and so forth. The member class `tensor_type`
     * of the `dealii::Tensor` class template is aliases the right value type,
     * i.e. it reduces to `double` for rank 0.
     */
    using value_type = typename Tensor<rank, dim>::tensor_type;

    /**
     * Same considerations as for `value_type` but for the gradient, i.e. the
     * gradient type of a scalar field is rank-1 tensor, for a vector field
     * rank-2 tensor, etc.
     */
    using gradient_type = typename Tensor<rank + 1, dim>::tensor_type;

    /**
     * This is a bit of template magic for selecting the right type to extract
     * the values of a finite element field, depending on whether it's a scalar
     * or a vector.
     * TODO: make it work for tensor fields too.
     */
    using extractor_type =
      typename std::conditional<rank == 0,
                                dealii::FEValuesExtractors::Scalar,
                                dealii::FEValuesExtractors::Vector>::type;

  protected:
    /**
     * Keep a pointer to the underlying discretization for this field. If the
     * discretization is destroyed before this field is done with it, deal.II
     * will throw an exception to let us know that something went wrong.
     */
    dealii::SmartPointer<const Discretization<dim>> discretization_;

    /**
     * The Galerkin expansion coefficients for this field.
     */
    Vector<double> coefficients_;
  };


  /**
   * Some sensible type aliases for different sorts of fields.
   */
  template <int dim> using Field = FieldType<0, dim, primal>;
  template <int dim> using VectorField = FieldType<1, dim, primal>;
  template <int dim> using DualField = FieldType<0, dim, dual>;
  template <int dim> using DualVectorField = FieldType<1, dim, dual>;



  /* --------------------------------------------------------
   * Interpolating functions to finite element representation
   * -------------------------------------------------------- */

  template <int dim>
  Field<dim> interpolate(
    const Discretization<dim>& discretization,
    const dealii::Function<dim>& phi
  );


  template <int dim>
  VectorField<dim>
  interpolate(
    const Discretization<dim>& discretization,
    const dealii::TensorFunction<1, dim>& phi
  );



  /* -----------------------------------------
   * Field norms, inner products, and adjoints
   * ----------------------------------------- */

  template <typename F>
  const auto& get_discretization(const F& phi)
  {
    return phi.discretization();
  }

  template <typename F, typename... Args>
  const auto& get_discretization(const F& phi, Args&&... args)
  {
    const auto& dsc1 = phi.discretization();
    const auto& dsc2 = get_discretization(args...);
    Assert(&dsc1 == &dsc2, dealii::ExcInternalError());

    return dsc1;
  }


  /**
   * Convert a primal field to a dual field. This amounts to multiplying the
   * Galerkin expansion coefficients of the primal field by the mass matrix.
   */
  template <int rank, int dim>
  FieldType<rank, dim, dual> transpose(const FieldType<rank, dim, primal>& phi)
  {
    const auto& discretization = phi.discretization();
    FieldType<rank, dim, dual> f(discretization);

    const auto& M = discretization(rank).mass_matrix();
    M.vmult(f.coefficients(), phi.coefficients());

    return f;
  }


  /**
   * Convert a dual field to a primal field. This amounts to multiplying the
   * Galerkin expansion coefficients of the dual field by the inverse of the
   * mass matrix.
   */
  template <int rank, int dim>
  FieldType<rank, dim, primal> transpose(const FieldType<rank, dim, dual>& f)
  {
    const auto& discretization = f.discretization();
    FieldType<rank, dim, primal> phi(discretization);

    const auto& M = discretization(rank).mass_matrix();

    // TODO: make these member variables of the discretization and let the user
    // set them if they want to.
    const unsigned int max_iterations = f.coefficients().size();
    const double tolerance = 1.0e-8;
    dealii::SolverControl solver_control(max_iterations, tolerance);
    solver_control.log_result(false);
    dealii::SolverCG<> solver(solver_control);
    const auto id = dealii::PreconditionIdentity();
    solver.solve(M, phi.coefficients(), f.coefficients(), id);

    discretization(rank).constraints().distribute(phi.coefficients());

    return phi;
  }


  /**
   * Compute the L2 norm of a field.
   */
  template <int rank, int dim>
  double norm(const FieldType<rank, dim>& phi)
  {
    const auto& M = phi.discretization()(rank).mass_matrix();
    return std::sqrt(M.matrix_norm_square(phi.coefficients()));
  }

  /**
   * Compute the L2 inner product of two fields.
   */
  template <int rank, int dim>
  double inner_product(
    const FieldType<rank, dim>& phi1,
    const FieldType<rank, dim>& phi2
  )
  {
    const auto& discretization = get_discretization(phi1, phi2);
    const auto& M = discretization(rank).mass_matrix();
    return M.matrix_scalar_product(phi1.coefficients(), phi2.coefficients());
  }

  /**
   * Compute the duality pairing between a linear functional `phi1` and a field
   * `phi2`.
   */
  template <int rank, int dim>
  double inner_product(
    const FieldType<rank, dim, dual>& phi1,
    const FieldType<rank, dim, primal>& phi2
  )
  {
    get_discretization(phi1, phi2);
    return phi1.coefficients() * phi2.coefficients();
  }

  /**
   * Compute the distance between two fields in the L2 norm. This is equal to
   * the norm of the difference of the fields.
   */
  template <int rank, int dim>
  double dist(
    const FieldType<rank, dim>& phi1,
    const FieldType<rank, dim>& phi2
  )
  {
    const auto& discretization = get_discretization(phi1, phi2);
    const auto& fe = discretization(rank).finite_element();
    const QGauss<dim> quad = discretization.quad();
    dealii::FEValues<dim> fe_values(fe, quad, DefaultFlags::flags);

    const unsigned int n_q_points = quad.size();
    using value_type = typename FieldType<rank, dim>::value_type;
    std::vector<value_type> phi1_values(n_q_points);
    std::vector<value_type> phi2_values(n_q_points);

    const typename FieldType<rank, dim>::extractor_type ex(0);

    double dist_squared = 0.0;
    for (auto cell: discretization(rank).dof_handler().active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[ex].get_function_values(phi1.coefficients(), phi1_values);
      fe_values[ex].get_function_values(phi2.coefficients(), phi2_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double dx = fe_values.JxW(q);
        const auto diff = phi1_values[q] - phi2_values[q];
        dist_squared += (diff * diff) * dx;
      }
    }

    return std::sqrt(dist_squared);
  }



  /* -----------------------------------
   * Expression templates:
   * Abandon hope, all ye who enter here
   * ----------------------------------- */

  template <int rank, int dim, Duality duality, class Expr>
  FieldExpr<rank, dim, duality, Expr>::operator const Expr&() const
  {
    return static_cast<const Expr&>(*this);
  }

  template <int rank, int dim, Duality duality, class Expr>
  double FieldExpr<rank, dim, duality, Expr>::coefficient(const size_t i) const
  {
    return static_cast<const Expr&>(*this).coefficient(i);
  }

  template <int rank, int dim, Duality duality, class Expr>
  const Discretization<dim>&
  FieldExpr<rank, dim, duality, Expr>::discretization() const
  {
    return static_cast<const Expr&>(*this).discretization();
  }


  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator *=(FieldType<rank, dim, duality>& phi, const double alpha)
  {
    phi.coefficients() *= alpha;
    return phi;
  }

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator /=(FieldType<rank, dim, duality>& phi, const double alpha)
  {
    phi.coefficients() *= 1.0/alpha;
    return phi;
  }


  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator +=(FieldType<rank, dim, duality>& phi,
              const FieldType<rank, dim, duality>& psi)
  {
    get_discretization(phi, psi);
    phi.coefficients().add(1.0, psi.coefficients());
    return phi;
  }


  template <int rank, int dim, Duality duality, class Expr>
  FieldType<rank, dim, duality>&
  operator +=(FieldType<rank, dim, duality>& phi,
              const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    get_discretization(phi, expr);
    Vector<double>& Phi = phi.coefficients();
    for (unsigned int i = 0; i < Phi.size(); ++i)
      Phi(i) += expr.coefficient(i);

    return phi;
  }


  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator -=(FieldType<rank, dim, duality>& phi,
              const FieldType<rank, dim, duality>& psi)
  {
    get_discretization(phi, psi);
    phi.coefficients().add(-1.0, psi.coefficients());
    return phi;
  }


  template <int rank, int dim, Duality duality, class Expr>
  FieldType<rank, dim, duality>&
  operator -=(FieldType<rank, dim, duality>& phi,
              const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    get_discretization(phi, expr);
    Vector<double>& Phi = phi.coefficients();
    for (unsigned int i = 0; i < Phi.size(); ++i)
      Phi(i) -= expr.coefficient(i);

    return phi;
  }


  template <int rank, int dim, Duality duality, class Expr>
  class ScalarMultiplyExpr :
    public FieldExpr<rank, dim, duality,
                     ScalarMultiplyExpr<rank, dim, duality, Expr>>
  {
  public:
    ScalarMultiplyExpr(const double alpha, const Expr& expr)
      : alpha(alpha), expr(expr)
    {}

    double coefficient(const size_t i) const
    {
      return alpha * expr.coefficient(i);
    }

    const Discretization<dim>& discretization() const
    {
      return expr.discretization();
    }

  protected:
    const double alpha;
    const Expr& expr;
  };


  template <int rank, int dim, Duality duality, class Expr>
  ScalarMultiplyExpr<rank, dim, duality, Expr>
  operator*(const double alpha, const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    return ScalarMultiplyExpr<rank, dim, duality, Expr>(alpha, expr);
  }


  template <int rank, int dim, Duality duality, class Expr>
  ScalarMultiplyExpr<rank, dim, duality, Expr>
  operator/(const FieldExpr<rank, dim, duality, Expr>& expr, const double alpha)
  {
    return ScalarMultiplyExpr<rank, dim, duality, Expr>(1.0/alpha, expr);
  }


  template <int rank, int dim, Duality duality, class Expr>
  ScalarMultiplyExpr<rank, dim, duality, Expr>
  operator-(const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    return ScalarMultiplyExpr<rank, dim, duality, Expr>(-1.0, expr);
  }


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  class AddExpr :
    public FieldExpr<rank, dim, duality,
                     AddExpr<rank, dim, duality, Expr1, Expr2>>
  {
  public:
    AddExpr(const Expr1& expr1, const Expr2& expr2)
      : expr1(expr1), expr2(expr2)
    {
      get_discretization(expr1, expr2);
    }

    double coefficient(const size_t i) const
    {
      return expr1.coefficient(i) + expr2.coefficient(i);
    }

    const Discretization<dim>& discretization() const
    {
      return expr1.discretization();
    }

  protected:
    const Expr1& expr1;
    const Expr2& expr2;
  };


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  AddExpr<rank, dim, duality, Expr1, Expr2>
  operator+(const FieldExpr<rank, dim, duality, Expr1>& expr1,
            const FieldExpr<rank, dim, duality, Expr2>& expr2)
  {
    return AddExpr<rank, dim, duality, Expr1, Expr2>(expr1, expr2);
  }


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  class SubtractExpr :
    public FieldExpr<rank, dim, duality,
                     SubtractExpr<rank, dim, duality, Expr1, Expr2> >
  {
  public:
    SubtractExpr(const Expr1& expr1, const Expr2& expr2)
      : expr1(expr1), expr2(expr2)
    {
      get_discretization(expr1, expr2);
    }

    double coefficient(const size_t i) const
    {
      return expr1.coefficient(i) - expr2.coefficient(i);
    }

    const Discretization<dim>& discretization() const
    {
      return expr1.discretization();
    }

  protected:
    const Expr1& expr1;
    const Expr2& expr2;
  };


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  SubtractExpr<rank, dim, duality, Expr1, Expr2>
  operator-(const FieldExpr<rank, dim, duality, Expr1>& expr1,
            const FieldExpr<rank, dim, duality, Expr2>& expr2)
  {
    return SubtractExpr<rank, dim, duality, Expr1, Expr2>(expr1, expr2);
  }


} // namespace icepack


#endif
