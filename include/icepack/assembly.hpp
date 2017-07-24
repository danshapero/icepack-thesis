
#ifndef ICEPACK_ASSEMBLY_HPP
#define ICEPACK_ASSEMBLY_HPP

#include <icepack/field.hpp>

namespace icepack
{
  /**
   * @brief Helper struct for storing the cell-wise values or gradients of a
   * single field during finite element assembly
   *
   * @note You should never need to actually refer to an object of this class
   * directly. You will create `Evaluator`s with `evaluate::function`,
   * `evaluate::gradient`, etc., and pass them directly to `make_assembly_data`
   * or `make_assembly_face_data`.
   *
   * The process for keeping updated cell-wise values of some scalar or vector
   * field during finite element assembly is roughly the same regardless of
   * what it is you're actually integrating. For example, to compute the
   * viscous power dissipation of an ice stream under the shallow shelf
   * approximation, you need the thickness and the strain rate (the symmetric
   * gradient of the velocity field) at the quadrature points of each cell of
   * the mesh. The number of values you need to keep updated is determined by
   * the quadrature rule you're using to approximate the integral. The type of
   * the values you store depends on what is you're computing, for example the
   * divergence of a vector field is a scalar and has type `double`, the strain
   * rate of a 2D flow is a symmetric rank-2 tensor, the gradient of a scalar
   * field is a rank-1 tensor, and so forth.
   *
   * The `Evaluator` class is used to manage all of the boilerplate necessary
   * for keeping updated values of scalar and vector fields on the cells of a
   * mesh. It has two members that we care about:
   *
   *   * `reinit()`: method to update the stored cell values given a view of a
   *   deal.II FE values object.
   *   * `values`: a `std::vector` of values of the field on the current cell
   *
   * In order to update the cell values, the evaluator uses a function that we
   * pass to the constructor; this is the `Eval` template argument. The actual
   * evaluation function that does the dirty work calls out to the appropriate
   * method of the FE values view that we pass in to `reinit()`.
   *
   * All of the different constraints between what the `value_type` should be,
   * what evaluation function we should use to update the field values should
   * be, and so forth are pretty complex. Rather than make the user deal with
   * all that, the class `evaluate` acts as a factory for `Evaluator` that
   * computes all of the types for you. You should probably never construct an
   * `Evaluator` object yourself -- use the routines in `evaluate` instead.
   * The result of calling one of the routines in `evaluate` should then be
   * passed directly to `make_assembly_data` or `make_assembly_face_data`; the
   * resulting assembly data object will aggregate together all of the
   * `Evaluators`.
   *
   * @ingroup assembly
   */
  template <int rank, int dim, typename T, typename Eval>
  struct Evaluator
  {
    using value_type = T;

    Evaluator(
      const FieldType<rank, dim>& field,
      const Eval& eval
    );

    void reinit(const typename FieldType<rank, dim>::view_type& view);

    const FieldType<rank, dim>& field;
    std::vector<T> values;
    const Eval eval;
  };


  /**
   * @brief Helper struct for evaluating shape functions
   */
  template <int rank, int dim, typename T>
  struct ShapeFn
  {
    using view_type = typename FieldType<rank, dim>::view_type;

    /**
     * Evaluate the stored method for the FE values view for a degree-of-
     * freedom at a quadrature point of the cell
     */
    T operator()(
      const view_type& view,
      const unsigned int dof_index,
      const unsigned int quadrature_point
    ) const;

    using method_type =
      T (view_type::*)(const unsigned int, const unsigned int) const;
    const method_type method;
  };


  /**
   * @brief Makes proxy objects for evaluating shape functions
   *
   * @ingroup assembly
   */
  template <int dim>
  struct shape_functions
  {
    /**
     * Make proxy objects for evaluating scalar shape functions
     */
    struct scalar
    {
      using view_type = typename Field<dim>::view_type;

      static ShapeFn<0, dim, double> value();

      static ShapeFn<0, dim, dealii::Tensor<1, dim>> gradient();
    };

    /**
     * Make proxy objects for evaluating vector shape functions
     */
    struct vector
    {
      using view_type = typename VectorField<dim>::view_type;

      static ShapeFn<1, dim, dealii::Tensor<1, dim>> value();

      static ShapeFn<1, dim, dealii::Tensor<2, dim>> gradient();

      static ShapeFn<1, dim, dealii::SymmetricTensor<2, dim>>
      symmetric_gradient();

      static ShapeFn<1, dim, double> divergence();
    };
  };



  /**
   * @brief Base class for keeping track of values and gradients of several
   * fields during finite element assembly
   *
   * This is a base class for different kinds of assembly data. There are two
   * types of assembly data: data on the cells of a mesh (`AssemblyData`) and
   * data on the faces of cells of a mesh (`AssemblyFaceData`). The procedures
   * to update cell-wise and face-wise assembly data are different. For cell-
   * wise assembly data, we only need the cell iterator, whereas for face-wise
   * data, we need the cell iterator and a face number. Additionally, cell-wise
   * assembly data stores `dealii::FEValues` objects, while face-wise assembly
   * data stores `dealii::FEFaceValues` objects.
   *
   * Nonetheless, there is a lot of common functionality between both cell-wise
   * and face-wise assembly data:
   *
   *   * `field_values()`: get a reference to the values of some field on the
   *   current cell
   *   * `fe_values_view()`: get a view of the scalar or vector
   *   `dealii::FEValues` object on the current cell
   *   * `JxW()`: get the Jacobian of the coordinate transformation at a
   *   quadrature point
   *
   * These methods are all identical, whether the assembly data is for cells or
   * faces of cells. In order to avoid re-implementing these functions in more
   * than one place, we use
   * [static polymorphism](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern#Static_polymorphism).
   * The use of static polymorphism is the reason for the `Derived` template
   * argument.
   *
   * Each class of the variadic `Args...` template argument is of type
   * `Evaluator` for some rank, cell value type, and evaluation function. See
   * the documentation for that class for more information about its role. The
   * assembly data classes aggregate together several evaluators, each of which
   * keeps track of the values, gradients, etc. of some field.
   *
   * @ingroup assembly
   */
  template <typename Derived, int dim, typename... Args>
  class AssemblyDataBase
  {
  public:
    /**
     * Return a tuple of the values of all the fields at the `q`-th quadrature
     * point of the current cell
     */
    std::tuple<typename Args::value_type...> values(const size_t q) const;

    /**
     * Return a reference to the underlying discretization for the input fields
     */
    const Discretization<dim>& discretization() const;

    /**
     * Return a reference to a view of the stored FE values object of the
     * rank specified in this method's template argument
     */
    template <int rank>
    const typename FieldType<rank, dim>::view_type& fe_values_view() const;

    /**
     * Return the weight of the coordinate transformation at a quadrature
     * point
     */
    double JxW(const unsigned int q) const;

  protected:
    AssemblyDataBase(const size_t n_q_points, Args&&... args);

    std::tuple<Args...> evaluators_;

    /**
     * Helper method to reinitialize an individual evaluator
     */
    template <int rank, typename T, typename Eval>
    void reinit_evaluator_(Evaluator<rank, dim, T, Eval>&);

    /**
     * Terminal case of template recursion in helper method to reinitialize all
     * of the stored evaluators
     */
    template <size_t N>
    typename std::enable_if<N == sizeof...(Args), void>::type reinit_();

    /**
     * Induction case of template recursion in helper method to reinitialize
     * all of the stored evaluators
     */
    template <size_t N>
    typename std::enable_if<N != sizeof...(Args), void>::type reinit_();


    template <size_t N>
    typename std::enable_if<N == sizeof...(Args), void>::type
    init_(const size_t n_q_points);

    template <size_t N>
    typename std::enable_if<N != sizeof...(Args), void>::type
    init_(const size_t n_q_points);
  };


  /**
   * @brief Keeps track of the cell-wise values and gradients of several fields
   * during finite element assembly
   *
   * @ingroup assembly
   */
  template <int dim, typename... Args>
  class AssemblyData :
    public AssemblyDataBase<AssemblyData<dim, Args...>, dim, Args...>
  {
  public:
    /**
     * Construct an assembly object given the number of quadrature points and a
     * variadic number of `Evaluator`s
     *
     * @note Use the function `make_assembly_data` instead of calling the
     * constructor for this class directly.
     */
    AssemblyData(const size_t n_q_points, Args&&... args);

    /**
     * Copy constructor
     */
    AssemblyData(const AssemblyData<dim, Args...>& assembly_data);

    /**
     * Reinitialize the stored `dealii::FEValues` objects and all of the field
     * evaluators for the current cell
     */
    void reinit(const typename Discretization<dim>::iterator& cell);

  protected:
    std::array<std::unique_ptr<dealii::FEValues<dim>>, 2> fe_values;

    friend AssemblyDataBase<AssemblyData<dim, Args...>, dim, Args...>;
  };


  /**
   * Construct an `AssemblyData` object from a variadic number of `Evaluator`
   * objects.
   *
   * @ingroup assembly
   */
  template <int dim, typename... Args>
  auto make_assembly_data(Args&&... args)
  {
    const dealii::QGauss<dim> qd = get_discretization(args.field...).quad();
    return AssemblyData<dim, Args...>(qd.size(), std::forward<Args>(args)...);
  }


  /**
   * @brief Keeps track of face-wise values and gradients of several fields
   * during finite element assembly
   *
   * @ingroup assembly
   */
  template <int dim, typename... Args>
  class AssemblyFaceData :
    public AssemblyDataBase<AssemblyFaceData<dim, Args...>, dim, Args...>
  {
  public:
    /**
     * Construct a face assembly object given the number of quadrature points
     * and a variadic number of empty `Evaluator`s
     *
     * @note Use the function `make_assembly_face_data` instead of calling the
     * constructor for this class directly.
     */
    AssemblyFaceData(const size_t n_q_points, Args&&... args);

    /**
     * Copy constructor
     */
    AssemblyFaceData(const AssemblyFaceData<dim, Args...>& assembly_face_data);

    /**
     * Reinitialize the stored `dealii::FEFaceValues` objects and all of the
     * field evaluators for the given face of the current cell
     */
    void reinit(
      const typename Discretization<dim>::iterator& cell,
      const unsigned int face_number
    );

    /**
     * Return the unit outward normal vector for the face at the given
     * quadrature point
     */
    dealii::Tensor<1, dim> normal_vector(const unsigned int q) const;

  protected:
    std::array<std::unique_ptr<dealii::FEFaceValues<dim>>, 2> fe_values;

    friend AssemblyDataBase<AssemblyFaceData<dim, Args...>, dim, Args...>;
  };


  /**
   * Construct an `AssemblyFaceData` object from a variadic number of
   * `Evaluator` objects.
   *
   * @ingroup assembly
   */
  template <int dim, typename... Args>
  auto make_assembly_face_data(Args&&... args)
  {
    const dealii::QGauss<dim - 1> qd =
      get_discretization(args.field...).face_quad();
    return
      AssemblyFaceData<dim, Args...>(qd.size(), std::forward<Args>(args)...);
  }


  /**
   * @brief Functions for creating `Evaluator` objects
   *
   * This class contains several static functions for constructing `Evaluator`
   * objects. There are several constraints on how an evaluator is constructed
   * -- what template arguments it takes, which evaluation method it uses for
   * the field in question, etc. Rather than make you figure this out every
   * time, the `evaluate` class contains functions that do all the work for
   * you and return an object of the right type.
   *
   * You shouldn't actually be storing the `Evaluator` object that these
   * functions create in some variable; instead, you'll most likely be passing
   * it on as one of the arguments to either `make_assembly_data(Args&&...)`
   * or `make_assembly_face_data(Args&&...)`. The concrete type of the
   * evaluator object doesn't matter, because the functions that make the
   * assembly data will just infer it for you through the magic of variadic
   * templates. Consequently, all of the functions in this class return an
   * object of type `auto`, since the actual type is both tedious to write out
   * and irrelevant.
   *
   * @ingroup assembly
   */
  struct evaluate
  {
    /**
     * Returns an object for computing the values of a finite element field on
     * cells of a triangulation (as opposed to any of its derivatives)
     */
    template <int rank, int dim>
    static auto function(const FieldType<rank, dim>& field);

    /**
     * Returns an object for computing the gradients of a finite element field
     * on cells of a triangulation
     */
    template <int rank, int dim>
    static auto gradient(const FieldType<rank, dim>& field);

    /**
     * Returns an object for compute the symmetrized gradient of a finite
     * element vector field on cells of a triangulation
     */
    template <int dim>
    static auto symmetric_gradient(const VectorField<dim>& field);

    /**
     * Returns an object for computing the divergence of a finite element
     * vector field on cells of a triangulation
     */
    template <int dim>
    static auto divergence(const VectorField<dim>& field);
  };


  namespace internal
  {
    template <typename F, typename Tuple, size_t... I>
    decltype(auto) apply_impl(F&& f, Tuple&& tuple, std::index_sequence<I...>)
    {
      return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(tuple))...);
    }

    template <typename F, typename Tuple>
    decltype(auto) apply(F&& f, Tuple&& tuple)
    {
      constexpr size_t N =
        std::tuple_size<typename std::remove_reference<Tuple>::type>::value;
      return apply_impl(f, tuple, std::make_index_sequence<N>());
    }
  }


  /**
   * Evaluate a functional of several fields given the kernel and some assembly
   * data
   *
   * @ingroup assembly
   */
  template <typename Functional, int dim, typename... Args>
  double integrate(
    Functional&& f,
    AssemblyData<dim, Args...>& assembly_data
  )
  {
    const Discretization<dim>& discretization = assembly_data.discretization();
    const dealii::QGauss<dim> quad = discretization.quad();
    const size_t n_q_points = quad.size();

    double integral = 0.0;
    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);

      for (size_t q = 0; q < n_q_points; ++q)
      {
        const double dx = assembly_data.JxW(q);
        const double cell_value = internal::apply(f, assembly_data.values(q));
        integral += cell_value * dx;
      }
    }

    return integral;
  }


  template <typename Functional, int rank, int dim, typename T, typename... Args>
  FieldType<rank, dim, dual> integrate(
    Functional&& F,
    const dealii::ConstraintMatrix& constraints,
    const ShapeFn<rank, dim, T>& shape_fn,
    AssemblyData<dim, Args...>& assembly_data
  )
  {
    const Discretization<dim>& discretization = assembly_data.discretization();
    const dealii::QGauss<dim> quad = discretization.quad();

    const size_t n_dofs = discretization(rank).finite_element().dofs_per_cell;
    dealii::Vector<double> cell_f(n_dofs);
    std::vector<dealii::types::global_dof_index> dof_ids(n_dofs);

    FieldType<rank, dim, dual> f(discretization);

    const auto& view = assembly_data.template fe_values_view<rank>();
    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);
      cell_f = 0;

      for (size_t q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);

        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const T phi_i = shape_fn(view, i, q);
          const double cell_value =
            internal::apply(F, std::tuple_cat(values, std::make_tuple(phi_i)));
          cell_f(i) += cell_value * dx;
        }
      }

      std::get<rank>(cell)->get_dof_indices(dof_ids);
      constraints.distribute_local_to_global(cell_f, dof_ids, f.coefficients());
    }

    return f;
  }


  template <typename Functional, int rank, int dim, typename T1, typename T2, typename... Args>
  dealii::SparseMatrix<double> integrate(
    Functional&& F,
    const dealii::ConstraintMatrix& constraints,
    const ShapeFn<rank, dim, T1>& shape_fn1,
    const ShapeFn<rank, dim, T2>& shape_fn2,
    AssemblyData<dim, Args...>& assembly_data
  )
  {
    const Discretization<dim>& discretization = assembly_data.discretization();
    const dealii::QGauss<dim> quad = discretization.quad();

    const size_t n_dofs = discretization(rank).finite_element().dofs_per_cell;
    dealii::FullMatrix<double> cell_m(n_dofs, n_dofs);
    std::vector<dealii::types::global_dof_index> dof_ids(n_dofs);

    dealii::SparseMatrix<double> M(discretization(rank).sparsity_pattern());

    const auto& view = assembly_data.template fe_values_view<rank>();
    for (const auto& cell: discretization)
    {
      assembly_data.reinit(cell);
      cell_m = 0;

      for (size_t q = 0; q < quad.size(); ++q)
      {
        const double dx = assembly_data.JxW(q);
        const auto values = assembly_data.values(q);

        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const T1 phi_i = shape_fn1(view, i, q);
          for (unsigned int j = 0; j < n_dofs; ++j)
          {
            const T2 phi_j = shape_fn2(view, j, q);
            const double cell_value =
              internal::apply(F, std::tuple_cat(values, std::make_tuple(phi_i, phi_j)));
            cell_m(i, j) += cell_value * dx;
          }
        }
      }

      std::get<rank>(cell)->get_dof_indices(dof_ids);
      constraints.distribute_local_to_global(cell_m, dof_ids, M);
    }

    return M;
  }



  /* ------------------------------------
   * Implementations of evaluator methods
   * ------------------------------------ */

  template <int rank, int dim>
  auto evaluate::function(const FieldType<rank, dim>& field)
  {
    using value_type = typename FieldType<rank, dim>::value_type;

    const auto
      eval = [](
        const typename FieldType<rank, dim>::view_type& view,
        const FieldType<rank, dim>& field,
        std::vector<value_type>& values
      )
      {
        view.get_function_values(field.coefficients(), values);
      };

    return Evaluator<rank, dim, value_type, decltype(eval)>(field, eval);
  }


  template <int rank, int dim>
  auto evaluate::gradient(const FieldType<rank, dim>& field)
  {
    using value_type = typename FieldType<rank, dim>::gradient_type;

    const auto
      eval = [](
        const typename FieldType<rank, dim>::view_type& view,
        const FieldType<rank, dim>& field,
        std::vector<value_type>& values
      )
      {
        view.get_function_gradients(field.coefficients(), values);
      };

    return Evaluator<rank, dim, value_type, decltype(eval)>(field, eval);
  }


  template <int dim>
  auto evaluate::symmetric_gradient(const VectorField<dim>& field)
  {
    using value_type = dealii::SymmetricTensor<2, dim>;

    const auto
      eval = [](
        const typename VectorField<dim>::view_type& view,
        const VectorField<dim>& field,
        std::vector<value_type>& values
      )
      {
        view.get_function_symmetric_gradients(field.coefficients(), values);
      };

    return Evaluator<1, dim, value_type, decltype(eval)>(field, eval);
  }


  template <int dim>
  auto evaluate::divergence(const VectorField<dim>& field)
  {
    using value_type = double;

    const auto
      eval = [](
        const typename VectorField<dim>::view_type& view,
        const VectorField<dim>& field,
        std::vector<value_type>& values
      )
      {
        view.get_function_divergences(field.coefficients(), values);
      };

    return Evaluator<1, dim, value_type, decltype(eval)>(field, eval);
  }


  template <int rank, int dim, typename T, typename Eval>
  Evaluator<rank, dim, T, Eval>::
  Evaluator(
    const FieldType<rank, dim>& field,
    const Eval& eval
  ) :
    field(field),
    values(0),
    eval(eval)
  {}


  template <int rank, int dim, typename T, typename Eval>
  void Evaluator<rank, dim, T, Eval>::reinit(
    const typename FieldType<rank, dim>::view_type& view
  )
  {
    eval(view, field, values);
  }



  /* ------------------------------------
   * Implementations of `ShapeFn` methods
   * ------------------------------------ */

  template <int rank, int dim, typename T>
  T ShapeFn<rank, dim, T>::operator()(
    const view_type& view,
    const unsigned int dof_index,
    const unsigned int quadrature_point
  ) const
  {
    return (view.*method)(dof_index, quadrature_point);
  }


  template <int dim>
  ShapeFn<0, dim, double> shape_functions<dim>::scalar::value()
  {
    return ShapeFn<0, dim, double>{&view_type::value};
  }


  template <int dim>
  ShapeFn<0, dim, dealii::Tensor<1, dim>>
  shape_functions<dim>::scalar::gradient()
  {
    return ShapeFn<0, dim, dealii::Tensor<1, dim>>{&view_type::gradient};
  }


  template <int dim>
  ShapeFn<1, dim, dealii::Tensor<1, dim>> shape_functions<dim>::vector::value()
  {
    return ShapeFn<1, dim, dealii::Tensor<1, dim>>{&view_type::value};
  }


  template <int dim>
  ShapeFn<1, dim, dealii::Tensor<2, dim>>
  shape_functions<dim>::vector::gradient()
  {
    return ShapeFn<1, dim, dealii::Tensor<2, dim>>{&view_type::gradient};
  }


  template <int dim>
  ShapeFn<1, dim, dealii::SymmetricTensor<2, dim>>
  shape_functions<dim>::vector::symmetric_gradient()
  {
    typename ShapeFn<1, dim, dealii::SymmetricTensor<2, dim>>::method_type
      method = &view_type::symmetric_gradient;
    return ShapeFn<1, dim, dealii::SymmetricTensor<2, dim>>{method};
  }


  template <int dim>
  ShapeFn<1, dim, double> shape_functions<dim>::vector::divergence()
  {
    return ShapeFn<1, dim, double>{&view_type::divergence};
  }



  /* ---------------------------------------------
   * Implementations of `AssemblyDataBase` methods
   * --------------------------------------------- */

  template <typename Derived, int dim, typename... Args>
  AssemblyDataBase<Derived, dim, Args...>::
  AssemblyDataBase(const size_t n_q_points, Args&&... args) :
    evaluators_(std::forward<Args>(args)...)
  {
    init_<0>(n_q_points);
  }


  namespace internal
  {
    template <typename... Args, size_t... Is>
    std::tuple<typename Args::value_type...>
    values_helper(
      const std::tuple<Args...>& evaluators,
      const size_t q,
      std::index_sequence<Is...>
    )
    {
      return std::make_tuple(std::get<Is>(evaluators).values[q]...);
    }
  }


  template <typename Derived, int dim, typename... Args>
  std::tuple<typename Args::value_type...>
  AssemblyDataBase<Derived, dim, Args...>::values(const size_t q) const
  {
    const auto index_sequence = std::index_sequence_for<Args...>{};
    return internal::values_helper(evaluators_, q, index_sequence);
  }


  namespace internal
  {
    template <size_t dim, typename Tuple, size_t... Is>
    const Discretization<dim>&
    assembly_data_discretization_helper(
      const Tuple& tuple,
      std::index_sequence<Is...>
    )
    {
      return get_discretization(std::get<Is>(tuple).field...);
    }
  }


  template <typename Derived, int dim, typename... Args>
  const Discretization<dim>&
  AssemblyDataBase<Derived, dim, Args...>::discretization() const
  {
    const auto index_sequence = std::index_sequence_for<Args...>();
    return internal::assembly_data_discretization_helper<dim>(evaluators_, index_sequence);
  }


  template <typename Derived, int dim, typename... Args>
  template <int rank>
  const typename FieldType<rank, dim>::view_type&
  AssemblyDataBase<Derived, dim, Args...>::fe_values_view() const
  {
    const typename FieldType<rank, dim>::extractor_type ex(0);
    const auto& fe_values =
      *std::get<rank>(static_cast<const Derived&>(*this).fe_values);
    return fe_values[ex];
  }


  template <typename Derived, int dim, typename... Args>
  double
  AssemblyDataBase<Derived, dim, Args...>::JxW(const unsigned int q) const
  {
    return std::get<0>(static_cast<const Derived&>(*this).fe_values)->JxW(q);
  }


  template <typename Derived, int dim, typename... Args>
  template <int rank, typename T, typename Eval>
  void AssemblyDataBase<Derived, dim, Args...>::
  reinit_evaluator_(Evaluator<rank, dim, T, Eval>& evaluator)
  {
    evaluator.reinit(fe_values_view<rank>());
  }


  template <typename Derived, int dim, typename... Args>
  template <size_t N>
  typename std::enable_if<N == sizeof...(Args), void>::type
  AssemblyDataBase<Derived, dim, Args...>::reinit_()
  {}


  template <typename Derived, int dim, typename... Args>
  template <size_t N>
  typename std::enable_if<N != sizeof...(Args), void>::type
  AssemblyDataBase<Derived, dim, Args...>::reinit_()
  {
    reinit_evaluator_(std::get<N>(evaluators_));
    reinit_<N + 1>();
  }


  template <typename Derived, int dim, typename... Args>
  template <size_t N>
  typename std::enable_if<N != sizeof...(Args), void>::type
  AssemblyDataBase<Derived, dim, Args...>::init_(const size_t n_q_points)
  {
    std::get<N>(evaluators_).values.resize(n_q_points);
    init_<N + 1>(n_q_points);
  }


  template <typename Derived, int dim, typename... Args>
  template <size_t N>
  typename std::enable_if<N == sizeof...(Args), void>::type
  AssemblyDataBase<Derived, dim, Args...>::init_(const size_t)
  {}



  /* -----------------------------------------
   * Implementations of `AssemblyData` methods
   * ----------------------------------------- */

  namespace internal
  {
    template <int dim>
    std::array<std::unique_ptr<dealii::FEValues<dim>>, 2>
    make_fe_values_pair(const Discretization<dim>& discretization)
    {
      const dealii::QGauss<dim> quad = discretization.quad();
      using DefaultFlags::flags;

      const auto& scalar_fe = discretization(0).finite_element();
      const auto& vector_fe = discretization(1).finite_element();

      auto scalar_fe_values =
        std::make_unique<dealii::FEValues<dim>>(scalar_fe, quad, flags);
      auto vector_fe_values =
        std::make_unique<dealii::FEValues<dim>>(vector_fe, quad, flags);

      return std::array<std::unique_ptr<dealii::FEValues<dim>>, 2>
        {{std::move(scalar_fe_values), std::move(vector_fe_values)}};
    }
  }


  template <int dim, typename... Args>
  AssemblyData<dim, Args...>::
  AssemblyData(const size_t n_q_points, Args&&... args) :
    AssemblyDataBase<AssemblyData<dim, Args...>, dim, Args...>
      (n_q_points, std::forward<Args>(args)...),
    fe_values(internal::make_fe_values_pair(get_discretization(args.field...)))
  {}


  template <int dim, typename...Args>
  AssemblyData<dim, Args...>::
  AssemblyData(const AssemblyData<dim, Args...>& assembly_data) :
    AssemblyDataBase<AssemblyData<dim, Args...>, dim, Args...>(assembly_data),
    fe_values(internal::make_fe_values_pair(assembly_data.discretization()))
  {}


  template <int dim, typename... Args>
  void AssemblyData<dim, Args...>::
  reinit(const typename Discretization<dim>::iterator& cell)
  {
    std::get<0>(fe_values)->reinit(std::get<0>(*cell));
    std::get<1>(fe_values)->reinit(std::get<1>(*cell));

    this->template reinit_<0>();
  }



  /* ---------------------------------------------
   * Implementations of `AssemblyFaceData` methods
   * --------------------------------------------- */

  namespace internal
  {
    template <int dim>
    std::array<std::unique_ptr<dealii::FEFaceValues<dim>>, 2>
    make_fe_face_values_pair(const Discretization<dim>& discretization)
    {
      const dealii::QGauss<dim - 1> quad = discretization.face_quad();
      using DefaultFlags::face_flags;

      const auto& scalar_fe = discretization(0).finite_element();
      const auto& vector_fe = discretization(1).finite_element();

      auto scalar_fe_values =
        std::make_unique<dealii::FEFaceValues<dim>>(scalar_fe, quad, face_flags);
      auto vector_fe_values =
        std::make_unique<dealii::FEFaceValues<dim>>(vector_fe, quad, face_flags);

      return std::array<std::unique_ptr<dealii::FEFaceValues<dim>>, 2>
        {{std::move(scalar_fe_values), std::move(vector_fe_values)}};
    }
  }


  template <int dim, typename... Args>
  AssemblyFaceData<dim, Args...>::
  AssemblyFaceData(const size_t n_q_points, Args&&... args) :
    AssemblyDataBase<AssemblyFaceData<dim, Args...>, dim, Args...>
      (n_q_points, std::forward<Args>(args)...),
    fe_values
      (internal::make_fe_face_values_pair(get_discretization(args.field...)))
  {}


  template <int dim, typename... Args>
  AssemblyFaceData<dim, Args...>::
  AssemblyFaceData(const AssemblyFaceData<dim, Args...>& assembly_face_data) :
    AssemblyDataBase<AssemblyFaceData<dim, Args...>, dim, Args...>
      (assembly_face_data),
    fe_values
      (internal::make_fe_face_values_pair(assembly_face_data.discretization()))
  {}


  template <int dim, typename... Args>
  void AssemblyFaceData<dim, Args...>::
  reinit(
    const typename Discretization<dim>::iterator& cell,
    const unsigned int face_number
  )
  {
    std::get<0>(fe_values)->reinit(std::get<0>(*cell), face_number);
    std::get<1>(fe_values)->reinit(std::get<1>(*cell), face_number);

    this->template reinit_<0>();
  }


  template <int dim, typename... Args>
  dealii::Tensor<1, dim>
  AssemblyFaceData<dim, Args...>::normal_vector(const unsigned int q) const
  {
    return fe_values[0]->normal_vector(q);
  }


}

#endif

