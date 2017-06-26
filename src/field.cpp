
#include <icepack/field.hpp>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>

namespace icepack
{
  using dealii::Function;
  using dealii::TensorFunction;

  /* -----------------------------------------
   * FieldType member function implementations
   * ----------------------------------------- */

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>::FieldType(const Discretization<dim>& dsc)
    : discretization_(&dsc),
      coefficients_(discretization()(rank).dof_handler().n_dofs())
  {
    coefficients_ = 0;
  }

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>::FieldType(FieldType<rank, dim, duality>&& phi)
    : discretization_(phi.discretization_),
      coefficients_(std::move(phi.coefficients_))
  {
    phi.discretization_ = nullptr;
    phi.coefficients_.reinit(0);
  }

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  FieldType<rank, dim, duality>::operator=(FieldType<rank, dim, duality>&& phi)
  {
    discretization_ = phi.discretization_;
    coefficients_ = std::move(phi.coefficients_);

    phi.discretization_ = nullptr;
    phi.coefficients_.reinit(0);

    return *this;
  }

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  FieldType<rank, dim, duality>::
  operator=(const FieldType<rank, dim, duality>& phi)
  {
    discretization_ = phi.discretization_;
    coefficients_ = phi.coefficients_;

    return *this;
  }

  template <int rank, int dim, Duality duality>
  const Discretization<dim>&
  FieldType<rank, dim, duality>::discretization() const
  {
    return *discretization_;
  }

  template <int rank, int dim, Duality duality>
  const dealii::Vector<double>&
  FieldType<rank, dim, duality>::coefficients() const
  {
    return coefficients_;
  }

  template <int rank, int dim, Duality duality>
  dealii::Vector<double>&
  FieldType<rank, dim, duality>::coefficients()
  {
    return coefficients_;
  }

  template <int rank, int dim, Duality duality>
  double
  FieldType<rank, dim, duality>::coefficient(const size_t i) const
  {
    return coefficients_[i];
  }


  // Explicitly instantiate a bunch of templates.
  template class FieldType<0, 2, primal>;
  template class FieldType<1, 2, primal>;
  template class FieldType<0, 2, dual>;
  template class FieldType<1, 2, dual>;



  /* -----------
   * File output
   * ----------- */

  template <int rank, int dim>
  void write_ucd(
    const FieldType<rank, dim>& u,
    const std::string& filename,
    const std::string& field_name
  )
  {
    std::ofstream stream(filename.c_str());

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(u.discretization()(rank).dof_handler());

    std::vector<std::string> component_names;
    if (rank == 1)
      for (unsigned int k = 0; k < dim; ++k)
        component_names.push_back(field_name + "_" + std::to_string(k + 1));
    else
      component_names.push_back(field_name);

    data_out.add_data_vector(u.coefficients(), component_names);
    data_out.build_patches();
    data_out.write_ucd(stream);
  }

  template
  void write_ucd<0, 2>(
    const Field<2>&, const std::string&, const std::string&
  );

  template
  void write_ucd<1, 2>(
    const VectorField<2>&, const std::string&, const std::string&
  );



  /* --------------------------------------------------------
   * Interpolating functions to finite element representation
   * -------------------------------------------------------- */

  template <int dim>
  Field<dim> interpolate(
    const Discretization<dim>& discretization,
    const dealii::Function<dim>& phi
  )
  {
    Field<dim> Phi(discretization);
    const auto& dof_handler = discretization.scalar().dof_handler();
    dealii::VectorTools::interpolate(dof_handler, phi, Phi.coefficients());
    return Phi;
  }


  template <int dim>
  VectorField<dim>
  interpolate(
    const Discretization<dim>& discretization,
    const dealii::TensorFunction<1, dim>& phi
  )
  {
    VectorField<dim> Phi(discretization);
    const dealii::VectorFunctionFromTensorFunction<dim> vphi(phi);
    const auto& dof_handler = discretization.vector().dof_handler();
    dealii::VectorTools::interpolate(dof_handler, vphi, Phi.coefficients());
    return Phi;
  }


  VectorField<2>
  interpolate(
    const Discretization<2>& discretization,
    const dealii::Function<2>& phi1,
    const dealii::Function<2>& phi2
  )
  {
    class Helper : public dealii::TensorFunction<1, 2>
    {
      dealii::SmartPointer<const dealii::Function<2>> phi1_;
      dealii::SmartPointer<const dealii::Function<2>> phi2_;

    public:
      Helper(const dealii::Function<2>& phi1, const dealii::Function<2>& phi2) :
        phi1_(&phi1), phi2_(&phi2)
      {}

      dealii::Tensor<1, 2> value(const dealii::Point<2>& x) const
      {
        return dealii::Tensor<1, 2>{{phi1_->value(x), phi2_->value(x)}};
      }
    };

    Helper helper(phi1, phi2);
    return interpolate(discretization, helper);
  }


  template Field<2> interpolate(const Discretization<2>&, const Function<2>&);

  template
  VectorField<2>
  interpolate(const Discretization<2>&, const TensorFunction<1, 2>&);

} // namespace icepack
