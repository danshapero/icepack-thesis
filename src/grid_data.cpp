
#include <fstream>
#include <icepack/grid_data.hpp>

namespace icepack
{

  namespace
  {
    dealii::Table<2, bool>
    make_missing_data_mask(
      const dealii::Table<2, double>& data,
      const double missing_data_value
    )
    {
      dealii::Table<2, bool> mask(data.size(0), data.size(1));

      for (size_t i = 0; i < data.size(0); ++i)
        for (size_t j = 0; j < data.size(1); ++j)
          mask(i, j) = (data(i, j) == missing_data_value);

      return mask;
    }
  }


  using dealii::Functions::InterpolatedTensorProductGridData;

  GridData::GridData(
    const std::array<std::vector<double>, 2>& coordinate_values,
    const dealii::Table<2, double>& data_values,
    const double missing_data_value
  ) : InterpolatedTensorProductGridData<2>(coordinate_values, data_values),
      xrange{{coordinate_values[0][0], coordinate_values[0].back()}},
      yrange{{coordinate_values[1][0], coordinate_values[1].back()}},
      missing_data_value(missing_data_value),
      mask(make_missing_data_mask(data_values, missing_data_value))
  {}


  double GridData::value(const dealii::Point<2>& x, const unsigned int) const
  {
    if (is_masked(x))
      return missing_data_value;

    return InterpolatedTensorProductGridData<2>::value(x);
  }


  bool GridData::is_masked(const dealii::Point<2>& x) const
  {
    const auto idx = table_index_of_point(x);
    return
      mask(idx[0], idx[1]) or
      mask(idx[0] + 1, idx[1]) or
      mask(idx[0], idx[1] + 1) or
      mask(idx[0] + 1, idx[1] + 1);
  }


  GridData readArcAsciiGrid(const std::string& filename)
  {
    unsigned int nx, ny;
    double x0, y0, dx, dy, missing;
    std::string dummy;

    std::ifstream file_stream(filename);
    file_stream >> dummy >> nx >> dummy >> ny;
    file_stream >> dummy >> x0 >> dummy >> y0;
    file_stream >> dummy >> dx >> dummy >> missing;
    dy = dx;

    std::vector<double> x(nx);
    std::vector<double> y(ny);
    dealii::Table<2, double> table(nx, ny);

    for (unsigned int i = 0; i < ny; ++i)
      y[i] = y0 + i * dy;

    for (unsigned int j = 0; j < nx; ++j)
      x[j] = x0 + j * dx;

    std::array<std::vector<double>, 2> coordinate_values = {{x, y}};

    for (unsigned int i = 0; i < ny; ++i)
      for (unsigned int j = 0; j < nx; ++j)
        file_stream >> table[j][ny - i - 1];

    file_stream.close();

    return GridData(coordinate_values, table, missing);
  }

}
