// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"

#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg {
template <size_t Dim>
void MortarData<Dim>::insert_local_geometric_quantities(
    const Scalar<DataVector>& local_volume_det_inv_jacobian,
    const Scalar<DataVector>& local_face_det_jacobian,
    const Scalar<DataVector>& local_face_normal_magnitude) {
  ASSERT(mortar_data_.has_value(),
         "Must set mortar data before setting the geometric quantities.");
  ASSERT(local_face_det_jacobian[0].size() ==
             local_face_normal_magnitude[0].size(),
         "The determinant of the local face Jacobian has "
             << local_face_det_jacobian[0].size()
             << " grid points, and the magnitude of the local face normal has "
             << local_face_normal_magnitude[0].size()
             << " but they must be the same");
  ASSERT(local_face_det_jacobian[0].size() ==
             std::get<0>(*mortar_data()).number_of_grid_points(),
         "The number of grid points ("
             << std::get<0>(*mortar_data()).number_of_grid_points()
             << ") on the local face must match the number of grid points "
                "passed in for the face Jacobian determinant and normal vector "
                "magnitude ("
             << local_face_det_jacobian[0].size() << ")");
  ASSERT(not using_only_face_normal_magnitude_,
         "The face normal, volume inverse Jacobian determinant, and face "
         "Jacobian determinant cannot be inserted because the only the face "
         "normal is being used.");
  using_volume_and_face_jacobians_ = true;
  const size_t required_storage_size = local_volume_det_inv_jacobian[0].size() +
                                       2 * local_face_det_jacobian[0].size();
  local_geometric_quantities_.destructive_resize(required_storage_size);

  std::copy(local_volume_det_inv_jacobian[0].begin(),
            local_volume_det_inv_jacobian[0].end(),
            local_geometric_quantities_.begin());
  std::copy(
      local_face_det_jacobian[0].begin(), local_face_det_jacobian[0].end(),
      local_geometric_quantities_.begin() +
          static_cast<std::ptrdiff_t>(local_volume_det_inv_jacobian[0].size()));
  std::copy(
      local_face_normal_magnitude[0].begin(),
      local_face_normal_magnitude[0].end(),
      local_geometric_quantities_.begin() +
          static_cast<std::ptrdiff_t>(local_volume_det_inv_jacobian[0].size() +
                                      local_face_det_jacobian[0].size()));
}

template <size_t Dim>
void MortarData<Dim>::insert_local_face_normal_magnitude(
    const Scalar<DataVector>& local_face_normal_magnitude) {
  ASSERT(mortar_data_.has_value(),
         "Must set mortar data before setting the local face normal.");
  ASSERT(not using_volume_and_face_jacobians_,
         "The face normal magnitude cannot be inserted if the face normal, "
         "volume inverse Jacobian determinant, and face Jacobian determinant "
         "are being used.");
  using_only_face_normal_magnitude_ = true;
  const size_t required_storage_size = local_face_normal_magnitude[0].size();
  local_geometric_quantities_.destructive_resize(required_storage_size);

  std::copy(local_face_normal_magnitude[0].begin(),
            local_face_normal_magnitude[0].end(),
            local_geometric_quantities_.begin());
}

template <size_t Dim>
void MortarData<Dim>::get_local_volume_det_inv_jacobian(
    const gsl::not_null<Scalar<DataVector>*> local_volume_det_inv_jacobian)
    const {
  ASSERT(mortar_data_.has_value(),
         "Must set mortar data before getting the local volume inverse "
         "Jacobian determinant.");
  ASSERT(
      local_geometric_quantities_.size() >
          2 * std::get<0>(*mortar_data()).number_of_grid_points(),
      "Cannot retrieve the volume inverse Jacobian determinant because it was "
      "not inserted.");
  ASSERT(
      using_volume_and_face_jacobians_,
      "Cannot retrieve the volume inverse Jacobian determinant because it was "
      "not inserted.");
  ASSERT(not using_only_face_normal_magnitude_,
         "Inconsistent internal state: we are apparently using both the volume "
         "and face Jacobians, as well as only the face normal.");
  const size_t num_face_points =
      std::get<0>(*mortar_data()).number_of_grid_points();
  const size_t num_volume_points =
      local_geometric_quantities_.size() - 2 * num_face_points;
  get(*local_volume_det_inv_jacobian)
      .set_data_ref(const_cast<double*>(  // NOLINT
                        local_geometric_quantities_.data()),
                    num_volume_points);
}

template <size_t Dim>
void MortarData<Dim>::get_local_face_det_jacobian(
    const gsl::not_null<Scalar<DataVector>*> local_face_det_jacobian) const {
  ASSERT(mortar_data_.has_value(),
         "Must set mortar data before getting the local face Jacobian "
         "determinant.");
  ASSERT(local_geometric_quantities_.size() >
             2 * std::get<0>(*mortar_data()).number_of_grid_points(),
         "Cannot retrieve the face Jacobian determinant because it was not "
         "inserted.");
  ASSERT(using_volume_and_face_jacobians_,
         "Cannot retrieve the face Jacobian determinant because it was not "
         "inserted.");
  ASSERT(not using_only_face_normal_magnitude_,
         "Inconsistent internal state: we are apparently using both the volume "
         "and face Jacobians, as well as only the face normal.");
  const size_t num_face_points =
      std::get<0>(*mortar_data()).number_of_grid_points();
  const size_t offset =
      local_geometric_quantities_.size() - 2 * num_face_points;
  get(*local_face_det_jacobian)
      .set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(  // NOLINTNEXTLINE
              local_geometric_quantities_.data() + offset),
          num_face_points);
}

template <size_t Dim>
void MortarData<Dim>::get_local_face_normal_magnitude(
    const gsl::not_null<Scalar<DataVector>*> local_face_normal_magnitude)
    const {
  ASSERT(mortar_data_.has_value(),
         "Must set local mortar data before getting the local face normal "
         "magnitude.");
  const size_t num_face_points =
      std::get<0>(*mortar_data()).number_of_grid_points();
  ASSERT(local_geometric_quantities_.size() == num_face_points or
             local_geometric_quantities_.size() > 2 * num_face_points,
         "Cannot retrieve the face normal magnitude because it was not "
         "inserted.");
  const size_t offset = local_geometric_quantities_.size() - num_face_points;
  get(*local_face_normal_magnitude)
      .set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(  // NOLINTNEXTLINE
              local_geometric_quantities_.data() + offset),
          num_face_points);
}

template <size_t Dim>
void MortarData<Dim>::pup(PUP::er& p) {
  p | mortar_data_;
  p | local_geometric_quantities_;
  p | using_volume_and_face_jacobians_;
  p | using_only_face_normal_magnitude_;
}

template <size_t Dim>
bool operator==(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return lhs.mortar_data() == rhs.mortar_data() and
         lhs.local_geometric_quantities_ == rhs.local_geometric_quantities_ and
         lhs.using_volume_and_face_jacobians_ ==
             rhs.using_volume_and_face_jacobians_ and
         lhs.using_only_face_normal_magnitude_ ==
             rhs.using_only_face_normal_magnitude_;
}

template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const MortarData<Dim>& mortar_data) {
  os << "MortarData: " << mortar_data.mortar_data() << "\n";
  return os;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                \
  template class MortarData<DIM(data)>;                       \
  template bool operator==(const MortarData<DIM(data)>& lhs,  \
                           const MortarData<DIM(data)>& rhs); \
  template bool operator!=(const MortarData<DIM(data)>& lhs,  \
                           const MortarData<DIM(data)>& rhs); \
  template std::ostream& operator<<(std::ostream& os,         \
                                    const MortarData<DIM(data)>& mortar_data);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
