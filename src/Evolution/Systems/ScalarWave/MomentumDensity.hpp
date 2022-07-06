// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarWave {
/// @{
/*!
 * \brief Computes the energy density of the scalar wave system.
 *
 * Below is the function used to calculate the energy density.
 *
 * \f{align*}
 * \epsilon = \frac{1}{2}\left( \Pi^{2} + \abs{\Phi}^{2} \right)
 * \f}
 */
template <size_t SpatialDim>
void momentum_density(
    gsl::not_null<Scalar<DataVector>*> result, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi);

template <size_t SpatialDim>
Scalar<DataVector> momentum_density(
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi);
/// @}

namespace Tags {
/// \brief Computes the momentum density using ScalarWave::momentum_density()
template <size_t SpatialDim>
struct MomentumDensityCompute : MomentumDensity<SpatialDim>, db::ComputeTag {
  using argument_tags = tmpl::list<Pi, Phi<SpatialDim>>;

  using return_type = Scalar<DataVector>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*> result,
      const Scalar<DataVector>&,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&)>(
      &momentum_density<SpatialDim>);

  using base = MomentumDensity<SpatialDim>;
};
}  // namespace Tags
}  // namespace ScalarWave
