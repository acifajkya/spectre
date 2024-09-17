// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"

#include <complex>
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::Ij<DataType, SpatialDim, Frame> up_down_from_lower_rank_2_tensor(
    const tnsr::ii<DataType, SpatialDim, Frame>& lower_rank_2_tensor,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  auto result = make_with_value<tnsr::Ij<DataType, SpatialDim, Frame>>(
      get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        result.get(j, k) +=
            lower_rank_2_tensor.get(i, k) * inverse_spatial_metric.get(i, j);
      }
    }
  }
  return result;
}
}  // namespace

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
void weyl_type_D1(
    const gsl::not_null<
        tnsr::ii<typename tenex::detail::get_complex_datatype<DataType>::type,
                 SpatialDim, Frame>*>
        weyl_type_D1,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magnetic,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  const std::complex<double> imag(0.0, 1.0);
  auto fancy_e = tenex::evaluate<ti::i, ti::j>(
      0.5 *
      ((weyl_electric(ti::i, ti::j)) - imag * (weyl_magnetic(ti::i, ti::j))));
  auto fancy_e_up_down = make_with_value<
      tnsr::Ij<typename tenex::detail::get_complex_datatype<DataType>::type,
               SpatialDim, Frame>>(get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        fancy_e_up_down.get(j, k) +=
            fancy_e.get(i, k) * inverse_spatial_metric.get(i, j);
      }
    }
  }

  auto upper_fancy_e = make_with_value<
      tnsr::II<typename tenex::detail::get_complex_datatype<DataType>::type,
               SpatialDim, Frame>>(get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t l = i; l < SpatialDim; ++l) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        upper_fancy_e.get(i, l) +=
            fancy_e_up_down.get(i, j) * inverse_spatial_metric.get(j, l);
      }
    }
  }

  // compute factor a = 16.0 * E_{ij} E^{ij}:
  auto a = tenex::evaluate(16.0 * (fancy_e(ti::i, ti::j)) *
                           (upper_fancy_e(ti::I, ti::J)));

  // Compute factor b = -64 E^k_i E^{ij} E_{jk}:
  auto b = make_with_value<
      Scalar<typename tenex::detail::get_complex_datatype<DataType>::type>>(
      get<0, 0>(inverse_spatial_metric), 0.0);

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        get(b) += fancy_e_up_down.get(k, i) * upper_fancy_e.get(i, j) *
                  fancy_e.get(j, k);
      }
    }
  }
  get(b) *= -64.0;

  // compute deviation from type D using measure D1:
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      weyl_type_D1->get(i, j) = (get(a) / 12.0) * spatial_metric.get(i, j) -
                                (get(b) / get(a) * fancy_e.get(i, j));
      for (size_t k = 0; k < SpatialDim; ++k) {
        weyl_type_D1->get(i, j) -=
            4.0 * (fancy_e_up_down.get(k, i) * fancy_e.get(j, k));
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<typename tenex::detail::get_complex_datatype<DataType>::type,
         SpatialDim, Frame>
weyl_type_D1(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magnetic,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  tnsr::ii<typename tenex::detail::get_complex_datatype<DataType>::type,
           SpatialDim, Frame>
      weyl_type_D1_result{};
  weyl_type_D1<DataType, SpatialDim, Frame>(
      make_not_null(&weyl_type_D1_result), weyl_electric, weyl_magnetic,
      spatial_metric, inverse_spatial_metric);
  return weyl_type_D1_result;
}

template <typename DataType, size_t SpatialDim, typename Frame>
void weyl_type_D1_scalar(
    const gsl::not_null<Scalar<DataType>*> weyl_type_D1_scalar_result,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  *weyl_type_D1_scalar_result =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);

  auto weyl_type_D1_up_down =
      up_down_from_lower_rank_2_tensor(weyl_type_D1, inverse_spatial_metric);

  for (size_t j = 0; j < SpatialDim; ++j) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      if (UNLIKELY(j == 0 and k == 0)) {
        get(*weyl_type_D1_scalar_result) =
            weyl_type_D1_up_down.get(j, k) * weyl_type_D1_up_down.get(k, j);
      } else {
        get(*weyl_type_D1_scalar_result) +=
            weyl_type_D1_up_down.get(j, k) * weyl_type_D1_up_down.get(k, j);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> weyl_type_D1_scalar(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  Scalar<DataType> weyl_type_D1_scalar_result{};
  weyl_type_D1_scalar<DataType, SpatialDim, Frame>(
      make_not_null(&weyl_type_D1_scalar_result), weyl_type_D1,
      inverse_spatial_metric);
  return weyl_type_D1_scalar_result;
}

}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template tnsr::ii<                                                        \
      typename tenex::detail::get_complex_datatype<DTYPE(data)>::type,      \
      DIM(data), FRAME(data)>                                               \
  gr::weyl_type_D1(                                                         \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_electric,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_magnetic,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,  \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_spatial_metric);                                          \
  template void gr::weyl_type_D1(                                           \
      const gsl::not_null<tnsr::ii<                                         \
          typename tenex::detail::get_complex_datatype<DTYPE(data)>::type,  \
          DIM(data), FRAME(data)>*>                                         \
          weyl_type_D1,                                                     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_electric,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_magnetic,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,  \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_spatial_metric);                                          \
  template void gr::weyl_type_D1_scalar(                                    \
      const gsl::not_null<Scalar<DTYPE(data)>*> weyl_type_D1_scalar_result, \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_type_D1,    \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_spatial_metric);                                          \
  template Scalar<DTYPE(data)> gr::weyl_type_D1_scalar(                     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_type_D1,    \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))
#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
