# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# def weyl_type_D1(
#     weyl_electric, weyl_magentic, spatial_metric, inverse_spatial_metric
# ):
#     inverse_weyl_electric = np.einsum(
#         "lk,ik,lj",
#         weyl_electric,
#         inverse_spatial_metric,
#         inverse_spatial_metric,
#     )
#     a = 16 * (np.einsum("ij,ij", weyl_electric, inverse_weyl_electric))
#     b = -64 * (
#         np.einsum(
#             "il,lk,ij,jk",
#             weyl_electric,
#             inverse_spatial_metric,
#             inverse_weyl_electric,
#             weyl_electric,
#         )
#     )
#     return (
#         (a / 12) * (np.einsum("ij", spatial_metric))
#         - (np.einsum("ij", weyl_electric) * (b / a))
#         - 4
#         * np.einsum(
#             "im,km,jk", weyl_electric, inverse_spatial_metric, weyl_electric
#         )
#     )


def weyl_type_D1(
    weyl_electric, weyl_magentic, spatial_metric, inverse_spatial_metric
):
    imag = 1j
    fancy_e = 0.5 * (weyl_electric - imag * weyl_magnetic)
    fancy_e_up_down = np.einsum("ik, ij->jk", fancy_e, inverse_spatial_metric)
    upper_fancy_e = np.einsum(
        "ij, jl->il", fancy_e_up_down, inverse_spatial_metric
    )

    a = 16.0 * (np.einsum("ij, ij", fancy_e, upper_fancy_e))
    b = -64.0 * (
        np.einsum("ki, ij, jk", fancy_e_up_down, upper_fancy_e, fancy_e)
    )
    return (
        (a / 12.0) * (np.einsum("ij", spacial_metric))
        - (b / a) * (np.einsum("ij", fancy_e))
        - 4.0(np.einsum("ki, jk", fancy_e_up_down, fancy_e))
    )
