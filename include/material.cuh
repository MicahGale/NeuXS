#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#pragma once
#include "cross_section.cuh"

namespace neuxs {

template <typename T> struct NuclideComponent {
  CrossSection<T> nuclide;
  float atom_dens;
};
template <typename T> class Material {
public:
  __device__ void get_cross_section(neuxs::f_vec *energy, neuxs::f_vec *sigma_s,
                                    neuxs::f_vec *sigma_c,
                                    neuxs::f_vec *sigma_f,
                                    neuxs::f_vec *sigma_t);
  __device__ void decide_if_collide(neuxs::f_vec *energies,
                                    neuxs::f_vec *distance_escape,
                                    thrust::device_vector<bool> *collide);
  __device__ void decide_collide_type(
      neuxs::f_vec *energy,
      thrust::device_vector<CrossSectionDataType> *collision_types);

private:
  thrust::device_vector<NuclideComponent<T>> _components;
};

} // namespace neuxs

#endif // NEUXS_MATERIAL_H
