#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#endif // NEUXS_MATERIAL_H

#pragma once
#include "cross_section.cuh"

namespace neuxs {

template <typename T> struct NuclideComponent {
  CrossSection<T> nuclide;
  float atom_dens;
};
template <typename T> class Material {
public:
  __device__ void get_cross_section(float *energy, float *sigma_s,
                                    float *sigma_c, float *sigma_f,
                                    float *sigma_t, unsigned int n_particles);
  __device__ void decide_if_collide(float *energies, float *distance_escape,
                                    bool *collide, unsigned int n_particles);
  __device__ void decide_collide_type(float *energies,
                                      CrossSectionDataType *collision_types,
                                      unsigned int n_particles);

private:
  thrust::device_vector<NuclideComponent<T>> _components;
};

} // namespace neuxs
