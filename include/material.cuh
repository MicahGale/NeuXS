#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#endif // NEUXS_MATERIAL_H

namespace neuxs {
template <typename Derived> class Material {
public:
  __device__ void get_total_cross_section(float *energies, float *sigma_t,
                                          unsigned int n_particles);
  __device__ void decide_if_collide(float *energies, float *distance_escape,
                                    bool *collide, unsigned int n_particles);
  __device__ void
  decide_collide_type(float *energies,
                      neuxs::CrossSectionDataType *collision_types,
                      unsigned int n_particles);
};

class ArrayStructMat : public Material<ArrayStructMat> {};

class StructArrayMat : public Material<StructArrayMat> {};

class SLBWMat : public Material<SLBWMat> {};

} // namespace neuxs
