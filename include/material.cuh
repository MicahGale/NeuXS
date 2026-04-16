#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#include "cross_section.cuh"

namespace neuxs {

template <typename T> using DeviceVector = thrust::device_vector<T>;
template <typename T> using HostVector = thrust::host_vector<T>;

class OpenMCCrossSectionReader;

template <typename T> struct NuclideComponent {

  const unsigned int _nuclide_id;
  const T _atom_dens;
};

template <typename T> class Material {
public:
  Material(OpenMCCrossSectionReader &cross_section_reader);

  __host__ void addIsotope() = 0;

  __host__ void buildEnergyGrid(std::string isotope_name);

  __device__ void getCrossSection(DeviceVector<float> *energy,
                                  DeviceVector<float> *sigma_s,
                                  DeviceVector<float> *sigma_c,
                                  DeviceVector<float> *sigma_f,
                                  DeviceVector<float> *sigma_t);

  __device__ void decideIfCollide(DeviceVector<float> *energies,
                                  DeviceVector<float> *distance_escape,
                                  thrust::device_vector<bool> *collide);

  __device__ void decideCollideType(
      DeviceVector<float> *energy,
      thrust::device_vector<CrossSectionDataType> *collision_types);

private:
  const OpenMCCrossSectionReader &_cross_section_reader;
  thrust::device_vector<NuclideComponent<T>> _components;
};

} // namespace neuxs

#endif // NEUXS_MATERIAL_H
