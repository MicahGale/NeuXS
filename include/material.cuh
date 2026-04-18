#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#include <cuda_runtime.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace neuxs {

template <typename T> using DeviceVector = thrust::device_vector<T>;
template <typename T> using HostVector = thrust::host_vector<T>;

enum class CollisionType { SCATTERING, FISSION, CAPTURE };

class OpenMCCrossSectionReader;

struct NuclideComponent {

  NuclideComponent(char *name, unsigned int id, float atom_density,
                   float temperature, bool allow_fission)
      : _nuclide_id(id), _atom_dens(atom_density), _temperature(temperature),
        _allows_fission(allow_fission) {}

  char *_name;
  const unsigned int _nuclide_id;
  const float _atom_dens;
  const float _temperature;
  const bool _allows_fission;
};

/*
 * Templated Material class
 * T1 what type of cross-section data structure will be used for example
 * AoSLinear<float> T2 Numeric value type
 */
template <typename T1, typename T2> class Material {
public:
  Material(OpenMCCrossSectionReader &cross_section_reader);

  __host__ void addIsotope() = 0;

  __host__ void buildEnergyGrid(std::string isotope_name);

  __device__ void getMacroscopicXS(T2 *energy, T1 *cross_section);

  __device__ void decideIfCollide(T2 *energy, bool *collides);

  __device__ CollisionType decideCollideType(T2 *energy);

private:
  const OpenMCCrossSectionReader &_cross_section_reader;

  // Device vector of nuclides
  DeviceVector<NuclideComponent> _nuclides;

  // templated data struct. We will define when we declare the material class.
  DeviceVector<T1> _cross_section_data;
};

} // namespace neuxs

#endif // NEUXS_MATERIAL_H
