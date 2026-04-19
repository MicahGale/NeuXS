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

template <typename FPrecision> struct NuclideComponent {

  NuclideComponent(char *name, FPrecision atom_density, FPrecision temperature,
                   bool allow_fission)
      : _name(name), _atom_dens(atom_density), _temperature(temperature),
        _allows_fission(allow_fission) {}

  char *_name;
  const FPrecision _atom_dens;
  const FPrecision _temperature;
  const FPrecision _allows_fission;
  const float _alpha;
};

/*
 * Templated Material class
 * XSType what type of cross-section data structure will be used for example
 * AoSLinear<float> FPrecision Numeric value type
 */

template <typename FPrecision> struct Collision {
  Collision(CollisionType type, NuclideComponent<FPrecision> *nuclide)
      : _type(type), _nuclide(nuclide) {}
  CollisionType _type;
  NuclideComponent<FPrecision> *_nuclide;
};

template <typename XSType, typename FPrecision> class Material {
public:
  Material(OpenMCCrossSectionReader &cross_section_reader);

  __host__ void addIsotope() = 0;

  __host__ void buildEnergyGrid(std::string isotope_name);

  __device__ void getMacroscopicXS(FPrecision *energy,
                                   FPrecision *cross_section);

  __device__ void decideIfCollide(FPrecision *energy, bool *collides);

  __device__ CollisionType decideCollideType(FPrecision *energy);

private:
  const OpenMCCrossSectionReader &_cross_section_reader;

  // Device vector of nuclides
  DeviceVector<NuclideComponent<FPrecision>> _nuclides;

  // templated data struct. We will define when we declare the material class.
  DeviceVector<XSType> _cross_section_data;
};

// explicit def

} // namespace neuxs

#endif // NEUXS_MATERIAL_H
