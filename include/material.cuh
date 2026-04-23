#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#include <cuda_runtime.h>
#include <string>

namespace neuxs {
enum class CollisionType { SCATTERING, FISSION, CAPTURE };

class OpenMCCrossSectionReader;

template <typename FPrecision> struct NuclideComponent {

  __host__ __device__ NuclideComponent()
      : _name(nullptr), _atom_dens(0), _temperature(0), _allows_fission(false) {
  }

  __host__ __device__ NuclideComponent(const char *name,
                                       FPrecision atom_density,
                                       FPrecision temperature,
                                       bool allow_fission)
      : _name(name), _atom_dens(atom_density), _temperature(temperature),
        _allows_fission(allow_fission) {}

  const char *_name;
  FPrecision _atom_dens;
  FPrecision _temperature;
  bool _allows_fission;
};

/*
 * Templated Material class
 * XSType what type of cross-section data structure will be used for example
 * AoSLinear<float> FPrecision Numeric value type
 */
template <typename XSClass, typename FPrecision> class Material {
public:
  Material(OpenMCCrossSectionReader &cross_section_reader,
           unsigned int num_isotope)
      : _cross_section_reader(cross_section_reader),
        _num_isotopes(num_isotope) {}

  __host__ void addIsotope(NuclideComponent<FPrecision> isotope) {}

  __device__ void getMacroscopicXS(FPrecision *energy,
                                   FPrecision *cross_section);

  __device__ void decideIfCollide(FPrecision *energy, bool *collides);

  __device__ CollisionType decideCollideType(FPrecision *energy);

  __host__ void setCrossSection(NuclideComponent<FPrecision> isotope);

  const OpenMCCrossSectionReader &_cross_section_reader;

  // Device vector of nuclides
  NuclideComponent<FPrecision> *_nuclides;

  // templated data struct. We will define when we declare the material class.
  XSClass *_cross_section_data;

  const unsigned int _num_isotopes;
};

// explicit def

} // namespace neuxs

#endif // NEUXS_MATERIAL_H