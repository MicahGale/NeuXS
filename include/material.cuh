#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "memory.cuh"

namespace neuxs {
enum class CollisionType { SCATTERING, FISSION, CAPTURE };

class OpenMCCrossSectionReader;

// Forward declaration
template <typename FPrecision> struct CrossSectionGridPoint;

template <typename FPrecision> struct NuclideComponent {
  __host__ __device__ NuclideComponent();

  __host__ __device__ NuclideComponent(const char *name,
                                       FPrecision atom_density,
                                       FPrecision temperature,
                                       bool allow_fission);

  const char *_name;
  FPrecision _atom_dens;
  FPrecision _temperature;
  bool _allows_fission;
};

// ==================== MaterialView (device-facing POD) =====================
// ===========================================================================
/*
 * Kernel-facing view of a Material. All pointers are device pointers and are
 * backed by DeviceBuffers owned by the host-side Material.
 *
 * Usage from a kernel (assuming XSViewType = AoSLinearView<double>):
 *
 *   auto& nuc  = material_view->_nuclides[i];     // NuclideComponent
 *   auto& xs   = material_view->_xs_views[i];     // the i-th view
 *   auto  grid = xs.getCrossSection(energy);      // device method
 *
 * `_nuclides` and `_xs_views` are parallel arrays (same index → same isotope).
 */
template <typename XSViewType, typename FPrecision> struct MaterialView {
  NuclideComponent<FPrecision> *_nuclides; // device ptr, length = _num_isotopes
  XSViewType *_xs_views;                   // device ptr, length = _num_isotopes
  unsigned int _num_isotopes;

  // Macroscopic total XS at a given energy:
  __device__ FPrecision getMacroscopicSigmaT(FPrecision energy) const;

  // Full macroscopic reaction breakdown — used when deciding which reaction
  // channel fires after a collision is known to occur.
  __device__ CrossSectionGridPoint<FPrecision>
  getMacroscopicXS(FPrecision energy) const;
};

/*
 * Templated Material class
 * XSType what type of cross-section data structure will be used for example
 * AoSLinear<float> FPrecision Numeric value type
 */
template <typename XSClass, typename FPrecision> class Material {
public:
  // Derive view types from the cross-section class's associated ViewType.
  // This is how new XS schemes plug in without changing Material at all.
  using XSViewType = typename XSClass::ViewType;
  using ViewType = MaterialView<XSViewType, FPrecision>;

  Material(OpenMCCrossSectionReader &cross_section_reader,
           unsigned int num_isotope);

  ~Material();

  // Non-copyable (owns raw arrays and device buffers).
  Material(const Material &) = delete;
  Material &operator=(const Material &) = delete;

  __host__ void addIsotope(NuclideComponent<FPrecision> isotope);

  /*
   * Deep-copy this material onto the device and return a device pointer to
   * its kernel-facing MaterialView. Idempotent — subsequent calls return the
   * same pointer without re-uploading. The device memory is owned by the
   * DeviceBuffer members and released when *this* object is destroyed.
   */
  __host__ ViewType *uploadToDevice();

  __device__ void getMacroscopicXS(FPrecision *energy,
                                   FPrecision *cross_section);

  __device__ void decideIfCollide(FPrecision *energy, bool *collides);

  __device__ CollisionType decideCollideType(FPrecision *energy);

  __host__ void setCrossSection(NuclideComponent<FPrecision> isotope);

  unsigned int numIsotopes() const;

  const OpenMCCrossSectionReader &_cross_section_reader;

  // Device vector of nuclides
  NuclideComponent<FPrecision> *_nuclides;

  // templated cross-section data struct
  XSClass *_cross_section_data;

  // ---------- device-side backing storage ----------
  DeviceBuffer<NuclideComponent<FPrecision>> _d_nuclides;
  DeviceBuffer<XSViewType> _d_xs_views;
  DeviceBuffer<ViewType> _d_self;

private:
  const unsigned int _num_isotopes;
  unsigned int _temp_isotope_counter = 0;
  bool _uploaded = false;
};

} // namespace neuxs

#endif // NEUXS_MATERIAL_H
