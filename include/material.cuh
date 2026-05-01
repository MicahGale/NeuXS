#ifndef NEUXS_MATERIAL_H
#define NEUXS_MATERIAL_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "memory.cuh"
#include "transport.cuh"

namespace neuxs {
enum class CollisionType { SCATTERING, FISSION, CAPTURE };

class OpenMCCrossSectionReader;

// Forward declaration
template <typename FPrecision> struct CrossSectionGridPoint;
template <typename FPrecision> struct Particle;

template <typename FPrecision> struct NuclideComponent {
  __host__ __device__ NuclideComponent();

  __host__ __device__ NuclideComponent(const char *name, size_t A,
                                       FPrecision atom_density,
                                       FPrecision temperature,
                                       bool allow_fission);

  const char *_name;
  FPrecision _atom_dens;
  FPrecision _temperature;
  bool _allows_fission;
  FPrecision _alpha;
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

  /*
   * first we sample the nuclide reaction type using a random_number.
   * total_sigma_t_of_material at (E)
   * auto sigma_t = 0;
   * for (size_t nuclide_index =0 ; nuclide_index < this->_num_isotopes;
   * nuclide_index++ ){ sigma_t +=
   * _xs_view[nuclide_index]->getTotalSigmaT(particle._energy); if
   * (random_number > sigma_t/total_sigma_t_of_material){ break; may not be
   * the best idea as we are gonna get thread divergence but then again my
   * loop isn't that big. So maybe it shouldn't matter
   *   }
   * }
   * then we sample the reaction type for which can just do it by microscopic
   * xs section
   *
   * */
  __device__ CollisionType decideCollideType(Particle<FPrecision> part) {

    FPrecision sigma_t_mat = this->getMacroscopicSigmaT(part._energy);
    FPrecision sigma_t_cumulative = 0;
    CrossSectionGridPoint<FPrecision> collision_nuclide_xs_grid;
    auto rand_num = part._rng.nextFloat();
    for (size_t nuclide_index = 0; nuclide_index < this->_num_isotopes;
         nuclide_index++) {

      collision_nuclide_xs_grid =
          this->_xs_views[nuclide_index].getCrossSection(part._energy);
      sigma_t_cumulative += collision_nuclide_xs_grid._sigma_t;
      if (rand_num < sigma_t_cumulative / sigma_t_mat)
        break;
    }

    // now we have
  }
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
