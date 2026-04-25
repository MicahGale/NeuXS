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

// Forward declaration so MaterialView's device methods can reference the
// return type without pulling in cross_section.cuh (which itself includes
// material.cuh).
template <typename FPrecision> struct CrossSectionGridPoint;

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
  __device__ __forceinline__ FPrecision
  getMacroscopicSigmaT(FPrecision energy) const {
    FPrecision sigma_t = static_cast<FPrecision>(0);
    for (unsigned int i = 0; i < _num_isotopes; ++i) {
      auto grid = _xs_views[i].getCrossSection(energy);
      sigma_t += _nuclides[i]._atom_dens * grid._sigma_t;
    }
    return sigma_t;
  }

  // Full macroscopic reaction breakdown — used when deciding which reaction
  // channel fires after a collision is known to occur.
  __device__ __forceinline__ CrossSectionGridPoint<FPrecision>
  getMacroscopicXS(FPrecision energy) const {
    CrossSectionGridPoint<FPrecision> total;
    total._sigma_s = static_cast<FPrecision>(0);
    total._sigma_f = static_cast<FPrecision>(0);
    total._sigma_c = static_cast<FPrecision>(0);
    total._sigma_t = static_cast<FPrecision>(0);

    for (unsigned int i = 0; i < _num_isotopes; ++i) {
      auto grid = _xs_views[i].getCrossSection(energy);
      FPrecision N = _nuclides[i]._atom_dens;
      total._sigma_s += N * grid._sigma_s;
      total._sigma_f += N * grid._sigma_f;
      total._sigma_c += N * grid._sigma_c;
      total._sigma_t += N * grid._sigma_t;
    }
    return total;
  }
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
           unsigned int num_isotope)
      : _cross_section_reader(cross_section_reader),
        _num_isotopes(num_isotope) {

    _nuclides = new NuclideComponent<FPrecision>[_num_isotopes];
    _cross_section_data = new XSClass[_num_isotopes];
  }

  ~Material() {
    delete[] _nuclides;
    delete[] _cross_section_data;
  }

  // Non-copyable (owns raw arrays and device buffers).
  Material(const Material &) = delete;
  Material &operator=(const Material &) = delete;

  __host__ void addIsotope(NuclideComponent<FPrecision> isotope) {
    if (_temp_isotope_counter < _num_isotopes) {
      this->_nuclides[_temp_isotope_counter] = isotope;
      _cross_section_data[_temp_isotope_counter].setCrossSection(
          _cross_section_reader, isotope);
      _temp_isotope_counter++;

    } else
      std::runtime_error("you can load more than isotopes");
  }

  /*
   * Deep-copy this material onto the device and return a device pointer to
   * its kernel-facing MaterialView. Idempotent — subsequent calls return the
   * same pointer without re-uploading. The device memory is owned by the
   * DeviceBuffer members and released when *this* object is destroyed.
   */
  __host__ ViewType *uploadToDevice() {
    if (_uploaded)
      return _d_self.get();

    // 1) Upload each isotope's cross-section data and collect the views.
    std::vector<XSViewType> xs_views_host;
    xs_views_host.reserve(_num_isotopes);
    for (unsigned int i = 0; i < _num_isotopes; ++i) {
      xs_views_host.push_back(_cross_section_data[i].uploadToDevice());
    }

    // 2) Upload the nuclide metadata and the xs-view array.
    _d_nuclides = DeviceBuffer<NuclideComponent<FPrecision>>::makeFromHost(
        _nuclides, _num_isotopes);
    _d_xs_views = DeviceBuffer<XSViewType>::makeFromHost(xs_views_host.data(),
                                                         _num_isotopes);

    // 3) Build the view struct on host, then upload a single copy of it so
    //    kernels can take its address.
    ViewType v;
    v._nuclides = _d_nuclides.get();
    v._xs_views = _d_xs_views.get();
    v._num_isotopes = _num_isotopes;
    _d_self = DeviceBuffer<ViewType>::makeSingle(v);

    _uploaded = true;
    return _d_self.get();
  }

  __device__ void getMacroscopicXS(FPrecision *energy,
                                   FPrecision *cross_section);

  __device__ void decideIfCollide(FPrecision *energy, bool *collides);

  __device__ CollisionType decideCollideType(FPrecision *energy);

  __host__ void setCrossSection(NuclideComponent<FPrecision> isotope);

  unsigned int numIsotopes() const { return _num_isotopes; }

  const OpenMCCrossSectionReader &_cross_section_reader;

  // Device vector of nuclides
  NuclideComponent<FPrecision> *_nuclides;

  // templated data struct. We will define when we declare the material class.
  XSClass *_cross_section_data;

  // ---------- device-side backing storage (owned via RAII) ----------
  DeviceBuffer<NuclideComponent<FPrecision>> _d_nuclides;
  DeviceBuffer<XSViewType> _d_xs_views;
  DeviceBuffer<ViewType> _d_self;

private:
  const unsigned int _num_isotopes;
  unsigned int _temp_isotope_counter = 0;
  bool _uploaded = false;
};

// explicit def

} // namespace neuxs

#endif // NEUXS_MATERIAL_H
