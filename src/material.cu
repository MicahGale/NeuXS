#include <stdexcept>
#include <vector>

#include "cross_section.cuh"
#include "cross_section_reader.h"
#include "material.cuh"

namespace neuxs {

// ============================================================================
//                          NuclideComponent
// ============================================================================

template <typename FPrecision>
__host__ __device__ NuclideComponent<FPrecision>::NuclideComponent()
    : _name(nullptr), _atom_dens(0), _temperature(0), _allows_fission(false) {}

template <typename FPrecision>
__host__ __device__ NuclideComponent<FPrecision>::NuclideComponent(
    const char *name, size_t A, FPrecision atom_density, FPrecision temperature,
    bool allow_fission)
    : _name(name), _atom_dens(atom_density), _temperature(temperature),
      _allows_fission(allow_fission) {
  FPrecision fraction =
      static_cast<FPrecision>(A - 1) / static_cast<FPrecision>(A + 1);
  this->_alpha = fraction * fraction;
}

// ============================================================================
//                              MaterialView
// ============================================================================

template <typename XSViewType, typename FPrecision>
__device__ FPrecision
MaterialView<XSViewType, FPrecision>::getMacroscopicSigmaT(
    FPrecision energy) const {
  FPrecision sigma_t = static_cast<FPrecision>(0);
  for (unsigned int i = 0; i < _num_isotopes; ++i) {
    auto grid = _xs_views[i].getCrossSection(energy);
    sigma_t += _nuclides[i]._atom_dens * grid._sigma_t;
  }
  return sigma_t;
}

// ============================================================================
//                                Material
// ============================================================================

template <typename XSClass, typename FPrecision>
Material<XSClass, FPrecision>::Material(
    OpenMCCrossSectionReader &cross_section_reader, unsigned int num_isotope)
    : _cross_section_reader(cross_section_reader), _num_isotopes(num_isotope) {
  _nuclides = new NuclideComponent<FPrecision>[_num_isotopes];
  _cross_section_data = new XSClass[_num_isotopes];
}

template <typename XSClass, typename FPrecision>
Material<XSClass, FPrecision>::~Material() {
  delete[] _nuclides;
  delete[] _cross_section_data;
}

template <typename XSClass, typename FPrecision>
__host__ void Material<XSClass, FPrecision>::addIsotope(
    NuclideComponent<FPrecision> isotope) {
  if (_temp_isotope_counter < _num_isotopes) {
    this->_nuclides[_temp_isotope_counter] = isotope;
    _cross_section_data[_temp_isotope_counter].setCrossSection(
        _cross_section_reader, isotope);
    _temp_isotope_counter++;

  } else
    std::runtime_error("you can load more than isotopes");
}

template <typename XSClass, typename FPrecision>
__host__ typename Material<XSClass, FPrecision>::ViewType *
Material<XSClass, FPrecision>::uploadToDevice() {
  if (_uploaded)
    return _d_self.get();

  // Upload each isotope's cross-section data and collect the views.
  // Upload the nuclide metadata and the xs-view array.
  // Build the view struct on host, then upload a single copy of it so
  // kernels can take its address.
  std::vector<XSViewType> xs_views_host;
  xs_views_host.reserve(_num_isotopes);
  for (unsigned int i = 0; i < _num_isotopes; ++i) {
    xs_views_host.push_back(_cross_section_data[i].uploadToDevice());
  }

  _d_nuclides = DeviceBuffer<NuclideComponent<FPrecision>>::makeFromHost(
      _nuclides, _num_isotopes);
  _d_xs_views = DeviceBuffer<XSViewType>::makeFromHost(xs_views_host.data(),
                                                       _num_isotopes);

  ViewType v;
  v._nuclides = _d_nuclides.get();
  v._xs_views = _d_xs_views.get();
  v._num_isotopes = _num_isotopes;
  _d_self = DeviceBuffer<ViewType>::makeSingle(v);

  _uploaded = true;
  return _d_self.get();
}

template <typename XSClass, typename FPrecision>
unsigned int Material<XSClass, FPrecision>::numIsotopes() const {
  return _num_isotopes;
}

// Explicit template instantiations

template struct NuclideComponent<float>;
template struct NuclideComponent<double>;

template struct MaterialView<AoSLinearView<float>, float>;
template struct MaterialView<AoSLinearView<double>, double>;
template struct MaterialView<SoALinearView<float>, float>;
template struct MaterialView<SoALinearView<double>, double>;
template struct MaterialView<LogarithmicHashAoSView<float>, float>;
template struct MaterialView<LogarithmicHashAoSView<double>, double>;

template class Material<AoSLinear<float>, float>;
template class Material<AoSLinear<double>, double>;
template class Material<SoALinear<float>, float>;
template class Material<SoALinear<double>, double>;
template class Material<LogarithmicHashAoS<float>, float>;
template class Material<LogarithmicHashAoS<double>, double>;

} // namespace neuxs
