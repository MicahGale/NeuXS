#include "cross_section.cuh"
#include "material.cuh"

namespace neuxs {

template class Material<AoSLinear<float>, float>;
template class Material<AoSLinear<double>, double>;
// template class Material<SoALinear<float>, float>;
// template class Material<SoALinear<double>, double>;

template <typename XSClass, typename FPrecision>
__host__ void Material<XSClass, FPrecision>::setCrossSection(
    NuclideComponent<FPrecision> isotope) {
  // Resize cross-section data to match number of nuclides
  _cross_section_data.resize(_nuclides.size());
  XSClass xs;
  xs.setCrossSection(_cross_section_reader, isotope);
  std::cout << xs._device_data.size();
  _cross_section_data[_nuclides.size() - 1] = xs;
}

} // namespace neuxs