#include "cross_section.cuh"
#include "material.cuh"

namespace neuxs {

template class Material<AoSLinear<float>, float>;
template class Material<AoSLinear<double>, double>;
template class Material<SoALinear<float>, float>;
template class Material<SoALinear<double>, double>;

template <typename XSClass, typename FPrecision>
__host__ void Material<XSClass, FPrecision>::setCrossSection(
    NuclideComponent<FPrecision> isotope) {}

} // namespace neuxs