#ifndef NEUXS_CROSS_SECTION_CUH
#define NEUXS_CROSS_SECTION_CUH

#include <cuda_runtime.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "hdf5.h"

namespace neuxs {

typedef thrust::device_vector<float> f_vec;

enum class CrossSectionDataType { ENERGY, SCATTERING, FISSION, CAPTURE, TOTAL };

__host__ inline int getMTNumber(CrossSectionDataType type) {
  switch (type) {
  case CrossSectionDataType::SCATTERING:
    return 2;
  case CrossSectionDataType::CAPTURE:
    return 102;
  case CrossSectionDataType::FISSION:
    return 18;
  default:
    return -1;
  }
}
template <typename T> class CrossSection {
public:
  __device__ void get_cross_section(f_vec *energy, f_vec *sigma_s,
                                    f_vec *sigma_c, f_vec *sigma_f,
                                    f_vec *sigma_t);
  __device__ void interpolate(float x1, float x2, float x_val, float y1,
                              float y2, float *y_val);
};

/*A wrapper class around for reading HDF5 cross-section data
 * mostly using hfd5 and openmc api */
class OpenMCCrossSectionReader {
public:
  explicit OpenMCCrossSectionReader(std::string cross_section_dir);

  std::vector<float> getEnergyDataPoints(const std::string &isotope_name,
                                         float temperature);

  std::vector<float> getCrossSectionDataPoints(const std::string &isotope_name,
                                               float temperature,
                                               CrossSectionDataType data_type);

  // Do I need to set the return type to host vector as well?
  // I need to ask Micah what he thinks.
  std::vector<float> readDataPointFromFile(const std::string &isotope_name,
                                           float temperature,
                                           CrossSectionDataType data_type);
  std::string buildFilePath(const std::string &isotope_name) const;
  std::string buildDatasetPath(float temperature,
                               CrossSectionDataType data_type,
                               int mt_number) const;
  void validateInputs(const std::string &isotope_name, float temperature) const;

private:
  const std::string cross_section_dir_;
  static constexpr std::string_view PARTICLE_TYPE = "neutron";
};

struct CrossSectionGridPoint {

  CrossSectionGridPoint(float energy, float sigma_s, float sigma_f,
                        float sigma_t, float sigma_c)
      : _energy(energy), _sigma_s(sigma_s), _sigma_f(sigma_f),
        _sigma_c(sigma_c), _sigma_t(sigma_t) {}

  // adding another constructor that automatically sets the total cross-section
  CrossSectionGridPoint(float energy, float sigma_s, float sigma_f,
                        float sigma_c)
      : _energy(energy), _sigma_s(sigma_s), _sigma_f(sigma_f),
        _sigma_c(sigma_c) {
    _sigma_t = _sigma_c + _sigma_f + _sigma_s;
  }

  float _energy;
  float _sigma_s;
  float _sigma_f;
  float _sigma_c;
  float _sigma_t;
};

class ArrayStructCrossSection : public CrossSection<ArrayStructCrossSection> {
public:
  __device__ void get_sigma(f_vec *energy, f_vec *sigma_s, f_vec *sigma_c,
                            f_vec *sigma_f, f_vec *sigma_t);

private:
  thrust::host_vector<CrossSectionGridPoint> _host_data;
  thrust::device_vector<CrossSectionGridPoint> _d_data;
};

/*array of struct.
 * A future todo note for us: We will come back and
 * implement struct of array data structure as well
 */
struct NuclideCrossSectionSet {

  NuclideCrossSectionSet(const std::vector<float> &energy,
                         const std::vector<float> &sigma_s,
                         const std::vector<float> &sigma_f,
                         const std::vector<float> &sigma_t,
                         const std::vector<float> &sigma_c);

  // check that length of every array are same.

  void preCheck(const std::vector<float> &energy,
                const std::vector<float> &sigma_s,
                const std::vector<float> &sigma_f,
                const std::vector<float> &sigma_t,
                const std::vector<float> &sigma_c);

  thrust::host_vector<CrossSectionGridPoint> _cross_section_grids;
  thrust::device_vector<CrossSectionGridPoint> _device_cross_section_grids;
};

class StructArrayCrossSection : CrossSection<StructArrayCrossSection> {
public:
  __device__ void get_sigma(f_vec *energy, f_vec *sigma_s, f_vec *sigma_c,
                            f_vec *sigma_f, f_vec *sigma_t);

private:
  NuclideCrossSectionSet _data;
};

__device__ void energy_binary_search(float *particle_energy,
                                     CrossSectionDataType reaction_type,
                                     float *cross_section);

/*This builds a 2D hashmap of the nuclides, and it's associated energy grid
 *
 * Nuclides/NuclideCrossSectionSets
 * N1 --E1---E2--E3--E4--E4
 * N2 --E1---E2--E4
 * N3 --E1--E3--E4
 *
 * This could be an acceleration technique when we sample reaction types.
 * We can do binary search for one nuclide get the energy grid index and use
 * that to look up cross-section for other nuclides.
 */
__host__ void build_nuclide_energy_grids(
    /*some argument but I am just a place-holder for now*/);

__device__ CrossSectionGridPoint
interpolate(CrossSectionGridPoint *grid_point_a,
            const CrossSectionGridPoint *grid_point_b);

__host__ void transfer_cross_section_data_to_device(unsigned int nuclide_index);

} // namespace neuxs

#endif // NEUXS_CROSS_SECTION_CUH
