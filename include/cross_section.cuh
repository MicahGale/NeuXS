#ifndef NEUXS_CROSS_SECTION_CUH
#define NEUXS_CROSS_SECTION_CUH

#include <cuda_runtime.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "hdf5.h"

namespace neuxs {

template <typename T> using DeviceVector = thrust::device_vector<T>;
template <typename T> using HostVector = thrust::host_vector<T>;

enum class CrossSectionDataType { ENERGY, SCATTERING, FISSION, CAPTURE, TOTAL };

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

template <typename T> class CrossSection {
public:
  __device__ void getCrossSection(DeviceVector<float> *energy,
                                  DeviceVector<float> *sigma_s,
                                  DeviceVector<float> *sigma_c,
                                  DeviceVector<float> *sigma_f,
                                  DeviceVector<float> *sigma_t);
  __device__ void interpolate(float x1, float x2, float x_val, float y1,
                              float y2, float *y_val);
};

class ArrayStructCrossSection : public CrossSection<ArrayStructCrossSection> {
public:
  __device__ void get_sigma(DeviceVector<float> *energy,
                            DeviceVector<float> *sigma_s,
                            DeviceVector<float> *sigma_c,
                            DeviceVector<float> *sigma_f,
                            DeviceVector<float> *sigma_t);

private:
  HostVector<CrossSectionGridPoint> _host_data;
  DeviceVector<CrossSectionGridPoint> _device_data;
};

struct NuclideCrossSectionSet {

  NuclideCrossSectionSet(const std::vector<float> &energy,
                         const std::vector<float> &sigma_s,
                         const std::vector<float> &sigma_f,
                         const std::vector<float> &sigma_t,
                         const std::vector<float> &sigma_c);

  void preCheck(const std::vector<float> &energy,
                const std::vector<float> &sigma_s,
                const std::vector<float> &sigma_f,
                const std::vector<float> &sigma_t,
                const std::vector<float> &sigma_c);

  HostVector<CrossSectionGridPoint> _cross_section_grids;
  DeviceVector<CrossSectionGridPoint> _device_cross_section_grids;
};

class StructArrayCrossSection : CrossSection<StructArrayCrossSection> {
public:
  __device__ void get_sigma(DeviceVector<float> *energy,
                            DeviceVector<float> *sigma_s,
                            DeviceVector<float> *sigma_c,
                            DeviceVector<float> *sigma_f,
                            DeviceVector<float> *sigma_t);

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
