#ifndef NEUXS_CROSS_SECTION_READER_H
#define NEUXS_CROSS_SECTION_READER_H

#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
#include <vector>

#include "hdf5.h"

namespace neuxs {

enum class CrossSectionDataType { ENERGY, SCATTERING, FISSION, CAPTURE, TOTAL };

inline int getMTNumber(CrossSectionDataType type) {
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

/* Template helper to map C++ types to HDF5 types
 */
template <typename T> struct HDF5TypeTraits;

template <> struct HDF5TypeTraits<float> {
  static hid_t get_type() { return H5T_NATIVE_FLOAT; }
};

template <> struct HDF5TypeTraits<double> {
  static hid_t get_type() { return H5T_NATIVE_DOUBLE; }
};

/*  A templated wrapper class for reading HDF5 cross-section data
 * using HDF5 and OpenMC API. Supports float and double types.
 */

class OpenMCCrossSectionReader {
public:
  /* This constructor will look for the
   * OPENMC_CROSS_SECTIONS environment variable in the system
   */
  OpenMCCrossSectionReader();

  explicit OpenMCCrossSectionReader(std::string cross_section_dir);

  template <typename T>
  std::vector<T> getEnergyDataPoints(const std::string &isotope_name,
                                     T temperature);

  template <typename T>
  std::vector<T> getCrossSectionDataPoints(const std::string &isotope_name,
                                           T temperature,
                                           CrossSectionDataType data_type);
  template <typename T>
  std::vector<T> readDataPointFromFile(const std::string &isotope_name,
                                       T temperature,
                                       CrossSectionDataType data_type);

  std::string buildFilePath(const std::string &isotope_name) const;

  std::string buildDatasetPath(float temperature,
                               CrossSectionDataType data_type,
                               int mt_number) const;
  void validateInputs(const std::string &isotope_name, float temperature) const;

private:
  std::string processSystemCrossSectionEnv();
  const std::string _cross_section_dir;
};

} // namespace neuxs
#endif // NEUXS_CROSS_SECTION_READER_H