#ifndef NEUXS_CROSS_SECTION_READER_H
#define NEUXS_CROSS_SECTION_READER_H

#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include "hdf5.h"

namespace neuxs {

extern "C" enum class CrossSectionDataType {
  ENERGY,
  SCATTERING,
  FISSION,
  CAPTURE,
  TOTAL
};

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

/*  A wrapper class around for reading HDF5 cross-section data
 * mostly using hfd5 and openmc api
 */
class OpenMCCrossSectionReader {
public:
  /*This constructor reader will look for the
   * OPENMC_CROSS_SECTIONS ENV variable in the system
   */
  OpenMCCrossSectionReader();

  explicit OpenMCCrossSectionReader(std::string cross_section_dir);

  std::vector<float> getEnergyDataPoints(const std::string &isotope_name,
                                         float temperature);

  std::vector<float> getCrossSectionDataPoints(const std::string &isotope_name,
                                               float temperature,
                                               CrossSectionDataType data_type);

  std::vector<float> readDataPointFromFile(const std::string &isotope_name,
                                           float temperature,
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
