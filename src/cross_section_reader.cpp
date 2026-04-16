#include <algorithm>
#include <filesystem>
#include <vector>

#include "cross_section_reader.h"

namespace neuxs {

template <typename T>
OpenMCCrossSectionReader<T>::OpenMCCrossSectionReader()
    : _cross_section_dir(processSystemCrossSectionEnv()) {
  if (_cross_section_dir.empty()) {
    throw std::invalid_argument("Cross-section path cannot be empty");
  }
}

template <typename T>
OpenMCCrossSectionReader<T>::OpenMCCrossSectionReader(
    std::string cross_section_dir)
    : _cross_section_dir(std::move(cross_section_dir)) {
  if (_cross_section_dir.empty()) {
    throw std::invalid_argument("Cross-section path cannot be empty");
  }
}

template <typename T>
std::vector<T> OpenMCCrossSectionReader<T>::getEnergyDataPoints(
    const std::string &isotope_name, T temperature) {

  validateInputs(isotope_name, temperature);

  auto data = readDataPointFromFile(isotope_name, temperature,
                                    CrossSectionDataType::ENERGY);

  return data;
}

template <typename T>
std::vector<T> OpenMCCrossSectionReader<T>::getCrossSectionDataPoints(
    const std::string &isotope_name, T temperature,
    CrossSectionDataType data_type) {

  if (data_type == CrossSectionDataType::ENERGY) {
    throw std::invalid_argument("Use getEnergyDataPoints for Energy data type");
  }

  validateInputs(isotope_name, temperature);

  auto data = readDataPointFromFile(isotope_name, temperature, data_type);

  return data;
}

template <typename T>
std::vector<T> OpenMCCrossSectionReader<T>::readDataPointFromFile(
    const std::string &isotope_name, T temperature,
    CrossSectionDataType data_type) {

  std::string file_path = buildFilePath(isotope_name);
  hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (file_id < 0) {
    throw std::runtime_error("Failed to open file: " + file_path);
  }

  int mt_number = 0;
  if (data_type != CrossSectionDataType::ENERGY) {
    mt_number = getMTNumber(data_type);
  }

  std::string dataset_path =
      isotope_name + buildDatasetPath(temperature, data_type, mt_number);
  hid_t dataset_id = H5Dopen(file_id, dataset_path.c_str(), H5P_DEFAULT);

  if (dataset_id < 0) {
    H5Fclose(file_id);
    throw std::runtime_error("Failed to open dataset: " + dataset_path);
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);

  int n_dims = H5Sget_simple_extent_ndims(dataspace_id);
  std::vector<hsize_t> dims(n_dims);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

  size_t total_size = 1;
  for (auto d : dims) {
    total_size *= d;
  }

  std::vector<T> data(total_size);
  herr_t status = H5Dread(dataset_id, HDF5TypeTraits<T>::get_type(), H5S_ALL,
                          H5S_ALL, H5P_DEFAULT, data.data());

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  if (status < 0) {
    throw std::runtime_error("Failed to read dataset");
  }

  return data;
}

template <typename T>
std::string OpenMCCrossSectionReader<T>::buildFilePath(
    const std::string &isotope_name) const {
  return _cross_section_dir + "/" + isotope_name + ".h5";
}

template <typename T>
std::string OpenMCCrossSectionReader<T>::buildDatasetPath(
    T temperature, CrossSectionDataType data_type, int mt_number) const {

  std::string temp_str = std::to_string(static_cast<int>(temperature)) + "K";

  if (data_type == CrossSectionDataType::ENERGY) {
    return "/energy/" + temp_str;
  }

  std::string mt_number_converted_to_string;
  if (mt_number < 10) {
    mt_number_converted_to_string = "00" + std::to_string(mt_number);
  } else if (mt_number < 100) {
    mt_number_converted_to_string = "0" + std::to_string(mt_number);
  } else {
    mt_number_converted_to_string = std::to_string(mt_number);
  }

  return "/reactions/reaction_" + mt_number_converted_to_string + "/" +
         temp_str + "/xs";
}

template <typename T>
void OpenMCCrossSectionReader<T>::validateInputs(
    const std::string &isotope_name, T temperature) const {

  if (isotope_name.empty()) {
    throw std::invalid_argument("Isotope name cannot be empty");
  }

  if (temperature < 0.0) {
    throw std::invalid_argument("Temperature must be positive");
  }
}

template <typename T>
std::string OpenMCCrossSectionReader<T>::processSystemCrossSectionEnv() {
  char *xml_path = std::getenv("OPENMC_CROSS_SECTIONS");
  if (xml_path) {
    std::string dir_path =
        std::filesystem::path(xml_path).parent_path().string();
    return (std::filesystem::path(dir_path) / "neutron").string();
  }
  return std::string{};
}

// Explicit template instantiations for float and double
// other-wise compiler will complain as it will have abs no idea
// what type it will be.
template class OpenMCCrossSectionReader<float>;
template class OpenMCCrossSectionReader<double>;

} // namespace neuxs