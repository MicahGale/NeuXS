#include "cross_section.cuh"
#include <algorithm>

namespace neuxs {

OpenMCCrossSectionReader::OpenMCCrossSectionReader(
    std::string cross_section_dir)
    : cross_section_dir_(std::move(cross_section_dir)) {
  if (cross_section_dir_.empty()) {
    throw std::invalid_argument("Cross-section path cannot be empty");
  }
}

std::vector<float>
OpenMCCrossSectionReader::getEnergyDataPoints(const std::string &isotope_name,
                                              float temperature) {

  validateInputs(isotope_name, temperature);

  auto data = readDataPointFromFile(isotope_name, temperature, CrossSectionDataType::Energy);

  std::vector<float> result(data.size());
  std::transform(data.begin(), data.end(), result.begin(), [](double d) { return static_cast<float>(d); });

  return result;
}

std::vector<float> OpenMCCrossSectionReader::getCrossSectionDataPoints(
    const std::string &isotope_name, float temperature,
    CrossSectionDataType data_type) {

  if (data_type == CrossSectionDataType::Energy)
    throw std::invalid_argument("Use getEnergyDataPoints for Energy data type");



  validateInputs(isotope_name, temperature);

  auto data = readDataPointFromFile(isotope_name, temperature,data_type);

  std::vector<float> result(data.size());
  std::transform(data.begin(), data.end(), result.begin(), [](float d) { return static_cast<float>(d); });

  return result;
}

std::vector<float> OpenMCCrossSectionReader::readDataPointFromFile(
    const std::string &isotope_name,
    float temperature, CrossSectionDataType data_type) {

  std::string file_path = buildFilePath(isotope_name);
  hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (file_id < 0) {
    throw std::runtime_error("Failed to open file: " + file_path);
  }

  int mt_number = 0;
  if (data_type != CrossSectionDataType::Energy) {
    mt_number = getMTNumber(data_type);
  }

  std::string dataset_path = isotope_name + buildDatasetPath(temperature, data_type, mt_number);
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

  std::vector<float> data(total_size);
  herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, data.data());

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  if (status < 0) {
    throw std::runtime_error("Failed to read dataset");
  }

  return data;
}

std::string
OpenMCCrossSectionReader::buildFilePath(const std::string &isotope_name) const {
  return cross_section_dir_ + "/" + isotope_name + ".h5";
}

std::string OpenMCCrossSectionReader::buildDatasetPath(
    float temperature, CrossSectionDataType data_type, int mt_number) const {

  std::string temp_str = std::to_string(static_cast<int>(temperature)) + "K";

  if (data_type == CrossSectionDataType::Energy)
    return "/energy/" + temp_str;

  // I miss python now :)
  std::string mt_number_converted_to_string;
  if (mt_number < 10)
    mt_number_converted_to_string = "00" + std::to_string(mt_number);
  else if (10 <= mt_number and mt_number < 100)
    mt_number_converted_to_string = "0" + std::to_string(mt_number);
  else
    mt_number_converted_to_string = std::to_string(mt_number);

  return "/reactions/reaction_" + mt_number_converted_to_string + "/" +
         temp_str + "/xs";
}

void OpenMCCrossSectionReader::validateInputs(const std::string &isotope_name,
                                              float temperature) const {

  if (isotope_name.empty()) {
    throw std::invalid_argument("Isotope name cannot be empty");
  }

  if (temperature < 0.0f) {
    throw std::invalid_argument("Temperature must be positive");
  }
}

NuclideCrossSectionSet::NuclideCrossSectionSet(
    const unsigned int material_id, const std::vector<float> &energy,
    const std::vector<float> &sigma_s, const std::vector<float> &sigma_f,
    const std::vector<float> &sigma_t, const std::vector<float> &sigma_c) {

  preCheck(energy, sigma_s, sigma_f, sigma_t, sigma_c);

  _material_id = material_id;
  const auto size = energy.size();
  _cross_section_grids.reserve(size);

  for (size_t i = 0; i < size; i++)
    _cross_section_grids.push_back(CrossSectionGridPoint(
        energy[i], sigma_s[i], sigma_f[i], sigma_t[i], sigma_c[i]));

  _cross_section_grids.shrink_to_fit();
}

void NuclideCrossSectionSet::preCheck(const std::vector<float> &energy,
                                      const std::vector<float> &sigma_s,
                                      const std::vector<float> &sigma_f,
                                      const std::vector<float> &sigma_t,
                                      const std::vector<float> &sigma_c) {

  const auto n = energy.size();
  const bool check = (sigma_s.size() == n && sigma_f.size() == n &&
                      sigma_t.size() == n && sigma_c.size() == n);

  if (!check)
    throw std::runtime_error(" Invalid cross section data point. Size of "
                             "reaction data don't match with each other");
}
}; // namespace neuxs