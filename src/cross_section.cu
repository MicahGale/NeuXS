#include "cross_section.cuh"
#include <algorithm>
#include <filesystem>

namespace neuxs {

template <typename T>
__device__ void
CrossSection<T>::get_cross_section(neuxs::f_vec *energy, neuxs::f_vec *sigma_s,
                                   neuxs::f_vec *sigma_c, neuxs::f_vec *sigma_f,
                                   neuxs::f_vec *sigma_t) {
  return static_cast<T>(this)->get_sigma(energy, sigma_s, sigma_c, sigma_f,
                                         sigma_t);
}

template <typename T>
__device__ void CrossSection<T>::interpolate(float x1, float x2, float x_val,
                                             float y1, float y2, float *y_val) {
  float x_delta = x2 - x1;
  float y_delta = y2 - y1;
  *y_val = y1 + y_delta * (x_val - x1) / x_delta;
}

OpenMCCrossSectionReader::OpenMCCrossSectionReader()
    : _cross_section_dir(processSystemCrossSectionEnv()) {
  if (_cross_section_dir.empty()) {
    throw std::invalid_argument("Cross-section path cannot be empty");
  }
}

OpenMCCrossSectionReader::OpenMCCrossSectionReader(
    std::string cross_section_dir)
    : _cross_section_dir(std::move(cross_section_dir)) {
  if (_cross_section_dir.empty()) {
    throw std::invalid_argument("Cross-section path cannot be empty");
  }
}

std::vector<float>
OpenMCCrossSectionReader::getEnergyDataPoints(const std::string &isotope_name,
                                              float temperature) {

  validateInputs(isotope_name, temperature);

  auto data = readDataPointFromFile(isotope_name, temperature,
                                    CrossSectionDataType::ENERGY);

  std::vector<float> result(data.size());
  std::transform(data.begin(), data.end(), result.begin(),
                 [](double d) { return static_cast<float>(d); });

  return result;
}

std::vector<float> OpenMCCrossSectionReader::getCrossSectionDataPoints(
    const std::string &isotope_name, float temperature,
    CrossSectionDataType data_type) {

  if (data_type == CrossSectionDataType::ENERGY)
    throw std::invalid_argument("Use getEnergyDataPoints for Energy data type");

  validateInputs(isotope_name, temperature);

  auto data = readDataPointFromFile(isotope_name, temperature, data_type);

  std::vector<float> result(data.size());
  std::transform(data.begin(), data.end(), result.begin(),
                 [](float d) { return static_cast<float>(d); });

  return result;
}

std::vector<float> OpenMCCrossSectionReader::readDataPointFromFile(
    const std::string &isotope_name, float temperature,
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
  return _cross_section_dir + "/" + isotope_name + ".h5";
}

std::string OpenMCCrossSectionReader::buildDatasetPath(
    float temperature, CrossSectionDataType data_type, int mt_number) const {

  std::string temp_str = std::to_string(static_cast<int>(temperature)) + "K";

  if (data_type == CrossSectionDataType::ENERGY)
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

std::string OpenMCCrossSectionReader::processSystemCrossSectionEnv() {
  char *xml_path = std::getenv("OPENMC_CROSS_SECTIONS");
  if (xml_path) {
    std::string dir_path = std::filesystem::path(xml_path).parent_path().string();
    return  (std::filesystem::path(dir_path) / "neutron").string();
  }
  return std::string{};
}

NuclideCrossSectionSet::NuclideCrossSectionSet(
    const std::vector<float> &energy, const std::vector<float> &sigma_s,
    const std::vector<float> &sigma_f, const std::vector<float> &sigma_t,
    const std::vector<float> &sigma_c) {

  preCheck(energy, sigma_s, sigma_f, sigma_t, sigma_c);

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
