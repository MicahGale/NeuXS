#include "reflective_pincell.cuh"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "./app")
              << " <aos|soa|log> <double|single>\n";
    return 1;
  }

  const std::string_view xs_type = argv[1];
  const std::string_view precision = argv[2];

  if (precision == "double")
    return pincell::dispatch_xs<double>(xs_type);
  if (precision == "single")
    return pincell::dispatch_xs<float>(xs_type);

  std::cerr << "Invalid precision: " << precision
            << " (expected double|single)\n";
  return 1;
}

template <typename XSDataStruct, typename FPrecision>
int pincell::run_simulation() {

  using Material = neuxs::NuclideComponent<XSDataStruct>;
  using Cell = neuxs::Cell<XSDataStruct, FPrecision>;
  neuxs::OpenMCCrossSectionReader reader;

  // TODO: build the model
  return 0;
}
