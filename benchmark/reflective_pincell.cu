// neuxs includes
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "timer.cuh"
#include "transport.cuh"

#include "reflective_pincell.cuh"

/* Usage:
 *   ./this_executable(I need to create another cmake for benchmark)
 * <aos|soa|log> <double|single>
 */

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
